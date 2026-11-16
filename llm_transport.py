"""Verified TLS, phase-aware timeouts, and sanitized transport diagnostics."""
import argparse
import errno
from functools import partial
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, HTTPSHandler, getproxies, proxy_bypass


class ConnectionTimeout(TimeoutError):
    pass


class ReadTimeout(TimeoutError):
    pass


class TLSConfigurationError(ValueError):
    pass


def create_tls_context(ca_bundle=None):
    """Keep system/explicit trust. Add certifi only when default trust is empty."""
    explicit_file = ca_bundle or os.environ.get('SSL_CERT_FILE')
    explicit_directory = os.environ.get('SSL_CERT_DIR')
    try:
        if explicit_file and not Path(explicit_file).is_file():
            raise TLSConfigurationError('Configured CA bundle is not a readable file; check --ca-bundle or SSL_CERT_FILE')
        if explicit_directory and not Path(explicit_directory).is_dir():
            raise TLSConfigurationError('Configured SSL_CERT_DIR is not a directory')
        context = ssl.create_default_context(cafile=explicit_file) if explicit_file else ssl.create_default_context()
        source = 'explicit_bundle' if ca_bundle else ('ssl_environment' if explicit_file or explicit_directory else 'system')
        # macOS python.org installations can have missing OpenSSL CA paths even
        # though curl's SecureTransport succeeds using the macOS trust store.
        if not context.cert_store_stats()['x509_ca'] and not explicit_file and not explicit_directory:
            import certifi
            context.load_verify_locations(cafile=certifi.where())
            source = 'certifi_fallback'
    except TLSConfigurationError:
        raise
    except (OSError, ValueError, ImportError):
        raise TLSConfigurationError('Could not initialize verified TLS trust; install requirements or supply a valid --ca-bundle') from None
    return context, {'trust_source': source, 'ca_certificates': context.cert_store_stats()['x509_ca']}


class TimeoutPhases:
    def __init__(self, *args, read_timeout=None, **kwargs):
        self.read_timeout = read_timeout
        super().__init__(*args, **kwargs)

    def connect(self):
        try:
            super().connect()
        except TimeoutError:
            raise ConnectionTimeout('Connection/TLS setup timed out') from None
        if self.sock is not None:
            if self.read_timeout is not None:
                self.sock.settimeout(self.read_timeout)
            elif self.timeout is None or isinstance(self.timeout, (int, float)):
                self.sock.settimeout(self.timeout)

    def getresponse(self):
        try:
            return super().getresponse()
        except TimeoutError:
            raise ReadTimeout('Waiting for response headers timed out') from None


class DiagnosticHTTPSConnection(TimeoutPhases, http.client.HTTPSConnection):
    pass


class DiagnosticHTTPConnection(TimeoutPhases, http.client.HTTPConnection):
    pass


class DiagnosticHTTPSHandler(HTTPSHandler):
    def __init__(self, context, read_timeout):
        super().__init__(context=context)
        self.read_timeout = read_timeout

    def https_open(self, request):
        return self.do_open(partial(DiagnosticHTTPSConnection, read_timeout=self.read_timeout),
                            request, context=self._context)


class DiagnosticHTTPHandler(HTTPHandler):
    def __init__(self, read_timeout):
        super().__init__()
        self.read_timeout = read_timeout

    def http_open(self, request):
        return self.do_open(partial(DiagnosticHTTPConnection, read_timeout=self.read_timeout), request)


def classify_transport_error(error, phase='open'):
    """Use exception types/codes; never echo arbitrary reasons, URLs or messages."""
    seen = set()
    while isinstance(error, URLError) and isinstance(error.reason, BaseException) and id(error) not in seen:
        seen.add(id(error))
        error = error.reason
    code = getattr(error, 'errno', None)
    result = {'category': 'connection_error', 'phase': phase}
    if isinstance(code, int):
        result['errno'] = code
    if isinstance(error, socket.gaierror):
        result.update(category='dns_error', phase='dns', detail='DNS name resolution failed; check resolver and network access')
    elif isinstance(error, ssl.SSLCertVerificationError):
        verify_code = getattr(error, 'verify_code', None)
        if isinstance(verify_code, int):
            result['verify_code'] = verify_code
        reasons = {10: 'certificate expired', 18: 'self-signed leaf certificate', 19: 'self-signed chain',
                   20: 'issuer certificate unavailable', 21: 'leaf signature cannot be verified', 62: 'hostname mismatch'}
        reason = reasons.get(verify_code, 'certificate trust or identity check failed')
        result.update(category='tls_certificate_error', phase='tls',
                      detail=f'TLS certificate verification failed ({reason}); check CA trust, certificate validity and hostname; verification remains enabled')
    elif isinstance(error, ssl.SSLError):
        result.update(category='tls_error', phase='tls', detail='TLS negotiation or encrypted connection failed')
    elif isinstance(error, ConnectionTimeout):
        result.update(category='connect_timeout', phase='connect', detail='Connection or TLS setup timed out')
    elif isinstance(error, ReadTimeout) or (isinstance(error, TimeoutError) and phase == 'read'):
        result.update(category='read_timeout', phase='read', detail='Response headers or body timed out while reading')
    elif isinstance(error, TimeoutError) or code == errno.ETIMEDOUT:
        result.update(category='open_timeout', detail='Request timed out during open/send; injected or uninstrumented transport cannot identify a narrower phase')
    elif code in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN):
        result.update(category='network_unreachable', detail='Network route unavailable; check IPv4/IPv6 routing and proxy configuration')
    elif isinstance(error, ConnectionRefusedError) or code == errno.ECONNREFUSED:
        result.update(category='connection_refused', detail='Connection was refused by the destination or configured proxy')
    elif isinstance(error, ConnectionResetError) or code == errno.ECONNRESET:
        result.update(category='connection_reset', detail='Peer or proxy reset the connection')
    elif isinstance(error, http.client.IncompleteRead):
        result.update(category='incomplete_response', phase='read', detail='HTTP response ended before its advertised length')
    elif isinstance(error, http.client.HTTPException):
        result.update(category='malformed_http_response', detail='Peer returned an invalid HTTP response')
    else:
        result['detail'] = 'Network connection failed; inspect local DNS, TLS and proxy configuration'
    return result


def http_error_details(status):
    category = 'http_server_error' if 500 <= status < 600 else 'http_client_error' if 400 <= status < 500 else 'http_error'
    advice = {400: 'Check model, output-token field, temperature and JSON-mode support',
              401: 'Check API key and selected provider', 403: 'Check model/account access or proxy policy',
              407: 'Configured proxy requires authentication', 408: 'Provider reported a request timeout',
              429: 'Provider rate limit or quota exceeded'}
    return {'category': category, 'phase': 'http', 'http_status': status,
            'detail': f'LLM HTTP {status}; ' + advice.get(status, 'Provider returned an unsuccessful HTTP status')}


def diagnostic_message(diagnostic):
    codes = ', '.join(f'{key}={diagnostic[key]}' for key in ('errno', 'verify_code') if key in diagnostic)
    label = f"{diagnostic['category']}; phase={diagnostic['phase']}" + (', ' + codes if codes else '')
    return f"[{label}] {diagnostic['detail']}"


def local_diagnostics(ca_bundle=None):
    paths = ssl.get_default_verify_paths()
    result = {'openssl': ssl.OPENSSL_VERSION,
              'default_ca_file_exists': bool(paths.cafile and Path(paths.cafile).is_file()),
              'default_ca_directory_exists': bool(paths.capath and Path(paths.capath).is_dir()),
              'environment_configured': {key: bool(os.environ.get(key)) for key in
                                        ('SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
                                         'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY',
                                         'https_proxy', 'http_proxy', 'all_proxy', 'no_proxy')},
              'platform_has_ipv6': socket.has_ipv6}
    try:
        result['default_ca_certificates'] = ssl.create_default_context().cert_store_stats()['x509_ca']
        _, result['effective_tls'] = create_tls_context(ca_bundle)
    except (OSError, ValueError):
        result['tls_configuration_error'] = True
    result['proxy_configured'] = {scheme: bool(value) for scheme, value in getproxies().items() if scheme in ('http', 'https', 'all')}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__ + ' No HTTP or API requests are made.')
    parser.add_argument('--host', default='api.groq.com')
    parser.add_argument('--dns', action='store_true', help='Allow DNS-only resolution')
    parser.add_argument('--tls-probe', action='store_true', help='Allow TCP/TLS handshakes only; sends no HTTP or credentials')
    parser.add_argument('--ca-bundle')
    args = parser.parse_args()
    if not args.host or any(char in args.host for char in '/@?#:'):
        parser.error('Use a bare DNS hostname, not a URL')
    result = local_diagnostics(args.ca_bundle)
    result['proxy_bypassed'] = proxy_bypass(args.host) if args.dns or args.tls_probe else 'not_checked_without_dns_opt_in'
    result['probe_note'] = 'TLS probes connect directly (not through HTTP proxies); the application retains urllib proxy handling'
    if args.dns or args.tls_probe:
        try:
            addresses = socket.getaddrinfo(args.host, 443, type=socket.SOCK_STREAM)
            result['dns_families'] = sorted({'IPv6' if family == socket.AF_INET6 else 'IPv4' for family, *_ in addresses})
            if args.tls_probe:
                context, _ = create_tls_context(args.ca_bundle)
                probes = []
                seen = set()
                for family, kind, protocol, _, address in addresses:
                    if family in seen:
                        continue
                    seen.add(family)
                    try:
                        with socket.socket(family, kind, protocol) as connection:
                            connection.settimeout(5)
                            connection.connect(address)
                            with context.wrap_socket(connection, server_hostname=args.host):
                                probes.append({'family': 'IPv6' if family == socket.AF_INET6 else 'IPv4', 'tls_verified': True})
                    except OSError as error:
                        probes.append(dict(family='IPv6' if family == socket.AF_INET6 else 'IPv4', **classify_transport_error(error)))
                result['tls_probes'] = probes
        except OSError as error:
            result['dns_or_tls_error'] = classify_transport_error(error)
        except TLSConfigurationError:
            result['tls_configuration_error'] = True
    print(json.dumps(result, indent=2))
    return 1 if result.get('tls_configuration_error') or result.get('dns_or_tls_error') or any(
        not probe.get('tls_verified') for probe in result.get('tls_probes', [])) else 0


if __name__ == '__main__':
    raise SystemExit(main())

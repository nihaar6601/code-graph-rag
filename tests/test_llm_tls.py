"""Offline regression checks for TLS trust and phase-aware HTTP transport."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import ssl
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request

import llm_transport as transport


class TLSTransportTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        # Fail closed if a test accidentally reaches a real network primitive.
        for name in ('socket.getaddrinfo', 'socket.create_connection', 'socket.socket'):
            guard = patch(name, side_effect=AssertionError('Unexpected network access'))
            guard.start()
            self.addCleanup(guard.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bundle = Path(self.directory.name) / 'ca.pem'
        self.bundle.write_text('placeholder: certificate loading is mocked')

    def context(self, count=3):
        context = Mock(spec=ssl.SSLContext)
        context.cert_store_stats.return_value = {'x509_ca': count}
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        return context

    def test_system_roots_are_retained(self):
        context = self.context()
        with patch.object(transport.ssl, 'create_default_context', return_value=context) as create:
            actual, diagnostic = transport.create_tls_context()
        self.assertIs(actual, context)
        create.assert_called_once_with()
        context.load_verify_locations.assert_not_called()
        self.assertEqual(diagnostic, {'trust_source': 'system', 'ca_certificates': 3})

    def test_empty_default_roots_load_certifi(self):
        context = self.context()
        context.cert_store_stats.side_effect = [{'x509_ca': 0}, {'x509_ca': 12}]
        with patch.object(transport.ssl, 'create_default_context', return_value=context), \
                patch('certifi.where', return_value=str(self.bundle)) as where:
            actual, diagnostic = transport.create_tls_context()
        where.assert_called_once_with()
        context.load_verify_locations.assert_called_once_with(cafile=str(self.bundle))
        self.assertIs(actual, context)
        self.assertEqual(diagnostic, {'trust_source': 'certifi_fallback', 'ca_certificates': 12})
        self.assertEqual(actual.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(actual.check_hostname)

    def test_explicit_bundle_wins_over_environment(self):
        os.environ['SSL_CERT_FILE'] = '/unused/environment.pem'
        context = self.context(0)
        with patch.object(transport.ssl, 'create_default_context', return_value=context) as create:
            _, diagnostic = transport.create_tls_context(str(self.bundle))
        create.assert_called_once_with(cafile=str(self.bundle))
        context.load_verify_locations.assert_not_called()
        self.assertEqual(diagnostic['trust_source'], 'explicit_bundle')

    def test_environment_file_preserved_without_fallback(self):
        os.environ['SSL_CERT_FILE'] = str(self.bundle)
        context = self.context(0)
        with patch.object(transport.ssl, 'create_default_context', return_value=context) as create:
            _, diagnostic = transport.create_tls_context()
        create.assert_called_once_with(cafile=str(self.bundle))
        context.load_verify_locations.assert_not_called()
        self.assertEqual(diagnostic['trust_source'], 'ssl_environment')

    def test_environment_directory_preserved_without_fallback(self):
        os.environ['SSL_CERT_DIR'] = self.directory.name
        context = self.context(0)
        with patch.object(transport.ssl, 'create_default_context', return_value=context) as create:
            _, diagnostic = transport.create_tls_context()
        create.assert_called_once_with()
        context.load_verify_locations.assert_not_called()
        self.assertEqual(diagnostic['trust_source'], 'ssl_environment')

    def test_invalid_explicit_paths_are_validated_and_sanitized(self):
        secret = 'private-test-marker'
        missing = str(Path(self.directory.name) / secret)
        cases = [({}, missing), ({'SSL_CERT_FILE': missing}, None),
                 ({'SSL_CERT_DIR': str(self.bundle)}, None)]
        for environment, bundle in cases:
            with self.subTest(environment=tuple(environment)), patch.dict(os.environ, environment, clear=True), \
                    patch.object(transport.ssl, 'create_default_context') as create:
                with self.assertRaises(transport.TLSConfigurationError) as caught:
                    transport.create_tls_context(bundle)
                self.assertNotIn(secret, str(caught.exception))
                self.assertNotIn(str(self.bundle), str(caught.exception))
                create.assert_not_called()

    def test_tls_initialization_failures_are_sanitized(self):
        for error in (OSError('private-test-marker'), ValueError('private-test-marker')):
            with self.subTest(error=type(error)), \
                    patch.object(transport.ssl, 'create_default_context', side_effect=error):
                with self.assertRaises(transport.TLSConfigurationError) as caught:
                    transport.create_tls_context()
                self.assertNotIn('private-test-marker', str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_certifi_loading_failure_is_sanitized(self):
        context = self.context(0)
        context.load_verify_locations.side_effect = ssl.SSLError('private-test-marker')
        with patch.object(transport.ssl, 'create_default_context', return_value=context), \
                patch('certifi.where', return_value=str(self.bundle)):
            with self.assertRaises(transport.TLSConfigurationError) as caught:
                transport.create_tls_context()
        self.assertNotIn('private-test-marker', str(caught.exception))

    def test_real_default_context_requires_verification_without_network(self):
        context, _ = transport.create_tls_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def connection_cases(self):
        return [(transport.DiagnosticHTTPConnection, transport.http.client.HTTPConnection),
                (transport.DiagnosticHTTPSConnection, transport.http.client.HTTPSConnection)]

    def test_connect_sets_read_timeout_only_after_parent_setup(self):
        for cls, parent in self.connection_cases():
            with self.subTest(connection=cls.__name__):
                connection = cls('example.invalid', timeout=2, read_timeout=7)
                sock = Mock()
                def connected():
                    sock.settimeout.assert_not_called()
                    connection.sock = sock
                with patch.object(parent, 'connect', side_effect=connected) as connect:
                    connection.connect()
                connect.assert_called_once_with()
                sock.settimeout.assert_called_once_with(7)

    def test_connect_timeout_is_classified(self):
        for cls, parent in self.connection_cases():
            with self.subTest(connection=cls.__name__), patch.object(parent, 'connect', side_effect=TimeoutError('private-test-marker')):
                with self.assertRaises(transport.ConnectionTimeout) as caught:
                    cls('example.invalid', timeout=2).connect()
                self.assertNotIn('private-test-marker', str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_read_timeout_defaults_to_connection_timeout(self):
        for cls, parent in self.connection_cases():
            with self.subTest(connection=cls.__name__), patch.object(parent, 'connect'):
                connection = cls('example.invalid', timeout=4)
                connection.sock = Mock()
                connection.connect()
                connection.sock.settimeout.assert_called_once_with(4)

    def test_connect_with_no_socket_is_safe(self):
        for cls, parent in self.connection_cases():
            with self.subTest(connection=cls.__name__), patch.object(parent, 'connect'):
                cls('example.invalid', read_timeout=9).connect()

    def test_getresponse_converts_timeout_and_preserves_success(self):
        for cls, parent in self.connection_cases():
            with self.subTest(connection=cls.__name__):
                connection = cls('example.invalid')
                response = object()
                with patch.object(parent, 'getresponse', return_value=response):
                    self.assertIs(connection.getresponse(), response)
                with patch.object(parent, 'getresponse', side_effect=TimeoutError('private-test-marker')):
                    with self.assertRaises(transport.ReadTimeout) as caught:
                        connection.getresponse()
                    self.assertNotIn('private-test-marker', str(caught.exception))

    def test_non_timeout_parent_errors_are_preserved(self):
        for cls, parent in self.connection_cases():
            for method in ('connect', 'getresponse'):
                error = ssl.SSLError('TLS test failure')
                with self.subTest(connection=cls.__name__, method=method), patch.object(parent, method, side_effect=error):
                    with self.assertRaises(ssl.SSLError) as caught:
                        getattr(cls('example.invalid'), method)()
                    self.assertIs(caught.exception, error)

    def test_handlers_bind_original_connection_classes_and_https_context(self):
        context = self.context()
        cases = [(transport.DiagnosticHTTPHandler(8), 'http_open', transport.DiagnosticHTTPConnection, 'http'),
                 (transport.DiagnosticHTTPSHandler(context, 8), 'https_open', transport.DiagnosticHTTPSConnection, 'https')]
        for handler, method, cls, scheme in cases:
            request = Request(f'{scheme}://example.invalid/')
            response = object()
            with self.subTest(scheme=scheme), patch.object(handler, 'do_open', return_value=response) as do_open:
                self.assertIs(getattr(handler, method)(request), response)
                args, kwargs = do_open.call_args
                self.assertIs(args[0].func, cls)
                self.assertEqual(args[0].keywords, {'read_timeout': 8})
                self.assertIs(args[1], request)
                self.assertEqual(kwargs, {'context': context} if scheme == 'https' else {})

    def test_local_diagnostics_are_offline_and_hide_proxy_credentials(self):
        os.environ['HTTPS_PROXY'] = 'https://user:private-test-marker@proxy.invalid:443'
        with patch.object(transport.ssl, 'create_default_context', return_value=self.context()), \
                patch.object(transport, 'getproxies', return_value={'https': os.environ['HTTPS_PROXY'], 'no': 'private-test-marker'}):
            result = transport.local_diagnostics()
        self.assertEqual(result['proxy_configured'], {'https': True})
        self.assertTrue(result['environment_configured']['HTTPS_PROXY'])
        self.assertTrue(all(isinstance(value, bool) for value in result['environment_configured'].values()))
        self.assertNotIn('private-test-marker', json.dumps(result))
        self.assertEqual(result['default_ca_certificates'], 3)

    def test_main_default_is_offline_and_dns_is_opt_in(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.0.2.1', 443)),
                     (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2001:db8::1', 443, 0, 0))]
        for flags in ([], ['--dns']):
            with self.subTest(flags=flags), patch('sys.argv', ['llm_transport.py'] + flags), \
                    patch.object(transport, 'local_diagnostics', return_value={}), \
                    patch.object(transport, 'proxy_bypass', return_value=False), \
                    patch.object(transport.socket, 'getaddrinfo', return_value=addresses) as dns, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(transport.main(), 0)
                result = json.loads(output.getvalue())
                if flags:
                    dns.assert_called_once_with('api.groq.com', 443, type=socket.SOCK_STREAM)
                    self.assertEqual(result['dns_families'], ['IPv4', 'IPv6'])
                else:
                    dns.assert_not_called()
                self.assertNotIn('tls_probes', result)

    def test_main_tls_probe_is_mocked_verified_and_deduplicated(self):
        address = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.0.2.1', 443))
        context = self.context()
        sock = Mock()
        socket_manager = Mock()
        socket_manager.__enter__ = Mock(return_value=sock)
        socket_manager.__exit__ = Mock(return_value=False)
        tls_manager = Mock()
        tls_manager.__enter__ = Mock(return_value=Mock())
        tls_manager.__exit__ = Mock(return_value=False)
        context.wrap_socket.return_value = tls_manager
        with patch('sys.argv', ['llm_transport.py', '--tls-probe']), \
                patch.object(transport, 'local_diagnostics', return_value={}), \
                patch.object(transport, 'proxy_bypass', return_value=False), \
                patch.object(transport, 'create_tls_context', return_value=(context, {})), \
                patch.object(transport.socket, 'getaddrinfo', return_value=[address, address]), \
                patch.object(transport.socket, 'socket', return_value=socket_manager) as create_socket, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(transport.main(), 0)
        create_socket.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM, 6)
        sock.settimeout.assert_called_once_with(5)
        sock.connect.assert_called_once_with(('192.0.2.1', 443))
        context.wrap_socket.assert_called_once_with(sock, server_hostname='api.groq.com')
        self.assertEqual(json.loads(output.getvalue())['tls_probes'], [{'family': 'IPv4', 'tls_verified': True}])


if __name__ == '__main__':
    unittest.main()

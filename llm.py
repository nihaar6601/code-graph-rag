"""Opt-in OpenAI-compatible Chat Completions transport; no provider SDK required."""
from dataclasses import asdict, dataclass
import http.client
import json
import math
import os
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from token_counter import count_tokens
from llm_transport import (DiagnosticHTTPHandler, DiagnosticHTTPSHandler, TLSConfigurationError,
                           classify_transport_error, create_tls_context, diagnostic_message, http_error_details)


class LLMError(RuntimeError):
    """Sanitized provider failure: never contains API keys or response bodies."""
    def __init__(self, message, attempts=0, usage=None, *, diagnostic=None):
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class LLMConfig:
    provider: str = 'compatible'
    model: str = ''
    base_url: str = ''
    api_key_env: str = 'LLM_API_KEY'
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout: float = 30.0
    retries: int = 1
    output_token_parameter: str = 'max_tokens'
    json_mode: bool = False
    omit_temperature: bool = False
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    ca_bundle: str | None = None
    connect_timeout: float | None = None
    read_timeout: float | None = None

    def validate(self):
        if self.provider not in ('openai', 'groq', 'compatible', 'local', 'mock'):
            raise ValueError('Unknown LLM provider')
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('Set an LLM model with --llm-model/--model or LLM_MODEL')
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (self.max_output_tokens, self.retries)):
            raise ValueError('Output token limit and retries must be integers')
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError('Temperature must be finite and in [0, 2]')
        if self.max_output_tokens < 1 or not math.isfinite(self.timeout) or self.timeout <= 0 or not 0 <= self.retries <= 5:
            raise ValueError('Output limit/timeout must be positive; retries must be in [0, 5]')
        if self.output_token_parameter not in ('max_tokens', 'max_completion_tokens'):
            raise ValueError('Unsupported output token parameter')
        for value in (self.connect_timeout, self.read_timeout):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Connect/read timeouts must be finite and positive')
        if self.ca_bundle is not None and (not isinstance(self.ca_bundle, str) or not self.ca_bundle.strip()):
            raise ValueError('CA bundle must be a nonempty file path')
        prices = (self.input_price_per_million, self.output_price_per_million)
        if any(price is not None and (not math.isfinite(price) or price < 0) for price in prices):
            raise ValueError('Configured prices must be finite and nonnegative')
        if (prices[0] is None) != (prices[1] is None):
            raise ValueError('Configure both input and output prices, or neither')
        if self.provider != 'mock':
            url = urlsplit(self.base_url)
            if not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('LLM base URL must have a host and no credentials, query or fragment')
            local = url.hostname in ('localhost', '127.0.0.1', '::1')
            if url.scheme != 'https' and not (url.scheme == 'http' and local and self.provider == 'local'):
                raise ValueError('Use HTTPS; HTTP is allowed only for a local loopback provider')
        return self


def config_from_environment(provider=None, model=None, base_url=None, api_key_env=None, **options):
    provider = provider or os.environ.get('LLM_PROVIDER', 'compatible')
    urls = {'openai': 'https://api.openai.com/v1', 'groq': 'https://api.groq.com/openai/v1',
            'local': 'http://localhost:8000/v1'}
    keys = {'openai': 'OPENAI_API_KEY', 'groq': 'GROQ_API_KEY'}
    options.setdefault('output_token_parameter', 'max_completion_tokens' if provider in ('openai', 'groq') else 'max_tokens')
    return LLMConfig(provider=provider, model=model or os.environ.get('LLM_MODEL', 'mock' if provider == 'mock' else ''),
                     base_url=base_url or os.environ.get('LLM_BASE_URL', urls.get(provider, '')),
                     api_key_env=api_key_env or os.environ.get('LLM_API_KEY_ENV', keys.get(provider, 'LLM_API_KEY')),
                     **options).validate()


def estimate_message_tokens(messages):
    # Chat wrappers/tokenizers differ by model. This is explicitly an estimate.
    return 3 + sum(3 + count_tokens(message['role']) + count_tokens(message['content']) for message in messages)


def token_accounting(messages, content, usage=None):
    usage = usage if isinstance(usage, dict) else {}
    actual = {}
    for field, key in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'), ('total_tokens', 'total_tokens')]:
        value = usage.get(key)
        actual[field] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    estimated = {'input_tokens': estimate_message_tokens(messages), 'output_tokens': count_tokens(content)}
    estimated['total_tokens'] = estimated['input_tokens'] + estimated['output_tokens']
    source = 'provider' if all(value is not None for value in actual.values()) else (
        'provider_partial' if any(value is not None for value in actual.values()) else 'estimated_only')
    return {'actual': actual, 'estimated': estimated, 'usage_source': source,
            'estimate_method': 'cl100k_base visible text + approximate chat framing; hidden/reasoning tokens unknown'}


def estimate_cost(config, tokens, mocked=False, attempts=1):
    if mocked:
        return {'estimated_usd': 0.0, 'basis': 'mock_no_api_call', 'retry_usage_unknown': False}
    if config.input_price_per_million is None:
        return {'estimated_usd': None, 'basis': 'pricing_not_configured', 'retry_usage_unknown': attempts > 1}
    actual, estimated = tokens['actual'], tokens['estimated']
    inputs = actual['input_tokens'] if actual['input_tokens'] is not None else estimated['input_tokens']
    outputs = actual['output_tokens'] if actual['output_tokens'] is not None else estimated['output_tokens']
    basis = 'provider_usage' if actual['input_tokens'] is not None and actual['output_tokens'] is not None else 'estimated_or_partial_usage'
    return {'estimated_usd': (inputs * config.input_price_per_million + outputs * config.output_price_per_million) / 1_000_000,
            'basis': basis, 'retry_usage_unknown': attempts > 1,
            'note': 'Configured flat rates; excludes cached-token discounts, tiers and unreported failed-attempt usage'}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


class ChatProvider:
    def __init__(self, config, opener=None, sleep=None):
        self.config = config.validate()
        self.tls_diagnostics = None
        if opener is not None:
            self.opener = opener
        else:
            try:
                context, self.tls_diagnostics = create_tls_context(config.ca_bundle)
            except TLSConfigurationError as error:
                raise LLMError(str(error), diagnostic={'category': 'tls_configuration_error', 'phase': 'configuration'}) from None
            read_timeout = config.read_timeout if config.read_timeout is not None else config.timeout
            self.opener = build_opener(NoRedirect(), DiagnosticHTTPSHandler(context, read_timeout),
                                      DiagnosticHTTPHandler(read_timeout)).open
        self.sleep = sleep or time.sleep

    def check_model(self):
        """Read-only authenticated catalog check; never a generation request."""
        key = os.environ.get(self.config.api_key_env)
        if not key and self.config.provider != 'local':
            raise LLMError(f'Missing API key environment variable: {self.config.api_key_env}')
        endpoint = self.config.base_url.rstrip('/')
        if endpoint.endswith('/chat/completions'):
            endpoint = endpoint[:-len('/chat/completions')]
        headers = {'Accept': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = Request(endpoint + '/models', headers=headers, method='GET')
        phase = 'open'
        try:
            with self.opener(request, timeout=self.config.connect_timeout or self.config.timeout) as response:
                phase = 'read'
                body = json.loads(response.read().decode('utf-8'))
            phase = 'parse'
            models = body['data']
            if not isinstance(models, list) or not all(isinstance(item, dict) and isinstance(item.get('id'), str) for item in models):
                raise ValueError('Invalid model catalog')
            if self.config.model not in {item['id'] for item in models}:
                raise LLMError('Configured model is not advertised by the provider model catalog', attempts=1)
            return {'status': 'model_advertised', 'model': self.config.model, 'http_attempts': 1,
                    'note': 'Catalog membership does not guarantee quota or support for every generation parameter'}
        except HTTPError as error:
            status = error.code
            error.close()
            diagnostic = http_error_details(status)
            raise LLMError('Model configuration check failed: ' + diagnostic_message(diagnostic), attempts=1, diagnostic=diagnostic) from None
        except (TimeoutError, URLError, OSError, http.client.HTTPException) as error:
            diagnostic = classify_transport_error(error, phase)
            raise LLMError('Model configuration check failed: ' + diagnostic_message(diagnostic), attempts=1, diagnostic=diagnostic) from None
        except (ValueError, KeyError, TypeError, UnicodeError):
            raise LLMError('Provider returned an invalid model catalog', attempts=1,
                           diagnostic={'category': 'malformed_response', 'phase': 'parse'}) from None

    def complete(self, messages):
        config = self.config
        key = os.environ.get(config.api_key_env)
        if not key and config.provider != 'local':
            raise LLMError(f'Missing API key environment variable: {config.api_key_env}')
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        payload = {'model': config.model, 'messages': messages, config.output_token_parameter: config.max_output_tokens}
        if not config.omit_temperature:
            payload['temperature'] = config.temperature
        if config.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        endpoint = config.base_url.rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            endpoint += '/chat/completions'
        try:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        except (ValueError, TypeError, UnicodeError):
            raise LLMError('LLM request cannot be encoded as UTF-8 JSON',
                           diagnostic={'category': 'invalid_request', 'phase': 'serialization'}) from None
        failures = []
        for attempt in range(config.retries + 1):
            request = Request(endpoint, data=data, headers=headers, method='POST')
            body = None
            phase = 'open'
            try:
                with self.opener(request, timeout=config.connect_timeout or config.timeout) as response:
                    phase = 'read'
                    body = json.loads(response.read().decode('utf-8'))
                phase = 'parse'
                if not isinstance(body, dict) or not isinstance(body.get('choices'), list) or not body['choices']:
                    raise ValueError('Missing choices')
                choice = body['choices'][0]
                if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                    raise ValueError('Missing message object')
                message = choice['message']
                content = message.get('content') or ''
                if not isinstance(content, str) or (not content and not message.get('refusal')):
                    raise ValueError('Missing completion text')
                return {'content': content, 'usage': body.get('usage'), 'model': body.get('model', config.model),
                        'finish_reason': choice.get('finish_reason'), 'refusal': bool(message.get('refusal')),
                        'request_id': body.get('id'), 'attempts': attempt + 1, 'mocked': False}
            except HTTPError as error:
                retry = error.code in (408, 429, 500, 502, 503, 504)
                diagnostic = http_error_details(error.code)
                error.close()
            except (TimeoutError, socket.timeout, URLError, OSError) as error:
                retry = True
                diagnostic = classify_transport_error(error, phase)
            except http.client.HTTPException as error:
                diagnostic = classify_transport_error(error, phase)
                raise LLMError(diagnostic_message(diagnostic), attempt + 1, diagnostic=diagnostic) from None
            except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
                raise LLMError('LLM returned an invalid Chat Completions response', attempt + 1,
                               body.get('usage') if isinstance(body, dict) else None,
                               diagnostic={'category': 'malformed_response', 'phase': 'parse'}) from None
            failures.append(dict(diagnostic))
            if not retry or attempt == config.retries:
                diagnostic = dict(diagnostic, attempt_failures=failures)
                raise LLMError(f'{diagnostic_message(diagnostic)} (attempts={attempt + 1}); usage for failed attempts is unknown',
                               attempt + 1, diagnostic=diagnostic) from None
            self.sleep(min(0.25 * 2 ** attempt, 2.0))
        raise LLMError('LLM request failed')


class MockProvider:
    """Fixed response for plumbing demonstrations; never uses benchmark labels."""
    def __init__(self, config=None, response=None):
        self.config = config or LLMConfig(provider='mock', model='mock')
        self.config.validate()
        self.response = response if response is not None else {'claims': [], 'insufficient_context': True,
                                                              'uncertainties': ['Mock run: no model inference was performed.']}

    def complete(self, messages):
        return {'content': json.dumps(self.response), 'usage': None, 'model': self.config.model,
                'finish_reason': 'stop', 'refusal': False, 'request_id': None, 'attempts': 0, 'mocked': True}


def create_provider(config, mock_response=None):
    if config.provider == 'mock':
        response = json.loads(Path(mock_response).read_text(encoding='utf-8')) if mock_response else None
        return MockProvider(config, response)
    return ChatProvider(config)


def public_config(provider):
    return asdict(provider.config)


def add_llm_arguments(parser):
    parser.add_argument('--provider', choices=('openai', 'groq', 'compatible', 'local', 'mock'))
    parser.add_argument('--llm-model', help='LLM model (LLM_MODEL also supported)')
    parser.add_argument('--base-url', help='OpenAI-compatible API base URL')
    parser.add_argument('--api-key-env', help='Name of the environment variable holding the API key, never its value')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-output-tokens', type=int, default=512)
    parser.add_argument('--request-timeout', type=float, default=30)
    parser.add_argument('--connect-timeout', type=float, help='Connection/TLS setup timeout; defaults to request-timeout')
    parser.add_argument('--read-timeout', type=float, help='Response headers/body socket timeout; defaults to request-timeout')
    parser.add_argument('--ca-bundle', help='Verified custom CA bundle; otherwise system trust or certifi when system trust is empty')
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--output-token-parameter', choices=('max_tokens', 'max_completion_tokens'))
    parser.add_argument('--json-mode', action='store_true', help='Request provider JSON-object output mode')
    parser.add_argument('--omit-temperature', action='store_true', help='For models that disallow temperature')
    parser.add_argument('--input-price-per-million', type=float)
    parser.add_argument('--output-price-per-million', type=float)
    parser.add_argument('--mock-response', type=Path, help='Fixed structured answer JSON for an explicitly mocked run')


def config_from_arguments(args, model=None, require_credentials=True):
    options = dict(temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                   timeout=args.request_timeout, retries=args.retries, json_mode=args.json_mode,
                   omit_temperature=args.omit_temperature, input_price_per_million=args.input_price_per_million,
                   output_price_per_million=args.output_price_per_million)
    options.update(ca_bundle=getattr(args, 'ca_bundle', None), connect_timeout=getattr(args, 'connect_timeout', None),
                   read_timeout=getattr(args, 'read_timeout', None))
    if args.output_token_parameter:
        options['output_token_parameter'] = args.output_token_parameter
    config = config_from_environment(args.provider, args.llm_model or model, args.base_url, args.api_key_env, **options)
    if require_credentials and config.provider not in ('local', 'mock') and not os.environ.get(config.api_key_env):
        raise ValueError(f'Missing API key environment variable: {config.api_key_env}')
    if args.mock_response and config.provider != 'mock':
        raise ValueError('--mock-response requires --provider mock')
    return config


def provider_from_arguments(args, model=None):
    return create_provider(config_from_arguments(args, model), args.mock_response)


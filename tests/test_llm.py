"""Offline transport, configuration and accounting checks for Phase 6."""
from dataclasses import FrozenInstanceError, replace
import io
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from llm import (ChatProvider, LLMConfig, LLMError, MockProvider, NoRedirect,
                 config_from_environment, estimate_cost, public_config,
                 token_accounting)


MESSAGES = [{'role': 'user', 'content': 'Explain café checkout.'}]
BODY = {'id': 'request-123', 'model': 'served-model',
        'choices': [{'message': {'content': '{"claims": []}'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}}


def response(body=BODY):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
    return io.BytesIO(raw)


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Unexpected default-opener construction must never permit real I/O.
        self.network = patch('llm.build_opener', side_effect=AssertionError('Unexpected network transport'))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.config = LLMConfig(model='test-model', base_url='https://example.invalid/v1')
        os.environ['LLM_API_KEY'] = 'secret-test-value'

    def complete(self, opener, config=None):
        return ChatProvider(config or self.config, opener=opener, sleep=Mock()).complete(MESSAGES)

    def test_config_frozen_and_validate_returns_self(self):
        self.assertIs(self.config.validate(), self.config)
        with self.assertRaises(FrozenInstanceError):
            self.config.model = 'changed'

    def test_openai_and_groq_defaults(self):
        for provider, url, key, field in [
            ('openai', 'https://api.openai.com/v1', 'OPENAI_API_KEY', 'max_completion_tokens'),
            ('groq', 'https://api.groq.com/openai/v1', 'GROQ_API_KEY', 'max_completion_tokens')]:
            with self.subTest(provider=provider):
                config = config_from_environment(provider=provider, model='chosen')
                self.assertEqual((config.base_url, config.api_key_env, config.output_token_parameter),
                                 (url, key, field))

    def test_environment_and_explicit_overrides(self):
        os.environ.update(LLM_PROVIDER='groq', LLM_MODEL='env-model',
                          LLM_BASE_URL='https://env.invalid/v1', LLM_API_KEY_ENV='CUSTOM_KEY')
        config = config_from_environment()
        self.assertEqual((config.provider, config.model, config.base_url, config.api_key_env),
                         ('groq', 'env-model', 'https://env.invalid/v1', 'CUSTOM_KEY'))
        config = config_from_environment(provider='openai', model='explicit',
                                         base_url='https://override.invalid', api_key_env='OTHER',
                                         temperature=0.5, max_output_tokens=99)
        self.assertEqual((config.provider, config.model, config.api_key_env, config.temperature,
                          config.max_output_tokens, config.base_url),
                         ('openai', 'explicit', 'OTHER', 0.5, 99, 'https://override.invalid'))

    def test_public_config_and_repr_do_not_expose_key_values(self):
        provider = ChatProvider(self.config, opener=Mock())
        self.assertEqual(public_config(provider)['api_key_env'], 'LLM_API_KEY')
        for representation in (repr(self.config), repr(provider), repr(public_config(provider))):
            self.assertNotIn('secret-test-value', representation)

    def test_https_and_loopback_validation(self):
        for host in ('localhost', '127.0.0.1', '[::1]'):
            self.assertIsInstance(replace(self.config, provider='local',
                                         base_url=f'http://{host}:8000/v1').validate(), LLMConfig)
        for url in ('http://example.com', 'http://localhost', 'https://user:pass@example.com',
                    'https://example.com?key=x', 'https://example.com#fragment', '/v1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                replace(self.config, base_url=url).validate()
        with self.assertRaises(ValueError):
            replace(self.config, provider='local', base_url='http://remote.invalid').validate()

    def test_invalid_configuration_options(self):
        for options in ({'provider': 'unknown'}, {'model': ''}, {'temperature': float('nan')},
                        {'temperature': 3}, {'max_output_tokens': 0}, {'timeout': 0},
                        {'timeout': float('inf')}, {'retries': -1}, {'retries': 6},
                        {'output_token_parameter': 'unsupported'},
                        {'input_price_per_million': 1},
                        {'input_price_per_million': -1, 'output_price_per_million': 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                replace(self.config, **options).validate()
        with self.assertRaises(ValueError):
            config_from_environment(provider='openai')

    def test_request_payload_headers_timeout_and_result(self):
        opener = Mock(return_value=response())
        result = self.complete(opener, replace(self.config, json_mode=True, temperature=0.3,
                                              max_output_tokens=71, timeout=12))
        request = opener.call_args.args[0]
        self.assertEqual(opener.call_args.kwargs, {'timeout': 12})
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(request.full_url, 'https://example.invalid/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret-test-value')
        self.assertEqual(request.get_header('Content-type'), 'application/json')
        self.assertEqual(json.loads(request.data), {'model': 'test-model', 'messages': MESSAGES,
                         'max_tokens': 71, 'temperature': 0.3,
                         'response_format': {'type': 'json_object'}})
        self.assertEqual(result, {'content': '{"claims": []}', 'usage': BODY['usage'],
                         'model': 'served-model', 'finish_reason': 'stop', 'refusal': False,
                         'request_id': 'request-123', 'attempts': 1, 'mocked': False})
        self.assertTrue(opener.return_value.closed)

    def test_optional_payload_fields_and_existing_endpoint(self):
        opener = Mock(return_value=response())
        self.complete(opener, replace(self.config, omit_temperature=True,
                                     output_token_parameter='max_completion_tokens',
                                     base_url='https://example.invalid/v1/chat/completions/'))
        payload = json.loads(opener.call_args.args[0].data)
        self.assertNotIn('temperature', payload)
        self.assertNotIn('response_format', payload)
        self.assertNotIn('max_tokens', payload)
        self.assertEqual(payload['max_completion_tokens'], 512)
        self.assertEqual(opener.call_args.args[0].full_url,
                         'https://example.invalid/v1/chat/completions')

    def test_missing_key_fails_before_request(self):
        os.environ.clear()
        opener = Mock()
        with self.assertRaises(LLMError):
            self.complete(opener)
        opener.assert_not_called()

    def test_loopback_without_auth(self):
        os.environ.clear()
        opener = Mock(return_value=response())
        self.complete(opener, replace(self.config, provider='local', base_url='http://127.0.0.1:8000/v1'))
        self.assertIsNone(opener.call_args.args[0].get_header('Authorization'))

    def test_default_opener_disables_redirects_without_network(self):
        fake = Mock()
        with patch('llm.build_opener', return_value=fake) as build:
            provider = ChatProvider(self.config)
        handler = build.call_args.args[0]
        self.assertIsInstance(handler, NoRedirect)
        self.assertEqual(provider.opener, fake.open)
        self.assertIsNone(handler.redirect_request(Request('https://example.invalid'), None,
                                                  302, 'Found', {}, 'https://other.invalid'))
        fake.open.assert_not_called()

    def test_transient_http_retries(self):
        for status in (429, 503):
            with self.subTest(status=status):
                error = HTTPError('https://example.invalid', status, 'secret-test-value', {}, io.BytesIO())
                opener = Mock(side_effect=[error, response()])
                sleep = Mock()
                result = ChatProvider(self.config, opener=opener, sleep=sleep).complete(MESSAGES)
                self.assertEqual(result['attempts'], 2)
                self.assertEqual(opener.call_count, 2)
                sleep.assert_called_once_with(0.25)

    def test_timeout_and_urlerror_retry(self):
        for error in (TimeoutError('secret-test-value'), URLError('secret-test-value')):
            with self.subTest(error=type(error).__name__):
                opener = Mock(side_effect=[error, response()])
                self.assertEqual(self.complete(opener)['attempts'], 2)

    def test_unauthorized_not_retried_and_sanitized(self):
        opener = Mock(side_effect=HTTPError('https://example.invalid', 401,
                                           'secret-test-value', {}, io.BytesIO()))
        with self.assertRaises(LLMError) as caught:
            self.complete(opener)
        self.assertIn('401', str(caught.exception))
        self.assertNotIn('secret-test-value', str(caught.exception))
        opener.assert_called_once()

    def test_retry_exhaustion_and_backoff(self):
        opener = Mock(side_effect=URLError('secret-test-value'))
        sleep = Mock()
        with self.assertRaises(LLMError) as caught:
            ChatProvider(replace(self.config, retries=2), opener=opener, sleep=sleep).complete(MESSAGES)
        self.assertIn('attempts=3', str(caught.exception))
        self.assertNotIn('secret-test-value', str(caught.exception))
        self.assertEqual(opener.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.25, 0.5])

    def test_invalid_json_choices_and_missing_response_keys(self):
        for body in (b'not JSON', {}, {'choices': []}, {'choices': [{}]},
                     {'choices': [{'message': {}}]}, {'choices': [{'message': {'content': 42}}]}):
            with self.subTest(body=body):
                opener = Mock(return_value=response(body))
                with self.assertRaises(LLMError):
                    self.complete(opener)
                opener.assert_called_once()

    def test_refusal_and_missing_optional_metadata(self):
        result = self.complete(Mock(return_value=response({'choices': [
            {'message': {'content': None, 'refusal': 'Cannot comply'}}]})))
        self.assertEqual(result['content'], '')
        self.assertTrue(result['refusal'])
        self.assertEqual(result['model'], self.config.model)
        self.assertIsNone(result['request_id'])
        self.assertIsNone(result['usage'])
        self.assertIsNone(result['finish_reason'])

    def test_complete_provider_token_usage(self):
        tokens = token_accounting(MESSAGES, 'answer', BODY['usage'])
        self.assertEqual(tokens['actual'], {'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})
        self.assertEqual(tokens['usage_source'], 'provider')
        self.assertGreater(tokens['estimated']['input_tokens'], 0)
        self.assertEqual(tokens['estimated']['total_tokens'],
                         tokens['estimated']['input_tokens'] + tokens['estimated']['output_tokens'])

    def test_partial_usage_is_not_filled_with_estimates(self):
        tokens = token_accounting(MESSAGES, 'answer', {'prompt_tokens': 10})
        self.assertEqual(tokens['actual'], {'input_tokens': 10, 'output_tokens': None, 'total_tokens': None})
        self.assertEqual(tokens['usage_source'], 'provider_partial')

    def test_invalid_usage_values_rejected(self):
        for value in (True, False, -1, 1.5, '10', None):
            with self.subTest(value=value):
                usage = dict.fromkeys(('prompt_tokens', 'completion_tokens', 'total_tokens'), value)
                tokens = token_accounting(MESSAGES, '', usage)
                self.assertEqual(tokens['actual'], dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens')))
                self.assertEqual(tokens['usage_source'], 'estimated_only')
        self.assertEqual(token_accounting(MESSAGES, '', dict.fromkeys(
            ('prompt_tokens', 'completion_tokens', 'total_tokens'), 0))['usage_source'], 'provider')

    def test_missing_and_non_dict_usage(self):
        for usage in (None, [], 'bad'):
            with self.subTest(usage=usage):
                tokens = token_accounting(MESSAGES, 'answer', usage)
                self.assertTrue(all(value is None for value in tokens['actual'].values()))
                self.assertEqual(tokens['usage_source'], 'estimated_only')

    def test_cost_configured_actual_and_partial_usage(self):
        config = replace(self.config, input_price_per_million=2, output_price_per_million=4)
        tokens = token_accounting(MESSAGES, 'answer', BODY['usage'])
        cost = estimate_cost(config, tokens)
        self.assertAlmostEqual(cost['estimated_usd'], 0.000036)
        self.assertEqual(cost['basis'], 'provider_usage')
        self.assertFalse(cost['retry_usage_unknown'])
        partial = token_accounting(MESSAGES, 'answer', {'prompt_tokens': 10})
        cost = estimate_cost(config, partial, attempts=2)
        self.assertAlmostEqual(cost['estimated_usd'], (20 + 4 * partial['estimated']['output_tokens']) / 1_000_000)
        self.assertEqual(cost['basis'], 'estimated_or_partial_usage')
        self.assertTrue(cost['retry_usage_unknown'])

    def test_unconfigured_pricing(self):
        cost = estimate_cost(self.config, token_accounting(MESSAGES, 'answer'), attempts=2)
        self.assertIsNone(cost['estimated_usd'])
        self.assertEqual(cost['basis'], 'pricing_not_configured')
        self.assertTrue(cost['retry_usage_unknown'])

    def test_mock_fixed_json_no_actual_usage_or_cost(self):
        provider = MockProvider()
        result = provider.complete(MESSAGES)
        self.assertTrue(result['mocked'])
        self.assertEqual(result['attempts'], 0)
        self.assertTrue(json.loads(result['content'])['insufficient_context'])
        self.assertIsNone(result['usage'])
        tokens = token_accounting(MESSAGES, result['content'], result['usage'])
        self.assertTrue(all(value is None for value in tokens['actual'].values()))
        self.assertEqual(estimate_cost(provider.config, tokens, mocked=True),
                         {'estimated_usd': 0.0, 'basis': 'mock_no_api_call', 'retry_usage_unknown': False})
        fixed = {'claims': ['fixed'], 'insufficient_context': False}
        self.assertEqual(json.loads(MockProvider(response=fixed).complete(MESSAGES)['content']), fixed)


if __name__ == '__main__':
    unittest.main()

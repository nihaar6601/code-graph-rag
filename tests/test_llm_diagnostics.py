"""Transport regressions and curl-equivalent request comparison, all HTTP mocked."""
from dataclasses import replace
import errno
import http.client
import io
import json
import os
import socket
import ssl
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from llm import ChatProvider, LLMConfig, LLMError, config_from_environment
from llm_transport import ConnectionTimeout, ReadTimeout, classify_transport_error

SECRET = 'DO-NOT-LOG-SECRET-KEY'
SOURCE = 'sensitive repository source must never appear in transport errors'
MESSAGES = [{'role': 'user', 'content': SOURCE}]
GOOD = {'choices': [{'message': {'content': 'Hello'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 8, 'completion_tokens': 1, 'total_tokens': 9}}


def good_response():
    return io.BytesIO(json.dumps(GOOD).encode())


def certificate_error():
    error = ssl.SSLCertVerificationError(1, SECRET + SOURCE)
    error.verify_code = 20
    return error


class LlmDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'LLM_API_KEY': SECRET}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = LLMConfig(model='test', base_url='https://example.invalid/v1')

    def assert_safe(self, value):
        text = str(value)
        for forbidden in (SECRET, SOURCE, 'Bearer', 'user:password@'):
            self.assertNotIn(forbidden, text)

    def test_nested_urlerror_preserves_dns_and_errno_without_reason_text(self):
        failure = URLError(URLError(socket.gaierror(socket.EAI_NONAME, SECRET + SOURCE)))
        opener = Mock(side_effect=failure)
        with self.assertRaises(LLMError) as caught:
            ChatProvider(self.config, opener=opener, sleep=Mock()).complete(MESSAGES)
        self.assertEqual(caught.exception.diagnostic['category'], 'dns_error')
        self.assertEqual(caught.exception.diagnostic['phase'], 'dns')
        self.assertEqual(caught.exception.diagnostic['errno'], socket.EAI_NONAME)
        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(opener.call_count, 2)
        self.assert_safe(caught.exception)
        self.assert_safe(caught.exception.diagnostic)

    def test_tls_verification_is_not_mislabeled_as_timeout_and_retries_unchanged(self):
        opener = Mock(side_effect=URLError(certificate_error()))
        sleep = Mock()
        with self.assertRaises(LLMError) as caught:
            ChatProvider(self.config, opener=opener, sleep=sleep).complete(MESSAGES)
        self.assertEqual(caught.exception.diagnostic['category'], 'tls_certificate_error')
        self.assertEqual(caught.exception.diagnostic['verify_code'], 20)
        self.assertIn('issuer certificate unavailable', str(caught.exception))
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(0.25)
        self.assert_safe(caught.exception)

    def test_tls_handshake_protocol_error(self):
        diagnostic = classify_transport_error(URLError(ssl.SSLError(1, SECRET)))
        self.assertEqual(diagnostic['category'], 'tls_error')
        self.assert_safe(diagnostic)

    def test_tagged_connect_and_read_header_timeouts(self):
        for error, category in ((ConnectionTimeout(SECRET), 'connect_timeout'), (ReadTimeout(SECRET), 'read_timeout')):
            with self.subTest(category=category):
                with self.assertRaises(LLMError) as caught:
                    ChatProvider(self.config, opener=Mock(side_effect=URLError(error)), sleep=Mock()).complete(MESSAGES)
                self.assertEqual(caught.exception.diagnostic['category'], category)
                self.assertEqual(caught.exception.attempts, 2)
                self.assert_safe(caught.exception)

    def test_raw_open_timeout_is_labeled_unknown_phase_not_guessed_connect(self):
        with self.assertRaises(LLMError) as caught:
            ChatProvider(replace(self.config, retries=0), opener=Mock(side_effect=TimeoutError(SECRET))).complete(MESSAGES)
        self.assertEqual(caught.exception.diagnostic['category'], 'open_timeout')
        self.assertEqual(caught.exception.diagnostic['phase'], 'open')

    def test_body_read_timeout_is_classified_and_retried(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.side_effect = TimeoutError(SECRET)
        opener = Mock(side_effect=[response, good_response()])
        result = ChatProvider(self.config, opener=opener, sleep=Mock()).complete(MESSAGES)
        self.assertEqual(result['attempts'], 2)
        response.read.assert_called_once()
        with self.assertRaises(LLMError) as caught:
            ChatProvider(replace(self.config, retries=0), opener=Mock(return_value=response)).complete(MESSAGES)
        self.assertEqual(caught.exception.diagnostic['category'], 'read_timeout')
        self.assert_safe(caught.exception)

    def test_route_refused_and_reset_errors(self):
        for error, category in ((OSError(errno.ENETUNREACH, SECRET), 'network_unreachable'),
                                (ConnectionRefusedError(errno.ECONNREFUSED, SECRET), 'connection_refused'),
                                (ConnectionResetError(errno.ECONNRESET, SECRET), 'connection_reset')):
            with self.subTest(category=category):
                diagnostic = classify_transport_error(URLError(error))
                self.assertEqual(diagnostic['category'], category)
                self.assert_safe(diagnostic)

    def test_http_4xx_5xx_categories_and_retry_policy_and_body_not_read(self):
        for code, category, calls in ((400, 'http_client_error', 1), (401, 'http_client_error', 1),
                                      (403, 'http_client_error', 1), (429, 'http_client_error', 2),
                                      (500, 'http_server_error', 2), (503, 'http_server_error', 2)):
            with self.subTest(code=code):
                body = Mock()
                def error_factory(*args, **kwargs):
                    return_error = HTTPError('https://user:password@example.invalid/?key=' + SECRET, code, SOURCE, {}, body)
                    raise return_error
                opener = Mock(side_effect=error_factory)
                with self.assertRaises(LLMError) as caught:
                    ChatProvider(self.config, opener=opener, sleep=Mock()).complete(MESSAGES)
                self.assertEqual(caught.exception.diagnostic['category'], category)
                self.assertEqual(caught.exception.diagnostic['http_status'], code)
                self.assertEqual(opener.call_count, calls)
                body.read.assert_not_called()
                self.assert_safe(caught.exception)
                self.assert_safe(caught.exception.diagnostic)

    def test_malformed_response_shapes_are_sanitized_without_attribute_errors(self):
        for body in ([], None, {'choices': None}, {'choices': [None]}, {'choices': ['secret']},
                     {'choices': [{'message': None}]}, {'choices': [{'message': []}]}):
            with self.subTest(body=body):
                opener = Mock(return_value=io.BytesIO(json.dumps(body).encode()))
                with self.assertRaises(LLMError) as caught:
                    ChatProvider(self.config, opener=opener).complete(MESSAGES)
                self.assertEqual(caught.exception.diagnostic['category'], 'malformed_response')
                opener.assert_called_once()
                self.assert_safe(caught.exception)

    def test_invalid_json_utf8_and_known_usage_are_kept_separate(self):
        for data in (b'\xff', ('not json ' + SECRET).encode()):
            with self.assertRaises(LLMError) as caught:
                ChatProvider(self.config, opener=Mock(return_value=io.BytesIO(data))).complete(MESSAGES)
            self.assertEqual(caught.exception.diagnostic['category'], 'malformed_response')
            self.assert_safe(caught.exception)
        body = {'choices': [None], 'usage': GOOD['usage']}
        with self.assertRaises(LLMError) as caught:
            ChatProvider(self.config, opener=Mock(return_value=io.BytesIO(json.dumps(body).encode()))).complete(MESSAGES)
        self.assertEqual(caught.exception.usage, GOOD['usage'])

    def test_incomplete_or_invalid_http_response_is_wrapped_without_partial_source_leak(self):
        for failure, expected in ((http.client.IncompleteRead((SECRET + SOURCE).encode(), 10), 'incomplete_response'),
                                  (http.client.BadStatusLine(SECRET + SOURCE), 'malformed_http_response')):
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.side_effect = failure
            opener = Mock(return_value=response)
            with self.assertRaises(LLMError) as caught:
                ChatProvider(self.config, opener=opener).complete(MESSAGES)
            self.assertEqual(caught.exception.diagnostic['category'], expected)
            opener.assert_called_once()
            self.assert_safe(caught.exception)

    def test_catalog_shares_dns_tls_and_read_diagnostics(self):
        for error, category in ((socket.gaierror(socket.EAI_AGAIN, SECRET), 'dns_error'),
                                (certificate_error(), 'tls_certificate_error'), (ReadTimeout(SECRET), 'read_timeout')):
            with self.subTest(category=category), self.assertRaises(LLMError) as caught:
                ChatProvider(self.config, opener=Mock(side_effect=URLError(error))).check_model()
            self.assertEqual(caught.exception.diagnostic['category'], category)
            self.assertEqual(caught.exception.attempts, 1)
            self.assert_safe(caught.exception)

    def test_custom_connect_read_timeouts_and_unified_timeout_backwards_compatibility(self):
        opener = Mock(return_value=good_response())
        ChatProvider(replace(self.config, timeout=60, connect_timeout=10, read_timeout=60), opener=opener).complete(MESSAGES)
        self.assertEqual(opener.call_args.kwargs['timeout'], 10)
        opener = Mock(return_value=good_response())
        ChatProvider(replace(self.config, timeout=42), opener=opener).complete(MESSAGES)
        self.assertEqual(opener.call_args.kwargs['timeout'], 42)
        for field in ('connect_timeout', 'read_timeout'):
            for value in (0, -1, float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    replace(self.config, **{field: value}).validate()

    def test_python_payload_matches_user_curl_without_sending_any_request(self):
        config = config_from_environment('groq', 'openai/gpt-oss-120b', max_output_tokens=256, retries=0)
        messages = [{'role': 'user', 'content': 'Reply with exactly: Hello'}]
        opener = Mock(return_value=good_response())
        with patch.dict(os.environ, {'GROQ_API_KEY': SECRET}):
            result = ChatProvider(config, opener=opener).complete(messages)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.groq.com/openai/v1/chat/completions')
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(json.loads(request.data), {'model': 'openai/gpt-oss-120b', 'messages': messages,
                                                  'temperature': 0.0, 'max_completion_tokens': 256})
        self.assertEqual(result['content'], 'Hello')
        self.assertEqual(config_from_environment('groq', 'chosen', output_token_parameter='max_tokens').output_token_parameter, 'max_tokens')

    def test_invalid_request_serialization_never_echoes_prompt_fragments(self):
        for messages in ([{'role': 'user', 'content': SECRET + '\ud800'}], [{'role': 'user', 'content': {object()}}]):
            opener = Mock(side_effect=AssertionError('No request'))
            with self.assertRaises(LLMError) as caught:
                ChatProvider(self.config, opener=opener).complete(messages)
            self.assertEqual(caught.exception.attempts, 0)
            self.assertEqual(caught.exception.diagnostic['category'], 'invalid_request')
            self.assert_safe(caught.exception)
            opener.assert_not_called()


if __name__ == '__main__':
    unittest.main()

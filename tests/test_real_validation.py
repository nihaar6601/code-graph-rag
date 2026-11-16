"""Phase 6.1 planning, preflight and controlled workflow: no live API/model calls."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import numpy as np

import evaluate_answers
from evaluation_controls import BudgetedProvider, EvaluationBudget
from evaluation_workflow import all_rows, checkout_annotation, run_stages, select_questions
from llm import ChatProvider, LLMConfig, LLMError, MockProvider
from strategies import prepare_search

ROOT = Path(__file__).resolve().parents[1]


class SmallEncoder:
    dimension = 2
    cache_identity = 'phase61-unit-encoder'
    model_name = 'fake-tests-only'
    load_ms = 0

    def encode(self, texts):
        return np.asarray([[text.lower().count('payment'), text.lower().count('stock')]
                           for text in texts], dtype=np.float32).reshape(len(texts), 2)


def response(body):
    result = MagicMock()
    result.__enter__.return_value.read.return_value = json.dumps(body).encode()
    return result


class CatalogTests(unittest.TestCase):
    def config(self):
        return LLMConfig(provider='compatible', model='unit-model', base_url='https://example.test/v1', api_key_env='UNIT_TEST_KEY')

    def test_model_catalog_uses_configured_url_and_auth_without_exposing_it(self):
        opener = MagicMock(return_value=response({'data': [{'id': 'unit-model'}]}))
        with patch.dict(os.environ, {'UNIT_TEST_KEY': 'NEVER_EXPORT_THIS_KEY'}):
            result = ChatProvider(self.config(), opener=opener).check_model()
        request = opener.call_args.args[0]
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(request.full_url, 'https://example.test/v1/models')
        self.assertEqual(request.get_header('Authorization'), 'Bearer NEVER_EXPORT_THIS_KEY')
        self.assertEqual(result['status'], 'model_advertised')
        self.assertNotIn('NEVER_EXPORT_THIS_KEY', json.dumps(result))
        opener.assert_called_once()

    def test_missing_model_malformed_catalog_and_failed_auth(self):
        for payload in ({'data': []}, {'data': ['invalid']}, []):
            with self.subTest(payload=payload), patch.dict(os.environ, {'UNIT_TEST_KEY': 'secret'}):
                with self.assertRaises(LLMError):
                    ChatProvider(self.config(), opener=MagicMock(return_value=response(payload))).check_model()
        error = HTTPError('https://example.test/v1/models', 401, 'secret response', {}, io.BytesIO(b'secret'))
        with patch.dict(os.environ, {'UNIT_TEST_KEY': 'secret'}):
            with self.assertRaisesRegex(LLMError, 'HTTP 401') as caught:
                ChatProvider(self.config(), opener=MagicMock(side_effect=error)).check_model()
        self.assertNotIn('secret', str(caught.exception))

    def test_catalog_timeout_and_missing_key(self):
        with patch.dict(os.environ, {'UNIT_TEST_KEY': 'secret'}):
            with self.assertRaisesRegex(LLMError, 'timed out'):
                ChatProvider(self.config(), opener=MagicMock(side_effect=TimeoutError('secret'))).check_model()
        with patch.dict(os.environ, {}, clear=True):
            opener = MagicMock(side_effect=AssertionError('No network'))
            with self.assertRaisesRegex(LLMError, 'Missing API key'):
                ChatProvider(self.config(), opener=opener).check_model()
            opener.assert_not_called()


class DryRunTests(unittest.TestCase):
    def invoke(self, flags):
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['evaluate_answers.py', str(ROOT / 'examples/shop'), *flags]), contextlib.redirect_stdout(stream):
            code = evaluate_answers.main()
        return code, stream.getvalue()

    def test_dry_run_without_credentials_or_provider_or_retrieval_even_if_answer_flag_present(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            flags = ['--provider', 'groq', '--model', 'unit-model', '--profile', 'initial', '--dry-run', '--answer',
                     '--checkout-smoke', '--compare-optimization', '--retries', '0', '--json', str(path)]
            with patch.dict(os.environ, {}, clear=True), patch('llm.create_provider', side_effect=AssertionError('No provider')):
                with patch('evaluate_answers.prepare_search', side_effect=AssertionError('No embedding/retrieval')):
                    code, text = self.invoke(flags)
            plan = json.loads(path.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(plan['mode'], 'dry_run')
            self.assertEqual(plan['plan']['question_count'], 8)
            self.assertEqual(plan['plan']['expected_api_requests'], 50)
            self.assertFalse(plan['plan']['credentials_present'])
            self.assertIsNone(plan['plan']['estimated_upper_bound_usd'])
            self.assertIn('no API requests', text)
            self.assertIn('Dry-run plan', path.with_suffix('.md').read_text())

    def test_dry_run_keys_never_exported_and_configured_price_estimate_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'GROQ_API_KEY': 'SECRET_REAL_KEY_MARKER'}):
            path = Path(directory) / 'plan.json'
            code, printed = self.invoke(['--provider', 'groq', '--model', 'unit-model', '--profile', 'initial',
                                         '--dry-run', '--retries', '0', '--input-price-per-million', '1',
                                         '--output-price-per-million', '2', '--max-total-cost', '10', '--json', str(path)])
            self.assertEqual(code, 0)
            self.assertNotIn('SECRET_REAL_KEY_MARKER', printed + path.read_text() + path.with_suffix('.md').read_text())
            self.assertGreater(json.loads(path.read_text())['plan']['estimated_upper_bound_usd'], 0)

    def test_outputs_cannot_overwrite_sources_or_annotations(self):
        original = (ROOT / 'examples/shop/src/shop/service/PaymentService.java').read_bytes()
        with self.assertRaises(SystemExit) as caught:
            self.invoke(['--provider', 'mock', '--dry-run', '--markdown', str(ROOT / 'examples/shop/src/shop/service/PaymentService.java')])
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual((ROOT / 'examples/shop/src/shop/service/PaymentService.java').read_bytes(), original)

    def test_no_real_evaluation_without_opt_in(self):
        with self.assertRaises(SystemExit) as caught:
            self.invoke(['--provider', 'local', '--model', 'unit-model'])
        self.assertEqual(caught.exception.code, 2)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.annotations = json.loads((ROOT / 'benchmarks/shop_answers.json').read_text())

    def test_profile_is_eight_exact_unmodified_existing_annotations(self):
        before = copy.deepcopy(self.annotations)
        selected = select_questions(self.annotations, 'initial')
        self.assertEqual(len(selected), 8)
        for item in selected:
            self.assertIn(item, before)
        smoke = checkout_annotation(self.annotations)
        self.assertEqual(smoke['question'], 'How does checkout handle a rejected payment?')
        original = next(item for item in before if item['question'] == smoke['annotation_question'])
        self.assertEqual(smoke['facts'], original['facts'])
        smoke['facts'][0]['description'] = 'mutated locally'
        self.assertEqual(self.annotations, before)

    def test_optimization_and_smoke_are_separate_identical_generation_settings(self):
        selected = select_questions(self.annotations, 'initial', limit=1)
        state = prepare_search([ROOT / 'examples/shop'], ['graph', 'hybrid_graph'], encoder=SmallEncoder(), cache_dir=self.root / 'cache')
        config = LLMConfig(provider='mock', model='mock')
        provider = BudgetedProvider(MockProvider(config), EvaluationBudget(config, 0, 1))
        result = run_stages(state, selected, ['graph', 'hybrid_graph'], provider, checkout=checkout_annotation(self.annotations),
                            checkout_smoke=True, compare_optimization=True, max_tokens=2000, top_k=3)
        self.assertEqual(len(result['results']), 2)
        self.assertEqual(len(result['smoke_report']['results']), 2)
        self.assertEqual(sum(len(report['results']) for report in result['optimization_reports'].values()), 4)
        self.assertEqual(result['total_recorded_trials'], 8)
        rows = all_rows(result)
        self.assertEqual(len({row['answer']['system_prompt_sha256'] for row in rows}), 1)
        self.assertEqual(len({json.dumps(row['answer']['llm'], sort_keys=True) for row in rows}), 1)
        self.assertTrue(all(row['answer']['metrics']['actual_input_tokens'] is None for row in rows))
        for mode, report in result['optimization_reports'].items():
            self.assertTrue(all(row['answer']['retrieval']['configuration']['optimization'] == mode for row in report['results']))
        self.assertEqual(len(result['optimization_comparison']), 2)
        self.assertTrue(all(pair['actual_input_tokens_delta'] is None for pair in result['optimization_comparison']))
        self.assertTrue(all(pair['independently_validated_answer_quality_delta'] is None for pair in result['optimization_comparison']))

    def test_real_provider_numeric_settings_require_integer_counts(self):
        for options in ({'max_output_tokens': True}, {'max_output_tokens': 1.5}, {'retries': False}, {'retries': 1.2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LLMConfig(provider='local', model='test', base_url='http://localhost:8000/v1', **options).validate()

    def test_budget_stops_are_exportable_partial_results_not_fair_completed_comparison(self):
        class HugeReportedUsage:
            config = LLMConfig(provider='local', model='test', base_url='http://localhost:8000/v1', retries=0,
                               input_price_per_million=1, output_price_per_million=1, max_output_tokens=32)
            calls = 0

            def complete(self, messages):
                self.calls += 1
                return {'content': '{"claims":[],"insufficient_context":true,"uncertainties":[]}',
                        'usage': {'prompt_tokens': 1000000, 'completion_tokens': 10, 'total_tokens': 1000010},
                        'model': 'test', 'attempts': 1, 'mocked': False}
        raw = HugeReportedUsage()
        budget = EvaluationBudget(raw.config, max_total_cost=0.1)
        state = prepare_search([ROOT / 'examples/shop'], ['graph'])
        selected = select_questions(self.annotations, 'initial', limit=2)
        result = run_stages(state, selected, ['graph'], BudgetedProvider(raw, budget), max_tokens=2000, top_k=1)
        self.assertEqual(raw.calls, 1)
        self.assertEqual(result['results'][1]['answer']['status'], 'budget_exhausted')
        self.assertEqual(result['results'][1]['answer']['metrics']['attempts'], 0)
        self.assertEqual(result['comparison_status'], 'incomplete_budget_do_not_rank_strategies')
        self.assertEqual(result['summary']['graph']['budget_stop_rate'], 0.5)
        json.dumps(result)

    def test_all_stage_request_limits_checked_before_provider_use(self):
        state = prepare_search([ROOT / 'examples/shop'], ['graph'])
        provider = MockProvider()
        with patch.object(provider, 'complete', side_effect=AssertionError('No request')):
            with self.assertRaises(ValueError):
                run_stages(state, select_questions(self.annotations, 'initial', limit=1), ['graph'], provider,
                           max_requests=1, checkout=checkout_annotation(self.annotations), checkout_smoke=True)

    def test_failed_smoke_does_not_continue_to_benchmark_or_optimization(self):
        state = prepare_search([ROOT / 'examples/shop'], ['graph'])
        provider = MockProvider(response={})
        with patch.object(provider, 'complete', wraps=provider.complete) as calls:
            report = run_stages(state, select_questions(self.annotations, 'initial', limit=1), ['graph'], provider,
                                checkout=checkout_annotation(self.annotations), checkout_smoke=True,
                                compare_optimization=True, top_k=1)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(report['comparison_status'], 'not_executed_smoke_failed')
        self.assertEqual(report['results'], [])
        self.assertNotIn('optimization_reports', report)
        self.assertEqual(len(report['smoke_report']['results']), 1)


if __name__ == '__main__':
    unittest.main()

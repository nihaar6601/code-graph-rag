"""Offline factual-point grading, optimization, QA integration and aggregation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from answer_grading import grade_answer, validate_answer_benchmark
from answering import answer_from_retrieval, evidence_sources
from evaluate_answers import answer_comparison, evaluate_answer_benchmark
from llm import MockProvider
from strategies import STRATEGIES, prepare_search, retrieve_strategy

ROOT = Path(__file__).resolve().parents[1]


class TestEncoder:
    dimension = 2
    cache_identity = 'answer-tests-v1'
    model_name = 'fake-test-only'
    load_ms = 0

    def encode(self, texts):
        return np.asarray([[text.lower().count('payment'), text.lower().count('audit')]
                           for text in texts], dtype=np.float32).reshape(len(texts), 2)


def benchmark_item():
    return {'question': 'How does processPayment work?', 'category': 'behavior',
            'expected': ['Service.processPayment()', 'Service.validatePayment()'], 'expect_insufficient': False,
            'reference_answer': 'processPayment calls validatePayment. validatePayment is empty.',
            'facts': [{'id': 'calls', 'description': 'Delegates validation',
                       'patterns': [r'processPayment.*calls.*validatePayment'], 'sources': ['Service.processPayment()']},
                      {'id': 'empty', 'description': 'Empty validation',
                       'patterns': [r'validatePayment.*empty'], 'sources': ['Service.validatePayment()']}]}


class AnswerEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'Service.java').write_text('''class Service {
    void processPayment() { validatePayment(); }
    void validatePayment() {}
}''')
        self.state = prepare_search([self.repo], STRATEGIES, encoder=TestEncoder(), cache_dir=self.root / 'cache')
        self.retrieval = retrieve_strategy(self.state, 'graph', 'How does processPayment work?', top_k=1)
        sources = evidence_sources(self.retrieval)
        self.response = {'claims': [
            {'text': 'processPayment calls validatePayment.', 'citations': [{'source_id': 'S1', 'start_line': sources[0]['start_line'], 'end_line': sources[0]['end_line']}]},
            {'text': 'validatePayment is empty.', 'citations': [{'source_id': 'S2', 'start_line': sources[1]['start_line'], 'end_line': sources[1]['end_line']}]}],
            'insufficient_context': False, 'uncertainties': []}

    def test_complete_facts_and_source_attribution(self):
        answer = answer_from_retrieval(self.retrieval, MockProvider(response=self.response))
        grade = grade_answer(answer, benchmark_item(), self.state['graph'])
        self.assertEqual(grade['correctness_proxy'], 1)
        self.assertEqual(grade['completeness'], 1)
        self.assertEqual(grade['grounded_completeness'], 1)
        self.assertEqual(grade['appropriate_insufficiency'], 1)
        self.assertEqual(grade['unverified_claim_count'], 0)

    def test_empty_mock_object_is_validated_not_replaced_by_default(self):
        answer = answer_from_retrieval(self.retrieval, MockProvider(response={}))
        self.assertEqual(answer['status'], 'invalid_response')

    def test_optimized_incoming_callers_are_still_supported(self):
        result = retrieve_strategy(self.state, 'graph', 'Who calls validatePayment?', top_k=1,
                                   include_callers=True, optimization='relevance')
        self.assertTrue(any('processPayment' in snippet['name'] for snippet in result['snippets']))

    def test_wrong_cited_method_does_not_get_grounding_credit(self):
        response = copy.deepcopy(self.response)
        response['claims'][1]['citations'] = copy.deepcopy(response['claims'][0]['citations'])
        answer = answer_from_retrieval(self.retrieval, MockProvider(response=response))
        grade = grade_answer(answer, benchmark_item(), self.state['graph'])
        self.assertEqual(grade['completeness'], 1)
        self.assertEqual(grade['grounded_completeness'], 0.5)

    def test_incomplete_context_gets_conditional_completeness(self):
        retrieval = retrieve_strategy(self.state, 'structure', 'processPayment', top_k=1)
        response = {'claims': [self.response['claims'][0]], 'insufficient_context': True,
                    'uncertainties': ['Validation implementation is not provided.']}
        answer = answer_from_retrieval(retrieval, MockProvider(response=response))
        grade = grade_answer(answer, benchmark_item(), self.state['graph'])
        self.assertTrue(grade['expected_insufficient_given_context'])
        self.assertEqual(grade['appropriate_insufficiency'], 1)
        self.assertEqual(grade['completeness'], 0.5)
        self.assertEqual(grade['available_fact_completeness'], 1)

    def test_rejected_claims_not_credited_and_unknown_claims_not_called_false(self):
        response = copy.deepcopy(self.response)
        response['claims'][1]['citations'][0]['source_id'] = 'S999'
        response['claims'].append({'text': 'A possible additional description.', 'citations': self.response['claims'][0]['citations']})
        answer = answer_from_retrieval(self.retrieval, MockProvider(response=response))
        grade = grade_answer(answer, benchmark_item(), self.state['graph'])
        self.assertEqual(grade['completeness'], 0.5)
        self.assertEqual(grade['rejected_claim_count'], 1)
        self.assertEqual(grade['unverified_claim_count'], 1)
        self.assertEqual(grade['known_false_claim_count'], 0)

    def test_known_contradiction_pattern_is_reported_separately(self):
        item = benchmark_item()
        item['forbidden_patterns'] = [{'pattern': 'always accepts invalid payments', 'description': 'Known fixture falsehood'}]
        response = copy.deepcopy(self.response)
        response['claims'].append({'text': 'It always accepts invalid payments.', 'citations': self.response['claims'][0]['citations']})
        grade = grade_answer(answer_from_retrieval(self.retrieval, MockProvider(response=response)), item, self.state['graph'])
        self.assertEqual(grade['known_false_claim_count'], 1)
        self.assertEqual(grade['correctness_proxy'], 2 / 3)

    def test_no_fact_evidence_acknowledgment(self):
        retrieval = retrieve_strategy(self.state, 'structure', 'quasar astrology')
        answer = answer_from_retrieval(retrieval, MockProvider())
        item = {'question': 'quasar astrology', 'expected': [], 'expect_insufficient': True,
                'reference_answer': 'Insufficient repository evidence.', 'facts': []}
        grade = grade_answer(answer, item, self.state['graph'])
        self.assertEqual(grade['completeness'], 1)
        self.assertEqual(grade['appropriate_insufficiency'], 1)

    def test_benchmark_validation_rejects_bad_labels_and_patterns(self):
        for field, value in [('sources', ['Missing.method()']), ('patterns', ['[bad'])]:
            item = benchmark_item()
            item['facts'][0][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, __import__('re').error)):
                validate_answer_benchmark(self.state['graph'], [item])

    def test_aggregation_preserves_unknown_actual_usage(self):
        report = evaluate_answer_benchmark(self.state, [benchmark_item()], ['structure', 'graph'], MockProvider(), repeats=2)
        self.assertEqual(report['mode'], 'mock_plumbing_only')
        self.assertEqual(len(report['results']), 4)
        for values in report['summary'].values():
            self.assertIsNone(values['mean_actual_input_tokens'])
            self.assertEqual(values['actual_input_usage_samples'], 0)
            self.assertEqual(values['mean_cost_usd'], 0)
            self.assertEqual(len(values['repeat_variation']), 1)
        self.assertIn('unknown', answer_comparison(report))
        json.dumps(report)

    def test_all_strategies_same_prompt_and_no_ground_truth_leakage(self):
        item = benchmark_item()
        item['reference_answer'] = 'REFERENCE_SENTINEL_NOT_FOR_MODEL'
        report = evaluate_answer_benchmark(self.state, [item], STRATEGIES, MockProvider(), top_k=1)
        systems = {row['answer']['prompt_messages'][0]['content'] for row in report['results']}
        self.assertEqual(len(systems), 1)
        for row in report['results']:
            self.assertNotIn('REFERENCE_SENTINEL_NOT_FOR_MODEL', str(row['answer']['prompt_messages']))

    def test_request_upper_bound_and_bad_repeats_rejected_before_calls(self):
        provider = MockProvider()
        with patch.object(provider, 'complete', side_effect=AssertionError('No call')):
            with self.assertRaises(ValueError):
                evaluate_answer_benchmark(self.state, [benchmark_item()], STRATEGIES, provider, max_requests=1)
            with self.assertRaises(ValueError):
                evaluate_answer_benchmark(self.state, [benchmark_item()], ['graph'], provider, repeats=0)

    def test_traditional_annotation_graph_uses_same_relative_paths(self):
        state = prepare_search([self.repo], ['traditional'], encoder=TestEncoder(), cache_dir=self.root / 'cache')
        report = evaluate_answer_benchmark(state, [benchmark_item()], ['traditional'], MockProvider())
        self.assertFalse(report['results'][0]['grade']['expected_insufficient_given_context'])
        self.assertIsNotNone(report['annotation_only_graph_ms'])


class OptimizationAndCliTests(unittest.TestCase):
    def test_rejected_payment_context_and_unmodified_baseline(self):
        state = prepare_search([ROOT / 'examples/shop'], ['graph'])
        question = 'How does checkout handle a rejected payment?'
        baseline = retrieve_strategy(state, 'graph', question, top_k=3)
        optimized = retrieve_strategy(state, 'graph', question, top_k=3, optimization='relevance')
        self.assertEqual(len(optimized['snippets']), 3)
        names = [s['name'] for s in optimized['snippets']]
        self.assertTrue(any('checkout(' in name for name in names))
        self.assertTrue(any('charge(' in name for name in names))
        self.assertTrue(any('release(' in name for name in names))
        self.assertFalse(any('Shipping' in name or 'Audit' in name or 'OrderRepository' in name for name in names))
        self.assertLess(optimized['token_count'], baseline['token_count'])
        self.assertEqual(retrieve_strategy(state, 'graph', question, top_k=3)['context'], baseline['context'])

    def test_implementation_lookup_does_not_expand_even_when_optimized(self):
        state = prepare_search([ROOT / 'examples/shop'], ['graph'])
        result = retrieve_strategy(state, 'graph', 'Show only checkout implementation', optimization='relevance', top_k=1)
        self.assertEqual(len(result['snippets']), 1)
        self.assertEqual(result['expanded_count'], 0)

    def test_mock_qa_cli_without_api_keys(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'answer'
            environment = {key: value for key, value in os.environ.items() if key not in
                           ('OPENAI_API_KEY', 'GROQ_API_KEY', 'LLM_API_KEY', 'LLM_MODEL', 'LLM_PROVIDER', 'LLM_BASE_URL')}
            run = subprocess.run([sys.executable, str(ROOT / 'main.py'), str(ROOT / 'examples/shop'),
                                  '--query', 'How does checkout handle a rejected payment?', '--strategy', 'graph',
                                  '--optimization', 'relevance', '--answer', '--provider', 'mock',
                                  '--mock-response', str(ROOT / 'examples/mock_checkout_answer.json'),
                                  '--output-dir', str(output)], capture_output=True, text=True, env=environment)
            self.assertEqual(run.returncode, 0, run.stderr)
            answer = json.loads((output / 'answer.json').read_text())
            self.assertEqual(answer['status'], 'ok')
            self.assertTrue(answer['mocked'])
            self.assertIsNone(answer['metrics']['actual_input_tokens'])
            self.assertIn('PAYMENT_FAILED', (output / 'answer.txt').read_text())
            self.assertIn('MOCK', run.stdout)

    def test_answer_evaluation_cli_is_explicitly_opt_in(self):
        run = subprocess.run([sys.executable, str(ROOT / 'evaluate_answers.py'), str(ROOT / 'examples/shop')], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn('opt in', run.stderr)

    def test_cli_answer_requires_query_and_valid_credentials(self):
        import os
        env = dict(os.environ)
        env.pop('OPENAI_API_KEY', None)
        for flags in (['--answer'], ['--query', 'checkout', '--answer', '--provider', 'openai', '--model', 'test-model']):
            run = subprocess.run([sys.executable, str(ROOT / 'main.py'), str(ROOT / 'examples/shop'), *flags],
                                 capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 2)
            self.assertNotIn('Traceback', run.stderr)

    def test_full_answer_annotations_match_fixture(self):
        state = prepare_search([ROOT / 'examples/shop'], ['structure'])
        benchmark = json.loads((ROOT / 'benchmarks/shop_answers.json').read_text())
        validate_answer_benchmark(state['graph'], benchmark)
        self.assertEqual(len(benchmark), 36)
        self.assertEqual(sum(len(item['facts']) for item in benchmark), 70)


if __name__ == '__main__':
    unittest.main()

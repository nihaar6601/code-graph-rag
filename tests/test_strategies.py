"""Five-strategy integration and shared source-range evaluation with fake vectors."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate import evaluate_strategies
from hybrid import classify_intent
from relevance import expected_records, merge_ranges, source_metrics
from strategies import STRATEGIES, prepare_search, retrieve_strategy
from token_counter import count_tokens

ROOT = Path(__file__).resolve().parents[1]


class FakeEncoder:
    dimension = 4
    cache_identity = 'deterministic-integration-v1'
    model_name = 'fake-test-only'
    load_ms = 0

    def encode(self, texts):
        groups = [('payment', 'charge', 'refund'), ('stock', 'inventory'), ('shipping', 'dispatch'), ('audit', 'record')]
        return np.asarray([[sum(text.lower().count(word) for word in group) for group in groups]
                           for text in texts], dtype=np.float32).reshape(len(texts), 4)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'src' / 'PaymentService.java'
        self.source.parent.mkdir()
        self.source.write_text('''class PaymentService {
    void processPayment() { validatePayment(); }
    void validatePayment() { Audit.record(); }
}
''')
        (self.source.parent / 'Audit.java').write_text('class Audit { static void record() {} }')
        self.encoder = FakeEncoder()

    def prepare(self, strategies=STRATEGIES, **kwargs):
        return prepare_search([self.root / 'src'], strategies, encoder=self.encoder,
                              cache_dir=self.root / 'cache', chunk_size=32, overlap=8, **kwargs)

    def test_all_five_strategies_and_budgets(self):
        state = self.prepare()
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                result = retrieve_strategy(state, strategy, 'How does payment processing work?', top_k=1, max_tokens=300)
                self.assertEqual(result['strategy'], strategy)
                self.assertTrue(result['snippets'])
                self.assertEqual(result['token_count'], count_tokens(result['context']))
                self.assertLessEqual(result['token_count'], 300)
                json.dumps(result)

    def test_traditional_is_independent_of_parser_and_graph(self):
        with patch('strategies.index_inputs', side_effect=AssertionError('No parser')):
            with patch('strategies.build_graph', side_effect=AssertionError('No graph')):
                state = self.prepare(['traditional'])
                result = retrieve_strategy(state, 'traditional', 'payment')
        self.assertIsNone(state['graph'])
        self.assertTrue(all(s['kind'] == 'chunk' for s in result['snippets']))

    def test_entity_semantics_and_fusion_metadata(self):
        state = self.prepare(['hybrid'])
        result = retrieve_strategy(state, 'hybrid', 'processPayment', top_k=1)
        self.assertEqual(result['snippets'][0]['name'], 'PaymentService.processPayment()')
        self.assertIn('lexical', result['candidates'][0]['fusion'])
        self.assertIn('semantic', result['candidates'][0]['fusion'])
        self.assertEqual(result['expanded_count'], 0)

    def test_semantic_failure_falls_back_to_lexical(self):
        with patch('strategies.LocalEncoder', side_effect=RuntimeError('Model not prepared')):
            state = prepare_search([self.root / 'src'], ['hybrid'])
        result = retrieve_strategy(state, 'hybrid', 'processPayment', top_k=1)
        self.assertTrue(result['snippets'])
        self.assertIn('semantic_unavailable', [d['kind'] for d in result['diagnostics']])
        self.assertNotIn('semantic', result['candidates'][0]['fusion'])

    def test_unavailable_traditional_is_marked_not_presented_as_semantic_success(self):
        with patch('strategies.LocalEncoder', side_effect=RuntimeError('Model not prepared')):
            state = prepare_search([self.root / 'src'], ['traditional'])
        result = retrieve_strategy(state, 'traditional', 'payment')
        self.assertEqual(result['semantic_status'], 'unavailable')
        self.assertFalse(result['snippets'])

    def test_no_semantic_or_lexical_matches(self):
        state = self.prepare()
        for strategy in STRATEGIES:
            result = retrieve_strategy(state, strategy, 'quasar zebra nebula', top_k=1)
            self.assertFalse(result['snippets'])
            self.assertIn('no_matches', [d['kind'] for d in result['diagnostics']])

    def test_auto_intent_and_explicit_overrides(self):
        state = self.prepare()
        for strategy in ('graph', 'hybrid_graph'):
            result = retrieve_strategy(state, strategy, 'Show only the processPayment implementation', top_k=1)
            self.assertEqual(result['expanded_count'], 0)
            result = retrieve_strategy(state, strategy, 'Show only the processPayment implementation', top_k=1, expansion='always')
            self.assertEqual(result['expanded_count'], 1)
            result = retrieve_strategy(state, strategy, 'Trace processPayment call chain', top_k=1, depth=2)
            self.assertEqual(result['expanded_count'], 2)
            result = retrieve_strategy(state, strategy, 'Trace processPayment call chain', top_k=1, expansion='never')
            self.assertEqual(result['expanded_count'], 0)
        self.assertEqual(classify_intent('processPayment'), 'lookup')
        self.assertEqual(classify_intent('Show checkout without dependencies'), 'implementation_only')

    def test_traditional_and_entity_source_change_rejected(self):
        state = self.prepare()
        self.source.write_text('// changed\n' + self.source.read_text())
        for strategy in ('traditional', 'hybrid'):
            result = retrieve_strategy(state, strategy, 'processPayment', top_k=1)
            self.assertFalse(result['snippets'])
            self.assertIn('source_unavailable', [d['kind'] for d in result['diagnostics']])

    def test_chunk_attribution_changes_are_rejected(self):
        from chunking import verify_chunk
        state = self.prepare(['traditional'])
        chunk = dict(state['chunks'][0])
        chunk['start_line'] += 1
        with self.assertRaises(ValueError):
            verify_chunk(chunk, state['chunk_indexing']['root'])

    def test_reusable_embedding_caches(self):
        first = self.prepare()
        second = self.prepare()
        for key in ('chunks', 'entities'):
            self.assertGreater(first['timings'][key]['cache_misses'], 0)
            self.assertEqual(second['timings'][key]['cache_misses'], 0)
            self.assertGreater(second['timings'][key]['cache_hits'], 0)

    def test_small_budget_and_tiny_budget(self):
        state = self.prepare()
        for strategy in STRATEGIES:
            result = retrieve_strategy(state, strategy, 'payment', max_tokens=1)
            self.assertEqual(result['context'], '')
            self.assertEqual(result['token_count'], 0)

    def test_deterministic_context_and_ranking_with_mock_embeddings(self):
        state = self.prepare()
        for strategy in STRATEGIES:
            first = retrieve_strategy(state, strategy, 'payment', top_k=2)
            second = retrieve_strategy(state, strategy, 'payment', top_k=2)
            self.assertEqual(first['candidates'], second['candidates'])
            self.assertEqual(first['context'], second['context'])

    def test_explicit_class_queries_allow_class_snippets(self):
        state = self.prepare()
        result = retrieve_strategy(state, 'structure', 'Where is the PaymentService class declared?', top_k=1)
        self.assertEqual(result['snippets'][0]['kind'], 'class')
        self.assertEqual(retrieve_strategy(state, 'structure', 'PaymentService', top_k=1)['snippets'][0]['kind'], 'class')

    def test_invalid_limits_and_strategy(self):
        state = self.prepare(['structure'])
        for options in ({'top_k': 0}, {'max_tokens': -1}, {'min_similarity': 2}, {'rrf_k': 0}, {'depth': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                retrieve_strategy(state, 'structure', 'payment', **options)
        with self.assertRaises(ValueError):
            retrieve_strategy(state, 'unknown', 'payment')

    def test_shared_evaluation_and_recall_at_k(self):
        benchmark = [{'question': 'Trace payment processing call chain', 'category': 'dependency',
                      'expected': ['PaymentService.processPayment()', 'PaymentService.validatePayment()']}]
        report = evaluate_strategies([self.root / 'src'], benchmark, STRATEGIES, top_k=1,
                                    encoder=self.encoder, cache_dir=self.root / 'cache')
        self.assertEqual(len(report['results']), 5)
        self.assertIn('graph_indexing_ms', report['overhead'])
        self.assertIn('embedding_setup_ms', report['overhead'])
        self.assertEqual(report['model']['name'], 'fake-test-only')
        graph_row = next(row for row in report['results'] if row['strategy'] == 'graph')
        self.assertGreaterEqual(graph_row['recall'], graph_row['recall_at_k'])
        self.assertEqual(graph_row['expanded_count'], 1)
        json.dumps(report)

    def test_legacy_cli_and_new_lexical_cli_no_model_loading(self):
        for strategy in ('structure', 'graph'):
            path = self.root / f'{strategy}.json'
            run = subprocess.run([sys.executable, str(ROOT / 'main.py'), str(self.root / 'src'),
                                  '--strategy', strategy, '--query', 'Show only processPayment implementation',
                                  '--top-k', '1', '--context-json', str(path)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(path.read_text())['expanded_count'], 0)
            self.assertNotIn('Loading weights', run.stderr)

    def test_shop_benchmark_labels_and_categories(self):
        from graph_builder import build_graph
        from indexer import index_inputs
        indexed = index_inputs([ROOT / 'examples/shop'])
        graph = build_graph(indexed['files'])
        benchmark = json.loads((ROOT / 'benchmarks/shop_queries.json').read_text())
        self.assertTrue(30 <= len(benchmark) <= 50)
        self.assertGreaterEqual(sum(not q['expected'] for q in benchmark), 3)
        self.assertEqual(len({q['category'] for q in benchmark}), 8)
        for item in benchmark:
            expected_records(graph, item)


def interval(start, end, path='A.java', label='method'):
    return {'file_path': path, 'start_byte': start, 'end_byte': end, 'label': label}


class SourceRelevanceTests(unittest.TestCase):
    def test_partial_chunk_is_not_a_whole_method(self):
        metrics = source_metrics([interval(150, 170)], [interval(100, 200)])
        self.assertEqual(metrics['precision'], 1)
        self.assertEqual(metrics['recall'], 0.2)
        self.assertEqual(metrics['relevant_entities'], 0)
        self.assertEqual(metrics['relevant_snippets'], 1)

    def test_large_chunk_receives_irrelevant_byte_penalty(self):
        metrics = source_metrics([interval(0, 400)], [interval(100, 200)])
        self.assertEqual((metrics['precision'], metrics['recall']), (0.25, 1))
        self.assertEqual(metrics['f1'], 0.4)
        self.assertEqual(metrics['relevant_entities'], 1)

    def test_overlapping_chunks_and_labels_are_unioned(self):
        metrics = source_metrics([interval(90, 160), interval(140, 210)],
                                 [interval(100, 180), interval(150, 200)])
        self.assertEqual(metrics['retrieved_source_bytes'], 120)
        self.assertEqual(metrics['expected_source_bytes'], 100)
        self.assertEqual(metrics['relevant_source_bytes'], 100)

    def test_different_files_do_not_overlap(self):
        metrics = source_metrics([interval(0, 100, 'other.java')], [interval(0, 100)])
        self.assertEqual((metrics['precision'], metrics['recall']), (0, 0))

    def test_empty_ground_truth_and_predictions(self):
        self.assertEqual(source_metrics([], [])['f1'], 1)
        self.assertEqual(source_metrics([interval(0, 5)], [])['precision'], 0)
        self.assertEqual(source_metrics([], [interval(0, 5)])['recall'], 0)

    def test_bad_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_ranges([interval(5, 5)])

    def test_coverage_threshold_is_explicit(self):
        metrics = source_metrics([interval(0, 79)], [interval(0, 100)])
        self.assertEqual(metrics['relevant_entities'], 0)
        self.assertEqual(source_metrics([interval(0, 80)], [interval(0, 100)])['relevant_entities'], 1)


if __name__ == '__main__':
    unittest.main()

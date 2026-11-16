"""Lexical ranking, expansion, verified source, budgets and evaluation checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluate import evaluate_graph, retrieval_metrics
from graph_builder import build_graph
from indexer import index_inputs
from parser import parse_file
from retrieval import (assemble_context, entity_label, expand_dependencies, identifier_tokens,
                       load_snippet, rank_entities, retrieve_context)
from token_counter import count_tokens, get_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def indexed_graph(directory):
    indexed = index_inputs([directory])
    graph = build_graph(indexed['files'])
    graph.graph['indexing'] = {'root': indexed['root']}
    return graph


def named_id(graph, name):
    return next(entity_id for entity_id, entity in graph.nodes(data=True)
                if entity.get('name') == name and entity['kind'] == 'method')


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'PaymentService.java'
        self.source.write_text('''class PaymentService {
    void processPayment() { validatePayment(); }
    void validatePayment() { finish(); }
    void finish() { processPayment(); }
    void unrelated() {}
}''')
        self.graph = indexed_graph(self.root)
        self.seed = named_id(self.graph, 'processPayment')

    def retrieve(self, query='processPayment', **kwargs):
        return retrieve_context(self.graph, query, top_k=1, **kwargs)

    def test_exact_method_name(self):
        self.assertEqual(rank_entities(self.graph, 'processPayment')[0]['id'], self.seed)

    def test_partial_method_name(self):
        self.assertEqual(rank_entities(self.graph, 'processPay')[0]['id'], self.seed)

    def test_exact_identifier_can_also_be_an_english_stopword(self):
        self.source.write_text('class Worker { void work() {} }')
        graph = indexed_graph(self.root)
        result = retrieve_context(graph, 'work', top_k=1)
        self.assertEqual(result['snippets'][0]['name'], 'Worker.work()')

    def test_camelcase_acronyms_and_underscores(self):
        self.assertEqual(identifier_tokens('HTTPPaymentService process_payment'), ['http', 'payment', 'service', 'process', 'payment'])
        self.assertEqual(rank_entities(self.graph, 'process payment')[0]['id'], self.seed)

    def test_natural_language_ranking(self):
        result = self.retrieve('How does payment processing work?')
        self.assertEqual(result['snippets'][0]['id'], self.seed)
        self.assertEqual(result['snippets'][0]['reason'], 'Direct query match')

    def test_classes_are_ranked_but_not_automatically_included(self):
        self.assertTrue(any(c['kind'] == 'class' for c in rank_entities(self.graph, 'PaymentService')))
        result = self.retrieve('PaymentService')
        self.assertTrue(all(s['kind'] != 'class' for s in result['snippets']))

    def test_explicit_class_snippets_deduplicate_overlapping_methods(self):
        result = retrieve_context(self.graph, 'PaymentService', include_classes=True, top_k=10, expand=False)
        self.assertEqual(len(result['snippets']), 1)
        self.assertEqual(result['snippets'][0]['kind'], 'class')
        self.assertTrue(any(d['kind'] == 'overlap' for d in result['diagnostics']))

    def test_depth_zero_one_two_and_cycles(self):
        self.assertEqual(len(self.retrieve(depth=0)['snippets']), 1)
        self.assertEqual(len(self.retrieve(depth=1)['snippets']), 2)
        self.assertEqual(len(self.retrieve(depth=2)['snippets']), 3)
        result = self.retrieve(depth=20)
        self.assertEqual(len(result['snippets']), 3)
        self.assertEqual(len({s['id'] for s in result['snippets']}), 3)

    def test_expansion_can_be_disabled(self):
        result = self.retrieve(expand=False, depth=20)
        self.assertEqual((result['direct_count'], result['expanded_count']), (1, 0))

    def test_incoming_relationships(self):
        result = self.retrieve('validatePayment', include_callers=True, depth=1)
        self.assertIn(self.seed, [s['id'] for s in result['snippets']])
        self.assertTrue(any(s['reason'].startswith('Calls ') for s in result['snippets']))

    def test_expansion_entity_cap(self):
        self.assertEqual(len(self.retrieve(depth=10, max_expanded=1)['snippets']), 2)
        self.assertEqual(len(self.retrieve(depth=10, max_expanded=0)['snippets']), 1)

    def test_repeated_edges_do_not_duplicate_snippets(self):
        target = named_id(self.graph, 'validatePayment')
        self.graph.add_edge(self.seed, target, kind='CALLS')
        self.assertEqual(len(self.retrieve(depth=1)['snippets']), 2)

    def test_multiple_direct_matches_stay_ahead_of_dependencies(self):
        result = retrieve_context(self.graph, 'payment', top_k=2, depth=2)
        distances = [s['distance'] for s in result['snippets']]
        self.assertEqual(distances[:2], [0, 0])
        self.assertTrue(all(distance > 0 for distance in distances[2:]))

    def test_exact_source_and_locations(self):
        snippet = load_snippet(self.graph, self.seed)
        self.assertEqual(snippet['source'], 'void processPayment() { validatePayment(); }')
        self.assertEqual((snippet['start_line'], snippet['end_line']), (2, 2))
        self.assertEqual(snippet['class_context'], ['PaymentService'])

    def test_complete_context_budget_includes_all_formatting(self):
        full = self.retrieve(depth=0)
        budget = full['token_count']
        result = self.retrieve(max_tokens=budget, depth=2)
        self.assertEqual(result['token_count'], count_tokens(result['context']))
        self.assertLessEqual(result['token_count'], budget)
        self.assertEqual(len(result['snippets']), 1)
        self.assertTrue(any(d['kind'] == 'budget_exhausted' for d in result['diagnostics']))
        self.assertEqual(result['snippets'][0]['source'], full['snippets'][0]['source'])

    def test_oversized_method_is_reported_without_truncation(self):
        full = self.retrieve(depth=0)
        result = self.retrieve(max_tokens=full['token_count'] - 1)
        self.assertEqual(result['snippets'], [])
        self.assertIn('oversized_method', [d['kind'] for d in result['diagnostics']])
        self.assertLessEqual(result['token_count'], result['max_tokens'])

    def test_tiny_and_zero_budgets(self):
        for budget in (0, 1, 3):
            result = self.retrieve(max_tokens=budget)
            self.assertEqual(result['context'], '')
            self.assertEqual(result['token_count'], 0)
            self.assertEqual(result['diagnostics'][0]['kind'], 'question_over_budget')

    def test_missing_source_file(self):
        self.source.unlink()
        result = self.retrieve()
        self.assertEqual(result['snippets'], [])
        self.assertEqual(result['diagnostics'][0]['kind'], 'source_unavailable')

    def test_modified_source_and_changes_outside_method(self):
        original = self.source.read_bytes()
        for modified in (original.replace(b'finish()', b'changed()'), b'// inserted\n' + original, original + b'\n'):
            self.source.write_bytes(modified)
            result = self.retrieve()
            self.assertFalse(result['snippets'])
            self.assertIn('changed after indexing', result['diagnostics'][0]['message'])

    def test_missing_fingerprint_requires_reindex(self):
        self.graph.graph['files'][0].pop('source_sha256')
        self.assertIn('fingerprint', self.retrieve()['diagnostics'][0]['message'])

    def test_incorrect_saved_locations_are_rejected(self):
        for field in ('start_line', 'start_column', 'start_byte', 'end_byte'):
            with self.subTest(field=field):
                original = self.graph.nodes[self.seed][field]
                self.graph.nodes[self.seed][field] += 1
                self.assertFalse(self.retrieve()['snippets'])
                self.graph.nodes[self.seed][field] = original

    def test_non_utf8_and_malformed_sources_are_not_retrieved(self):
        for data in (b'class Broken { void processPayment() {', b'//\xff\nclass Broken { void processPayment() {} }'):
            self.source.write_bytes(data)
            graph = indexed_graph(self.root)
            self.assertEqual(retrieve_context(graph, 'processPayment')['snippets'], [])

    def test_no_matches_and_stopword_only_query(self):
        for query in ('warehousezebra inventoryquartz', 'how does the'):
            result = self.retrieve(query)
            self.assertFalse(result['snippets'])
            self.assertEqual(result['diagnostics'][0]['kind'], 'no_matches')

    def test_deterministic_context_and_ranking(self):
        first = self.retrieve(depth=2)
        second = self.retrieve(depth=2)
        self.assertEqual(first['context'], second['context'])
        self.assertEqual(first['snippets'], second['snippets'])
        self.assertEqual(first['candidates'], second['candidates'])
        reversed_graph = self.graph.__class__()
        reversed_graph.graph.update(copy.deepcopy(self.graph.graph))
        reversed_graph.add_nodes_from(reversed(list(self.graph.nodes(data=True))))
        reversed_graph.add_edges_from(self.graph.edges(data=True))
        self.assertEqual(rank_entities(self.graph, 'payment'), rank_entities(reversed_graph, 'payment'))

    def test_result_is_json_serializable(self):
        result = self.retrieve()
        self.assertEqual(json.loads(json.dumps(result))['context'], result['context'])

    def test_no_graph_mutation(self):
        before = copy.deepcopy(dict(self.graph.nodes(data=True)))
        self.retrieve()
        self.assertEqual(dict(self.graph.nodes(data=True)), before)

    def test_invalid_options(self):
        for options in ({'max_tokens': -1}, {'depth': -1}, {'max_expanded': -1}, {'query': ''}, {'top_k': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                retrieve_context(self.graph, **dict({'query': 'payment'}, **options))

    def test_special_token_like_source_is_counted_as_plain_text(self):
        self.assertGreater(count_tokens('<|endoftext|>'), 0)
        self.assertEqual(count_tokens('hello world'), len(get_tokenizer().encode('hello world')))

    def test_absolute_file_mode_and_explicit_root(self):
        graph = build_graph(parse_file(self.source))
        self.assertTrue(retrieve_context(graph, 'processPayment')['snippets'])
        self.graph.graph.pop('indexing')
        self.assertFalse(self.retrieve()['snippets'])
        self.assertTrue(retrieve_context(self.graph, 'processPayment', root=self.root)['snippets'])

    def test_relative_path_escape_rejected(self):
        self.graph.nodes[self.seed]['file_path'] = '../outside.java'
        self.graph.graph['files'].append({'file_path': '../outside.java', 'source_sha256': 'unused'})
        self.assertIn('escapes', self.retrieve()['diagnostics'][0]['message'])

    def test_unreadable_file(self):
        with patch('retrieval.Path.read_bytes', side_effect=PermissionError('denied')):
            self.assertIn('denied', self.retrieve()['diagnostics'][0]['message'])


class ExampleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.graph = indexed_graph(ROOT / 'examples/repository')

    def test_cross_file_expansion(self):
        result = retrieve_context(self.graph, 'controller', top_k=1, depth=1)
        self.assertEqual(len(result['snippets']), 3)
        self.assertEqual(len({s['file_path'] for s in result['snippets']}), 3)
        self.assertEqual(result['expanded_count'], 2)

    def test_depth_two_reaches_validation(self):
        result = retrieve_context(self.graph, 'controller', top_k=1, depth=2)
        self.assertEqual(len(result['snippets']), 4)
        self.assertTrue(any('validatePayment' in s['name'] for s in result['snippets']))

    def test_benchmark_precision_recall_and_strategy_configuration(self):
        benchmark = json.loads((ROOT / 'benchmarks/payment_queries.json').read_text())
        report = evaluate_graph(self.graph, benchmark)
        self.assertEqual(len(report['results']), 8)
        structure, enhanced = report['results'][:2]
        self.assertEqual((structure['precision'], structure['recall']), (1.0, 0.5))
        self.assertEqual((enhanced['precision'], enhanced['recall']), (1.0, 1.0))
        self.assertEqual(report['results'][5]['precision'], 0.5)
        self.assertEqual(report['summary']['structure_only']['macro_precision'], 1.0)
        self.assertEqual(report['summary']['graph_enhanced']['macro_precision'], 0.875)
        self.assertTrue(all(row['context_tokens'] <= report['max_tokens'] for row in report['results']))
        json.dumps(report)

    def test_empty_retrieval_metrics(self):
        self.assertEqual(retrieval_metrics([], []), (1.0, 1.0))
        self.assertEqual(retrieval_metrics([], ['expected']), (0.0, 0.0))
        self.assertEqual(retrieval_metrics(['unexpected'], []), (0.0, 1.0))

    def test_missing_benchmark_entity_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            evaluate_graph(self.graph, [{'question': 'anything', 'expected': ['missing']}])


class RetrievalCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'main.py'), str(ROOT / 'examples/repository'), *map(str, args)],
                              capture_output=True, text=True)

    def test_cli_query_and_output_directory(self):
        run = self.run_cli('--query', 'processPayment', '--top-k', 1, '--output-dir', self.output)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads((self.output / 'retrieval.json').read_text())
        self.assertEqual(result['context'], (self.output / 'context.txt').read_text())
        self.assertEqual((result['direct_count'], result['expanded_count']), (1, 1))
        self.assertIn('source_sha256', result['files'][0])

    def test_context_and_graph_export_remain_separate(self):
        graph_path, context_path = self.output / 'graph.json', self.output / 'context.json'
        run = self.run_cli('--query', 'controller', '--depth', 2, '--top-k', 1, '--no-expand',
                           '--context-json', context_path, '--json', graph_path)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('entities', json.loads(graph_path.read_text()))
        result = json.loads(context_path.read_text())
        self.assertEqual(result['expanded_count'], 0)
        self.assertFalse(result['configuration']['expand'])

    def test_cli_budget_and_incoming_options(self):
        run = self.run_cli('--query', 'validatePayment', '--include-callers', '--max-tokens', 150,
                           '--context-json', self.output / 'context.json')
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads((self.output / 'context.json').read_text())
        self.assertLessEqual(result['token_count'], 150)
        self.assertTrue(result['configuration']['include_callers'])

    def test_cli_bad_options(self):
        for options in (('--query', ''), ('--query', 'payment', '--depth', -1), ('--context-json', self.output / 'x.json')):
            run = self.run_cli(*options)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertNotIn('Traceback', run.stderr)

    def test_cli_conflicting_outputs_rejected(self):
        run = self.run_cli('--query', 'payment', '--json', self.output / 'same.json',
                           '--context-json', self.output / 'same.json')
        self.assertEqual(run.returncode, 2)
        self.assertIn('distinct paths', run.stderr)

    def test_cli_retrieval_output_cannot_overwrite_source(self):
        source = ROOT / 'examples/repository/src/demo/service/PaymentService.java'
        before = source.read_bytes()
        run = self.run_cli('--query', 'payment', '--context-json', source)
        self.assertEqual(run.returncode, 1)
        self.assertEqual(source.read_bytes(), before)

    def test_evaluation_cli_export(self):
        path = self.output / 'evaluation.json'
        run = subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--json', str(path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(path.read_text())
        self.assertEqual(len(report['results']), 8)
        self.assertIn('not answer accuracy', run.stdout)


if __name__ == '__main__':
    unittest.main()

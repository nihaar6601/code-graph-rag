"""Regression and end-to-end validation beyond the initial examples."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from graph_builder import build_graph, graph_to_dict
from parser import parse_file, parse_source

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def calls(graph):
    return [(source, target) for source, target, edge in graph.edges(data=True)
            if edge['kind'] == 'CALLS']


class ValidationTests(unittest.TestCase):
    def test_recursion_and_mutual_recursion(self):
        result = parse_source('class A { void a() { a(); b(); } void b() { a(); } }')
        a, b = result['classes'][0]['methods']
        self.assertCountEqual(calls(build_graph(result)), [(a['id'], a['id']), (a['id'], b['id']), (b['id'], a['id'])])

    def test_nested_duplicate_names_resolve_to_their_own_class(self):
        result = parse_source('class A { void f() { this.f(); } class B { void f() { f(); } } }')
        outer, inner = [cls['methods'][0]['id'] for cls in result['classes']]
        self.assertCountEqual(calls(build_graph(result)), [(outer, outer), (inner, inner)])

    def test_duplicate_declaration_calls_are_unresolved(self):
        graph = build_graph(parse_source('class A { void f() {} void f() {} void g() { f(); } }'))
        self.assertEqual(len(graph), 4)
        self.assertFalse(calls(graph))
        self.assertIn('duplicate', graph.graph['invocations'][0]['unresolved_reason'])

    def test_comments_are_not_arguments(self):
        for arguments, count in [('/* empty */', 0), ('1 /* first */, // second\n 2', 2)]:
            with self.subTest(arguments=arguments):
                parameters = '' if count == 0 else 'int a, int b'
                graph = build_graph(parse_source(f'class A {{ void f() {{ g({arguments}); }} void g({parameters}) {{}} }}'))
                self.assertEqual(graph.graph['invocations'][0]['argument_count'], count)
                self.assertEqual(len(calls(graph)), 1)

    def test_comments_do_not_corrupt_varargs_types(self):
        method = parse_source('class A { void f(String /* note */ ... args) {} }')['classes'][0]['methods'][0]
        self.assertEqual(method['signature'], 'f(String...)')
        self.assertEqual(method['parameters'][0]['name'], 'args')

    def test_receiver_parameter_is_not_a_call_argument(self):
        result = parse_source('class A { void f(A this, int x) {} void g() { f(1); } }')
        method = result['classes'][0]['methods'][0]
        self.assertEqual(method['parameters'][0], {'kind': 'receiver_parameter', 'name': 'this', 'type': 'A'})
        self.assertEqual(method['signature'], 'f(int)')
        self.assertEqual(len(calls(build_graph(result))), 1)

    def test_anonymous_owner_does_not_leak_nested_class(self):
        result = parse_source('class A { Object x = new Object() { class Hidden {} }; class Visible {} }')
        self.assertEqual([cls['qualified_name'] for cls in result['classes']], ['A', 'A.Visible'])

    def test_unsupported_owner_does_not_leak_nested_class(self):
        result = parse_source('interface I { class Hidden {} } class Visible {}')
        self.assertEqual([cls['qualified_name'] for cls in result['classes']], ['Visible'])

    def test_local_enum_calls_do_not_leak_to_enclosing_method(self):
        result = parse_source('class A { void f() { enum E { X; void g() { external(); } } } }')
        self.assertFalse(result['has_error'])
        self.assertEqual(result['classes'][0]['methods'][0]['calls'], [])

    def test_constructors_are_source_declared_only(self):
        result = parse_file(ROOT / 'examples/PaymentService.java')
        constructors = [m for m in result['classes'][0]['methods'] if m['kind'] == 'constructor']
        self.assertEqual(len(constructors), 1)
        constructor = constructors[0]
        self.assertEqual(constructor['name'], 'PaymentService')
        self.assertEqual(constructor['source'], 'PaymentService() {}')
        self.assertEqual(constructor['start_line'], 2)
        self.assertIsNone(constructor['return_type'])
        self.assertEqual(parse_source('class A {}')['classes'][0]['methods'], [])

    def test_multiline_source_locations_with_utf8_byte_columns(self):
        source = 'class Café {\n  String f(String x) {\n    return x.trim();\n  }\n}\n'
        result = parse_source(source)
        entities = [result['classes'][0], result['classes'][0]['methods'][0]]
        invocation = entities[1]['calls'][0]
        entities.append(invocation)
        lines = source.encode().splitlines(keepends=True)
        for entity in entities:
            start = sum(map(len, lines[:entity['start_line'] - 1])) + entity['start_column'] - 1
            end = sum(map(len, lines[:entity['end_line'] - 1])) + entity['end_column'] - 1
            self.assertEqual(source.encode()[start:end].decode(), entity['source'])
        method = entities[1]
        self.assertEqual((method['start_line'], method['end_line']), (2, 4))
        same_line = parse_source('class Café { void f() { g(); } }')['classes'][0]['methods'][0]['calls'][0]
        self.assertEqual(same_line['start_column'], len('class Café { void f() { '.encode()) + 1)

    def test_chained_calls_are_preserved_without_false_edges(self):
        graph = build_graph(parse_source('class A { void f() { factory().external(); other.f(); } Object factory() { return null; } }'))
        self.assertEqual(len(graph.graph['invocations']), 3)
        self.assertEqual(len(calls(graph)), 1)
        self.assertEqual(sum(c['resolved_target'] is None for c in graph.graph['invocations']), 2)

    def test_different_arity_overloads_remain_unresolved(self):
        graph = build_graph(parse_source('class A { void f() {} void f(int x) {} void g() { f(); f(1); } }'))
        self.assertFalse(calls(graph))
        self.assertEqual(len(graph), 4)
        self.assertEqual(len(graph.graph['invocations']), 2)

    def test_invalid_paths_and_directories_have_diagnostics(self):
        for path in (str(ROOT), 'bad\0path.java'):
            with self.subTest(path=path):
                result = parse_file(path)
                self.assertTrue(result['has_error'])
                self.assertEqual(result['diagnostics'][0]['kind'], 'file_error')

    def test_all_examples_cli_json(self):
        examples = sorted((ROOT / 'examples').glob('*.java'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'graph.json'
            run = run_cli(*examples, '--json', output)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(output.read_text())
        self.assertEqual(len(data['files']), 3)
        self.assertEqual(len(data['entities']), 14)
        self.assertEqual(len(data['relationships']), 13)
        self.assertEqual(len(data['invocations']), 4)
        entities = {e['id']: e for e in data['entities']}
        self.assertEqual(len(entities), len(data['entities']))
        for entity in entities.values():
            for field in ('file_path', 'source', 'start_line', 'end_line', 'start_column', 'end_column'):
                self.assertIn(field, entity)
        for edge in data['relationships']:
            self.assertIn(edge['source'], entities)
            self.assertIn(edge['target'], entities)
            if edge['kind'] == 'CALLS':
                self.assertEqual(edge['invocation']['caller'], edge['source'])
                self.assertEqual(edge['invocation']['resolved_target'], edge['target'])
                self.assertEqual(entities[edge['source']]['class_id'], entities[edge['target']]['class_id'])
                self.assertEqual(edge['invocation']['start_line'], entities[edge['source']]['start_line'] + (1 if entities[edge['target']]['name'] == 'validatePayment' else 2))
            else:
                self.assertEqual(entities[edge['source']]['kind'], 'class')
        self.assertEqual(sum(c['resolved_target'] is None for c in data['invocations']), 2)

    def test_cli_path_aliases_do_not_duplicate_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'graph.json'
            run = run_cli('examples/PaymentService.java', ROOT / 'examples/PaymentService.java', '--json', output)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(output.read_text())
        self.assertEqual(len(data['files']), 1)
        self.assertEqual(len(data['relationships']), 6)
        self.assertEqual(len(data['invocations']), 3)

    def test_cli_multiple_files_do_not_cross_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'A.java', Path(directory) / 'B.java'
            a.write_text('class Same { void f() { g(); } }')
            b.write_text('class Same { void g() {} }')
            output = Path(directory) / 'graph.json'
            run = run_cli(a, b, '--json', output)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(output.read_text())
        self.assertEqual(len({e['id'] for e in data['entities']}), 4)
        self.assertFalse(any(e['kind'] == 'CALLS' for e in data['relationships']))
        self.assertIsNone(data['invocations'][0]['resolved_target'])

    def test_cli_diagnostics_and_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / 'Invalid.java'
            invalid.write_text('class A { void f() {')
            output = Path(directory) / 'partial.json'
            run = run_cli(invalid, Path(directory) / 'Missing.java', '--json', output)
            self.assertEqual(run.returncode, 1, run.stderr)
            data = json.loads(output.read_text())
        self.assertTrue(all(f['has_error'] and f['diagnostics'] for f in data['files']))
        self.assertIn('Problem', run.stdout)

    def test_cli_export_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = run_cli('--json', directory)
        self.assertEqual(run.returncode, 1)
        self.assertIn('Could not export graph', run.stderr)
        self.assertNotIn('Traceback', run.stderr)

    def test_cli_does_not_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'A.java'
            original = 'class A {}'
            source.write_text(original)
            run = run_cli(source, '--json', source)
            self.assertEqual(run.returncode, 1)
            self.assertIn('must not overwrite', run.stderr)
            self.assertEqual(source.read_text(), original)

    def test_default_cli_and_parser_demo(self):
        run = run_cli()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('CALLS -> validatePayment()', run.stdout)
        run = subprocess.run([sys.executable, str(ROOT / 'parser.py')], cwd=ROOT,
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('add(int,int)', run.stdout)


if __name__ == '__main__':
    unittest.main()

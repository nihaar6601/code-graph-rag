"""Repository discovery and conservative cross-file resolution regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from graph_builder import build_graph, graph_to_dict
from indexer import IGNORED_DIRECTORIES, index_inputs
from parser import parse_source

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'package p; public class Service { public void run() {} public static void ping() {} }'


def source_graph(sources):
    return build_graph([parse_source(source, path) for path, source in sources.items()])


def resolved_pairs(graph):
    return [(graph.nodes[a]['name'], graph.nodes[b]['name'])
            for a, b, edge in graph.edges(data=True) if edge['kind'] == 'CALLS']


def simulate_scan_error(directory, followlinks, onerror):
    onerror(PermissionError(13, 'Permission denied', str(Path(directory) / 'denied')))
    return []


class ResolutionTests(unittest.TestCase):
    def test_field_receiver_same_package(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java':
                              'package p; class Client { Service service; void f() { service.run(); } }'})
        self.assertEqual(resolved_pairs(graph), [('f', 'run')])

    def test_imported_field_local_parameter_and_static_receivers(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package app;
            import p.Service;
            class Client { Service field; void f(Service parameter) {
                field.run(); Service local = new Service(); local.run();
                parameter.run(); Service.ping(); this.field.run();
            } }'''})
        self.assertCountEqual(resolved_pairs(graph), [('f', 'run')] * 4 + [('f', 'ping')])

    def test_package_and_import_metadata(self):
        result = parse_source('package app; import p.Service; import p.*; import static p.Service.ping; class C {}')
        self.assertEqual(result['package'], 'app')
        self.assertEqual([(i['name'], i['static'], i['wildcard']) for i in result['imports']],
                         [('p.Service', False, False), ('p', False, True), ('p.Service.ping', True, False)])
        self.assertTrue(all(i['source'].startswith('import') and i['start_line'] == 1 for i in result['imports']))

    def test_duplicate_simple_names_use_import_not_global_guess(self):
        graph = source_graph({'p/Service.java': SERVICE,
                              'other/Service.java': SERVICE.replace('package p;', 'package other;'),
                              'Client.java': 'package app; import p.Service; class Client { void f(Service s) { s.run(); } }'})
        target = next(c['resolved_target'] for c in graph.graph['invocations'])
        self.assertEqual(graph.nodes[graph.nodes[target]['class_id']]['qualified_name'], 'p.Service')
        self.assertEqual(sum(e['kind'] == 'class' for _, e in graph.nodes(data=True)), 3)

    def test_same_package_lookup(self):
        graph = source_graph({'p/Service.java': SERVICE, 'other/Service.java': SERVICE.replace('package p;', 'package other;'),
                              'Client.java': 'package other; class Client { void f(Service s) { s.run(); } }'})
        target = graph.graph['invocations'][0]['resolved_target']
        self.assertEqual(graph.nodes[graph.nodes[target]['class_id']]['qualified_name'], 'other.Service')

    def test_unimported_class_is_not_guessed(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package app; class Client { void f(Service s) { s.run(); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertTrue(graph.graph['invocations'][0]['unresolved_reason'])

    def test_fully_qualified_static_and_parameter_types(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package app; class Client { void f(p.Service s) { p.Service.ping(); s.run(); } }'})
        self.assertCountEqual(resolved_pairs(graph), [('f', 'ping'), ('f', 'run')])

    def test_instance_method_cannot_use_class_receiver(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package p; class Client { void f() { Service.run(); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertIn('static', graph.graph['invocations'][0]['unresolved_reason'])

    def test_this_and_bare_static_calls(self):
        graph = source_graph({'A.java': 'class A { void f() { this.g(); } void g() {} static void a() { b(); } static void b() {} }'})
        self.assertCountEqual(resolved_pairs(graph), [('f', 'g'), ('a', 'b')])

    def test_invalid_instance_calls_in_static_context(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p; class Client {
            Service s; static Service shared; static void f() { s.run(); this.shared.run(); g(); }
            void g() {} }'''})
        self.assertFalse(resolved_pairs(graph))
        self.assertEqual(len(graph.graph['invocations']), 3)

    def test_local_shadowing_and_block_scope(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p; class Client {
            Service receiver; void f() { receiver.run(); { External receiver = null; receiver.run(); }
            receiver.run(); } }'''})
        self.assertEqual(len(resolved_pairs(graph)), 2)
        self.assertIsNone(graph.graph['invocations'][1]['resolved_target'])

    def test_parameter_shadows_field(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package p; class Client { Service s; void f(External s) { s.run(); this.s.run(); } }'})
        self.assertIsNone(graph.graph['invocations'][0]['resolved_target'])
        self.assertIsNotNone(graph.graph['invocations'][1]['resolved_target'])

    def test_local_not_yet_declared_is_not_field_fallback(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package p; class Client { Service s; void f() { s.run(); External s = null; } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertIn('not-yet-declared', graph.graph['invocations'][0]['unresolved_reason'])

    def test_for_and_catch_bindings(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p; class Client {
            void f(Service[] values) { for (Service s : values) { s.run(); }
            try {} catch (Service e) { e.run(); } } }'''})
        # Syntax-level declared types only; compiler validation is deliberately separate.
        self.assertEqual(len(resolved_pairs(graph)), 2)

    def test_overloads_remain_unresolved(self):
        overloaded = SERVICE.replace('public void run() {}', 'public void run() {} public void run(int x) {}')
        graph = source_graph({'Service.java': overloaded,
                              'Client.java': 'package p; class Client { void f(Service s) { s.run(); s.run(1); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertTrue(all('Overloaded' in call['unresolved_reason'] for call in graph.graph['invocations']))
        self.assertEqual(sum(e.get('name') == 'run' for _, e in graph.nodes(data=True)), 2)

    def test_wrong_argument_count(self):
        graph = source_graph({'Service.java': SERVICE,
                              'Client.java': 'package p; class Client { void f(Service s) { s.run(1); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertIn('count', graph.graph['invocations'][0]['unresolved_reason'])

    def test_ambiguous_explicit_imports(self):
        graph = source_graph({'Service.java': SERVICE, 'Other.java': SERVICE.replace('package p;', 'package other;'),
                              'Client.java': 'package app; import p.Service; import other.Service; class Client { void f(Service s) { s.run(); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertIn('Ambiguous', graph.graph['invocations'][0]['unresolved_reason'])

    def test_ambiguous_qualified_class(self):
        graph = source_graph({'one.java': SERVICE, 'two.java': SERVICE,
                              'Client.java': 'package p; class Client { void f(Service s) { s.run(); } }'})
        self.assertFalse(resolved_pairs(graph))
        self.assertIn('Ambiguous class', graph.graph['invocations'][0]['unresolved_reason'])

    def test_wildcard_static_imports_and_external_dependencies(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package app;
            import p.*; import static p.Service.ping;
            class Client { void f(Service s, java.util.List list) { s.run(); ping(); list.size(); } }'''})
        self.assertFalse(resolved_pairs(graph))
        self.assertEqual(len(graph.graph['invocations']), 3)
        self.assertTrue(all(call['unresolved_reason'] for call in graph.graph['invocations']))

    def test_inferred_generic_and_array_types_are_unresolved(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p;
            class Client { void f(Service<String> generic, Service[] array) {
            var inferred = new Service(); inferred.run(); generic.run(); array.run(); } }'''})
        self.assertFalse(resolved_pairs(graph))
        self.assertEqual(len(graph.graph['invocations']), 3)

    def test_generic_type_parameter_is_not_same_named_class(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p;
            class Client<Service> { void f(Service s) { s.run(); } }
            class Another { <Service> void g(Service s) { s.run(); } }'''})
        self.assertFalse(resolved_pairs(graph))
        self.assertTrue(all('Generic type parameter' in c['unresolved_reason'] for c in graph.graph['invocations']))

    def test_unsupported_pattern_binding_blocks_field_guess(self):
        graph = source_graph({'Service.java': SERVICE, 'Client.java': '''package p;
            class Client { Service s; void f(Object o) {
                if (o instanceof External s) { s.run(); }
            } }'''})
        self.assertFalse(resolved_pairs(graph))

    def test_nested_class_type_lookup(self):
        graph = source_graph({'Outer.java': 'package p; class Outer { class Inner { public void run() {} } void f(Inner x) { x.run(); } }'})
        self.assertEqual(resolved_pairs(graph), [('f', 'run')])

    def test_local_class_not_exported_as_repository_type(self):
        graph = source_graph({'Outer.java': 'package p; class Outer { void f() { class Hidden { public void run() {} } } }',
                              'Client.java': 'package p; class Client { void f(Hidden x) { x.run(); } }'})
        self.assertFalse(resolved_pairs(graph))

    def test_malformed_target_and_caller_do_not_create_edges(self):
        for malformed in ('target', 'caller'):
            with self.subTest(malformed=malformed):
                service = SERVICE + (' class Broken {' if malformed == 'target' else '')
                client = 'package p; class Client { void f(Service s) { s.run(); } }'
                if malformed == 'caller':
                    client += ' class Broken {'
                graph = source_graph({'Service.java': service, 'Client.java': client})
                self.assertFalse(resolved_pairs(graph))
                self.assertTrue(graph.graph['invocations'][0]['unresolved_reason'])

    def test_private_and_nonpublic_cross_package_targets(self):
        for access in ('private', ''):
            with self.subTest(access=access):
                service = SERVICE.replace('public void run()', f'{access} void run()')
                graph = source_graph({'Service.java': service, 'Client.java':
                                      'package app; import p.Service; class Client { void f(Service s) { s.run(); } }'})
                self.assertFalse(resolved_pairs(graph))

    def test_repeated_input_graphs_are_deduplicated(self):
        result = parse_source('class A { void f() { f(); } }', 'A.java')
        self.assertEqual(graph_to_dict(build_graph([result, result])), graph_to_dict(build_graph(result)))


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, relative_path, source='class A {}'):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

    def test_recursive_scanning_and_relative_ids(self):
        self.write('src/p/A.java')
        self.write('src/p/sub/B.java', 'class B {}')
        self.write('notes.txt')
        result = index_inputs([self.root])
        self.assertEqual(result['discovered'], 2)
        self.assertEqual([f['file_path'] for f in result['files']], ['src/p/A.java', 'src/p/sub/B.java'])
        graph = build_graph(result['files'])
        self.assertTrue(all(not Path(e['file_path']).is_absolute() and str(self.root) not in e['id']
                            for _, e in graph.nodes(data=True)))

    def test_all_ignored_directories(self):
        for name in IGNORED_DIRECTORIES:
            self.write(f'{name}/Ignored.java')
            self.write(f'src/{name}/Deep.java')
        self.write('src/Keep.java')
        result = index_inputs([self.root])
        self.assertEqual(result['discovered'], 1)
        self.assertEqual(len(result['skipped']), len(IGNORED_DIRECTORIES) * 2)

    def test_symlink_files_and_directory_cycles_are_skipped(self):
        self.write('src/A.java')
        (self.root / 'loop').symlink_to(self.root, target_is_directory=True)
        (self.root / 'Alias.java').symlink_to(self.root / 'src/A.java')
        result = index_inputs([self.root])
        self.assertEqual(result['discovered'], 1)
        self.assertEqual(len(result['skipped']), 2)

    def test_overlapping_roots_and_file_aliases_deduplicate(self):
        path = self.write('src/A.java', 'class A { void f() { f(); } }')
        result = index_inputs([self.root, self.root / 'src', path, path])
        self.assertEqual(len(result['files']), 1)
        self.assertEqual(len(build_graph(result['files']).graph['invocations']), 1)

    def test_partial_indexing_with_malformed_and_missing_files(self):
        self.write('Good.java')
        self.write('Bad.java', 'class Bad { void f() {')
        result = index_inputs([self.root, self.root / 'Missing.java'])
        self.assertEqual(len(result['files']), 3)
        self.assertEqual(sum(f['has_error'] for f in result['files']), 2)
        self.assertTrue(build_graph(result['files']).number_of_nodes() >= 2)

    def test_unreadable_file_does_not_abort(self):
        self.write('A.java')
        self.write('B.java', 'class B {}')
        with patch('parser.Path.read_bytes', side_effect=[PermissionError('denied'), b'class B {}']):
            result = index_inputs([self.root])
        self.assertTrue(result['files'][0]['has_error'])
        self.assertFalse(result['files'][1]['has_error'])
        self.assertEqual(result['files'][0]['file_path'], 'A.java')

    def test_scan_errors_are_reported(self):
        with patch('indexer.os.walk', side_effect=simulate_scan_error):
            result = index_inputs([self.root])
        self.assertEqual(result['diagnostics'][0]['kind'], 'scan_error')
        self.assertEqual(result['diagnostics'][0]['file_path'], 'denied')

    def test_empty_repository(self):
        result = index_inputs([self.root])
        self.assertEqual(result['discovered'], 0)
        self.assertEqual(build_graph(result['files']).number_of_nodes(), 0)

    def test_deterministic_output_across_reindex_and_input_order(self):
        first = self.write('A.java', 'class A { void f() { B.g(); } }')
        second = self.write('B.java', 'class B { public static void g() {} }')
        a = graph_to_dict(build_graph(index_inputs([self.root])['files']))
        b = graph_to_dict(build_graph(index_inputs([self.root, second, first])['files']))
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(a), json.dumps(graph_to_dict(build_graph(index_inputs([self.root])['files']))))

    def test_repository_identity_is_portable(self):
        source = self.root / 'one'
        destination = self.root / 'two'
        self.write('one/src/A.java', 'class A { void f() { f(); } }')
        shutil.copytree(source, destination)
        a = graph_to_dict(build_graph(index_inputs([source])['files']))
        b = graph_to_dict(build_graph(index_inputs([destination])['files']))
        self.assertEqual(a, b)

    def test_example_repository_cli_and_cross_file_edges(self):
        output = self.root / 'graph.json'
        command = [sys.executable, str(ROOT / 'main.py'), str(ROOT / 'examples/repository'), '--json', str(output)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('3 files discovered; 3 parsed successfully', run.stdout)
        data = json.loads(output.read_text())
        entities = {e['id']: e for e in data['entities']}
        edges = [e for e in data['relationships'] if e['kind'] == 'CALLS']
        self.assertEqual(len(edges), 5)
        self.assertEqual(sum(c['resolved_target'] is None for c in data['invocations']), 1)
        self.assertEqual(len(data['entities']), 7)
        self.assertEqual(len(data['relationships']), 9)
        self.assertEqual(data['indexing']['diagnostics'], [])
        self.assertTrue(all(not Path(f['file_path']).is_absolute() for f in data['files']))
        for edge in edges:
            self.assertEqual(edge['invocation']['caller'], edge['source'])
            self.assertEqual(edge['invocation']['resolved_target'], edge['target'])
            self.assertTrue(entities[edge['source']]['start_line'] <= edge['invocation']['start_line'] <= entities[edge['source']]['end_line'])
        cross_file = [e for e in edges if entities[e['source']]['file_path'] != entities[e['target']]['file_path']]
        self.assertEqual(len(cross_file), 4)
        before = output.read_bytes()
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(output.read_bytes(), before)

    def test_repository_cli_partial_export_and_input_protection(self):
        source = self.write('Good.java')
        self.write('Bad.java', 'class Bad {')
        output = self.root / 'graph.json'
        command = [sys.executable, str(ROOT / 'main.py'), str(self.root), '--json', str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertTrue(json.loads(output.read_text())['files'][0]['has_error'])
        command[-1] = str(source)
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertIn('must not overwrite', run.stderr)
        self.assertEqual(source.read_text(), 'class A {}')


if __name__ == '__main__':
    unittest.main()

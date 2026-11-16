import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from parser import parse_file, parse_source


class ParserTests(unittest.TestCase):
    def test_calculator(self):
        result = parse_file(Path(__file__).resolve().parents[1] / 'examples/Calculator.java')
        self.assertFalse(result['has_error'])
        self.assertEqual([m['name'] for m in result['classes'][0]['methods']], ['add', 'multiply'])
        json.dumps(result)

    def test_method_details(self):
        source = 'class A {\n String greet(int count, String name) { return name.trim(); }\n void stop() {}\n}'
        method = parse_source(source)['classes'][0]['methods'][0]
        self.assertEqual(method['return_type'], 'String')
        self.assertEqual([(p['name'], p['type']) for p in method['parameters']], [('count', 'int'), ('name', 'String')])
        self.assertEqual((method['start_line'], method['end_line']), (2, 2))
        self.assertEqual(method['source'], 'String greet(int count, String name) { return name.trim(); }')
        call = method['calls'][0]
        self.assertEqual((call['name'], call['receiver'], call['source']), ('trim', 'name', 'name.trim()'))
        self.assertEqual(call['caller'], method['id'])

    def test_constructor(self):
        method = parse_source('class A { A(int x) { init(); } void init() {} }')['classes'][0]['methods'][0]
        self.assertEqual(method['kind'], 'constructor')
        self.assertIsNone(method['return_type'])
        self.assertEqual(method['calls'][0]['name'], 'init')

    def test_nested_classes(self):
        classes = parse_source('package demo; class Outer { void a() {} class Inner { void b() {} } }')['classes']
        self.assertEqual([c['qualified_name'] for c in classes], ['demo.Outer', 'demo.Outer.Inner'])
        self.assertEqual(classes[1]['parent_id'], classes[0]['id'])
        self.assertEqual([m['name'] for m in classes[0]['methods']], ['a'])
        self.assertEqual([m['name'] for m in classes[1]['methods']], ['b'])

    def test_local_classes_have_distinct_ids(self):
        classes = parse_source('class A { void a() { class Local {} } void b() { class Local {} } }')['classes']
        self.assertEqual(len(set(c['id'] for c in classes)), 3)

    def test_nested_call_scopes(self):
        result = parse_source('class A { void f() { outside(); Runnable x = () -> later(); Object y = new Object() { void g() { inside(); } }; class Local { void h() { local(); } } } }')
        calls = result['classes'][0]['methods'][0]['calls']
        self.assertEqual([c['name'] for c in calls], ['outside'])
        self.assertEqual(result['classes'][1]['methods'][0]['calls'][0]['name'], 'local')

    def test_array_and_varargs_parameters(self):
        methods = parse_source('class A { int[] f(int values[]) { return values; } void g(final String... args) {} }')['classes'][0]['methods']
        self.assertEqual(methods[0]['parameters'][0]['type'], 'int[]')
        self.assertEqual(methods[0]['return_type'], 'int[]')
        self.assertEqual(methods[1]['parameters'][0]['name'], 'args')
        self.assertEqual(methods[1]['parameters'][0]['type'], 'String...')

    def test_invalid_and_incomplete(self):
        for source in ('class A { void f() {', 'class A { void f() { int x = ; } }', 'class A { void f() { return } }'):
            with self.subTest(source=source):
                result = parse_source(source)
                self.assertTrue(result['has_error'])
                self.assertTrue(result['diagnostics'])
                json.dumps(result)

    def test_missing_nodes(self):
        result = parse_source('class A { int x }')
        self.assertIn('missing_node', [d['kind'] for d in result['diagnostics']])

    def test_missing_empty_and_unreadable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'A.java'
            self.assertEqual(parse_file(path)['diagnostics'][0]['kind'], 'file_error')
            path.write_text('')
            self.assertEqual(parse_file(path)['diagnostics'][0]['kind'], 'empty_source')
            with patch.object(Path, 'read_bytes', side_effect=PermissionError('denied')):
                self.assertEqual(parse_file(path)['diagnostics'][0]['kind'], 'file_error')

    def test_unicode_and_encoding(self):
        result = parse_source('class Café { void salué() {} }')
        self.assertEqual(result['classes'][0]['methods'][0]['name'], 'salué')
        result = parse_source(b'//\xff\nclass A {}')
        self.assertTrue(result['has_error'])
        self.assertEqual(result['diagnostics'][0]['kind'], 'encoding_error')


if __name__ == '__main__':
    unittest.main()

import copy
import json
import unittest

from graph_builder import build_graph, graph_to_dict
from parser import parse_source


def call_edges(graph):
    return [(a, b, data) for a, b, data in graph.edges(data=True) if data['kind'] == 'CALLS']


class GraphTests(unittest.TestCase):
    def test_same_class_calls_and_constructor(self):
        result = parse_source('class PaymentService { PaymentService() { validatePayment(); } void processPayment() { validatePayment(); chargeCard(); } void validatePayment() {} void chargeCard() {} }')
        graph = build_graph(result)
        self.assertEqual(len(graph), 5)
        self.assertEqual(len(call_edges(graph)), 3)
        self.assertEqual(sum(d['kind'] == 'CONTAINS' for _, _, d in graph.edges(data=True)), 4)

    def test_repeated_calls_preserve_locations(self):
        graph = build_graph(parse_source('class A { void a() { b(); b(); } void b() {} }'))
        edges = call_edges(graph)
        self.assertEqual(len(edges), 2)
        self.assertNotEqual(edges[0][2]['invocation']['start_column'], edges[1][2]['invocation']['start_column'])

    def test_unresolved_receivers_and_unknown_calls(self):
        graph = build_graph(parse_source('class A { void a() { missing(); obj.b(); System.out.println("x"); this.b(); } void b() {} }'))
        self.assertEqual(len(call_edges(graph)), 1)
        self.assertEqual(sum(c['resolved_target'] is None for c in graph.graph['invocations']), 3)
        self.assertEqual(len(graph), 3)

    def test_overloads_are_distinct_and_unresolved(self):
        graph = build_graph(parse_source('class A { void f(int x) {} void f(String x) {} void g() { f(1); } }'))
        self.assertEqual(len(graph), 4)
        self.assertEqual(len(call_edges(graph)), 0)
        signatures = [data.get('signature') for _, data in graph.nodes(data=True)]
        self.assertIn('f(int)', signatures)
        self.assertIn('f(String)', signatures)

    def test_no_cross_file_or_nested_resolution(self):
        first = parse_source('class A { void f() { g(); } class Inner { void g() {} } }', 'one.java')
        second = parse_source('class A { void g() {} }', 'two.java')
        graph = build_graph([first, second])
        self.assertEqual(len(graph), 6)
        self.assertFalse(call_edges(graph))
        inner = first['classes'][1]
        self.assertTrue(graph.has_edge(first['classes'][0]['id'], inner['id']))

    def test_arity_varargs_and_invalid_source_are_unresolved(self):
        for source in ('class A { void g(int x) {} void f() { g(); } }', 'class A { void g(String... args) {} void f() { g("x"); } }', 'class A { void g() {} void f() { g(); int x = ; } }'):
            with self.subTest(source=source):
                graph = build_graph(parse_source(source))
                self.assertFalse(call_edges(graph))
                self.assertIsNone(graph.graph['invocations'][0]['resolved_target'])

    def test_json_and_no_input_mutation(self):
        result = parse_source('class A { void f() { g(); } void g() {} }')
        before = copy.deepcopy(result)
        graph = build_graph(result)
        exported = graph_to_dict(graph)
        json.dumps(exported)
        self.assertEqual(result, before)
        self.assertEqual(set(exported), {'files', 'entities', 'relationships', 'invocations'})
        self.assertTrue(all('id' in entity for entity in exported['entities']))

    def test_duplicate_declarations_do_not_merge(self):
        graph = build_graph(parse_source('class A { void f() {} void f() {} }'))
        self.assertEqual(len(graph), 3)


if __name__ == '__main__':
    unittest.main()

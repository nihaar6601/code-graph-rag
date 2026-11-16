"""Offline tests for static, untrusted-input-safe Markdown reporting."""
import copy
import unittest

from answer_reporting import render_answer_report


def sample():
    return {
        'mode': 'real_provider', 'comparison_status': 'real_provider_proxy_metrics',
        'provider': {'provider': 'local', 'model': 'demo', 'temperature': 0},
        'query_count': 1, 'planned_trials': 1, 'repeats': 1,
        'setup': {'model_load_ms': 15}, 'diagnostics': [],
        'summary': {'graph': {'samples': 1, 'correctness_proxy': .5,
                              'mean_actual_input_tokens': None, 'mean_estimated_input_tokens': 40}},
        'results': [{'question': 'How does checkout work?', 'category': 'flow',
                     'strategy': 'graph', 'repeat': 1, 'reference_answer': 'It checks stock.',
                     'reference_facts': [{'id': 'f1', 'description': 'Checks inventory',
                                          'sources': ['Cart.checkout'], 'patterns': ['stock']}],
                     'answer': {'answer': 'Stock is checked.', 'status': 'ok',
                                'insufficient_context': False,
                                'retrieval': {'context': 'ORIGINAL CONTEXT'},
                                'sources': [{'source_id': 'S1', 'file_path': 'Cart.java',
                                             'start_line': 2, 'end_line': 9, 'text': 'checkStock();'}],
                                'claims': [{'text': 'Checks stock', 'citations': [{'source_id': 'S1'}],
                                            'invalid_citations': []}],
                                'metrics': {'actual_input_tokens': None, 'estimated_input_tokens': 40,
                                            'generation_latency_ms': 20},
                                'cost': {'estimated_usd': None}},
                     'grade': {'fact_results': [{'id': 'f1', 'lexically_covered': True}]}}]}


class AnswerReportingTests(unittest.TestCase):
    def test_evidence_annotations_and_answer(self):
        text = render_answer_report(sample())
        for expected in ('How does checkout work?', 'It checks stock', 'Checks inventory',
                         'Cart.checkout', 'patterns', 'ORIGINAL CONTEXT', 'checkStock();',
                         'Cart.java', 'start_line', 'S1', 'Stock is checked', 'lexically_covered',
                         'generation_latency_ms', 'estimated_usd'):
            self.assertIn(expected, text)

    def test_unknown_usage_not_zero(self):
        text = render_answer_report(sample())
        self.assertIn('"actual_input_tokens": "unknown"', text)
        self.assertIn('"estimated_input_tokens": 40', text)
        self.assertIn('graph | 1 | 0\\.5 | unknown | 40', text)

    def test_none_usage(self):
        report = sample()
        report['results'][0]['answer']['metrics'] = None
        self.assertIn('Missing usage is unknown, never zero', render_answer_report(report))

    def test_missing_reference_facts(self):
        report = sample()
        del report['results'][0]['reference_facts']
        self.assertIn('Reference facts (annotations)\n\nUnavailable', render_answer_report(report))

    def test_every_trial_has_review(self):
        report = sample()
        report['results'].append(copy.deepcopy(report['results'][0]))
        text = render_answer_report(report)
        self.assertEqual(text.count('- [ ] Factual correctness'), 2)
        for field in ('Completeness', 'Semantic grounding', 'Unsupported claims', 'Insufficiency handling'):
            self.assertEqual(text.count('- [ ] ' + field), 2)
        self.assertNotIn('- [x]', text)
        self.assertIn('Not independently reviewed', text)

    def test_mock_mode_warning(self):
        report = sample()
        report['mode'] = 'mock_plumbing_only'
        self.assertIn('MOCK: plumbing validation only', render_answer_report(report))

    def test_mock_row_warning(self):
        report = sample()
        report['results'][0]['answer']['mocked'] = True
        self.assertIn('MOCK:', render_answer_report(report))

    def test_fences_cannot_be_closed_by_source(self):
        report = sample()
        source = '```\n<script>alert(1)</script>\n````\n# injected'
        report['results'][0]['answer']['retrieval']['context'] = source
        text = render_answer_report(report)
        self.assertIn('`````text\n' + source + '\n`````', text)

    def test_html_and_table_controls_escaped(self):
        report = sample()
        report['results'][0]['reference_answer'] = '<img src=x onerror=alert(1)>'
        report['results'][0]['answer']['answer'] = '<script>bad</script>'
        report['summary']['a|b\n<script>'] = report['summary'].pop('graph')
        text = render_answer_report(report)
        self.assertIn('&lt;img', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertIn('a\\|b<br>', text)
        self.assertNotIn('<script>', text)
        self.assertNotIn('<img', text)

    def test_credentials_and_unrelated_fields_omitted(self):
        report = sample()
        report['provider'].update(api_key='SECRET_A', token='SECRET_B', authorization='SECRET_C',
                                  api_key_env='SECRET_ENV', unknown='SECRET_D')
        report['extra'] = {'api_key': 'SECRET_E'}
        report['setup']['nested'] = {'authorization': 'SECRET_F', 'elapsed_ms': 8}
        text = render_answer_report(report)
        self.assertNotIn('SECRET_', text)
        self.assertIn('elapsed_ms', text)

    def test_url_credentials_removed(self):
        report = sample()
        report['provider']['base_url'] = 'https://user:SECRET@host/v1?api_key=SECRET#SECRET'
        text = render_answer_report(report)
        self.assertNotIn('SECRET', text)
        self.assertIn('https://host/v1', text)

    def test_budget_blocked(self):
        report = sample()
        report['budget'] = {'status': 'budget_blocked', 'spent_usd': None}
        report['comparison_status'] = 'blocked'
        self.assertIn('budget_blocked', render_answer_report(report))

    def test_dry_run_no_results(self):
        report = sample()
        report['mode'] = 'dry_run'
        report['plan'] = {'flexible': {'requests': 4, 'token': 'SECRET'}, 'strategies': ['graph']}
        text = render_answer_report(report)
        self.assertIn('Dry-run plan', text)
        self.assertIn('"requests": 4', text)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('### Trial', text)
        self.assertNotIn('Summary comparison', text)

    def test_nested_reports_separate_not_recursive(self):
        report = sample()
        nested = sample()
        nested['optimization_reports'] = {'cycle': report}
        report['smoke_report'] = nested
        report['optimization_reports'] = {'relevance': nested, 'self': report}
        text = render_answer_report(report)
        self.assertEqual(text.count('### Trial 1'), 3)
        self.assertIn('Smoke run', text)
        self.assertIn('Optimization run: relevance', text)
        self.assertEqual(text.count('quality metrics are not combined'), 2)

    def test_empty_report_and_no_mutation(self):
        self.assertIn('Summary unavailable', render_answer_report({}))
        report = sample()
        before = copy.deepcopy(report)
        render_answer_report(report)
        self.assertEqual(report, before)

    def test_no_configuration_accessor(self):
        class Provider:
            @property
            def config(self):
                raise AssertionError('Must not access provider configuration')
        report = sample()
        report['provider'] = Provider()
        render_answer_report(report)

    def test_cycle_in_diagnostics(self):
        report = sample()
        report['diagnostics'].append(report['diagnostics'])
        self.assertIn('cycle omitted', render_answer_report(report))


if __name__ == '__main__':
    unittest.main()

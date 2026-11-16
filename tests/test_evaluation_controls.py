"""Deterministic, transport-free planning and evaluation budget regression tests."""
from dataclasses import replace
from decimal import Decimal
import unittest
from unittest.mock import Mock, patch

from evaluation_controls import (BudgetExceeded, BudgetedProvider, EvaluationBudget,
                                 monetary, plan_evaluation, token_reservation)
from llm import LLMConfig, LLMError, MockProvider, token_accounting


MESSAGES = [{'role': 'user', 'content': 'Explain checkout.'}]
STRATEGIES = ['traditional', 'structure', 'graph', 'hybrid', 'hybrid_graph']
BENCHMARK = [{'question': f'Question {i}'} for i in range(8)]


class EvaluationControlsTests(unittest.TestCase):
    def setUp(self):
        self.config = LLMConfig(provider='local', model='offline-test',
                                base_url='http://localhost:8000/v1', retries=0,
                                max_output_tokens=20, input_price_per_million=1,
                                output_price_per_million=2).validate()
        # No tokenizer initialization, transport construction, or actual HTTP.
        for target, options in [
            ('evaluation_controls.input_token_bound', {'return_value': 100}),
            ('llm.estimate_message_tokens', {'return_value': 5}),
            ('llm.count_tokens', {'return_value': 2}),
            ('llm.build_opener', {'side_effect': AssertionError('Unexpected transport')}),
        ]:
            guard = patch(target, **options)
            guard.start()
            self.addCleanup(guard.stop)

    def plan(self, config=None, **kwargs):
        return plan_evaluation(config or self.config, BENCHMARK, STRATEGIES, **kwargs)

    def wrapped(self, config=None, usage=None, attempts=1, **limits):
        config = config or self.config
        response = {'content': 'answer', 'usage': usage, 'attempts': attempts}
        underlying = Mock(config=config)
        underlying.complete.return_value = response
        budget = EvaluationBudget(config, **limits)
        return BudgetedProvider(underlying, budget), underlying, budget

    def usage(self, inputs=10, outputs=5):
        return {'prompt_tokens': inputs, 'completion_tokens': outputs,
                'total_tokens': inputs + outputs}

    def test_profile_smoke_optimization_and_catalog_counts(self):
        plan = self.plan(checkout_smoke=True, compare_optimization=True)
        self.assertEqual(plan['components'], {'benchmark': 40, 'checkout_smoke': 5,
                                              'optimization_control': 4})
        self.assertEqual(plan['planned_trials'], 49)
        self.assertEqual(plan['maximum_completion_http_attempts'], 49)
        self.assertEqual(plan['model_preflight_requests'], 1)
        self.assertEqual(plan['expected_api_requests'], 50)
        self.assertEqual(plan['maximum_http_attempts'], 50)

    def test_retry_caps_include_preflight(self):
        config = replace(self.config, retries=2)
        plan = self.plan(config, max_api_requests=121)
        self.assertEqual(plan['maximum_http_attempts'], 121)
        self.assertEqual(plan['expected_api_requests'], 41)
        with self.assertRaises(ValueError):
            self.plan(config, max_api_requests=120)
        self.assertEqual(self.plan(config, skip_model_check=True,
                                   max_api_requests=120)['maximum_http_attempts'], 120)

    def test_trial_cap_counts_all_components(self):
        with self.assertRaises(ValueError):
            self.plan(checkout_smoke=True, compare_optimization=True, max_requests=48)
        self.assertEqual(self.plan(checkout_smoke=True, compare_optimization=True,
                                   max_requests=49)['planned_trials'], 49)

    def test_mock_plan_has_zero_http_and_cost(self):
        plan = self.plan(LLMConfig(provider='mock', model='mock', retries=5),
                         checkout_smoke=True, compare_optimization=True,
                         max_api_requests=1, max_total_cost=0)
        for key in ('expected_api_requests', 'maximum_http_attempts',
                    'maximum_completion_http_attempts', 'model_preflight_requests',
                    'estimated_upper_bound_usd'):
            self.assertEqual(plan[key], 0)
        self.assertTrue(plan['fits_configured_cost_estimate'])

    def test_unknown_pricing_and_cost_cap_rejection(self):
        config = replace(self.config, input_price_per_million=None,
                         output_price_per_million=None)
        self.assertIsNone(self.plan(config)['estimated_upper_bound_usd'])
        self.assertIsNone(token_reservation(config, 100, 20))
        with self.assertRaises(ValueError):
            self.plan(config, max_total_cost=1)
        with self.assertRaises(ValueError):
            EvaluationBudget(config, max_total_cost=1)

    def test_monetary_rejects_bool_negative_and_nonfinite(self):
        for value in (True, False, -1, '-0.01', float('nan'), float('inf'),
                      Decimal('NaN'), 'invalid', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                monetary(value)
        self.assertEqual(monetary('0.001'), Decimal('0.001'))
        self.assertEqual(monetary(0), Decimal(0))

    def test_invalid_monetary_caps_rejected_at_both_entrypoints(self):
        for value in (True, -1, float('nan')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.plan(max_total_cost=value)
                with self.assertRaises(ValueError):
                    EvaluationBudget(self.config, max_total_cost=value)

    def test_plan_validates_selection_and_limits(self):
        for options in ({'repeats': 0}, {'max_tokens': -1}, {'max_requests': 0},
                        {'max_api_requests': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)
        for benchmark, strategies in (([], STRATEGIES), (BENCHMARK, [])):
            with self.assertRaises(ValueError):
                plan_evaluation(self.config, benchmark, strategies)
        with self.assertRaises(ValueError):
            plan_evaluation(self.config, BENCHMARK, ['structure'], compare_optimization=True)

    def test_cost_cap_blocks_before_underlying_call(self):
        wrapper, underlying, budget = self.wrapped(max_total_cost='0.000139')
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_not_called()
        self.assertEqual(budget.used_attempts, 0)
        self.assertEqual(budget.halt_reason, 'cost_limit')

    def test_actual_usage_settlement_releases_unused_reservation(self):
        wrapper, underlying, budget = self.wrapped(usage=self.usage(),
                                                    max_total_cost='0.00016')
        wrapper.complete(MESSAGES)
        self.assertEqual(budget.charged, Decimal('0.00002'))
        wrapper.complete(MESSAGES)
        self.assertEqual(underlying.complete.call_count, 2)
        self.assertEqual(budget.charged, Decimal('0.00004'))
        self.assertEqual(budget.observed_samples, 2)
        self.assertIsNone(budget.halt_reason)

    def test_cumulative_cost_cap_stops_next_request(self):
        wrapper, underlying, budget = self.wrapped(usage=self.usage(100, 20),
                                                    max_total_cost='0.00028')
        wrapper.complete(MESSAGES)
        wrapper.complete(MESSAGES)
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        self.assertEqual(underlying.complete.call_count, 2)
        self.assertEqual(budget.charged, Decimal('0.00028'))

    def test_retry_reservation_blocks_before_call(self):
        wrapper, underlying, budget = self.wrapped(replace(self.config, retries=2),
                                                    max_total_cost='0.000419')
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_not_called()
        self.assertEqual(budget.charged, 0)

    def test_retried_success_retains_unknown_attempt_reservation(self):
        wrapper, _, budget = self.wrapped(replace(self.config, retries=2),
                                          usage=self.usage(), attempts=2,
                                          max_total_cost=1)
        wrapper.complete(MESSAGES)
        self.assertEqual(budget.charged, Decimal('0.00016'))
        self.assertEqual(budget.observed, Decimal('0.00002'))
        self.assertEqual(budget.used_attempts, 2)
        self.assertFalse(budget.events[0]['failed'])

    def test_failed_request_retains_reservation_and_fails_closed(self):
        wrapper, underlying, budget = self.wrapped(replace(self.config, retries=1),
                                                    max_total_cost=1)
        error = LLMError('offline failure', attempts=2)
        underlying.complete.side_effect = error
        with self.assertRaises(LLMError) as caught:
            wrapper.complete(MESSAGES)
        self.assertIs(caught.exception, error)
        self.assertEqual(budget.charged, Decimal('0.00028'))
        self.assertEqual(budget.halt_reason, 'usage_unavailable_under_cost_cap')
        self.assertTrue(budget.events[0]['failed'])
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_called_once()

    def test_partial_or_missing_usage_fails_closed_under_cap(self):
        for usage in (None, {'prompt_tokens': 10}, {'completion_tokens': 5},
                      {'prompt_tokens': True, 'completion_tokens': 5}):
            with self.subTest(usage=usage):
                wrapper, underlying, budget = self.wrapped(usage=usage, max_total_cost=1)
                wrapper.complete(MESSAGES)
                self.assertEqual(budget.charged, Decimal('0.00014'))
                self.assertIsNone(budget.snapshot()['observed_completion_cost_usd'])
                with self.assertRaises(BudgetExceeded):
                    wrapper.complete(MESSAGES)
                underlying.complete.assert_called_once()

    def test_unexpected_failure_accounts_all_reserved_attempts(self):
        wrapper, underlying, budget = self.wrapped(replace(self.config, retries=2),
                                                    max_total_cost=1)
        underlying.complete.side_effect = RuntimeError('offline failure')
        with self.assertRaises(RuntimeError):
            wrapper.complete(MESSAGES)
        self.assertEqual(budget.used_attempts, 3)
        self.assertEqual(budget.charged, Decimal('0.00042'))
        self.assertEqual(budget.halt_reason, 'usage_unavailable_under_cost_cap')

    def test_unknown_cost_reported_none(self):
        config = replace(self.config, input_price_per_million=None,
                         output_price_per_million=None)
        wrapper, _, budget = self.wrapped(config, usage=self.usage())
        wrapper.complete(MESSAGES)
        snapshot = budget.snapshot()
        self.assertIsNone(snapshot['conservatively_accounted_usd'])
        self.assertIsNone(snapshot['observed_completion_cost_usd'])
        self.assertIsNone(snapshot['events'][0]['observed_cost_usd'])
        self.assertEqual(snapshot['used_http_attempts'], 1)

    def test_model_preflight_consumes_request_limit(self):
        wrapper, underlying, budget = self.wrapped(max_api_requests=1)
        budget.preflight()
        self.assertEqual(budget.events[0]['kind'], 'model_catalog')
        self.assertIsNone(budget.events[0]['completion_cost_usd'])
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_not_called()
        self.assertEqual(budget.used_attempts, 1)
        self.assertEqual(budget.halt_reason, 'api_request_limit')

    def test_retry_attempt_limit_reserved_before_call(self):
        wrapper, underlying, budget = self.wrapped(replace(self.config, retries=1),
                                                    max_api_requests=2)
        budget.preflight()
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_not_called()
        with self.assertRaises(ValueError):
            EvaluationBudget(self.config, max_api_requests=0)

    def test_provider_overrun_stops_subsequent_requests(self):
        wrapper, underlying, budget = self.wrapped(usage=self.usage(1000, 20),
                                                    max_total_cost=1)
        wrapper.complete(MESSAGES)
        self.assertEqual(budget.charged, Decimal('0.00104'))
        self.assertEqual(budget.halt_reason, 'provider_cost_exceeded_reservation_or_cap')
        with self.assertRaises(BudgetExceeded):
            wrapper.complete(MESSAGES)
        underlying.complete.assert_called_once()

    def test_mock_response_unchanged_and_actual_usage_remains_none(self):
        provider = MockProvider()
        expected = provider.complete(MESSAGES)
        budget = EvaluationBudget(provider.config, max_total_cost=0, max_api_requests=1)
        result = BudgetedProvider(provider, budget).complete(MESSAGES)
        self.assertEqual(result, expected)
        self.assertIsNone(result['usage'])
        self.assertEqual(token_accounting(MESSAGES, result['content'], result['usage'])['actual'],
                         {'input_tokens': None, 'output_tokens': None, 'total_tokens': None})
        self.assertEqual(budget.used_attempts, 0)
        self.assertEqual(budget.charged, 0)
        self.assertIsNone(budget.halt_reason)

    def test_snapshots_are_defensive_copies(self):
        wrapper, _, budget = self.wrapped(usage=self.usage())
        wrapper.complete(MESSAGES)
        snapshot = budget.snapshot()
        snapshot['events'][0]['http_attempts'] = 999
        snapshot['events'].append({'kind': 'invented'})
        snapshot['halt_reason'] = 'invented'
        self.assertEqual(budget.snapshot()['events'][0]['http_attempts'], 1)
        self.assertEqual(len(budget.events), 1)
        self.assertIsNone(budget.halt_reason)
        earlier = budget.snapshot()
        wrapper.complete(MESSAGES)
        self.assertEqual(len(earlier['events']), 1)

    def test_mismatched_config_rejected(self):
        underlying = Mock(config=replace(self.config, model='different'))
        with self.assertRaises(ValueError):
            BudgetedProvider(underlying, EvaluationBudget(self.config))
        underlying.complete.assert_not_called()

    def test_failed_response_with_complete_usage_settles_actual_cost(self):
        wrapper, underlying, budget = self.wrapped(max_total_cost=1)
        underlying.complete.side_effect = LLMError('invalid response', attempts=1,
                                                    usage=self.usage())
        with self.assertRaises(LLMError):
            wrapper.complete(MESSAGES)
        self.assertEqual(budget.charged, Decimal('0.00002'))
        self.assertEqual(budget.observed_samples, 1)
        self.assertTrue(budget.events[0]['failed'])
        self.assertIsNone(budget.halt_reason)


if __name__ == '__main__':
    unittest.main()

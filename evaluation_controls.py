"""Credential-free planning and conservative per-request evaluation budgets."""
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import math
import os

from llm import LLMError, estimate_message_tokens


class BudgetExceeded(LLMError):
    budget_blocked = True


def monetary(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('Cost values must be numeric') from None
    if not result.is_finite() or result < 0:
        raise ValueError('Cost limits must be finite and nonnegative')
    return result


def token_reservation(config, inputs, outputs):
    if config.provider == 'mock':
        return Decimal(0)
    if config.input_price_per_million is None:
        return None
    return (monetary(config.input_price_per_million) * inputs +
            monetary(config.output_price_per_million) * outputs) / Decimal(1_000_000)


def input_token_bound(messages):
    # A deliberately conservative *client assumption*, not a provider guarantee:
    # count actual UTF-8 prompt bytes plus generous hidden-chat framing headroom.
    return max(estimate_message_tokens(messages), sum(len(m['content'].encode('utf-8')) + len(m['role']) for m in messages) + 1024)


def plan_evaluation(config, benchmark, strategies, repeats=1, max_tokens=2000,
                    checkout_smoke=False, compare_optimization=False, skip_model_check=False,
                    max_requests=None, max_api_requests=None, max_total_cost=None):
    from answering import SYSTEM_PROMPT
    config.validate()
    if repeats < 1 or max_tokens < 0 or not strategies:
        raise ValueError('Positive repeats/strategies and nonnegative context budget required')
    if max_requests is not None and max_requests < 1:
        raise ValueError('max_requests must be positive')
    if max_api_requests is not None and max_api_requests < 1:
        raise ValueError('max_api_requests must be positive')
    graph_strategies = [s for s in strategies if s in ('graph', 'hybrid_graph')]
    if compare_optimization and not graph_strategies:
        raise ValueError('Optimization comparison requires graph or hybrid_graph')
    components = {'benchmark': len(benchmark) * len(strategies) * repeats,
                  'checkout_smoke': len(strategies) * repeats if checkout_smoke else 0,
                  'optimization_control': len(graph_strategies) * 2 * repeats if compare_optimization else 0}
    trials = sum(components.values())
    if not benchmark or not trials:
        raise ValueError('Select at least one benchmark question')
    if max_requests is not None and trials > max_requests:
        raise ValueError('Planned trials exceed --max-requests (including smoke/optimization runs)')
    real = config.provider != 'mock'
    preflight = int(real and not skip_model_check)
    expected_requests = (trials + preflight) if real else 0
    attempts = trials * (config.retries + 1) if real else 0
    total_attempts = attempts + preflight
    if max_api_requests is not None and total_attempts > max_api_requests:
        raise ValueError('Planned HTTP attempts including retries/preflight exceed --max-api-requests')
    # Lightweight planning does not initialize tiktoken, models or retrieval.
    question_bytes = max(len(item['question'].encode('utf-8')) for item in benchmark)
    if checkout_smoke or compare_optimization:
        from evaluation_workflow import CHECKOUT_QUESTION
        question_bytes = max(question_bytes, len(CHECKOUT_QUESTION.encode('utf-8')))
    approximate_input = max_tokens + math.ceil(len(SYSTEM_PROMPT.encode('utf-8')) / 3) + 256
    reserved_input = max_tokens * 4 + len(SYSTEM_PROMPT.encode('utf-8')) + question_bytes * 6 + 2048
    cost = token_reservation(config, reserved_input, config.max_output_tokens)
    upper_cost = cost * attempts if cost is not None else None
    limit = monetary(max_total_cost) if max_total_cost is not None else None
    if limit is not None and cost is None:
        raise ValueError('--max-total-cost requires both explicitly configured token prices')
    return {'provider': asdict(config), 'credentials_present': bool(os.environ.get(config.api_key_env)),
            'credentials_required': config.provider not in ('local', 'mock'), 'model_validation': 'configuration_only_not_remotely_verified',
            'question_count': len(benchmark), 'strategy_count': len(strategies), 'strategies': list(strategies),
            'questions': [item['question'] for item in benchmark], 'repeats': repeats, 'components': components,
            'planned_trials': trials, 'expected_api_requests': expected_requests,
            'maximum_completion_http_attempts': attempts, 'model_preflight_requests': preflight,
            'maximum_http_attempts': total_attempts, 'max_context_tokens': max_tokens,
            'max_output_tokens': config.max_output_tokens, 'approximate_input_tokens_per_request': approximate_input,
            'approximate_max_input_tokens': approximate_input * attempts,
            'maximum_output_tokens': config.max_output_tokens * attempts,
            'approximate_max_total_tokens': (approximate_input + config.max_output_tokens) * attempts,
            'reservation_input_tokens_per_request': reserved_input,
            'estimated_upper_bound_usd': float(upper_cost) if upper_cost is not None else None,
            'pricing_source': 'explicit_user_rates_not_provider_verified' if config.input_price_per_million is not None else 'not_configured',
            'max_total_cost_usd': float(limit) if limit is not None else None,
            'fits_configured_cost_estimate': upper_cost <= limit if limit is not None else None,
            'note': 'Planning uses approximate token/byte assumptions, not actual provider usage. Runtime reserves against each actual prompt. '
                    'Provider-hidden framing, tokenizers and pricing can differ; this is not an invoice guarantee. No API/model/retrieval calls in dry run.'}


class EvaluationBudget:
    def __init__(self, config, max_total_cost=None, max_api_requests=None):
        config.validate()
        self.config = config
        self.limit = monetary(max_total_cost) if max_total_cost is not None else None
        if self.limit is not None and token_reservation(config, 1, 1) is None:
            raise ValueError('A cost cap requires configured input/output prices')
        if max_api_requests is not None and max_api_requests < 1:
            raise ValueError('max_api_requests must be positive')
        self.max_api_requests = max_api_requests
        self.charged = Decimal(0)
        self.observed = Decimal(0)
        self.used_attempts = 0
        self.halt_reason = None
        self.events = []
        self.observed_samples = 0

    def reserve(self, attempts, amount):
        if self.halt_reason:
            raise BudgetExceeded('Evaluation stopped: ' + self.halt_reason)
        if self.max_api_requests is not None and self.used_attempts + attempts > self.max_api_requests:
            self.halt_reason = 'api_request_limit'
            raise BudgetExceeded('Next request would exceed --max-api-requests')
        if self.limit is not None and self.charged + amount > self.limit:
            self.halt_reason = 'cost_limit'
            raise BudgetExceeded('Next request reservation would exceed --max-total-cost')

    def preflight(self):
        self.reserve(1, Decimal(0))
        self.used_attempts += 1
        self.events.append({'kind': 'model_catalog', 'http_attempts': 1, 'completion_cost_usd': None})

    def snapshot(self):
        return {'max_total_cost_usd': float(self.limit) if self.limit is not None else None,
                'max_api_requests': self.max_api_requests, 'used_http_attempts': self.used_attempts,
                'conservatively_accounted_usd': float(self.charged) if token_reservation(self.config, 1, 1) is not None else None,
                'observed_completion_cost_usd': float(self.observed) if self.observed_samples else None,
                'observed_cost_samples': self.observed_samples,
                'halt_reason': self.halt_reason, 'events': [dict(event) for event in self.events],
                'note': 'Configured completion-token rates only. Unknown retry usage retains reservations; missing usage under a cost cap stops further calls. '
                        'Actual provider costs may exceed an assumed reservation; such overruns stop subsequent requests but cannot undo a charge.'}


class BudgetedProvider:
    """Wrap the existing providers; MockProvider is intentionally unchanged."""
    def __init__(self, provider, budget):
        if provider.config != budget.config:
            raise ValueError('Provider and budget configurations must match')
        self.provider = provider
        self.config = provider.config
        self.budget = budget

    def complete(self, messages):
        mocked = self.config.provider == 'mock'
        max_attempts = 0 if mocked else self.config.retries + 1
        bound = input_token_bound(messages)
        per_attempt = token_reservation(self.config, bound, self.config.max_output_tokens)
        reservation = (per_attempt or Decimal(0)) * max_attempts
        self.budget.reserve(max_attempts, reservation)
        try:
            response = self.provider.complete(messages)
        except LLMError as error:
            attempts = max(0, min(getattr(error, 'attempts', max_attempts), max_attempts))
            self._settle(bound, per_attempt, attempts, getattr(error, 'usage', None), failed=True)
            raise
        except Exception:
            self._settle(bound, per_attempt, max_attempts, None, failed=True)
            raise
        value = response.get('attempts', max_attempts)
        attempts = min(max_attempts, max(1, value)) if isinstance(value, int) and not isinstance(value, bool) else max_attempts
        self._settle(bound, per_attempt, attempts, response.get('usage'), failed=False)
        return response

    def _settle(self, bound, per_attempt, attempts, usage, failed):
        from llm import token_accounting
        actual = token_accounting([], '', usage)['actual']
        complete = actual['input_tokens'] is not None and actual['output_tokens'] is not None
        observed = token_reservation(self.config, actual['input_tokens'], actual['output_tokens']) if complete else None
        reserve = per_attempt or Decimal(0)
        if self.config.provider == 'mock':
            observed = Decimal(0)
            charged = Decimal(0)
        elif observed is not None:
            charged = observed + reserve * max(0, attempts - 1)
        else:
            charged = reserve * attempts
        self.budget.used_attempts += attempts
        self.budget.charged += charged
        self.budget.observed += observed or Decimal(0)
        self.budget.observed_samples += int(observed is not None)
        if self.budget.limit is not None and attempts:
            if not complete:
                self.budget.halt_reason = 'usage_unavailable_under_cost_cap'
            elif observed > reserve or self.budget.charged > self.budget.limit:
                self.budget.halt_reason = 'provider_cost_exceeded_reservation_or_cap'
        self.budget.events.append({'kind': 'completion', 'http_attempts': attempts,
                                   'input_token_reservation': bound, 'reported_usage_complete': complete,
                                   'observed_cost_usd': float(observed) if observed is not None else None,
                                   'conservatively_accounted_usd': float(charged), 'failed': failed})

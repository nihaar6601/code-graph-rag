"""Small representative profiles and controlled, separate evaluation stages."""
import copy
import json
from pathlib import Path

CHECKOUT_QUESTION = 'How does checkout handle a rejected payment?'
CHECKOUT_ANNOTATION = 'How is reserved stock returned when a checkout payment is rejected?'
PROFILE_PATH = Path(__file__).parent / 'benchmarks' / 'phase61_initial.json'


def select_questions(benchmark, profile='all', limit=None):
    if profile == 'initial':
        manifest = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
        selected = []
        for entry in manifest['questions']:
            matches = [item for item in benchmark if item['question'] == entry['question']]
            if len(matches) != 1:
                raise ValueError('Initial profile requires its exact existing benchmark questions')
            selected.append(matches[0])
    elif profile == 'all':
        selected = list(benchmark)
    else:
        raise ValueError('Unknown benchmark profile')
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        selected = selected[:limit]
    if not selected:
        raise ValueError('Benchmark selection is empty')
    return selected


def checkout_annotation(benchmark):
    matches = [item for item in benchmark if item['question'] == CHECKOUT_ANNOTATION]
    if len(matches) != 1:
        raise ValueError('Checkout experiment requires the existing rejected-payment annotation')
    item = copy.deepcopy(matches[0])
    item['annotation_question'] = item['question']
    item['question'] = CHECKOUT_QUESTION
    return item


def run_stages(state, benchmark, strategies, provider, repeats=1, max_requests=None,
               checkout=None, checkout_smoke=False, compare_optimization=False, **options):
    from evaluate_answers import evaluate_answer_benchmark
    count = len(benchmark) * len(strategies) * repeats
    count += len(strategies) * repeats if checkout_smoke else 0
    count += 2 * sum(strategy in ('graph', 'hybrid_graph') for strategy in strategies) * repeats if compare_optimization else 0
    if max_requests is not None and count > max_requests:
        raise ValueError('Planned stages exceed max_requests')
    if (checkout_smoke or compare_optimization) and checkout is None:
        raise ValueError('Checkout stages require a reused source annotation')
    smoke = None
    if checkout_smoke:
        smoke = evaluate_answer_benchmark(state, [checkout], strategies, provider, repeats, None, **options)
        if any(row['answer']['status'] in ('provider_error', 'invalid_response', 'provider_refusal', 'truncated_response', 'partial', 'budget_exhausted')
               for row in smoke['results']):
            blocked = dict(smoke, results=[], summary={}, smoke_report=smoke, query_count=len(benchmark),
                           planned_trials=len(benchmark) * len(strategies) * repeats,
                           comparison_status='not_executed_smoke_failed')
            if hasattr(provider, 'budget') and provider.budget.halt_reason:
                blocked['comparison_status'] = 'incomplete_budget_do_not_rank_strategies'
            blocked['total_recorded_trials'] = len(smoke['results'])
            return blocked
    report = evaluate_answer_benchmark(state, benchmark, strategies, provider, repeats, None, **options)
    if smoke is not None:
        report['smoke_report'] = smoke
    if compare_optimization:
        modes = {}
        graph_strategies = [strategy for strategy in strategies if strategy in ('graph', 'hybrid_graph')]
        for mode in ('none', 'relevance'):
            trial_options = dict(options, optimization=mode)
            modes[mode] = evaluate_answer_benchmark(state, [checkout], graph_strategies, provider, repeats, None, **trial_options)
        report['optimization_reports'] = modes
        report['optimization_comparison'] = optimization_differences(modes['none'], modes['relevance'])
    if hasattr(provider, 'budget'):
        report['budget'] = provider.budget.snapshot()
        if provider.budget.halt_reason:
            report['comparison_status'] = 'incomplete_budget_do_not_rank_strategies'
    report['total_recorded_trials'] = len(all_rows(report))
    return report


def all_rows(report):
    rows = list(report.get('results', []))
    if report.get('smoke_report'):
        rows.extend(report['smoke_report']['results'])
    for component in report.get('optimization_reports', {}).values():
        rows.extend(component['results'])
    return rows


def optimization_differences(baseline, optimized):
    records = []
    by_key = {(row['strategy'], row['repeat']): row for row in baseline['results']}
    for row in optimized['results']:
        previous = by_key[(row['strategy'], row['repeat'])]
        first, second = previous['answer'], row['answer']
        if first['llm'] != second['llm'] or first['system_prompt_sha256'] != second['system_prompt_sha256']:
            raise ValueError('Optimization trials must share model settings and prompt template')
        before, after = first['metrics'], second['metrics']
        actual_inputs = (before['actual_input_tokens'], after['actual_input_tokens'])
        records.append({'question': row['question'], 'strategy': row['strategy'], 'repeat': row['repeat'],
                        'baseline_context_tokens': before['retrieved_context_tokens'],
                        'optimized_context_tokens': after['retrieved_context_tokens'],
                        'context_tokens_delta': after['retrieved_context_tokens'] - before['retrieved_context_tokens'],
                        'baseline_actual_input_tokens': actual_inputs[0], 'optimized_actual_input_tokens': actual_inputs[1],
                        'actual_input_tokens_delta': actual_inputs[1] - actual_inputs[0] if None not in actual_inputs else None,
                        'correctness_proxy_delta': row['grade']['correctness_proxy'] - previous['grade']['correctness_proxy'],
                        'completeness_proxy_delta': row['grade']['completeness'] - previous['grade']['completeness'],
                        'independently_validated_answer_quality_delta': None,
                        'mocked': first['mocked'] or second['mocked'],
                        'note': 'Automatic proxy differences, not semantic accuracy; fewer tokens do not establish better answers'})
    return records

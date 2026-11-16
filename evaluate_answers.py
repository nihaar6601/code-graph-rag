"""Explicit opt-in answer benchmarking; retrieval-only evaluate.py is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from answer_grading import grade_answer, validate_answer_benchmark
from answering import PROMPT_VERSION, SYSTEM_PROMPT, answer_question
from graph_builder import build_graph
from indexer import index_inputs
from llm import add_llm_arguments, public_config
from strategies import STRATEGIES, prepare_search


FAILURE_STATUSES = {'provider_error', 'invalid_response', 'provider_refusal', 'truncated_response'}


def mean_known(values):
    known = [value for value in values if value is not None]
    return statistics.mean(known) if known else None


def aggregate_answers(rows, strategies):
    summary = {}
    for strategy in strategies:
        selected = [row for row in rows if row['strategy'] == strategy]
        if not selected:
            continue
        metrics = [row['answer']['metrics'] for row in selected]
        grades = [row['grade'] for row in selected]
        count = len(selected)
        summary[strategy] = {
            'samples': count,
            'correctness_proxy': statistics.mean(grade['correctness_proxy'] for grade in grades),
            'completeness': statistics.mean(grade['completeness'] for grade in grades),
            'grounded_completeness': statistics.mean(grade['grounded_completeness'] for grade in grades),
            'citation_validity': statistics.mean(grade['citation_validity'] for grade in grades),
            'appropriate_insufficiency': statistics.mean(grade['appropriate_insufficiency'] for grade in grades),
            'unsupported_claim_proxy_count': sum(grade['unsupported_claim_proxy_count'] for grade in grades),
            'unverified_claim_count': sum(grade['unverified_claim_count'] for grade in grades),
            'failure_rate': sum(row['answer']['status'] in FAILURE_STATUSES for row in selected) / count,
            'insufficient_context_rate': sum(row['answer'].get('insufficient_context') is True for row in selected) / count,
            'mean_cost_usd': mean_known([row['answer']['cost']['estimated_usd'] for row in selected]),
            'cost_samples': sum(row['answer']['cost']['estimated_usd'] is not None for row in selected),
            'actual_input_usage_samples': sum(metric['actual_input_tokens'] is not None for metric in metrics),
            'actual_output_usage_samples': sum(metric['actual_output_tokens'] is not None for metric in metrics),
            'retry_usage_unknown_samples': sum(row['answer']['cost'].get('retry_usage_unknown', False) for row in selected),
            'budget_stop_rate': sum(row['answer']['status'] == 'budget_exhausted' for row in selected) / count,
        }
        for field in ('retrieved_context_tokens', 'actual_input_tokens', 'actual_output_tokens', 'actual_total_tokens',
                      'estimated_input_tokens', 'estimated_output_tokens', 'retrieval_latency_ms',
                      'generation_latency_ms', 'end_to_end_latency_ms'):
            summary[strategy]['mean_' + field] = mean_known([metric[field] for metric in metrics])
        summary[strategy]['end_to_end_latency_stddev_ms'] = statistics.pstdev(metric['end_to_end_latency_ms'] for metric in metrics)
        # Variation within repeated runs of the same question, not query complexity.
        variations = []
        for question in sorted({row['question'] for row in selected}):
            trials = [row['grade']['completeness'] for row in selected if row['question'] == question]
            if len(trials) > 1:
                variations.append({'question': question, 'repeats': len(trials),
                                   'mean_completeness': statistics.mean(trials),
                                   'completeness_stddev': statistics.pstdev(trials),
                                   'latency_stddev_ms': statistics.pstdev(row['answer']['metrics']['end_to_end_latency_ms']
                                                                        for row in selected if row['question'] == question),
                                   'output_tokens_stddev': statistics.pstdev(
                                       row['answer']['metrics']['actual_output_tokens'] for row in selected
                                       if row['question'] == question and row['answer']['metrics']['actual_output_tokens'] is not None)
                                   if all(row['answer']['metrics']['actual_output_tokens'] is not None
                                          for row in selected if row['question'] == question) else None})
        summary[strategy]['repeat_variation'] = variations
    return summary


def evaluate_answer_benchmark(state, benchmark, strategies, provider, repeats=1, max_requests=None, **retrieval_options):
    if repeats < 1 or not strategies or set(strategies) - set(STRATEGIES):
        raise ValueError('Positive repeats and valid strategies required')
    planned = len(benchmark) * len(strategies) * repeats
    if max_requests is not None and (max_requests < 1 or planned > max_requests):
        raise ValueError('Planned query/strategy trials exceed max_requests; reduce --limit/repeats or increase the explicit limit')
    graph = state['graph']
    annotation_ms = None
    if graph is None:
        from time import perf_counter
        started = perf_counter()
        from parser import parse_file
        graph = build_graph([parse_file(path, relative_to=state['chunk_indexing']['root'])
                             for path in state['chunk_indexing']['source_paths']])
        annotation_ms = (perf_counter() - started) * 1000
    validate_answer_benchmark(graph, benchmark)
    rows = []
    for repeat in range(repeats):
        for number, item in enumerate(benchmark):
            # Rotate ordering to reduce a fixed provider/cache ordering advantage.
            shift = (number + repeat) % len(strategies)
            order = list(strategies[shift:]) + list(strategies[:shift])
            for strategy in order:
                answer = answer_question(state, strategy, item['question'], provider, **retrieval_options)
                grade = grade_answer(answer, item, graph)
                rows.append({'question': item['question'], 'category': item.get('category'), 'strategy': strategy,
                             'repeat': repeat + 1, 'reference_answer': item['reference_answer'],
                             'reference_facts': item['facts'], 'annotation_question': item.get('annotation_question', item['question']),
                             'relevant_source_labels': item.get('expected', []),
                             'answer': answer, 'grade': grade})
    configuration = public_config(provider)
    benchmark_digest = hashlib.sha256(json.dumps(benchmark, sort_keys=True).encode()).hexdigest()
    report = {'mode': 'mock_plumbing_only' if provider.config.provider == 'mock' else 'real_provider',
              'results': rows, 'summary': aggregate_answers(rows, strategies), 'repeats': repeats,
              'query_count': len(benchmark), 'planned_trials': planned, 'provider': configuration,
              'prompt_version': PROMPT_VERSION, 'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              'benchmark_sha256': benchmark_digest, 'retrieval_options': retrieval_options,
              'setup': state['timings'], 'embedding_model': state.get('model'),
              'setup_note': 'model_load_ms is local retrieval-embedding startup, not server-side LLM startup; setup is excluded from query latency',
              'source_files': state['files'], 'indexing': state['indexing'],
              'diagnostics': state['diagnostics'], 'annotation_only_graph_ms': annotation_ms,
              'comparison_status': 'mock_only_not_model_quality' if provider.config.provider == 'mock' else 'real_provider_proxy_metrics',
              'observed_models': sorted({row['answer']['returned_model'] for row in rows if row['answer'].get('returned_model')}),
              'note': 'Deterministic lexical factual-point/citation proxies, not semantic answer accuracy. '
                      'Mock results validate plumbing only. Missing provider usage/pricing remains null; repeats do not guarantee determinism.'}
    if hasattr(provider, 'budget'):
        report['budget'] = provider.budget.snapshot()
        if provider.budget.halt_reason or any(row['answer']['status'] == 'budget_exhausted' for row in rows):
            report['comparison_status'] = 'incomplete_budget_do_not_rank_strategies'
    return report


def answer_comparison(report):
    lines = [f"Mode: {report['mode']}; factual/citation proxies only, not verified answer accuracy",
             f"Comparison status: {report.get('comparison_status', 'unknown')}",
             'strategy          correct* complete grounded ctx_tokens input_actual output_actual cost_usd latency_ms fail insuff']
    for strategy, values in report['summary'].items():
        actual_in = values['mean_actual_input_tokens']
        actual_out = values['mean_actual_output_tokens']
        cost = values['mean_cost_usd']
        lines.append(f"{strategy:<17} {values['correctness_proxy']:.3f}    {values['completeness']:.3f}    "
                     f"{values['grounded_completeness']:.3f}    {values['mean_retrieved_context_tokens']:.1f}      "
                     f"{str(round(actual_in, 1)) if actual_in is not None else 'unknown':<12} "
                     f"{str(round(actual_out, 1)) if actual_out is not None else 'unknown':<13} "
                     f"{f'{cost:.6f}' if cost is not None else 'unknown':<8} "
                     f"{values['mean_end_to_end_latency_ms']:.2f}      {values['failure_rate']:.2f} {values['insufficient_context_rate']:.2f}")
    return '\n'.join(lines) + '\n'


def export_answer_report(report, json_path=None, markdown_path=None, table_path=None):
    from answer_reporting import render_answer_report
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_answer_report(report), encoding='utf-8')
    if table_path and report.get('results'):
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(answer_comparison(report), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('--answer', action='store_true', help='Explicit opt-in for real model requests')
    parser.add_argument('--strategies', default=','.join(STRATEGIES))
    parser.add_argument('--benchmark', type=Path, default=Path(__file__).parent / 'benchmarks/shop_answers.json')
    parser.add_argument('--model', help='LLM model shorthand; use --embedding-model for local retrieval embeddings')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--max-tokens', type=int, default=2000)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--optimization', choices=('none', 'relevance'), default='none')
    parser.add_argument('--expansion', choices=('auto', 'always', 'never'), default='auto')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Use the first N benchmark queries for a small controlled run')
    parser.add_argument('--max-requests', type=int, help='Bound logical query/strategy trials; retries may add HTTP attempts')
    parser.add_argument('--json', type=Path, dest='json_path')
    parser.add_argument('--markdown', type=Path, help='Human-review report (defaults beside --json)')
    parser.add_argument('--dry-run', action='store_true', help='Plan requests/tokens/cost without credentials, retrieval, models or API calls')
    parser.add_argument('--profile', choices=('all', 'initial'), default='all')
    parser.add_argument('--checkout-smoke', action='store_true', help='First run the checkout-rejection question with every selected strategy')
    parser.add_argument('--compare-optimization', action='store_true', help='Separate none/relevance runs for graph-based strategies')
    parser.add_argument('--max-api-requests', type=int, help='Maximum HTTP attempts, including retries and model-catalog preflight')
    parser.add_argument('--max-total-cost', type=float, help='Client-side USD reservation cap; explicit token prices required')
    parser.add_argument('--skip-model-check', action='store_true', help='Skip GET /models for servers without a catalog (recorded as unverified)')
    add_llm_arguments(parser)
    args = parser.parse_args()
    if not args.dry_run and not args.answer and args.provider != 'mock':
        parser.error('Use --answer to opt in to real LLM evaluation, or --provider mock for offline plumbing')
    try:
        from dataclasses import asdict
        from evaluation_controls import plan_evaluation, EvaluationBudget, BudgetedProvider
        from evaluation_workflow import select_questions, checkout_annotation, run_stages, all_rows
        from llm import config_from_arguments, create_provider, LLMError
        from indexer import discover_inputs
        config = config_from_arguments(args, args.model, require_credentials=not args.dry_run)
        selected = list(dict.fromkeys(strategy.strip() for strategy in args.strategies.split(',')))
        if not selected or set(selected) - set(STRATEGIES):
            raise ValueError('Unknown or empty strategy selection')
        if args.top_k < 1 or args.depth < 0 or args.max_tokens < 0:
            raise ValueError('top-k must be positive; depth/budget must be nonnegative')
        if not args.repository.exists():
            raise ValueError('Repository path does not exist')
        annotations = json.loads(args.benchmark.read_text(encoding='utf-8'))
        benchmark = select_questions(annotations, args.profile, args.limit)
        checkout = checkout_annotation(annotations) if args.checkout_smoke or args.compare_optimization else None
        plan = plan_evaluation(config, benchmark, selected, args.repeats, args.max_tokens,
                               args.checkout_smoke, args.compare_optimization, args.skip_model_check,
                               args.max_requests, args.max_api_requests, args.max_total_cost)
        plan['profile'] = args.profile
        plan['optimization'] = args.optimization
        plan['top_k'] = args.top_k
        plan['depth'] = args.depth
        markdown_path = args.markdown or (args.json_path.with_suffix('.md') if args.json_path else None)
        table_path = args.json_path.with_suffix('.comparison.txt') if args.json_path else None
        outputs = [path for path in (args.json_path, markdown_path, table_path) if path is not None]
        protected = set(discover_inputs([args.repository])['source_paths']) | {str(args.benchmark.resolve())}
        protected.add(str((Path(__file__).parent / 'benchmarks/phase61_initial.json').resolve()))
        if args.mock_response:
            protected.add(str(args.mock_response.resolve()))
        resolved_outputs = [str(path.resolve()) for path in outputs]
        if set(resolved_outputs) & protected or len(resolved_outputs) != len(set(resolved_outputs)):
            raise ValueError('Report outputs must be distinct and must not overwrite source/benchmark/mock inputs')
        if args.dry_run:
            report = {'mode': 'dry_run', 'comparison_status': 'not_executed', 'provider': asdict(config),
                      'plan': plan, 'query_count': len(benchmark), 'planned_trials': plan['planned_trials'],
                      'repeats': args.repeats, 'results': [], 'summary': {},
                      'note': 'No API requests, provider instantiation, retrieval or embedding models were used.'}
            print('DRY RUN — no API requests\n' + json.dumps(plan, indent=2))
            export_answer_report(report, args.json_path, markdown_path, table_path)
            return 0
        if plan['fits_configured_cost_estimate'] is False:
            raise ValueError('Estimated full-plan reservation exceeds --max-total-cost; inspect --dry-run and adjust limits/pricing')
        provider = create_provider(config, args.mock_response)
        budget = EvaluationBudget(config, args.max_total_cost, args.max_api_requests)
        state = prepare_search([args.repository], selected, model=args.embedding_model)
        if state['graph'] is not None:
            validate_answer_benchmark(state['graph'], benchmark)
            if checkout:
                validate_answer_benchmark(state['graph'], [checkout])
        # Never pay for an allegedly fair comparison with unavailable source/semantic indexes.
        if any(file.get('has_error') for file in state['files']) or state['indexing'].get('diagnostics') or any(
                problem['kind'] == 'semantic_unavailable' for problem in state['diagnostics']):
            raise ValueError('Indexing/embedding setup is incomplete; resolve diagnostics before LLM evaluation')
        preflight = {'status': 'mock_no_network' if config.provider == 'mock' else 'user_skipped_not_remotely_verified'}
        if config.provider != 'mock' and not args.skip_model_check:
            budget.preflight()
            try:
                preflight = provider.check_model()
            except LLMError as error:
                report = {'mode': 'preflight_failed', 'comparison_status': 'not_executed', 'provider': asdict(config),
                          'plan': plan, 'budget': budget.snapshot(), 'preflight': {'status': 'failed', 'error': str(error)},
                          'setup': state['timings'], 'results': [], 'summary': {}}
                export_answer_report(report, args.json_path, markdown_path, table_path)
                print('Model preflight failed; no generation requests were made: ' + str(error))
                return 1
        controlled = BudgetedProvider(provider, budget)
        report = run_stages(state, benchmark, selected, controlled, args.repeats, args.max_requests,
                            checkout=checkout, checkout_smoke=args.checkout_smoke,
                            compare_optimization=args.compare_optimization, max_tokens=args.max_tokens,
                            top_k=args.top_k, depth=args.depth, expansion=args.expansion, optimization=args.optimization)
        report['plan'], report['preflight'] = plan, preflight
        print(answer_comparison(report))
        print(report['note'])
        print('Budget: ' + json.dumps(report['budget']))
        export_answer_report(report, args.json_path, markdown_path, table_path)
        failed = any(row['answer']['status'] in FAILURE_STATUSES | {'budget_exhausted'} for row in all_rows(report))
        failed = failed or bool(budget.halt_reason)
        failed = failed or report.get('comparison_status') == 'not_executed_smoke_failed'
        failed = failed or any(file.get('has_error') for file in state['files']) or state['indexing'].get('diagnostics')
        failed = failed or any(problem['kind'] == 'semantic_unavailable' for problem in state['diagnostics'])
        return 1 if failed else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'Answer evaluation failed: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())

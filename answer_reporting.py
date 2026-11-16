"""Static Markdown answer reports. No provider access, I/O, or human grading."""
import html
import json
import re
from urllib.parse import urlsplit, urlunsplit


PUBLIC_PROVIDER_KEYS = (
    'provider', 'model', 'base_url', 'temperature', 'max_output_tokens',
    'timeout', 'retries', 'output_token_parameter', 'json_mode',
    'omit_temperature', 'input_price_per_million', 'output_price_per_million',
)
SUMMARY_KEYS = (
    'samples', 'correctness_proxy', 'completeness', 'grounded_completeness',
    'citation_validity', 'appropriate_insufficiency', 'unsupported_claim_proxy_count',
    'unverified_claim_count', 'failure_rate', 'insufficient_context_rate',
    'mean_cost_usd', 'cost_samples', 'actual_input_usage_samples',
    'actual_output_usage_samples', 'retry_usage_unknown_samples',
    'mean_retrieved_context_tokens', 'mean_actual_input_tokens',
    'mean_actual_output_tokens', 'mean_actual_total_tokens',
    'mean_estimated_input_tokens', 'mean_estimated_output_tokens',
    'mean_retrieval_latency_ms', 'mean_generation_latency_ms',
    'mean_end_to_end_latency_ms', 'end_to_end_latency_stddev_ms', 'repeat_variation',
    'budget_stop_rate',
)
COMPARISON_KEYS = (
    'samples', 'correctness_proxy', 'completeness', 'grounded_completeness', 'citation_validity',
    'mean_retrieved_context_tokens', 'mean_actual_input_tokens', 'mean_estimated_input_tokens', 'mean_actual_output_tokens',
    'mean_cost_usd', 'mean_end_to_end_latency_ms', 'failure_rate', 'insufficient_context_rate', 'budget_stop_rate',
)


def _private(key):
    name = str(key).lower().replace('-', '_')
    return any(word in name for word in ('api_key', 'authorization', 'password', 'secret', 'credential')) or name in (
        'token', 'access_token', 'refresh_token', 'bearer_token', 'headers', 'cookie', 'cookies')


def _clean(value, active=None):
    """Filter credential fields and cycles without invoking configuration accessors."""
    active = set() if active is None else active
    if isinstance(value, (dict, list, tuple)):
        if id(value) in active:
            return '[cycle omitted]'
        active.add(id(value))
        if isinstance(value, dict):
            result = {str(key): _clean(item, active) for key, item in value.items() if not _private(key)}
        else:
            result = [_clean(item, active) for item in value]
        active.remove(id(value))
        return result
    if value is None:
        return 'unknown'
    if isinstance(value, (str, int, float, bool)):
        return value
    return '[unavailable]'


def _text(value):
    """Escape HTML and Markdown control characters in ordinary report text."""
    text = html.escape(str('unknown' if value is None else value), quote=True)
    return re.sub(r'([\\`*_{}\[\]()#+.!|>~\-])', r'\\\1', text).replace('\n', '<br>')


def _block(value, language='text'):
    # A fence longer than every source backtick run cannot be closed by source.
    text = value if isinstance(value, str) else json.dumps(_clean(value), indent=2, ensure_ascii=False)
    fence = '`' * max(3, 1 + max((len(run) for run in re.findall(r'`+', text)), default=0))
    return f'{fence}{language}\n{text}\n{fence}\n'


def _section(title, value):
    return f'\n#### {title}\n\n' + ('Unavailable (unknown)\n' if value is None else _block(_clean(value), 'json'))


def _provider(value):
    value = value if isinstance(value, dict) else {}
    result = {key: value[key] for key in PUBLIC_PROVIDER_KEYS if key in value}
    if 'base_url' in result:
        try:
            url = urlsplit(str(result['base_url']))
            # Userinfo, query parameters and fragments may carry credentials.
            host = url.netloc.rsplit('@', 1)[-1]
            result['base_url'] = urlunsplit((url.scheme, host, url.path, '', ''))
        except ValueError:
            result['base_url'] = '[invalid URL omitted]'
    return result


def _row(row, number):
    row = row if isinstance(row, dict) else {}
    answer = row.get('answer') or {}
    retrieval = answer.get('retrieval') or {}
    lines = [f'\n### Trial {number}\n',
             'Original question: ' + _text(row.get('question')) + '\n',
             'Category: ' + _text(row.get('category')) + '; strategy: ' + _text(row.get('strategy')) +
             '; repeat: ' + _text(row.get('repeat')) + '\n',
             'Reference answer: ' + _text(row.get('reference_answer')) + '\n',
             _section('Reference facts (annotations)', row.get('reference_facts')),
             '\n#### Retrieved context (original text)\n\n']
    if row.get('annotation_question') and row['annotation_question'] != row.get('question'):
        lines.insert(4, 'Annotation reused unchanged from: ' + _text(row['annotation_question']) + '\n')
    context = retrieval.get('context', retrieval.get('text'))
    lines.append(_block(context) if context is not None else 'Unavailable\n')
    sources = answer.get('sources')
    if sources is None:
        sources = retrieval.get('snippets')
    lines.append(_section('Evidence blocks (original source and locations)', sources))
    claims = answer.get('claims')
    lines.append(_section('Citations and claims (including rejected citations)', claims))
    lines.extend(['\n#### Rendered answer\n\n', _text(answer.get('answer', 'Unavailable')) + '\n',
                  'Status: ' + _text(answer.get('status')) +
                  '; insufficient context: ' + _text(answer.get('insufficient_context')) + '\n',
                  _section('Usage and latency (actual and estimated tokens are separate)', answer.get('metrics')),
                  _section('Cost estimate (not a verified bill)', answer.get('cost')),
                  _section('Fact-check proxy results (not semantic validation)', row.get('grade')),
                  _section('Answer diagnostics', answer.get('diagnostics')),
                  '\n#### Human review\n\nNot independently reviewed\n\n',
                  '- [ ] Factual correctness\n- [ ] Completeness\n- [ ] Semantic grounding\n'
                  '- [ ] Unsupported claims\n- [ ] Insufficiency handling\n'])
    return '\n'.join(lines)


def _render(report, title):
    """Render one run only; nested reports are handled separately by the caller."""
    mode = report.get('mode', 'unknown')
    lines = [f'## {title}\n', 'Mode: ' + _text(mode) + '\n',
             'Comparison status: ' + _text(report.get('comparison_status')) + '\n']
    if 'mock' in str(mode).lower() or _provider(report.get('provider')).get('provider') == 'mock' or any(
            (row.get('answer') or {}).get('mocked') for row in report.get('results', [])):
        lines.append('**MOCK: plumbing validation only; not real-model answer quality.**\n')
    lines.append('Live provider results, when present, are automated proxies only. Regex fact coverage and '
                 'citation-range validity do not establish factual correctness, semantic grounding, or absence '
                 'of hallucinations. No human validation is implied. Not independently reviewed.\n')
    lines.append(_section('Provider settings (public fields only)', _provider(report.get('provider'))))
    for field in ('query_count', 'planned_trials', 'repeats'):
        lines.append(field + ': ' + _text(report.get(field)) + '\n')
    for field in ('budget', 'preflight', 'setup', 'diagnostics'):
        lines.append(_section(field.capitalize(), report.get(field)))
    if report.get('optimization_comparison'):
        lines.append(_section('Separate optimization differences (not verified quality)', report['optimization_comparison']))
    if mode == 'dry_run':
        lines.append(_section('Dry-run plan (no answer quality results)', report.get('plan')))
        return '\n'.join(lines)
    if report.get('plan'):
        lines.append(_section('Request and cost plan', report['plan']))
    lines.append('\n### Summary comparison (proxy metrics only)\n\n'
                 'Missing usage is unknown, never zero. Actual tokens and estimated tokens are separate; '
                 'cost is an estimate. Setup is separate from per-query latency.\n\n')
    summary = report.get('summary') or {}
    keys = [key for key in COMPARISON_KEYS if any(key in values for values in summary.values())]
    if keys:
        lines.append('| Strategy | ' + ' | '.join(_text(key) for key in keys) + ' |')
        lines.append('| --- | ' + ' | '.join('---' for _ in keys) + ' |')
        for strategy, values in summary.items():
            cells = [json.dumps(_clean(values.get(key)), ensure_ascii=False) if isinstance(values.get(key), (dict, list))
                     else values.get(key) for key in keys]
            lines.append('| ' + _text(strategy) + ' | ' + ' | '.join(_text(cell) for cell in cells) + ' |')
    else:
        lines.append('Summary unavailable.\n')
    detailed = {html.escape(str(strategy), quote=True): {key: values[key] for key in SUMMARY_KEYS if key in values}
                for strategy, values in summary.items()}
    lines.append(_section('Detailed summary (including estimated usage and repeat variation)', detailed))
    for number, row in enumerate(report.get('results') or [], 1):
        lines.append(_row(row, number))
    return '\n'.join(lines)


def render_answer_report(report):
    """Return readable, static Markdown for an evaluate_answer_benchmark report.

    Only explicit public fields are selected. Nested smoke/optimization runs are
    rendered once, separately, without pooling quality metrics or following cycles.
    Source text is preserved inside dynamically sized code fences (never raw HTML).
    """
    if not isinstance(report, dict):
        raise TypeError('report must be a dictionary')
    lines = ['# Answer quality benchmark report\n', _render(report, 'Main run')]
    smoke = report.get('smoke_report')
    if isinstance(smoke, dict) and smoke is not report:
        lines.extend(['\nSeparate run; quality metrics are not combined.\n', _render(smoke, 'Smoke run')])
    for mode, nested in (report.get('optimization_reports') or {}).items():
        if isinstance(nested, dict) and nested is not report:
            lines.extend(['\nSeparate run; quality metrics are not combined.\n',
                          _render(nested, 'Optimization run: ' + _text(mode))])
    return '\n'.join(lines).rstrip() + '\n'

"""Compare lexical and graph-enhanced retrieval against a small manual benchmark."""
import argparse
import json
from pathlib import Path

from graph_builder import build_graph
from indexer import index_inputs
from retrieval import entity_label, retrieve_context
from token_counter import ENCODING_NAME, get_tokenizer


def retrieval_metrics(retrieved, expected):
    retrieved, expected = set(retrieved), set(expected)
    correct = len(retrieved & expected)
    precision = correct / len(retrieved) if retrieved else (1.0 if not expected else 0.0)
    recall = correct / len(expected) if expected else 1.0
    return precision, recall


def evaluate_graph(graph, benchmark, max_tokens=2000, top_k=1, depth=1):
    get_tokenizer().encode('warmup', disallowed_special=())
    labels = {}
    for entity_id in graph.nodes:
        labels.setdefault(entity_label(graph, entity_id), []).append(entity_id)
    rows = []
    for item in benchmark:
        expected_ids = set()
        for label in item['expected']:
            if len(labels.get(label, [])) != 1:
                raise ValueError(f'Benchmark expected method is missing or ambiguous: {label}')
            expected_ids.add(labels[label][0])
        for strategy, expansion in [('structure_only', False), ('graph_enhanced', True)]:
            result = retrieve_context(graph, item['question'], max_tokens=max_tokens,
                                      top_k=top_k, depth=depth, expand=expansion)
            retrieved = {snippet['id'] for snippet in result['snippets']}
            precision, recall = retrieval_metrics(retrieved, expected_ids)
            rows.append({
                'question': item['question'], 'strategy': strategy,
                'retrieved_entities': [{'id': s['id'], 'name': s['name']} for s in result['snippets']],
                'context_tokens': result['token_count'], 'latency_ms': result['latency_ms'],
                'direct_count': result['direct_count'], 'expanded_count': result['expanded_count'],
                'expected': sorted(item['expected']), 'expected_retrieved': sorted(entity_label(graph, e) for e in retrieved & expected_ids),
                'missed': sorted(entity_label(graph, e) for e in expected_ids - retrieved),
                'unexpected': sorted(entity_label(graph, e) for e in retrieved - expected_ids),
                'precision': precision, 'recall': recall, 'diagnostics': result['diagnostics'],
            })
    summary = {}
    for strategy in ('structure_only', 'graph_enhanced'):
        selected = [row for row in rows if row['strategy'] == strategy]
        count = len(selected)
        summary[strategy] = {
            'macro_precision': sum(row['precision'] for row in selected) / count if count else 0,
            'macro_recall': sum(row['recall'] for row in selected) / count if count else 0,
            'mean_context_tokens': sum(row['context_tokens'] for row in selected) / count if count else 0,
            'mean_latency_ms': sum(row['latency_ms'] for row in selected) / count if count else 0,
        }
    return {'tokenizer': ENCODING_NAME, 'max_tokens': max_tokens, 'top_k': top_k,
            'depth': depth, 'results': rows, 'summary': summary,
            'note': 'Retrieval quality on a small manually labeled benchmark, not answer accuracy. Lower token use is not inherently better.'}


def print_evaluation(report):
    for row in report['results']:
        names = ', '.join(entity['name'] for entity in row['retrieved_entities']) or '(none)'
        print(f"\n{row['strategy']}: {row['question']}\n  Retrieved: {names}")
        print(f"  Tokens={row['context_tokens']}, latency={row['latency_ms']:.3f}ms, "
              f"direct={row['direct_count']}, expanded={row['expanded_count']}, "
              f"precision={row['precision']:.3f}, recall={row['recall']:.3f}")
    print('\nMacro averages:')
    for strategy, scores in report['summary'].items():
        print(f"  {strategy}: precision={scores['macro_precision']:.3f}, recall={scores['macro_recall']:.3f}, "
              f"tokens={scores['mean_context_tokens']:.1f}, latency={scores['mean_latency_ms']:.3f}ms")
    print(report['note'])


def evaluate_strategies(inputs, benchmark, strategies, max_tokens=2000, top_k=3, depth=1,
                        encoder=None, **options):
    """Use source-byte relevance for every strategy, not chunk==method counts."""
    from time import perf_counter
    from strategies import prepare_search, retrieve_strategy
    from relevance import expected_records, source_metrics
    get_tokenizer().encode('warmup', disallowed_special=())
    setup_keys = ('model', 'model_cache', 'cache_dir', 'download_model', 'revision', 'chunk_size', 'overlap')
    setup = {key: options.pop(key) for key in setup_keys if key in options}
    state = prepare_search(inputs, strategies, encoder=encoder, **setup)
    graph = state['graph']
    if graph is None:
        started = perf_counter()
        indexed = index_inputs(inputs)
        graph = build_graph(indexed['files'])
        state['timings']['annotation_only_graph_ms'] = (perf_counter() - started) * 1000
    rows = []
    for item in benchmark:
        expected = expected_records(graph, item)
        for strategy in strategies:
            result = retrieve_strategy(state, strategy, item['question'], max_tokens, top_k, depth, **options)
            metrics = source_metrics(result['snippets'], expected)
            records = {record['id']: record for record in state['chunks']} if strategy == 'traditional' else dict(graph.nodes(data=True))
            initial = [records[candidate['id']] for candidate in result['candidates'][:top_k] if candidate['id'] in records]
            recall_at_k = source_metrics(initial, expected)['recall']
            rows.append(dict(metrics, question=item['question'], category=item.get('category', 'unspecified'),
                             strategy=strategy, context_tokens=result['token_count'], latency_ms=result['latency_ms'],
                             semantic_status=result['semantic_status'],
                             snippet_count=len(result['snippets']), expanded_count=result['expanded_count'],
                             direct_count=result['direct_count'], recall_at_k=recall_at_k,
                             intent=result.get('intent'), expansion_policy=result.get('expansion_policy'),
                             retrieved=[{key: snippet[key] for key in ('id', 'name', 'file_path', 'start_line', 'end_line', 'start_byte', 'end_byte')}
                                        for snippet in result['snippets']], diagnostics=result['diagnostics']))
    summary = {}
    for strategy in strategies:
        selected = [row for row in rows if row['strategy'] == strategy]
        count = len(selected)
        summary[strategy] = {key: sum(row[key] for row in selected) / count if count else 0 for key in
                             ('precision', 'recall', 'f1', 'recall_at_k', 'context_tokens', 'latency_ms',
                              'snippet_count', 'relevant_entities', 'relevant_snippets', 'expanded_count')}
    contrasts = []
    if 'structure' in strategies and 'hybrid_graph' in strategies:
        for item in benchmark:
            matched = [row for row in rows if row['question'] == item['question']]
            first = next(row for row in matched if row['strategy'] == 'structure')
            second = next(row for row in matched if row['strategy'] == 'hybrid_graph')
            contrasts.append({'question': item['question'], 'structure_f1': first['f1'],
                              'hybrid_graph_f1': second['f1'], 'delta': second['f1'] - first['f1']})
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'sentence-transformers', 'torch', 'tiktoken', 'transformers')}
    return {'results': rows, 'summary': summary, 'overhead': state['timings'], 'versions': versions,
            'contrasts': sorted(contrasts, key=lambda item: item['delta']),
            'diagnostics': state['diagnostics'], 'model': state.get('model'), 'indexing': state['indexing'],
            'files': state['files'], 'max_tokens': max_tokens, 'top_k': top_k, 'depth': depth,
            'configuration': {key: str(value) if isinstance(value, Path) else value
                              for key, value in dict(setup, **options).items()}, 'tokenizer': ENCODING_NAME,
            'metric_rule': 'Union of UTF-8 source byte ranges; precision/recall/F1 shared by chunks and entities. '
                           'Relevant entities require >=80% byte coverage. Recall@k measures initial candidates before budget/expansion.',
            'note': 'Retrieval quality, not answer accuracy. Macro query averages; latency excludes setup. '
                    'Empty-ground-truth recall is 1; empty predictions precision is 1 only when ground truth is empty.'}


def comparison_table(report):
    lines = ['strategy          precision  recall     F1         recall@k   tokens     latency_ms']
    for strategy, values in report['summary'].items():
        lines.append(f"{strategy:<17} {values['precision']:<10.4f} {values['recall']:<10.4f} "
                     f"{values['f1']:<10.4f} {values['recall_at_k']:<10.4f} "
                     f"{values['context_tokens']:<10.1f} {values['latency_ms']:.3f}")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', nargs='?', type=Path, default=Path(__file__).parent / 'examples/repository')
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--max-tokens', type=int, default=2000)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--json', type=Path, dest='json_path')
    parser.add_argument('--strategies', help='Comma-separated traditional,structure,graph,hybrid,hybrid_graph')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--model-cache', type=Path, default=Path(__file__).parent / '.cache/models')
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).parent / '.cache/embeddings')
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--revision')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--chunk-overlap', type=int, default=24)
    parser.add_argument('--lexical-candidates', type=int, default=20)
    parser.add_argument('--semantic-candidates', type=int, default=20)
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--min-similarity', type=float, default=0.2)
    parser.add_argument('--expansion', choices=('auto', 'always', 'never'), default='auto')
    args = parser.parse_args()
    if args.benchmark is None:
        name = 'shop_queries.json' if args.repository.name == 'shop' else 'payment_queries.json'
        args.benchmark = Path(__file__).parent / 'benchmarks' / name
    try:
        if args.strategies:
            benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
            selected = list(dict.fromkeys(strategy.strip() for strategy in args.strategies.split(',')))
            report = evaluate_strategies([args.repository], benchmark, selected, args.max_tokens, args.top_k, args.depth,
                                        model=args.model, model_cache=args.model_cache, cache_dir=args.cache_dir,
                                        download_model=args.download_model, revision=args.revision,
                                        chunk_size=args.chunk_size, overlap=args.chunk_overlap,
                                        lexical_candidates=args.lexical_candidates, semantic_candidates=args.semantic_candidates,
                                        rrf_k=args.rrf_k, min_similarity=args.min_similarity, expansion=args.expansion)
            print(comparison_table(report))
            print('Setup overhead (ms and embedding cache statistics): ' + json.dumps(report['overhead']))
            print(report['metric_rule'])
            print(report['note'])
            if args.json_path:
                protected = set(index_inputs([args.repository])['source_paths']) | {str(args.benchmark.resolve())}
                table_path = args.json_path.with_suffix('.comparison.txt')
                if str(args.json_path.resolve()) in protected or str(table_path.resolve()) in protected:
                    parser.exit(1, 'Evaluation output must not overwrite source or benchmark inputs\n')
                args.json_path.parent.mkdir(parents=True, exist_ok=True)
                args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
                table_path.write_text(comparison_table(report), encoding='utf-8')
            failed = any(item.get('has_error') for item in report['files']) or report['indexing'].get('diagnostics')
            return 1 if failed or any(d['kind'] == 'semantic_unavailable' for d in report['diagnostics']) else 0
        indexing = index_inputs([args.repository])
        graph = build_graph(indexing['files'])
        graph.graph['indexing'] = {key: value for key, value in indexing.items() if key not in ('files', 'source_paths')}
        benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
        report = evaluate_graph(graph, benchmark, args.max_tokens, args.top_k, args.depth)
        report['indexing'] = graph.graph['indexing']
        report['files'] = graph.graph['files']
        print_evaluation(report)
        if args.json_path:
            protected = set(indexing['source_paths']) | {str(args.benchmark.resolve())}
            if str(args.json_path.resolve()) in protected:
                parser.exit(1, 'Evaluation output must not overwrite source or benchmark inputs\n')
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')
    return 1 if indexing['diagnostics'] or any(file['has_error'] for file in indexing['files']) else 0


if __name__ == '__main__':
    raise SystemExit(main())

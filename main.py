"""Command-line entry point for Java parsing and graph export."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from graph_builder import build_graph, graph_to_dict
from indexer import index_inputs


def print_graph(graph):
    for file_info in graph.graph['files']:
        print(f"File: {file_info['file_path']}")
        for diagnostic in file_info['diagnostics']:
            print(f"  Problem ({diagnostic['kind']}): {diagnostic['message']}")
    for node_id, entity in graph.nodes(data=True):
        if entity['kind'] != 'class':
            continue
        print(f"\nClass: {entity['qualified_name']}")
        for _, target, edge in graph.out_edges(node_id, data=True):
            if edge['kind'] == 'CONTAINS':
                member = graph.nodes[target]
                label = member.get('signature', member.get('qualified_name'))
                print(f"  CONTAINS -> {label} [{member['kind']}]")
    for call in graph.graph['invocations']:
        caller_entity = graph.nodes[call['caller']]
        caller_class = graph.nodes[caller_entity['class_id']]['qualified_name']
        caller = f"{caller_class}.{caller_entity['signature']}"
        if call['resolved_target']:
            target_entity = graph.nodes[call['resolved_target']]
            target = target_entity['signature']
            target_class = graph.nodes[target_entity['class_id']]['qualified_name']
            print(f"  {caller} CALLS -> {target} (line {call['start_line']})")
            if caller_entity['class_id'] != target_entity['class_id']:
                print(f"    Target class: {target_class}")
        else:
            print(f"  {caller} UNRESOLVED -> {call['source']} (line {call['start_line']}: {call['unresolved_reason']})")


def print_summary(graph, indexing, show_file_problems=False):
    entities = list(graph.nodes.values())
    successful = sum(not file['has_error'] for file in graph.graph['files'])
    classes = sum(entity['kind'] == 'class' for entity in entities)
    methods = sum(entity['kind'] == 'method' for entity in entities)
    constructors = sum(entity['kind'] == 'constructor' for entity in entities)
    resolved = sum(call['resolved_target'] is not None for call in graph.graph['invocations'])
    unresolved = len(graph.graph['invocations']) - resolved
    print(f"\nSummary: {indexing['discovered']} files discovered; {successful} parsed successfully; "
          f"{len(graph.graph['files']) - successful} with errors; {len(indexing['skipped'])} skipped")
    print(f'{classes} classes, {methods} methods, {constructors} constructors; '
          f'{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
    print(f'{resolved} resolved calls, {unresolved} unresolved calls; '
          f"{len(indexing['diagnostics'])} indexing diagnostics")
    for problem in indexing['diagnostics']:
        print(f"  Problem ({problem['kind']}): {problem['message']}")
    if show_file_problems:
        for file in graph.graph['files']:
            for problem in file['diagnostics']:
                print(f"  Problem ({problem['kind']}) in {file['file_path']}: {problem['message']}")


def main():
    arguments = argparse.ArgumentParser(description='Build a Java structural code graph.')
    arguments.add_argument('files', nargs='*', help='Java files or repository directories')
    arguments.add_argument('--json', type=Path, dest='json_path', help='Export entities, relationships and invocations')
    arguments.add_argument('--verbose', action='store_true', help='List individual entities and calls in directory mode')
    arguments.add_argument('--query', help='Retrieve source context for a natural-language question')
    arguments.add_argument('--max-tokens', type=int, default=2000, help='Maximum assembled-context tokens')
    arguments.add_argument('--top-k', type=int, default=3, help='Number of initial lexical candidates')
    arguments.add_argument('--depth', type=int, default=1, help='CALLS expansion depth')
    arguments.add_argument('--no-expand', action='store_true', help='Use structure-only lexical retrieval')
    arguments.add_argument('--include-callers', action='store_true', help='Also traverse incoming CALLS edges')
    arguments.add_argument('--max-expanded', type=int, default=10, help='Maximum graph-expanded entities')
    arguments.add_argument('--include-classes', action='store_true', help='Allow complete class snippets explicitly')
    arguments.add_argument('--context-json', type=Path, help='Export retrieval context and metadata')
    arguments.add_argument('--output-dir', type=Path, help='Save context.txt and retrieval.json, e.g. output/')
    arguments.add_argument('--strategy', choices=('traditional', 'structure', 'graph', 'hybrid', 'hybrid_graph'))
    arguments.add_argument('--expansion', choices=('auto', 'always', 'never'), help='Query-aware graph policy for explicit strategies')
    arguments.add_argument('--model', help='Embedding model in retrieval-only mode; LLM shorthand with --answer')
    arguments.add_argument('--embedding-model', help='Separate local embedding model, especially with --answer')
    arguments.add_argument('--model-cache', type=Path, default=Path(__file__).parent / '.cache/models')
    arguments.add_argument('--cache-dir', type=Path, default=Path(__file__).parent / '.cache/embeddings')
    arguments.add_argument('--download-model', action='store_true', help='Explicitly permit local model downloads')
    arguments.add_argument('--revision', help='Optional pinned model revision')
    arguments.add_argument('--chunk-size', type=int, default=128)
    arguments.add_argument('--chunk-overlap', type=int, default=24)
    arguments.add_argument('--lexical-candidates', type=int, default=20)
    arguments.add_argument('--semantic-candidates', type=int, default=20)
    arguments.add_argument('--rrf-k', type=int, default=60)
    arguments.add_argument('--min-similarity', type=float, default=0.2)
    arguments.add_argument('--answer', action='store_true', help='Opt in to LLM answering (may call the configured provider)')
    arguments.add_argument('--answer-json', type=Path, help='Export answer, citations, retrieval and metrics')
    arguments.add_argument('--optimization', choices=('none', 'relevance'), default='none')
    from llm import add_llm_arguments, provider_from_arguments
    add_llm_arguments(arguments)
    args = arguments.parse_args()
    if args.max_tokens < 0 or args.top_k < 1 or args.depth < 0 or args.max_expanded < 0:
        arguments.error('top-k must be positive; max-tokens, depth and max-expanded must be nonnegative')
    if args.query is not None and not args.query.strip():
        arguments.error('query must not be empty')
    if (args.context_json or args.output_dir) and args.query is None:
        arguments.error('context-json and output-dir require --query')
    if args.answer and args.query is None:
        arguments.error('--answer requires --query')
    if args.answer_json and not args.answer:
        arguments.error('--answer-json requires --answer')
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        arguments.error('chunk overlap must be nonnegative and less than chunk size')
    if min(args.lexical_candidates, args.semantic_candidates, args.rrf_k) < 1 or not -1 <= args.min_similarity <= 1:
        arguments.error('Candidate counts/rrf-k must be positive; min-similarity must be in [-1,1]')
    files = args.files or [Path(__file__).parent / 'examples' / 'PaymentService.java']
    provider = None
    if args.answer:
        try:
            provider = provider_from_arguments(args, args.model)
        except (OSError, ValueError) as error:
            arguments.error(str(error))
        args.strategy = args.strategy or 'graph'
    embedding_model = args.embedding_model or (args.model if not args.answer and args.model else 'sentence-transformers/all-MiniLM-L6-v2')
    setup_options = dict(model=embedding_model, model_cache=args.model_cache, cache_dir=args.cache_dir,
                         download_model=args.download_model, revision=args.revision,
                         chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    state = None
    indexing_started = perf_counter()
    if args.query is not None and args.strategy == 'traditional' and args.json_path is None:
        from strategies import prepare_search
        state = prepare_search(files, ['traditional'], **setup_options)
        indexing = state['chunk_indexing']
    else:
        indexing = index_inputs(files)
    results = indexing['files']
    seen_paths = set(indexing['source_paths'])
    outputs = [path for path in (args.json_path, args.context_json, args.answer_json) if path is not None]
    if args.output_dir:
        outputs.extend([args.output_dir / 'context.txt', args.output_dir / 'retrieval.json'])
        if args.answer:
            outputs.extend([args.output_dir / 'answer.txt', args.output_dir / 'answer.json'])
    try:
        output_paths = [str(path.resolve()) for path in outputs]
    except (OSError, ValueError, RuntimeError) as error:
        arguments.exit(1, f'Could not export graph/context: {error}\n')
    if set(output_paths) & seen_paths:
        arguments.exit(1, 'JSON output must not overwrite an input source file\n')
    if args.mock_response and str(args.mock_response.resolve()) in output_paths:
        arguments.exit(1, 'Answer output must not overwrite mock-response input\n')
    if len(set(output_paths)) != len(output_paths):
        arguments.error('Graph, context and metadata outputs must use distinct paths')
    graph = None if state is not None else build_graph(results)
    graph_indexing_ms = (perf_counter() - indexing_started) * 1000 if state is None else 0
    index_metadata = {key: value for key, value in indexing.items() if key not in ('files', 'source_paths')}
    if graph is not None:
        graph.graph['indexing'] = index_metadata
    verbose = args.verbose or indexing['root'] is None
    if verbose and graph is not None:
        print_graph(graph)
    if graph is not None:
        print_summary(graph, indexing, show_file_problems=not verbose)
    else:
        print(f"Traditional index: {len(results)} files, {len(state['chunks'])} chunks; no Tree-sitter parsing")
    answer_result = None
    if args.query is not None:
        from retrieval import retrieve_context
        try:
            if args.strategy or args.expansion or args.optimization != 'none':
                from strategies import prepare_search, retrieve_strategy
                strategy = args.strategy or 'graph'
                if state is None:
                    state = prepare_search(files, [strategy], graph=graph, **setup_options)
                    state['timings']['graph_indexing_ms'] = graph_indexing_ms
                    state['timings']['indexing_ms'] += graph_indexing_ms
                result = retrieve_strategy(state, strategy, args.query, args.max_tokens, args.top_k, args.depth,
                                           expansion='never' if args.no_expand else (args.expansion or 'auto'),
                                           include_callers=args.include_callers, max_expanded=args.max_expanded,
                                           include_classes=args.include_classes, lexical_candidates=args.lexical_candidates,
                                           semantic_candidates=args.semantic_candidates, rrf_k=args.rrf_k,
                                           min_similarity=args.min_similarity, optimization=args.optimization)
                result['setup'] = state['timings']
                result['model'] = state.get('model')
            else:
                result = retrieve_context(graph, args.query, args.max_tokens, args.top_k, args.depth,
                                          not args.no_expand, args.include_callers, args.max_expanded,
                                          args.include_classes)
            result['indexing'] = index_metadata
            result['files'] = results
            print('\n' + result['context'])
            print(f"Context: {result['token_count']}/{args.max_tokens} tokens; "
                  f"{result['direct_count']} direct, {result['expanded_count']} expanded snippets")
            for diagnostic in result['diagnostics']:
                print(f"Retrieval ({diagnostic['kind']}): {diagnostic['message']}")
            if args.answer:
                from answering import answer_from_retrieval
                answer_result = answer_from_retrieval(result, provider)
                answer_result['setup'] = result.get('setup', {})
                print('\nAnswer' + (' [MOCK: no model inference]' if answer_result['mocked'] else '') + ':\n' + answer_result['answer'])
                for diagnostic in answer_result['diagnostics']:
                    print(f"Answer ({diagnostic['kind']}): {diagnostic['message']}")
                print('Usage: ' + json.dumps(answer_result['tokens']) + '\nCost: ' + json.dumps(answer_result['cost']))
                if args.answer_json:
                    args.answer_json.write_text(json.dumps(answer_result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if args.context_json:
                args.context_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / 'context.txt').write_text(result['context'], encoding='utf-8')
                (args.output_dir / 'retrieval.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                if answer_result is not None:
                    (args.output_dir / 'answer.txt').write_text(answer_result['answer'] + '\n', encoding='utf-8')
                    (args.output_dir / 'answer.json').write_text(json.dumps(answer_result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        except (OSError, ValueError, RuntimeError) as error:
            arguments.exit(1, f'Could not retrieve/export context: {error}\n')
    if args.json_path:
        try:
            if str(args.json_path.resolve()) in seen_paths:
                arguments.exit(1, 'JSON output must not overwrite an input source file\n')
            args.json_path.write_text(json.dumps(graph_to_dict(graph), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        except (OSError, ValueError, RuntimeError) as error:
            arguments.exit(1, f'Could not export graph: {error}\n')
        print(f'\nJSON exported to {args.json_path}')
    semantic_failed = args.query is not None and args.strategy == 'traditional' and result.get('semantic_status') == 'unavailable'
    answer_failed = answer_result is not None and answer_result['status'] in ('provider_error', 'invalid_response', 'provider_refusal', 'truncated_response')
    return 1 if answer_failed or semantic_failed or indexing['diagnostics'] or any(item['has_error'] for item in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Shared setup and five retrieval strategies, without changing Phase 4 APIs."""
from pathlib import Path
import re
from time import perf_counter

from chunking import index_chunks, verify_chunk
from embeddings import DEFAULT_CACHE, DEFAULT_MODEL, DEFAULT_MODEL_CACHE, LocalEncoder, embed_texts, cosine_search
from graph_builder import build_graph
from hybrid import entity_documents, reciprocal_rank_fusion, semantic_rank, should_expand
from indexer import index_inputs
from retrieval import assemble_context, rank_entities, retrieve_context
from token_counter import ENCODING_NAME, count_tokens

STRATEGIES = ('traditional', 'structure', 'graph', 'hybrid', 'hybrid_graph')


def prepare_search(inputs, strategies=STRATEGIES, graph=None, encoder=None, model=DEFAULT_MODEL,
                   model_cache=DEFAULT_MODEL_CACHE, cache_dir=DEFAULT_CACHE, download_model=False,
                   revision=None, chunk_size=128, overlap=24):
    started = perf_counter()
    requested = set(strategies)
    if not requested or requested - set(STRATEGIES):
        raise ValueError('Unknown or empty strategy list')
    state = {'graph': graph, 'chunks': [], 'documents': [], 'diagnostics': [],
             'timings': {}, 'encoder': encoder, 'chunk_matrix': None, 'entity_matrix': None}
    state['prepared_strategies'] = sorted(requested)
    if requested - {'traditional'} and graph is None:
        graph_started = perf_counter()
        indexed = index_inputs(inputs)
        graph = build_graph(indexed['files'])
        graph.graph['indexing'] = {key: value for key, value in indexed.items() if key not in ('files', 'source_paths')}
        state['graph'] = graph
        state['timings']['graph_indexing_ms'] = (perf_counter() - graph_started) * 1000
    if graph is not None:
        state['indexing'] = graph.graph.get('indexing', {})
        state['files'] = graph.graph['files']
    if 'traditional' in requested:
        chunks_started = perf_counter()
        chunks = index_chunks(inputs, chunk_size, overlap)
        state['chunks'] = chunks.pop('chunks')
        state['chunk_indexing'] = chunks
        if graph is None:
            state['indexing'], state['files'] = chunks, chunks['files']
        state['timings']['chunk_indexing_ms'] = (perf_counter() - chunks_started) * 1000
    if requested & {'hybrid', 'hybrid_graph'}:
        state['documents'], errors = entity_documents(state['graph'])
        state['diagnostics'].extend(errors)
    state['timings']['indexing_ms'] = (perf_counter() - started) * 1000
    if requested & {'traditional', 'hybrid', 'hybrid_graph'}:
        embedding_started = perf_counter()
        try:
            if encoder is None:
                encoder = LocalEncoder(model, model_cache, download_model, revision)
            state['encoder'] = encoder
            state['model'] = {'name': getattr(encoder, 'model_name', model), 'identity': encoder.cache_identity,
                              'revision': getattr(encoder, 'resolved_revision', None),
                              'dimension': encoder.dimension, 'pooling': 'mean token windows; L2 normalized'}
            state['timings']['model_load_ms'] = getattr(encoder, 'load_ms', 0)
            for documents, matrix_key, cache_key in [
                (state['chunks'], 'chunk_matrix', 'chunks'), (state['documents'], 'entity_matrix', 'entities')]:
                if not documents:
                    continue
                texts = [document['text'] if 'text' in document else f"{document['file_path']}\n{document['source']}"
                         for document in documents]
                matrix, statistics = embed_texts(texts, encoder, cache_dir)
                state[matrix_key] = matrix
                state['timings'][cache_key] = statistics
            encoder.encode(['warmup'])
        except (OSError, ValueError, RuntimeError) as error:
            state['diagnostics'].append({'kind': 'semantic_unavailable', 'message': str(error)})
        state['timings']['embedding_setup_ms'] = (perf_counter() - embedding_started) * 1000
    return state


def retrieve_traditional(state, query, max_tokens=2000, top_k=3, min_similarity=0.2):
    started = perf_counter()
    if not query.strip() or max_tokens < 0 or top_k < 1:
        raise ValueError('Nonempty query, nonnegative budget and positive top_k required')
    snippets, diagnostics, candidates = [], list(state['diagnostics']), []
    if state['chunk_matrix'] is not None and state['encoder'] is not None:
        try:
            vector = state['encoder'].encode([query])[0]
            candidates = cosine_search(state['chunk_matrix'], vector, [c['id'] for c in state['chunks']], top_k, min_similarity)
        except (ValueError, RuntimeError) as error:
            diagnostics.append({'kind': 'semantic_unavailable', 'message': str(error)})
    by_id = {chunk['id']: chunk for chunk in state['chunks']}
    root = state.get('chunk_indexing', {}).get('root')
    header_fits = count_tokens(assemble_context(query, [])) <= max_tokens
    cache = {}
    if not header_fits:
        diagnostics.append({'kind': 'question_over_budget', 'message': 'Question/header exceeds budget'})
    elif not candidates:
        diagnostics.append({'kind': 'no_matches', 'message': 'No chunks meet semantic similarity threshold or model unavailable'})
    for candidate in candidates if header_fits else []:
        try:
            snippet = verify_chunk(by_id[candidate['id']], root, cache)
            snippet.update(score=candidate['score'], reason='Dense chunk similarity', distance=0, via=None)
        except (OSError, ValueError) as error:
            diagnostics.append({'id': candidate['id'], 'kind': 'source_unavailable', 'message': str(error)})
            continue
        if count_tokens(assemble_context(query, snippets + [snippet])) > max_tokens:
            diagnostics.append({'id': candidate['id'], 'kind': 'budget_exhausted', 'message': 'Complete chunk omitted; no truncation'})
            continue
        snippets.append(snippet)
    context = assemble_context(query, snippets) if header_fits else ''
    return {'question': query, 'context': context, 'snippets': snippets, 'candidates': candidates,
            'diagnostics': diagnostics, 'token_count': count_tokens(context), 'tokenizer': ENCODING_NAME,
            'max_tokens': max_tokens, 'direct_count': len(snippets), 'expanded_count': 0,
            'latency_ms': (perf_counter() - started) * 1000}


def retrieve_strategy(state, strategy, query, max_tokens=2000, top_k=3, depth=1,
                      expansion='auto', include_callers=False, max_expanded=10, include_classes=False,
                      lexical_candidates=20, semantic_candidates=20, rrf_k=60, min_similarity=0.2,
                      optimization='none'):
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown retrieval strategy: {strategy}')
    if strategy not in state['prepared_strategies']:
        raise ValueError('Strategy was not prepared; call prepare_search with this strategy')
    if not query.strip() or min(top_k, lexical_candidates, semantic_candidates, rrf_k) < 1:
        raise ValueError('Query must be nonempty and candidate counts/rrf_k positive')
    if min(max_tokens, depth, max_expanded) < 0 or not -1 <= min_similarity <= 1:
        raise ValueError('Budgets/depth/counts must be nonnegative; similarity must be in [-1, 1]')
    if optimization not in ('none', 'relevance'):
        raise ValueError('Optimization must be none or relevance')
    started = perf_counter()
    if strategy == 'traditional':
        result = retrieve_traditional(state, query, max_tokens, top_k, min_similarity)
    else:
        graph = state['graph']
        # Explicit class/responsibility requests may include a complete class;
        # ordinary method queries retain the Phase 4 whole-method policy.
        include_classes = include_classes or bool(re.search(
            r'\b(class|classes|responsib\w*|owns?|maintain\w*|keeps?)\b', query.lower()))
        class_names = {name.lower() for _, entity in graph.nodes(data=True) if entity['kind'] == 'class'
                       for name in (entity['name'], entity['qualified_name'])}
        include_classes = include_classes or query.strip().strip('`').lower() in class_names
        diagnostics = list(state['diagnostics'])
        ranking = None
        if strategy in ('hybrid', 'hybrid_graph'):
            if lexical_candidates < 1 or semantic_candidates < 1:
                raise ValueError('Candidate counts must be positive')
            lexical = [candidate for candidate in rank_entities(graph, query)
                       if include_classes or candidate['kind'] != 'class'][:lexical_candidates]
            semantic = []
            if state['entity_matrix'] is not None and state['encoder'] is not None:
                try:
                    vector = state['encoder'].encode([query])[0]
                    semantic = semantic_rank(state['documents'], state['entity_matrix'], vector,
                                             semantic_candidates, min_similarity, include_classes)
                except (ValueError, RuntimeError) as error:
                    diagnostics.append({'kind': 'semantic_unavailable', 'message': str(error)})
            ranking = reciprocal_rank_fusion(lexical, semantic, rrf_k)
        decision, intent = should_expand(query, expansion)
        expand = strategy in ('graph', 'hybrid_graph') and decision
        priorities = None
        if optimization == 'relevance' and strategy in ('graph', 'hybrid_graph'):
            from context_optimizer import optimize_ranking, dependency_priorities
            ranking = optimize_ranking(graph, query, rank_entities(graph, query) if ranking is None else ranking)
            priorities = dependency_priorities(graph, query, include_callers)
        result = retrieve_context(graph, query, max_tokens, top_k, depth, expand, include_callers,
                                  max_expanded, include_classes, ranked_candidates=ranking,
                                  direct_reason='Hybrid rank fusion match' if strategy in ('hybrid', 'hybrid_graph') else 'Direct query match',
                                  dependency_priorities=priorities)
        result['diagnostics'] = diagnostics + result['diagnostics']
        result['intent'] = intent
        result['expansion_policy'] = expansion
        result['configuration'].update(lexical_candidates=lexical_candidates, semantic_candidates=semantic_candidates,
                                       rrf_k=rrf_k, min_similarity=min_similarity, optimization=optimization)
    result['strategy'] = strategy
    unavailable = any(d['kind'] == 'semantic_unavailable' for d in result['diagnostics'])
    result['semantic_status'] = ('unavailable' if strategy == 'traditional' else 'lexical_fallback') if unavailable else (
        'available' if strategy in ('traditional', 'hybrid', 'hybrid_graph') else 'not_required')
    result['latency_ms'] = (perf_counter() - started) * 1000
    return result

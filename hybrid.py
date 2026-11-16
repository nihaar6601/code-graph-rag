"""Semantic entity ranking, reciprocal rank fusion and query-intent rules."""
import re

from embeddings import cosine_search
from retrieval import entity_label, load_snippet


def classify_intent(query):
    text = query.lower()
    if re.search(r'\b(only|just|implementation)\b|\bmethod body\b|\b(no|without|exclude|excluding)\b.*\b(depend\w*|callers|callees|related)\b', text):
        return 'implementation_only'
    if re.search(r'\b(depend\w*|call\w*|invok\w*|chain|downstream|upstream|trace|follow|interact\w*)\b', text):
        return 'dependency'
    if re.search(r'\b(how|why|flow|behavio\w*|workflow|happen\w*|process|processing|responsibilit\w*|explain|together|across|compare)\b', text):
        return 'behavior'
    return 'lookup'


def should_expand(query, policy='auto'):
    if policy not in ('auto', 'always', 'never'):
        raise ValueError('Expansion policy must be auto, always or never')
    intent = classify_intent(query)
    return (policy == 'always' or (policy == 'auto' and intent in ('dependency', 'behavior'))), intent


def entity_documents(graph):
    documents, diagnostics = [], []
    file_cache = {}
    for entity_id in sorted(graph.nodes):
        try:
            snippet = load_snippet(graph, entity_id, file_cache=file_cache)
            entity = graph.nodes[entity_id]
            text = f"{entity_label(graph, entity_id)}\n{entity.get('signature', '')}\n{snippet['source']}"
            documents.append({'id': entity_id, 'kind': entity['kind'], 'name': snippet['name'], 'text': text})
        except (OSError, ValueError, KeyError) as error:
            diagnostics.append({'id': entity_id, 'kind': 'source_unavailable', 'message': str(error)})
    return documents, diagnostics


def semantic_rank(documents, matrix, query_vector, top_k=20, min_similarity=0.2, include_classes=False):
    positions = [i for i, document in enumerate(documents) if include_classes or document['kind'] != 'class']
    records = [documents[i] for i in positions]
    ranked = cosine_search(matrix[positions], query_vector, [document['id'] for document in records],
                           top_k, min_similarity)
    by_id = {document['id']: document for document in records}
    return [dict(candidate, kind=by_id[candidate['id']]['kind'], name=by_id[candidate['id']]['name']) for candidate in ranked]


def reciprocal_rank_fusion(lexical, semantic, rrf_k=60):
    if rrf_k < 1:
        raise ValueError('rrf_k must be positive')
    combined = {}
    for source, candidates in [('lexical', lexical), ('semantic', semantic)]:
        seen = set()
        for rank, candidate in enumerate(candidates, 1):
            entity_id = candidate['id']
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entry = combined.setdefault(entity_id, dict(candidate, score=0.0, fusion={}))
            entry['score'] += 1 / (rrf_k + rank)
            entry['fusion'][source] = {'rank': rank, 'score': candidate['score']}
    return sorted(combined.values(), key=lambda candidate: (-candidate['score'], candidate['id']))

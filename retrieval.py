"""Lexical entity ranking, bounded dependency expansion and safe context assembly."""
from collections import Counter, deque
import hashlib
import math
from pathlib import Path
import re
from time import perf_counter

from token_counter import ENCODING_NAME, count_tokens

STOP_WORDS = set('a an the how what which where why when does do did is are was were to of for and or in on with from by can explain please me tell about work works'.split())


def normalize_word(word):
    word = word.lower()
    if len(word) > 5 and word.endswith('ing'):
        word = word[:-3]
    elif len(word) > 4 and word.endswith('s') and not word.endswith('ss'):
        word = word[:-1]
    return word


def identifier_tokens(text):
    # Split acronyms too: HTTPPaymentService -> HTTP, Payment, Service.
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return [normalize_word(word) for word in re.findall(r'[A-Za-z0-9]+', text)
            if word.lower() not in STOP_WORDS]


def entity_label(graph, entity_id):
    entity = graph.nodes[entity_id]
    if entity['kind'] == 'class':
        return entity['qualified_name']
    owner = graph.nodes[entity['class_id']]
    return f"{owner['qualified_name']}.{entity['signature']}"


def entity_fields(graph, entity_id):
    entity = graph.nodes[entity_id]
    owner = entity if entity['kind'] == 'class' else graph.nodes[entity['class_id']]
    # Do not index entire class bodies: otherwise every contained method matches.
    return [
        (entity.get('name', ''), 8.0),
        (entity.get('signature', ''), 3.0),
        (owner.get('qualified_name', ''), 1.5),
        (' '.join([entity.get('file_path', ''), entity.get('return_type') or '',
                   ' '.join(p.get('type') or '' for p in entity.get('parameters', []))]), 0.5),
    ]


def ranking_key(candidate):
    return (-candidate['score'], candidate['id'])


def rank_entities(graph, query):
    """Return positive lexical matches, including classes, with stable tie breaks."""
    query_terms = set(identifier_tokens(query))
    documents = {}
    frequencies = Counter()
    for entity_id in sorted(graph.nodes):
        entity = graph.nodes[entity_id]
        if entity.get('has_error') or entity.get('kind') not in ('class', 'method', 'constructor'):
            continue
        fields = [(set(identifier_tokens(text)), weight) for text, weight in entity_fields(graph, entity_id)]
        documents[entity_id] = fields
        frequencies.update(set().union(*(tokens for tokens, _ in fields)))
    ranked = []
    count = len(documents)
    compact_query = re.sub(r'[^a-z0-9]', '', query.lower())
    for entity_id, fields in documents.items():
        score = 0.0
        matched = set()
        for tokens, weight in fields:
            for term in query_terms & tokens:
                score += weight * (1 + math.log((count + 1) / (frequencies[term] + 1)))
                matched.add(term)
        name = re.sub(r'[^a-z0-9]', '', graph.nodes[entity_id].get('name', '').lower())
        if name and compact_query == name:
            score += 30
            matched.add(name)
        elif len(compact_query) >= 4 and compact_query in name:
            score += 8
            matched.add(compact_query)
        if score:
            ranked.append({'id': entity_id, 'name': entity_label(graph, entity_id),
                           'kind': graph.nodes[entity_id]['kind'], 'score': round(score, 6),
                           'matched_terms': sorted(matched)})
    return sorted(ranked, key=ranking_key)


def expand_dependencies(graph, seed_ids, depth=1, include_callers=False, max_expanded=10):
    """Breadth-first CALLS traversal, bounded by depth and entity count."""
    if depth < 0 or max_expanded < 0:
        raise ValueError('depth and max_expanded must be nonnegative')
    seen = set(seed_ids)
    queue = deque((entity_id, 0) for entity_id in seed_ids)
    expanded = []
    while queue and len(expanded) < max_expanded:
        current, distance = queue.popleft()
        if distance >= depth:
            continue
        neighbors = {(target, 'Called by') for _, target, edge in graph.out_edges(current, data=True)
                     if edge['kind'] == 'CALLS'}
        if include_callers:
            neighbors.update((source, 'Calls') for source, _, edge in graph.in_edges(current, data=True)
                             if edge['kind'] == 'CALLS')
        for neighbor, relation in sorted(neighbors):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            expanded.append({'id': neighbor, 'distance': distance + 1,
                             'via': current, 'reason': f'{relation} {entity_label(graph, current)}'})
            queue.append((neighbor, distance + 1))
            if len(expanded) == max_expanded:
                break
    return expanded


def class_context(graph, entity):
    current = entity['id']
    owners, seen = [], set()
    while current and current not in seen:
        seen.add(current)
        owner = graph.nodes[current]
        if owner['kind'] == 'class':
            owners.append(owner['qualified_name'])
        parents = sorted({source for source, _, edge in graph.in_edges(current, data=True)
                          if edge['kind'] == 'CONTAINS' and graph.nodes[source]['kind'] == 'class'})
        current = parents[0] if len(parents) == 1 else None
    return list(reversed(owners))


def load_snippet(graph, entity_id, root=None, file_cache=None):
    """Verify the indexed snapshot before using exact source byte locations."""
    entity = graph.nodes[entity_id]
    file_path = entity['file_path']
    file_info = next((file for file in graph.graph['files'] if file['file_path'] == file_path), {})
    if file_info.get('has_error'):
        raise ValueError('Source has indexing diagnostics; repair and re-index it')
    expected_hash = file_info.get('source_sha256')
    if not expected_hash:
        raise ValueError('Indexed source fingerprint is missing; re-index before retrieval')
    base = root or graph.graph.get('indexing', {}).get('root')
    path = Path(file_path)
    if not path.is_absolute():
        if not base:
            raise ValueError('A repository root is required for relative source paths')
        base = Path(base).resolve()
        path = (base / path).resolve()
        if not path.is_relative_to(base):
            raise ValueError('Source path escapes the repository root')
    cache = file_cache if file_cache is not None else {}
    if str(path) not in cache:
        cache[str(path)] = path.read_bytes()
    source = cache[str(path)]
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError('Source changed after indexing; re-index before retrieval')
    start, end = entity['start_byte'], entity['end_byte']
    if not 0 <= start < end <= len(source):
        raise ValueError('Indexed source byte range is invalid')
    snippet = source[start:end].decode('utf-8')
    if snippet != entity['source']:
        raise ValueError('Indexed source range does not match the entity')
    for prefix, offset in (('start', start), ('end', end)):
        before = source[:offset]
        line = before.count(b'\n') + 1
        column = len(before.rsplit(b'\n', 1)[-1]) + 1
        if (line, column) != (entity[f'{prefix}_line'], entity[f'{prefix}_column']):
            raise ValueError('Indexed source lines/columns do not match the entity')
    return {'id': entity_id, 'name': entity_label(graph, entity_id), 'kind': entity['kind'],
            'file_path': file_path, 'start_line': entity['start_line'], 'end_line': entity['end_line'],
            'start_byte': start, 'end_byte': end, 'source': snippet,
            'class_context': class_context(graph, entity)}


def assemble_context(query, snippets):
    context = f'Question: {query}\n\nRelevant Source Code:\n'
    for number, snippet in enumerate(snippets, 1):
        context += (f"\n[{number}] {snippet['name']}\nID: {snippet['id']}\n"
                    f"File: {snippet['file_path']}\nLines: {snippet['start_line']}-{snippet['end_line']}\n"
                    f"Class context: {' > '.join(snippet['class_context'])}\n"
                    f"Reason: {snippet['reason']}\nScore: {snippet['score']:.6f}\n\n"
                    f"{snippet['source']}\n")
    return context


def overlaps(snippet, selected):
    return any(snippet['file_path'] == other['file_path'] and
               snippet['start_byte'] < other['end_byte'] and other['start_byte'] < snippet['end_byte']
               for other in selected)


def retrieve_context(graph, query, max_tokens=2000, top_k=3, depth=1, expand=True,
                     include_callers=False, max_expanded=10, include_classes=False, root=None,
                     ranked_candidates=None, direct_reason='Direct query match',
                     dependency_priorities=None):
    """Return budgeted context, snippets, ranking and explicit omission diagnostics."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    if max_tokens < 0 or top_k < 1 or depth < 0 or max_expanded < 0:
        raise ValueError('Invalid retrieval limits: top_k must be positive and other limits nonnegative')
    started = perf_counter()
    ranked = rank_entities(graph, query) if ranked_candidates is None else ranked_candidates
    candidates = [candidate for candidate in ranked if include_classes or candidate['kind'] != 'class'][:top_k]
    snippets, diagnostics = [], []
    file_cache = {}
    header = assemble_context(query, [])
    header_fits = count_tokens(header) <= max_tokens
    if not header_fits:
        diagnostics.append({'kind': 'question_over_budget', 'message': 'Question and formatting exceed the token budget'})
    elif not candidates:
        diagnostics.append({'kind': 'no_matches', 'message': 'No candidates meet the retrieval thresholds'})
    direct = [dict(candidate, reason=direct_reason, distance=0) for candidate in candidates] if header_fits else []
    pending = list(direct)
    expanded = False
    while pending or (expand and not expanded and header_fits):
        if not pending:
            seed_ids = [snippet['id'] for snippet in snippets if snippet['distance'] == 0]
            pool_size = max_expanded * 4 if dependency_priorities is not None else max_expanded
            dependencies = expand_dependencies(graph, seed_ids, depth, include_callers, pool_size)
            if dependency_priorities is not None:
                dependencies = [item for item in dependencies if dependency_priorities.get(item['id'], 0) > 0]
                dependencies.sort(key=lambda item: (-dependency_priorities[item['id']], item['distance'], item['id']))
                dependencies = dependencies[:max_expanded]
            scores = {candidate['id']: candidate['score'] for candidate in direct}
            seed_score = max(scores.values(), default=0)
            pending = []
            for item in dependencies:
                score = scores[item['via']] / 2 if item['via'] in scores else seed_score / 2 ** item['distance']
                scores[item['id']] = round(score, 6)
                if item['id'] not in {candidate['id'] for candidate in direct}:
                    pending.append(dict(item, score=scores[item['id']]))
            expanded = True
            if not pending:
                break
        candidate = pending.pop(0)
        entity_id = candidate['id']
        try:
            snippet = load_snippet(graph, entity_id, root, file_cache)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            diagnostics.append({'id': entity_id, 'kind': 'source_unavailable', 'message': str(error)})
            continue
        snippet.update(score=candidate['score'], reason=candidate['reason'], distance=candidate['distance'],
                       via=candidate.get('via'))
        if 'fusion' in candidate:
            snippet['fusion'] = candidate['fusion']
        if overlaps(snippet, snippets):
            diagnostics.append({'id': entity_id, 'kind': 'overlap', 'message': 'Overlaps an already selected snippet'})
            continue
        total_tokens = count_tokens(assemble_context(query, snippets + [snippet]))
        if total_tokens > max_tokens:
            standalone = count_tokens(assemble_context(query, [snippet]))
            oversized = 'oversized_class' if snippet['kind'] == 'class' else 'oversized_method'
            diagnostics.append({'id': entity_id, 'kind': oversized if standalone > max_tokens else 'budget_exhausted',
                                'message': 'Complete snippet omitted; source was not truncated',
                                'required_tokens_alone': standalone, 'attempted_context_tokens': total_tokens})
            continue
        snippets.append(snippet)
    context = assemble_context(query, snippets) if header_fits else ''
    return {
        'question': query, 'context': context, 'snippets': snippets, 'candidates': candidates,
        'diagnostics': diagnostics, 'token_count': count_tokens(context), 'tokenizer': ENCODING_NAME,
        'max_tokens': max_tokens, 'direct_count': sum(s['distance'] == 0 for s in snippets),
        'expanded_count': sum(s['distance'] > 0 for s in snippets),
        'latency_ms': round((perf_counter() - started) * 1000, 3),
        'configuration': {'top_k': top_k, 'depth': depth, 'expand': expand,
                          'include_callers': include_callers, 'max_expanded': max_expanded,
                          'include_classes': include_classes},
    }

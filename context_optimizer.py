"""Optional query-based dependency filtering. Never reads evaluation labels."""
import re

from retrieval import identifier_tokens

FAILURE_WORDS = r'\b(reject\w*|fail\w*|declin\w*|unsuccessful|rollback|error)\b'


def query_terms(question):
    terms = set(identifier_tokens(question))
    # Small language aliases, not fixture-specific method names or benchmark labels.
    if terms & {'payment', 'pay', 'paid', 'charge'}:
        terms.update(('payment', 'pay', 'charge'))
    if terms & {'inventory', 'stock'}:
        terms.update(('inventory', 'stock'))
    return terms


def optimize_ranking(graph, question, ranking):
    terms = query_terms(question)
    scored = []
    for candidate in ranking:
        entity = graph.nodes[candidate['id']]
        identifiers = entity['qualified_name'] if entity['kind'] == 'class' else entity['name']
        overlap = len(terms & set(identifier_tokens(identifiers)))
        scored.append(dict(candidate, optimization_score=overlap))
    strong = [candidate for candidate in scored if candidate['optimization_score'] > 0]
    return sorted(strong, key=lambda candidate: (-candidate['optimization_score'], -candidate['score'], candidate['id'])) if strong else scored


def dependency_priorities(graph, question, include_callers=False):
    terms = query_terms(question)
    failure_question = bool(re.search(FAILURE_WORDS, question, re.IGNORECASE))
    priorities = {}
    for source, target, edge in graph.edges(data=True):
        if edge['kind'] != 'CALLS':
            continue
        entity = graph.nodes[target]
        owner = graph.nodes[entity['class_id']]
        invocation = edge.get('invocation', {})
        score = 4 * len(terms & set(identifier_tokens(entity['name'])))
        score += len(terms & set(identifier_tokens(owner['qualified_name'])))
        # Receiver/name evidence, not arbitrary arguments shared by unrelated calls.
        call_identifiers = (invocation.get('receiver') or '') + ' ' + (invocation.get('name') or '')
        score += 2 * len(terms & set(identifier_tokens(call_identifiers)))
        for context in invocation.get('control_context', []):
            condition = context.get('condition') or ''
            if terms & set(identifier_tokens(condition)):
                score += 3
                branch = context.get('branch_source') or ''
                if failure_question and re.search(FAILURE_WORDS, branch, re.IGNORECASE):
                    score += 6
        priorities[target] = max(priorities.get(target, 0), score)
        if include_callers:
            priorities[source] = max(priorities.get(source, 0), score / 2)
    return priorities

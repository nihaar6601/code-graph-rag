"""Deterministic factual-point/citation proxies, not a semantic LLM judge."""
import re

from relevance import intersection_size, merge_ranges, range_size
from retrieval import entity_label


def resolve_fact_sources(graph, labels):
    by_label = {}
    for node_id, node in graph.nodes(data=True):
        by_label.setdefault(entity_label(graph, node_id), []).append(node)
    resolved = []
    for label in labels:
        matches = by_label.get(label, [])
        if len(matches) != 1:
            raise ValueError(f'Answer benchmark source is missing or ambiguous: {label}')
        resolved.append(matches[0])
    return resolved


def validate_answer_benchmark(graph, benchmark):
    seen = set()
    for item in benchmark:
        if not isinstance(item.get('question'), str) or not item['question'].strip() or not isinstance(item.get('expect_insufficient'), bool):
            raise ValueError('Each benchmark item needs question and expect_insufficient')
        if not isinstance(item.get('reference_answer'), str) or not isinstance(item.get('facts'), list):
            raise ValueError('Each answer benchmark item needs reference_answer and facts')
        resolve_fact_sources(graph, item.get('expected', []))
        for fact in item['facts']:
            if fact['id'] in seen or not fact['patterns'] or not fact['sources']:
                raise ValueError('Fact IDs must be unique with nonempty patterns and sources')
            seen.add(fact['id'])
            for pattern in fact['patterns']:
                re.compile(pattern, re.IGNORECASE)
            resolve_fact_sources(graph, fact['sources'])
        for forbidden in item.get('forbidden_patterns', []):
            re.compile(forbidden['pattern'], re.IGNORECASE)


def citation_supports(claim, required, source_registry):
    """Check range attribution, never claim semantic entailment from overlap alone."""
    for record in required:
        supported = False
        for citation in claim['citations']:
            source = source_registry[citation['source_id']]
            if (source['file_path'] == record['file_path'] and
                    source['start_byte'] < record['end_byte'] and record['start_byte'] < source['end_byte'] and
                    citation['start_line'] <= record['end_line'] and record['start_line'] <= citation['end_line']):
                supported = True
        if not supported:
            return False
    return bool(required)


def grade_answer(answer, item, graph, evidence_threshold=0.8):
    claims = answer.get('claims', [])
    rendered = [claim for claim in claims if claim['attribution_valid']]
    registry = {source['source_id']: source for source in answer.get('sources', [])}
    retrieved = merge_ranges(answer.get('sources', []))
    fact_results = []
    matched_claims = set()
    for fact in item['facts']:
        required = resolve_fact_sources(graph, fact['sources'])
        available = all(intersection_size(retrieved, merge_ranges([record])) / range_size(merge_ranges([record])) >= evidence_threshold
                        for record in required)
        matched = [index for index, claim in enumerate(rendered)
                   if any(re.search(pattern, claim['text'], re.IGNORECASE) for pattern in fact['patterns'])]
        matched_claims.update(matched)
        grounded = any(citation_supports(rendered[index], required, registry) for index in matched)
        fact_results.append({'id': fact['id'], 'description': fact['description'],
                             'lexically_covered': bool(matched), 'citation_backed': grounded,
                             'evidence_available': available})
    false_indices = {index for index, claim in enumerate(rendered) if any(
        re.search(rule['pattern'], claim['text'], re.IGNORECASE) for rule in item.get('forbidden_patterns', []))}
    rejected = sum(not claim['attribution_valid'] for claim in claims)
    context_missing = item['expect_insufficient'] or any(not fact['evidence_available'] for fact in fact_results)
    successful = answer['status'] in ('ok', 'partial', 'insufficient_context')
    acknowledgment = successful and answer.get('insufficient_context') == context_missing
    facts = len(fact_results)
    if facts:
        completeness = sum(fact['lexically_covered'] for fact in fact_results) / facts
        grounded_completeness = sum(fact['citation_backed'] for fact in fact_results) / facts
        correctness = len(matched_claims - false_indices) / len(rendered) if rendered else 0.0
    else:
        completeness = grounded_completeness = float(successful and answer.get('insufficient_context') is True and not rendered)
        correctness = completeness
    available_facts = [fact for fact in fact_results if fact['evidence_available']]
    conditional = sum(fact['citation_backed'] for fact in available_facts) / len(available_facts) if available_facts else None
    citations = sum(len(claim['citations']) + len(claim['invalid_citations']) for claim in claims)
    valid_citations = sum(len(claim['citations']) for claim in claims)
    return {'correctness_proxy': correctness, 'completeness': completeness,
            'grounded_completeness': grounded_completeness, 'available_fact_completeness': conditional,
            'citation_validity': valid_citations / citations if citations else (1.0 if not claims else 0.0),
            'known_false_claim_count': len(false_indices), 'unsupported_claim_proxy_count': rejected + len(false_indices),
            'unverified_claim_count': len(rendered) - len(matched_claims), 'rejected_claim_count': rejected,
            'appropriate_insufficiency': float(acknowledgment), 'expected_insufficient_given_context': context_missing,
            'fact_results': fact_results,
            'limitations': 'Regex coverage and citation-range checks cannot establish semantic correctness or detect all hallucinations; unmatched claims require review.'}

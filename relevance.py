"""Common source-byte relevance: chunks are never counted as whole methods."""
from retrieval import entity_label


def merge_ranges(records):
    by_file = {}
    for record in records:
        start, end = record['start_byte'], record['end_byte']
        if end <= start:
            raise ValueError('Source relevance ranges must be nonempty')
        by_file.setdefault(record['file_path'], []).append((start, end))
    merged = {}
    for path, ranges in by_file.items():
        intervals = []
        for start, end in sorted(ranges):
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        merged[path] = intervals
    return merged


def range_size(intervals):
    return sum(end - start for ranges in intervals.values() for start, end in ranges)


def intersection_size(first, second):
    return sum(max(0, min(end, other_end) - max(start, other_start))
               for path, ranges in first.items() for start, end in ranges
               for other_start, other_end in second.get(path, []))


def expected_records(graph, item):
    labels = {}
    for entity_id, entity in graph.nodes(data=True):
        labels.setdefault(entity_label(graph, entity_id), []).append(entity)
    records = []
    for label in item.get('expected', []):
        matches = labels.get(label, [])
        if len(matches) != 1:
            raise ValueError(f'Expected entity missing or ambiguous: {label}')
        records.append(dict(matches[0], label=label))
    for record in item.get('expected_ranges', []):
        records.append(dict(record, label=record.get('label', record['file_path'])))
    return records


def source_metrics(snippets, expected, coverage_threshold=0.8):
    retrieved_ranges, expected_ranges = merge_ranges(snippets), merge_ranges(expected)
    retrieved_bytes, expected_bytes = range_size(retrieved_ranges), range_size(expected_ranges)
    relevant_bytes = intersection_size(retrieved_ranges, expected_ranges)
    precision = relevant_bytes / retrieved_bytes if retrieved_bytes else float(expected_bytes == 0)
    recall = relevant_bytes / expected_bytes if expected_bytes else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    coverage = []
    for record in expected:
        relevant = intersection_size(retrieved_ranges, merge_ranges([record]))
        fraction = relevant / (record['end_byte'] - record['start_byte'])
        coverage.append({'label': record['label'], 'coverage': fraction, 'covered': fraction >= coverage_threshold})
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'retrieved_source_bytes': retrieved_bytes, 'expected_source_bytes': expected_bytes,
            'relevant_source_bytes': relevant_bytes, 'entity_coverage': coverage,
            'relevant_entities': sum(record['covered'] for record in coverage),
            'relevant_snippets': sum(intersection_size(merge_ranges([snippet]), expected_ranges) > 0 for snippet in snippets)}

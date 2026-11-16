"""Build a structural graph from parsed files and declared-type resolution."""
import copy

import networkx as nx
from resolver import create_symbol_index, resolve_call, resolve_invocation


def build_graph(parsed_files):
    """A multigraph preserves separate invocation locations for repeated calls."""
    if isinstance(parsed_files, dict):
        parsed_files = [parsed_files]
    # Repeated inputs do not multiply containment or invocation edges.
    parsed_files = sorted({result['file_path']: result for result in parsed_files}.values(),
                          key=lambda result: result['file_path'])
    index = create_symbol_index(parsed_files)
    graph = nx.MultiDiGraph()
    graph.graph['files'] = []
    graph.graph['invocations'] = []
    for result in parsed_files:
        graph.graph['files'].append({key: copy.deepcopy(result[key]) for key in
                                    ('file_path', 'has_error', 'diagnostics', 'package', 'imports', 'source_sha256') if key in result})
        for cls in result['classes']:
            class_data = {key: copy.deepcopy(value) for key, value in cls.items() if key != 'methods'}
            graph.add_node(cls['id'], **class_data)
            if cls['parent_id']:
                graph.add_edge(cls['parent_id'], cls['id'], kind='CONTAINS', file_path=result['file_path'], start_line=cls['start_line'])
            for method in cls['methods']:
                method_data = {key: copy.deepcopy(value) for key, value in method.items() if key != 'calls'}
                method_data['file_path'] = result['file_path']
                graph.add_node(method['id'], **method_data)
                graph.add_edge(cls['id'], method['id'], kind='CONTAINS', file_path=result['file_path'], start_line=method['start_line'])
    for result in parsed_files:
        for cls in result['classes']:
            for method in cls['methods']:
                for original_call in method['calls']:
                    call = copy.deepcopy(original_call)
                    if result['has_error']:
                        target, reason = None, 'Source has parsing diagnostics'
                    else:
                        target, reason = resolve_invocation(call, method, cls, index)
                    call.update(resolved_target=target, unresolved_reason=reason, file_path=result['file_path'])
                    graph.graph['invocations'].append(call)
                    if target:
                        graph.add_edge(method['id'], target, kind='CALLS', invocation=call)
    return graph


def graph_to_dict(graph):
    data = {
        'files': copy.deepcopy(graph.graph['files']),
        'entities': [copy.deepcopy(data) for _, data in graph.nodes(data=True)],
        'relationships': [dict(source=source, target=target, **copy.deepcopy(data)) for source, target, data in graph.edges(data=True)],
        'invocations': copy.deepcopy(graph.graph['invocations']),
    }
    if 'indexing' in graph.graph:
        data['indexing'] = copy.deepcopy(graph.graph['indexing'])
    return data

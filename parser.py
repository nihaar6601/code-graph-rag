"""Extract Java structure and syntactic calls using Tree-sitter."""
from pathlib import Path
import hashlib

from tree_sitter import Language, Parser
import tree_sitter_java


def get_node_text(node, source_code):
    """Missing fields can occur while Tree-sitter recovers incomplete code."""
    if node is None:
        return None
    return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def location(node):
    """Lines and byte columns are one-based; the end position is exclusive."""
    return {
        'start_line': node.start_point.row + 1,
        'end_line': node.end_point.row + 1,
        'start_column': node.start_point.column + 1,
        'end_column': node.end_point.column + 1,
        'start_byte': node.start_byte,
        'end_byte': node.end_byte,
    }


def extract_modifiers(node, source_code):
    modifiers = next((child for child in node.named_children if child.type == 'modifiers'), None)
    if modifiers is None:
        return []
    return [get_node_text(child, source_code) for child in modifiers.children if not child.is_named]


def extract_type_parameters(node, source_code):
    parameters = node.child_by_field_name('type_parameters')
    if parameters is None:
        return []
    names = []
    for parameter in parameters.named_children:
        name = next((child for child in parameter.named_children if child.type == 'type_identifier'), None)
        if name:
            names.append(get_node_text(name, source_code))
    return names


def extract_declarations(node, source_code):
    """Read field/local declarators using their shared declared type."""
    declared_type = get_node_text(node.child_by_field_name('type'), source_code)
    declarations = []
    for child in node.named_children:
        if child.type == 'variable_declarator':
            dimensions = get_node_text(child.child_by_field_name('dimensions'), source_code) or ''
            declarations.append({
                'name': get_node_text(child.child_by_field_name('name'), source_code),
                'type': declared_type + dimensions if declared_type else None,
                'start_byte': child.start_byte,
                'modifiers': extract_modifiers(node, source_code),
            })
    return declarations


def extract_local_variables(node, source_code, scope=None):
    """Keep lexical scopes so a local variable cannot leak into another block."""
    if node is None or node.type in (
        'class_body', 'class_declaration', 'lambda_expression', 'enum_declaration',
        'record_declaration', 'interface_declaration', 'annotation_type_declaration',
    ):
        return []
    if node.type in ('block', 'constructor_body', 'for_statement', 'enhanced_for_statement',
                     'catch_clause', 'try_with_resources_statement', 'switch_block'):
        scope = node
    variables = []
    if node.type in ('local_variable_declaration', 'resource'):
        if node.type == 'resource':
            declarations = [{'name': get_node_text(node.child_by_field_name('name'), source_code),
                             'type': get_node_text(node.child_by_field_name('type'), source_code),
                             'start_byte': node.start_byte}]
        else:
            declarations = extract_declarations(node, source_code)
        for declaration in declarations:
            if scope is not None:
                scope_end = scope.end_byte
                if node.type == 'resource':
                    body = scope.child_by_field_name('body')
                    scope_end = body.end_byte if body else scope_end
                variables.append(dict(declaration, scope_start=scope.start_byte, scope_end=scope_end))
    if node.type == 'instanceof_expression' and node.child_by_field_name('name') and scope:
        # Pattern-variable flow scopes need semantic analysis. Block fallback guesses.
        variables.append({
            'name': get_node_text(node.child_by_field_name('name'), source_code),
            'type': None, 'start_byte': node.start_byte,
            'scope_start': scope.start_byte, 'scope_end': scope.end_byte,
        })
    if node.type in ('enhanced_for_statement', 'catch_clause'):
        parameter = node if node.type == 'enhanced_for_statement' else next(
            (child for child in node.named_children if child.type == 'catch_formal_parameter'), node)
        type_node = parameter.child_by_field_name('type')
        if type_node is None:
            type_node = next((child for child in parameter.named_children if child.type == 'catch_type'), None)
        body = node.child_by_field_name('body')
        if body:
            variables.append({
                'name': get_node_text(parameter.child_by_field_name('name'), source_code),
                'type': get_node_text(type_node, source_code), 'start_byte': parameter.start_byte,
                'scope_start': body.start_byte, 'scope_end': body.end_byte,
            })
    for child in node.named_children:
        variables.extend(extract_local_variables(child, source_code, scope))
    return variables


def extract_file_metadata(root_node, source_code):
    package = ''
    imports = []
    for child in root_node.named_children:
        if child.type not in ('package_declaration', 'import_declaration'):
            continue
        name_node = next((part for part in child.named_children
                          if part.type in ('identifier', 'scoped_identifier')), None)
        name = get_node_text(name_node, source_code)
        if child.type == 'package_declaration':
            package = name or ''
        else:
            imports.append({
                'name': name, 'static': any(part.type == 'static' for part in child.children),
                'wildcard': any(part.type == 'asterisk' for part in child.named_children),
                'source': get_node_text(child, source_code), **location(child),
            })
    return package, imports


def extract_parameters(method_node, source_code):
    parameters = []
    parameter_list = method_node.child_by_field_name('parameters')
    if parameter_list is None:
        return parameters
    for parameter in parameter_list.named_children:
        if parameter.type not in ('formal_parameter', 'spread_parameter', 'receiver_parameter'):
            continue
        name = parameter.child_by_field_name('name')
        type_node = parameter.child_by_field_name('type')
        # The Java grammar represents varargs with a variable_declarator.
        if parameter.type == 'spread_parameter':
            for child in parameter.named_children:
                if child.type == 'variable_declarator':
                    name = child.child_by_field_name('name')
                elif child.type not in ('modifiers', 'annotation', 'marker_annotation', 'line_comment', 'block_comment'):
                    type_node = child
        if parameter.type == 'receiver_parameter':
            # Receiver parameters have no name/type fields in this grammar.
            parts = [child for child in parameter.named_children if child.type not in
                     ('annotation', 'marker_annotation', 'line_comment', 'block_comment')]
            type_node = parts[0] if parts else None
            name = next((child for child in parts if child.type == 'this'), None)
        parameter_type = get_node_text(type_node, source_code)
        dimensions = parameter.child_by_field_name('dimensions')
        if parameter_type and dimensions:
            parameter_type += get_node_text(dimensions, source_code)
        if parameter_type and parameter.type == 'spread_parameter':
            parameter_type += '...'
        parameters.append({
            'name': get_node_text(name, source_code),
            'type': parameter_type,
            'kind': parameter.type,
        })
    return parameters


def call_control_context(node, source_code):
    """Record enclosing if branches for optional, query-aware failure-path ranking."""
    contexts = []
    parent = node.parent
    while parent and parent.type not in ('method_declaration', 'constructor_declaration', 'class_body'):
        if parent.type == 'if_statement':
            condition = parent.child_by_field_name('condition')
            consequence = parent.child_by_field_name('consequence')
            alternative = parent.child_by_field_name('alternative')
            branch = 'condition'
            body = consequence
            if alternative and alternative.start_byte <= node.start_byte < alternative.end_byte:
                branch, body = 'alternative', alternative
            elif consequence and consequence.start_byte <= node.start_byte < consequence.end_byte:
                branch = 'consequence'
            contexts.append({'condition': get_node_text(condition, source_code), 'branch': branch,
                             'branch_source': get_node_text(body, source_code)})
        parent = parent.parent
    return contexts


def extract_calls(node, source_code, caller_id):
    """Walk a method body without attributing nested class/lambda calls to it."""
    calls = []
    if node is None or node.type in (
        'class_body', 'class_declaration', 'lambda_expression', 'enum_declaration',
        'record_declaration', 'interface_declaration', 'annotation_type_declaration',
    ):
        return calls
    if node.type == 'method_invocation':
        arguments = node.child_by_field_name('arguments')
        calls.append({
            'name': get_node_text(node.child_by_field_name('name'), source_code),
            'caller': caller_id,
            'receiver': get_node_text(node.child_by_field_name('object'), source_code),
            'argument_count': sum(child.type not in ('line_comment', 'block_comment')
                                  for child in arguments.named_children) if arguments else None,
            'source': get_node_text(node, source_code),
            'resolved_target': None,
            'control_context': call_control_context(node, source_code),
            **location(node),
        })
    for child in node.named_children:
        calls.extend(extract_calls(child, source_code, caller_id))
    return calls


def extract_methods(class_node, source_code, class_id=''):
    """Only direct members belong to this class, not nested class members."""
    methods = []
    body = class_node.child_by_field_name('body')
    if body is None:
        return methods
    for child in body.named_children:
        if child.type not in ('method_declaration', 'constructor_declaration'):
            continue
        name = get_node_text(child.child_by_field_name('name'), source_code)
        parameters = extract_parameters(child, source_code)
        parameter_types = [p['type'] or '?' for p in parameters if p['kind'] != 'receiver_parameter']
        kind = 'constructor' if child.type == 'constructor_declaration' else 'method'
        signature = f"{name or '?'}({','.join(parameter_types)})"
        method_id = f'{class_id}::{kind}:{signature}'
        # Even invalid duplicate declarations must not overwrite another graph node.
        if any(method['id'] == method_id for method in methods):
            method_id += f'@{child.start_byte}'
        return_type = get_node_text(child.child_by_field_name('type'), source_code)
        dimensions = child.child_by_field_name('dimensions')
        if return_type and dimensions:
            return_type += get_node_text(dimensions, source_code)
        methods.append({
            'id': method_id,
            'class_id': class_id,
            'name': name,
            'kind': kind,
            'signature': signature,
            'return_type': None if kind == 'constructor' else return_type,
            'parameters': parameters,
            'source': get_node_text(child, source_code),
            'has_error': child.has_error,
            'modifiers': extract_modifiers(child, source_code),
            'type_parameters': extract_type_parameters(child, source_code),
            'local_variables': extract_local_variables(child.child_by_field_name('body'), source_code),
            'calls': extract_calls(child.child_by_field_name('body'), source_code, method_id),
            **location(child),
        })
    return methods


def traverse(node, source_code, classes, file_path='', parent=None, scope=None):
    """Discover classes and keep the enclosing named class relationship."""
    if node.type in ('interface_declaration', 'enum_declaration', 'record_declaration', 'annotation_type_declaration'):
        # These owners are not modeled; do not attach their classes to another owner.
        return
    if node.type == 'class_declaration':
        name = get_node_text(node.child_by_field_name('name'), source_code) or '<missing>'
        qualified_name = f'{scope}.{name}' if scope else name
        class_id = f'{file_path}::class:{qualified_name}'
        if any(cls['id'] == class_id for cls in classes):
            class_id += f'@{node.start_byte}'
        class_info = {
            'id': class_id, 'name': name, 'qualified_name': qualified_name,
            'parent_id': parent, 'file_path': file_path, 'kind': 'class',
            'source': get_node_text(node, source_code), 'has_error': node.has_error,
            'methods': extract_methods(node, source_code, class_id), **location(node),
            'fields': [],
            'type_parameters': extract_type_parameters(node, source_code),
            'is_local': '@' in qualified_name,
        }
        body = node.child_by_field_name('body')
        if body:
            for child in body.named_children:
                if child.type == 'field_declaration':
                    class_info['fields'].extend(extract_declarations(child, source_code))
        classes.append(class_info)
        parent = class_id
        scope = qualified_name
    elif node.type in ('method_declaration', 'constructor_declaration'):
        # Local classes with the same name in different methods need distinct scopes.
        name = get_node_text(node.child_by_field_name('name'), source_code)
        scope = f'{scope}.{name}@{node.start_byte}'
    for child in node.named_children:
        # Anonymous bodies have no 'body' field in the installed Java grammar.
        if node.type == 'object_creation_expression' and child.type == 'class_body':
            continue
        traverse(child, source_code, classes, file_path, parent, scope)


def extract_structure(root_node, source_code, file_path=''):
    classes = []
    package, _ = extract_file_metadata(root_node, source_code)
    traverse(root_node, source_code, classes, file_path, scope=package)
    return classes


def collect_diagnostics(node, source_code):
    diagnostics = []
    if node.type == 'ERROR' or node.is_missing:
        diagnostics.append({
            'kind': 'missing_node' if node.is_missing else 'syntax_error',
            'message': f'Missing {node.type}' if node.is_missing else 'Invalid Java syntax',
            'source': get_node_text(node, source_code), **location(node),
        })
    # Missing punctuation is often an unnamed child.
    for child in node.children:
        diagnostics.extend(collect_diagnostics(child, source_code))
    return diagnostics


def parse_source(source_code, file_path='<memory>'):
    """Return JSON-serializable data, including partial results and diagnostics."""
    if isinstance(source_code, str):
        source_code = source_code.encode('utf-8')
    file_path = str(file_path)
    parser = Parser(Language(tree_sitter_java.language()))
    tree = parser.parse(source_code)
    diagnostics = collect_diagnostics(tree.root_node, source_code)
    if not source_code.strip():
        diagnostics.append({'kind': 'empty_source', 'message': 'Source file is empty'})
    try:
        source_code.decode('utf-8')
    except UnicodeDecodeError:
        diagnostics.append({'kind': 'encoding_error', 'message': 'Source is not valid UTF-8'})
    if tree.root_node.has_error and not diagnostics:
        diagnostics.append({'kind': 'syntax_error', 'message': 'Tree-sitter reported a syntax error'})
    package, imports = extract_file_metadata(tree.root_node, source_code)
    return {
        'file_path': file_path, 'has_error': bool(diagnostics),
        'source_sha256': hashlib.sha256(source_code).hexdigest(),
        'package': package, 'imports': imports,
        'diagnostics': diagnostics,
        'classes': extract_structure(tree.root_node, source_code, file_path),
    }


def parse_file(file_path, relative_to=None):
    path = Path(file_path)
    identity = str(path)
    try:
        path = path.expanduser().resolve()
        identity = path.relative_to(Path(relative_to).resolve()).as_posix() if relative_to else str(path)
        source_code = path.read_bytes()
    except (OSError, ValueError, RuntimeError) as error:
        return {
            'file_path': identity,
            'has_error': True, 'classes': [], 'package': '', 'imports': [],
            'diagnostics': [{'kind': 'file_error', 'message': str(error)}],
        }
    return parse_source(source_code, identity)


def print_structure(classes):
    for class_info in classes:
        print(f"Class: {class_info['qualified_name']}")
        for method in class_info['methods']:
            print(f"  {method['kind']}: {method['signature']} (line {method['start_line']})")


def main():
    result = parse_file(Path(__file__).with_name('Calculator.java'))
    print_structure(result['classes'])
    for diagnostic in result['diagnostics']:
        print(f"Problem: {diagnostic['message']}")


if __name__ == '__main__':
    main()

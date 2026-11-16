"""Conservative declared-type lookup, not Java compiler or runtime resolution."""
import re


def create_symbol_index(parsed_files):
    index = {'classes': {}, 'qualified': {}, 'files': {}}
    for result in parsed_files:
        index['files'][result['file_path']] = result
        for cls in result['classes']:
            index['classes'][cls['id']] = cls
            if not cls.get('is_local'):
                index['qualified'].setdefault(cls['qualified_name'], []).append(cls)
    return index


def lookup_class(type_name, cls, index):
    """Resolve explicit imports, lexical member types and same-package names only."""
    if not type_name or not re.fullmatch(r'[\w$]+(?:\.[\w$]+)*', type_name):
        return None, 'Unsupported receiver type (inferred, generic, array or compound)'
    if type_name in ('var', 'void', 'boolean', 'byte', 'char', 'short', 'int', 'long', 'float', 'double'):
        return None, 'Receiver type is not an identifiable class'
    file_info = index['files'][cls['file_path']]
    names = []
    # Nested types are considered in lexical order, not by global simple name.
    owner = cls
    while owner:
        if type_name == owner['name']:
            names.append(owner['qualified_name'])
        names.append(f"{owner['qualified_name']}.{type_name}")
        owner = index['classes'].get(owner['parent_id'])
    for name in names:
        matches = index['qualified'].get(name, [])
        if matches:
            return unique_class(matches)
    first, _, suffix = type_name.partition('.')
    imports = [entry for entry in file_info.get('imports', [])
               if not entry['static'] and not entry['wildcard'] and entry['name']
               and entry['name'].split('.')[-1] == first]
    if imports:
        imported_names = {entry['name'] + ('.' + suffix if suffix else '') for entry in imports}
        if len(imported_names) != 1:
            return None, 'Ambiguous explicit imports'
        return unique_class(index['qualified'].get(next(iter(imported_names)), []))
    package = file_info.get('package', '')
    package_name = f'{package}.{type_name}' if package else type_name
    matches = index['qualified'].get(package_name, [])
    if matches:
        return unique_class(matches)
    if '.' in type_name:
        matches = index['qualified'].get(type_name, [])
        if matches:
            return unique_class(matches)
    return None, 'Class not indexed or import unsupported (wildcard/static imports are not resolved)'


def unique_class(matches):
    if len(matches) == 1:
        return matches[0], None
    return None, 'Ambiguous class identity' if matches else 'Class not indexed (external or missing dependency)'


def receiver_binding(name, call, method, cls, fields_only=False):
    """A matching unknown binding must shadow fields and class names, not fall back."""
    if not fields_only:
        position = call.get('start_byte', -1)
        locals_in_scope = [local for local in method.get('local_variables', [])
                           if local['name'] == name and local['scope_start'] <= position < local['scope_end']]
        if locals_in_scope:
            width = min(local['scope_end'] - local['scope_start'] for local in locals_in_scope)
            nearest = [local for local in locals_in_scope if local['scope_end'] - local['scope_start'] == width]
            if len(nearest) != 1 or nearest[0]['start_byte'] > position:
                return True, None, 'Ambiguous or not-yet-declared local receiver'
            return True, nearest[0]['type'], None
        parameters = [p for p in method['parameters'] if p['name'] == name and p['kind'] != 'receiver_parameter']
        if parameters:
            return True, parameters[0]['type'] if len(parameters) == 1 else None, None
    fields = [field for field in cls.get('fields', []) if field['name'] == name]
    if fields:
        if len(fields) != 1:
            return True, None, 'Ambiguous field receiver'
        if 'static' in method.get('modifiers', []) and 'static' not in fields[0]['modifiers']:
            return True, None, 'Instance field in static context'
        return True, fields[0]['type'], None
    return False, None, None


def resolve_call(call, methods):
    """Select only unique fixed-arity declarations; do not infer argument types."""
    candidates = [m for m in methods if m['kind'] == 'method' and m['name'] == call['name']]
    if len(candidates) != 1:
        return None, 'No declared target' if not candidates else 'Overloaded or duplicate method'
    target = candidates[0]
    parameters = [p for p in target['parameters'] if p['kind'] != 'receiver_parameter']
    if target['has_error'] or any(p['type'] is None or p['type'].endswith('...') for p in parameters):
        return None, 'Incomplete or variable-argument signature'
    if len(parameters) != call['argument_count']:
        return None, 'Argument count does not match'
    return target['id'], None


def resolve_invocation(call, method, cls, index):
    receiver = call['receiver']
    if receiver and (receiver == 'this' or receiver.startswith('this.')) and 'static' in method.get('modifiers', []):
        return None, 'this receiver in static context'
    static_receiver = False
    target_class = cls
    if receiver not in (None, 'this'):
        fields_only = receiver.startswith('this.') and receiver.count('.') == 1
        name = receiver[5:] if fields_only else receiver
        if not re.fullmatch(r'[\w$]+(?:\.[\w$]+)*', name):
            return None, 'Receiver expression requires semantic analysis'
        found, declared_type, reason = receiver_binding(name, call, method, cls, fields_only)
        if reason:
            return None, reason
        if fields_only and not found:
            return None, 'No declared field for receiver'
        static_receiver = not found
        type_name = declared_type if found else name
        type_parameters = set(method.get('type_parameters', []))
        owner = cls
        while owner:
            type_parameters.update(owner.get('type_parameters', []))
            owner = index['classes'].get(owner['parent_id'])
        if type_name and type_name.split('.')[0] in type_parameters:
            return None, 'Generic type parameter requires semantic analysis'
        target_class, reason = lookup_class(type_name, cls, index)
        if reason:
            return None, reason
    file_info = index['files'][target_class['file_path']]
    if file_info['has_error']:
        return None, 'Target source has parsing diagnostics'
    target_id, reason = resolve_call(call, target_class['methods'])
    if reason:
        return None, reason
    target = next(member for member in target_class['methods'] if member['id'] == target_id)
    if static_receiver and 'static' not in target.get('modifiers', []):
        return None, 'Class-qualified receiver requires a static method'
    if receiver in (None, 'this') and 'static' in method.get('modifiers', []):
        if receiver == 'this' or 'static' not in target.get('modifiers', []):
            return None, 'Instance call in static context'
    if target_class['id'] != cls['id']:
        modifiers = target.get('modifiers', [])
        if 'private' in modifiers:
            return None, 'Private target outside declaring class'
        caller_package = index['files'][cls['file_path']].get('package', '')
        if 'public' not in modifiers and caller_package != file_info.get('package', ''):
            return None, 'Non-public cross-package target requires access analysis'
    return target_id, None

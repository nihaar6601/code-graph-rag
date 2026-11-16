"""Traditional token-window chunks; no syntax tree, entity or graph construction."""
import hashlib
from bisect import bisect_right
from pathlib import Path

from indexer import discover_inputs
from token_counter import get_tokenizer


def chunk_source(source, file_path, chunk_size=128, overlap=24):
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError('chunk_size must be positive; overlap must be in [0, chunk_size)')
    text = source.decode('utf-8')
    encoding = get_tokenizer()
    tokens = encoding.encode(text, disallowed_special=())
    # Token byte boundaries can bisect Unicode characters. Move boundaries inward
    # to valid UTF-8 offsets; original bytes and line attribution remain exact.
    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1] + len(encoding.decode_single_token_bytes(token)))
    boundaries = {0}
    position = 0
    for character in text:
        position += len(character.encode('utf-8'))
        boundaries.add(position)
    fingerprint = hashlib.sha256(source).hexdigest()
    chunks = []
    valid = [i for i, offset in enumerate(offsets) if offset in boundaries]
    start = 0
    while start < len(tokens):
        end_index = bisect_right(valid, start + chunk_size) - 1
        # A Unicode code point can itself require multiple tokenizer tokens.
        if valid[end_index] <= start:
            end_index = bisect_right(valid, start)
        end = valid[end_index]
        byte_start, byte_end = offsets[start], offsets[end]
        while byte_start not in boundaries and byte_start < byte_end:
            byte_start += 1
        while byte_end not in boundaries and byte_end > byte_start:
            byte_end -= 1
        if byte_end > byte_start:
            body = source[byte_start:byte_end].decode('utf-8')
            chunks.append({'id': f'{file_path}::chunk:{byte_start}-{byte_end}', 'name': f'{file_path} chunk {len(chunks) + 1}',
                           'kind': 'chunk', 'file_path': file_path, 'source': body,
                           'source_sha256': fingerprint, 'start_byte': byte_start, 'end_byte': byte_end,
                           'start_line': source[:byte_start].count(b'\n') + 1,
                           'end_line': source[:byte_end - 1].count(b'\n') + 1,
                           'token_start': start, 'token_end': end, 'class_context': []})
        if end == len(tokens):
            break
        next_index = bisect_right(valid, max(start + 1, end - overlap)) - 1
        start = valid[next_index] if valid[next_index] > start else end
    return chunks


def index_chunks(inputs, chunk_size=128, overlap=24):
    discovery = discover_inputs(inputs)
    chunks, files = [], []
    root = discovery['root']
    for value in discovery['source_paths']:
        path = Path(value)
        identity = path.relative_to(root).as_posix() if root else str(path)
        try:
            source = path.read_bytes()
            new_chunks = chunk_source(source, identity, chunk_size, overlap)
            chunks.extend(new_chunks)
            files.append({'file_path': identity, 'has_error': False, 'diagnostics': [],
                          'source_sha256': hashlib.sha256(source).hexdigest()})
        except (OSError, UnicodeError) as error:
            files.append({'file_path': identity, 'has_error': True,
                          'diagnostics': [{'kind': 'file_error', 'message': str(error)}]})
    files.extend(discovery.pop('failed'))
    return dict(discovery, files=files, chunks=chunks)


def verify_chunk(chunk, root=None, cache=None):
    path = Path(chunk['file_path'])
    if not path.is_absolute():
        if root is None:
            raise ValueError('Relative chunk paths need a repository root')
        base = Path(root).resolve()
        path = (base / path).resolve()
        if not path.is_relative_to(base):
            raise ValueError('Chunk path escapes repository root')
    cache = cache if cache is not None else {}
    if str(path) not in cache:
        cache[str(path)] = path.read_bytes()
    source = cache[str(path)]
    if hashlib.sha256(source).hexdigest() != chunk['source_sha256']:
        raise ValueError('Chunk source changed after indexing; re-index it')
    start, end = chunk['start_byte'], chunk['end_byte']
    if not 0 <= start < end <= len(source):
        raise ValueError('Chunk byte range is invalid')
    if source[start:end].decode('utf-8') != chunk['source']:
        raise ValueError('Chunk byte range does not match source')
    if (source[:start].count(b'\n') + 1, source[:end - 1].count(b'\n') + 1) != (chunk['start_line'], chunk['end_line']):
        raise ValueError('Chunk line range does not match source')
    return dict(chunk)

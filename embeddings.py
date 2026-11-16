"""Explicit local Sentence Transformers loading and content-addressed vector caching."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_CACHE = Path(__file__).parent / '.cache' / 'embeddings'
DEFAULT_MODEL_CACHE = Path(__file__).parent / '.cache' / 'models'


class LocalEncoder:
    """No model network access unless allow_download is explicitly enabled."""
    def __init__(self, model=DEFAULT_MODEL, model_cache=DEFAULT_MODEL_CACHE,
                 allow_download=False, revision=None):
        started = perf_counter()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(1)
            self.model = SentenceTransformer(model, device='cpu', cache_folder=str(model_cache),
                                             local_files_only=not allow_download,
                                             revision=revision, trust_remote_code=False)
        except Exception as error:
            raise RuntimeError(f'Local embedding model unavailable: {model}. '
                               'Run embeddings.py --download-model explicitly to prepare it. '
                               f'Details: {error}') from error
        dimension_method = getattr(self.model, 'get_embedding_dimension', None)
        self.dimension = dimension_method() if dimension_method else self.model.get_sentence_embedding_dimension()
        self.model_name = model
        self.max_sequence_length = self.model.max_seq_length
        config = getattr(getattr(self.model[0], 'auto_model', None), 'config', None)
        self.resolved_revision = getattr(config, '_commit_hash', None) or revision
        if self.resolved_revision is None and not Path(model).is_dir():
            reference = Path(model_cache) / ('models--' + model.replace('/', '--')) / 'refs' / 'main'
            if reference.is_file():
                self.resolved_revision = reference.read_text().strip()
        # Hash actual loaded weights/configuration, not a mutable model-name alias.
        digest = hashlib.sha256()
        digest.update(json.dumps({'model': model, 'revision': revision, 'policy': 'mean_windows_v1',
                                  'max_sequence_length': self.max_sequence_length}, sort_keys=True).encode())
        digest.update(json.dumps(self.model.tokenizer.get_vocab(), sort_keys=True).encode())
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.cache_identity = digest.hexdigest()
        self.load_ms = (perf_counter() - started) * 1000

    def encode(self, texts):
        """Mean-pool token windows so long entity text is not silently truncated."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokenizer = self.model.tokenizer
        limit = self.max_sequence_length - tokenizer.num_special_tokens_to_add(pair=False)
        windows, spans = [], []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            start = len(windows)
            for offset in range(0, max(len(tokens), 1), limit):
                windows.append(tokenizer.decode(tokens[offset:offset + limit], skip_special_tokens=True))
            spans.append((start, len(windows)))
        vectors = self.model.encode(windows, convert_to_numpy=True, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=32)
        pooled = np.asarray([vectors[start:end].mean(axis=0) for start, end in spans], dtype=np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.divide(pooled, norms, out=np.zeros_like(pooled), where=norms != 0)


def embed_texts(texts, encoder, cache_dir=DEFAULT_CACHE):
    """Cache by text and actual model identity. Read npy files without pickle."""
    started = perf_counter()
    folder = Path(cache_dir) / encoder.cache_identity
    folder.mkdir(parents=True, exist_ok=True)
    keys = [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]
    vectors, missing = {}, {}
    hits = 0
    for key, text in zip(keys, texts):
        if key in vectors or key in missing:
            continue
        try:
            vector = np.load(folder / f'{key}.npy', allow_pickle=False)
            if vector.shape != (encoder.dimension,) or not np.isfinite(vector).all():
                raise ValueError('Invalid cached vector')
            vectors[key] = vector.astype(np.float32)
            hits += 1
        except (OSError, ValueError, EOFError):
            missing[key] = text
    generation_started = perf_counter()
    if missing:
        generated = np.asarray(encoder.encode(list(missing.values())), dtype=np.float32)
        if generated.shape != (len(missing), encoder.dimension) or not np.isfinite(generated).all():
            raise ValueError('Embedding model returned an invalid matrix')
        for key, vector in zip(missing, generated):
            vectors[key] = vector
            with tempfile.NamedTemporaryFile(dir=folder, suffix='.npy', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, vector, allow_pickle=False)
            temporary_path.replace(folder / f'{key}.npy')
    generation_ms = (perf_counter() - generation_started) * 1000
    matrix = np.asarray([vectors[key] for key in keys], dtype=np.float32).reshape(len(keys), encoder.dimension)
    return matrix, {'cache_hits': hits, 'cache_misses': len(missing), 'generation_ms': generation_ms,
                    'total_ms': (perf_counter() - started) * 1000}


def cosine_search(matrix, query_vector, ids, top_k=20, min_similarity=0.2):
    if top_k < 1 or not -1 <= min_similarity <= 1:
        raise ValueError('top_k must be positive and similarity threshold must be in [-1, 1]')
    matrix, query_vector = np.asarray(matrix), np.asarray(query_vector)
    if matrix.ndim != 2 or len(matrix) != len(ids) or query_vector.shape != (matrix.shape[1],):
        raise ValueError('Vector dimensions do not match')
    if not np.isfinite(matrix).all() or not np.isfinite(query_vector).all():
        raise ValueError('Vectors must be finite')
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    scores = np.divide(matrix @ query_vector, norms, out=np.zeros(len(ids), dtype=np.float32), where=norms != 0)
    ranked = [{'id': entity_id, 'score': float(np.clip(score, -1, 1))} for entity_id, score, norm
              in zip(ids, scores, norms) if norm > 0 and score >= min_similarity]
    return sorted(ranked, key=lambda result: (-result['score'], result['id']))[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true', help='Explicitly allow model/vocabulary downloads')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--model-cache', type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument('--revision')
    args = parser.parse_args()
    try:
        encoder = LocalEncoder(args.model, args.model_cache, args.download_model, args.revision)
        encoder.encode(['local embedding warmup'])
        size = sum(path.stat().st_size for path in args.model_cache.rglob('*') if path.is_file() and not path.is_symlink())
        print(f'Ready locally: {args.model}; dimension={encoder.dimension}; model cache={size / 1024**2:.1f} MiB')
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()

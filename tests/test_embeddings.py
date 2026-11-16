"""Deterministic Phase 5 unit checks; no embedding model or tokenizer is loaded."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from embeddings import cosine_search, embed_texts
from chunking import chunk_source, index_chunks, verify_chunk
from hybrid import classify_intent, reciprocal_rank_fusion, semantic_rank, should_expand


class ByteTokenizer:
    """One token per UTF-8 byte, deliberately splitting multibyte characters."""

    def encode(self, text, disallowed_special=()):
        return list(text.encode('utf-8'))

    def decode_single_token_bytes(self, token):
        return bytes([token])


def mock_encoder(identity='model-a'):
    encoder = Mock(cache_identity=identity, dimension=3)
    encoder.encode.side_effect = lambda texts: np.asarray(
        [[len(text), sum(text.encode('utf-8')) % 17, 1] for text in texts],
        dtype=np.float32,
    ).reshape(len(texts), 3)
    return encoder


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.encoder = mock_encoder()
        # Fail immediately if a future change attempts model construction.
        guard = patch('embeddings.LocalEncoder', side_effect=AssertionError('No model loading'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_cache_hits_do_not_reencode(self):
        first, stats = embed_texts(['alpha', 'beta'], self.encoder, self.cache)
        self.assertEqual((stats['cache_hits'], stats['cache_misses']), (0, 2))
        second, stats = embed_texts(['beta', 'alpha'], self.encoder, self.cache)
        np.testing.assert_array_equal(second, first[::-1])
        self.assertEqual((stats['cache_hits'], stats['cache_misses']), (2, 0))
        self.encoder.encode.assert_called_once_with(['alpha', 'beta'])

    def test_changed_text_invalidates_only_changed_entry(self):
        original, _ = embed_texts(['alpha', 'beta'], self.encoder, self.cache)
        self.encoder.encode.reset_mock()
        changed, stats = embed_texts(['alpha', 'beta!'], self.encoder, self.cache)
        self.encoder.encode.assert_called_once_with(['beta!'])
        self.assertEqual((stats['cache_hits'], stats['cache_misses']), (1, 1))
        np.testing.assert_array_equal(changed[0], original[0])
        self.assertFalse(np.array_equal(changed[1], original[1]))

    def test_model_identity_isolates_cache(self):
        embed_texts(['alpha'], self.encoder, self.cache)
        other = mock_encoder('model-b')
        _, stats = embed_texts(['alpha'], other, self.cache)
        other.encode.assert_called_once_with(['alpha'])
        self.assertEqual((stats['cache_hits'], stats['cache_misses']), (0, 1))
        self.assertEqual({p.name for p in self.cache.iterdir()}, {'model-a', 'model-b'})

    def test_duplicates_encode_once_and_empty_input_never_encodes(self):
        matrix, stats = embed_texts(['a', 'a'], self.encoder, self.cache)
        self.encoder.encode.assert_called_once_with(['a'])
        np.testing.assert_array_equal(matrix[0], matrix[1])
        self.assertEqual(stats['cache_misses'], 1)
        self.encoder.encode.reset_mock()
        matrix, stats = embed_texts([], self.encoder, self.cache)
        self.assertEqual(matrix.shape, (0, 3))
        self.assertEqual((stats['cache_hits'], stats['cache_misses']), (0, 0))
        self.encoder.encode.assert_not_called()

    def test_corrupt_wrong_shape_and_nonfinite_cache_entries_are_rebuilt(self):
        key = hashlib.sha256(b'alpha').hexdigest()
        for corruption in ('garbage', 'shape', 'nan'):
            with self.subTest(corruption=corruption):
                expected, _ = embed_texts(['alpha'], self.encoder, self.cache)
                path = self.cache / self.encoder.cache_identity / f'{key}.npy'
                if corruption == 'garbage':
                    path.write_bytes(b'not an npy file')
                else:
                    np.save(path, np.ones(2) if corruption == 'shape' else np.full(3, np.nan))
                self.encoder.encode.reset_mock()
                actual, stats = embed_texts(['alpha'], self.encoder, self.cache)
                self.encoder.encode.assert_called_once_with(['alpha'])
                self.assertEqual((stats['cache_hits'], stats['cache_misses']), (0, 1))
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected[0])

    def test_invalid_generated_vectors_are_rejected_without_cache_entries(self):
        for value in ([1, 2, 3], [[1, 2]], [[1, 2, np.nan]], [[1, 2, np.inf]]):
            with self.subTest(value=value):
                encoder = mock_encoder()
                encoder.encode.side_effect = None
                encoder.encode.return_value = value
                with self.assertRaises(ValueError):
                    embed_texts(['alpha'], encoder, self.cache)
                self.assertEqual(list(self.cache.rglob('*.npy')), [])


class CosineTests(unittest.TestCase):
    def test_normalized_scores_stable_ties_threshold_and_top_k(self):
        matrix = np.array([[2., 0.], [1., 0.], [1., 1.], [-1., 0.]])
        ids = ['z', 'a', 'diagonal', 'opposite']
        ranked = cosine_search(matrix, np.array([3., 0.]), ids, top_k=3, min_similarity=.7)
        self.assertEqual([r['id'] for r in ranked], ['a', 'z', 'diagonal'])
        self.assertAlmostEqual(ranked[0]['score'], 1)
        self.assertAlmostEqual(ranked[2]['score'], 1 / np.sqrt(2))
        self.assertEqual(len(cosine_search(matrix, [1., 0.], ids, top_k=1)), 1)
        self.assertEqual([r['id'] for r in cosine_search(matrix, [1., 0.], ids, min_similarity=1)], ['a', 'z'])

    def test_zero_vectors_and_empty_matrix_produce_no_spurious_matches(self):
        matrix = np.array([[0., 0.], [1., 0.]])
        self.assertEqual(cosine_search(matrix, [0., 0.], ['zero', 'unit'], min_similarity=-1), [])
        self.assertEqual([r['id'] for r in cosine_search(matrix, [1., 0.], ['zero', 'unit'], min_similarity=-1)], ['unit'])
        self.assertEqual(cosine_search(np.empty((0, 2)), [1., 0.], []), [])

    def test_invalid_dimensions_nonfinite_vectors_and_options(self):
        cases = [([1., 0.], [1., 0.], ['a'], {}),
                 ([[1., 0.]], [1.], ['a'], {}),
                 ([[1., 0.]], [1., 0.], [], {}),
                 ([[np.nan, 0.]], [1., 0.], ['a'], {}),
                 ([[1., 0.]], [np.inf, 0.], ['a'], {}),
                 ([[1., 0.]], [1., 0.], ['a'], {'top_k': 0}),
                 ([[1., 0.]], [1., 0.], ['a'], {'min_similarity': 1.1}),
                 ([[1., 0.]], [1., 0.], ['a'], {'min_similarity': -1.1})]
        for matrix, query, ids, options in cases:
            with self.subTest(matrix=matrix, query=query, ids=ids, options=options):
                with self.assertRaises(ValueError):
                    cosine_search(matrix, query, ids, **options)


class ChunkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        tokenizer = patch('chunking.get_tokenizer', return_value=ByteTokenizer())
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def test_overlap_token_windows_and_exact_source_ranges(self):
        source = b'abcdefghij'
        chunks = chunk_source(source, 'A.java', chunk_size=4, overlap=2)
        self.assertEqual([(c['token_start'], c['token_end']) for c in chunks], [(0, 4), (2, 6), (4, 8), (6, 10)])
        self.assertEqual([c['source'] for c in chunks], ['abcd', 'cdef', 'efgh', 'ghij'])
        self.assertEqual(len({c['id'] for c in chunks}), 4)
        for chunk in chunks:
            self.assertEqual(source[chunk['start_byte']:chunk['end_byte']], chunk['source'].encode())
            self.assertEqual(chunk['file_path'], 'A.java')
            self.assertEqual(chunk['kind'], 'chunk')
            self.assertEqual(chunk['source_sha256'], hashlib.sha256(source).hexdigest())

    def test_unicode_byte_boundaries_and_line_attribution(self):
        source = 'abé\n中xyz\nlast'.encode('utf-8')
        chunks = chunk_source(source, 'Unicode.java', chunk_size=7, overlap=3)
        covered = set()
        for chunk in chunks:
            start, end = chunk['start_byte'], chunk['end_byte']
            self.assertEqual(chunk['source'], source[start:end].decode('utf-8'))
            self.assertEqual(chunk['start_line'], source[:start].count(b'\n') + 1)
            self.assertEqual(chunk['end_line'], source[:end - 1].count(b'\n') + 1)
            self.assertEqual(start, chunk['token_start'])
            self.assertEqual(end, chunk['token_end'])
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(len(source))))

    def test_unicode_zero_overlap_has_contiguous_ranges_without_gaps(self):
        source = 'aé中🙂\n雪e\u0301z'.encode('utf-8')
        for size in (1, 2, 3, 4, 5, 7):
            with self.subTest(chunk_size=size):
                chunks = chunk_source(source, 'Unicode.java', chunk_size=size, overlap=0)
                self.assertTrue(chunks)
                self.assertEqual(chunks[0]['start_byte'], 0)
                self.assertEqual(chunks[-1]['end_byte'], len(source))
                self.assertEqual(''.join(c['source'] for c in chunks).encode('utf-8'), source)
                for previous, current in zip(chunks, chunks[1:]):
                    self.assertEqual(previous['end_byte'], current['start_byte'])
                    self.assertEqual(previous['token_end'], current['token_start'])
                for chunk in chunks:
                    start, end = chunk['start_byte'], chunk['end_byte']
                    self.assertGreater(end, start)
                    self.assertEqual(chunk['source'], source[start:end].decode('utf-8'))
                    self.assertEqual((chunk['token_start'], chunk['token_end']), (start, end))
                    self.assertEqual(chunk['start_line'], source[:start].count(b'\n') + 1)
                    self.assertEqual(chunk['end_line'], source[:end - 1].count(b'\n') + 1)

    def test_empty_source_and_invalid_options(self):
        self.assertEqual(chunk_source(b'', 'Empty.java'), [])
        for size, overlap in ((0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_source(b'abc', 'A.java', size, overlap)
        with self.assertRaises(UnicodeDecodeError):
            chunk_source(b'\xff', 'Bad.java')

    def test_indexing_is_independent_of_parser_and_handles_malformed_java(self):
        (self.root / 'Bad.java').write_bytes(b'class Broken { nonsense (')
        (self.root / 'A.java').write_bytes(b'class A {}')
        with patch('indexer.parse_file', side_effect=AssertionError('Parser must not run')) as parse:
            result = index_chunks([self.root], chunk_size=16, overlap=4)
        parse.assert_not_called()
        self.assertEqual(Path(result['root']), self.root)
        self.assertEqual([f['file_path'] for f in result['files']], ['A.java', 'Bad.java'])
        self.assertTrue(all(not f['has_error'] for f in result['files']))
        self.assertEqual({c['file_path'] for c in result['chunks']}, {'A.java', 'Bad.java'})
        self.assertEqual(result['chunks'], index_chunks([self.root], 16, 4)['chunks'])

    def test_indexing_reports_missing_and_invalid_utf8_files(self):
        bad = self.root / 'Bad.java'
        bad.write_bytes(b'\xff')
        result = index_chunks([bad, self.root / 'Missing.java'])
        self.assertEqual(result['chunks'], [])
        self.assertEqual(len(result['files']), 2)
        for record in result['files']:
            self.assertTrue(record['has_error'])
            self.assertEqual(record['diagnostics'][0]['kind'], 'file_error')

    def test_verify_chunk_and_changed_source_rejection(self):
        path = self.root / 'A.java'
        path.write_bytes(b'class A {}')
        chunk = chunk_source(path.read_bytes(), 'A.java')[0]
        cache = {}
        self.assertEqual(verify_chunk(chunk, self.root, cache), chunk)
        self.assertEqual(cache[str(path)], b'class A {}')
        path.write_bytes(b'class B {}')
        # A verification cache is scoped to one source snapshot, so use a fresh one.
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_chunk(chunk, self.root, {})

    def test_verify_rejects_tampering_missing_root_and_path_escape(self):
        path = self.root / 'A.java'
        path.write_bytes(b'class A {}')
        chunk = chunk_source(path.read_bytes(), 'A.java')[0]
        with self.assertRaises(ValueError):
            verify_chunk(dict(chunk, source='forged'), self.root, {})
        with self.assertRaises(ValueError):
            verify_chunk(chunk)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            verify_chunk(dict(chunk, file_path='../A.java'), self.root, {})


class HybridTests(unittest.TestCase):
    def test_fusion_combines_ranks_not_raw_score_and_preserves_inputs(self):
        lexical = [{'id': 'a', 'score': 999.}, {'id': 'b', 'score': 1.}]
        semantic = [{'id': 'b', 'score': .9}, {'id': 'c', 'score': .8}]
        result = reciprocal_rank_fusion(lexical, semantic)
        self.assertEqual([r['id'] for r in result], ['b', 'a', 'c'])
        self.assertAlmostEqual(result[0]['score'], 1 / 62 + 1 / 61)
        self.assertEqual(result[0]['fusion'], {'lexical': {'rank': 2, 'score': 1.}, 'semantic': {'rank': 1, 'score': .9}})
        self.assertNotIn('fusion', lexical[0])
        self.assertEqual(lexical[0]['score'], 999.)

    def test_fusion_duplicates_empty_inputs_and_deterministic_ties(self):
        a = {'id': 'a', 'score': 1.}
        b = {'id': 'b', 'score': .5}
        result = reciprocal_rank_fusion([a, a], [b])
        self.assertEqual([r['id'] for r in result], ['a', 'b'])
        self.assertAlmostEqual(result[0]['score'], 1 / 61)
        self.assertEqual(result[0]['fusion']['lexical']['rank'], 1)
        self.assertEqual(reciprocal_rank_fusion([], []), [])
        self.assertEqual(reciprocal_rank_fusion([], [a])[0]['id'], 'a')
        self.assertEqual(reciprocal_rank_fusion([a], [])[0]['id'], 'a')
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_intent_categories_and_implementation_only_precedence(self):
        cases = {'Show only the implementation of processPayment': 'implementation_only',
                 'implementation-only payment code': 'implementation_only',
                 'Just the method body and its calls': 'implementation_only',
                 'only': 'implementation_only',
                 'just': 'implementation_only',
                 'implementation': 'implementation_only',
                 'Why only charge?': 'implementation_only',
                 'Find validatePayment': 'lookup',
                 'processPayment': 'lookup',
                 'processPayment()': 'lookup',
                 'Find processPayment': 'lookup',
                 'processingPayment': 'lookup',
                 'Which methods call charge?': 'dependency',
                 'dependency': 'dependency',
                 'Trace dependencies downstream': 'dependency',
                 'Follow checkout': 'dependency',
                 'How do I follow checkout?': 'dependency',
                 'How does checkout work?': 'behavior',
                 'Why checkout?': 'behavior',
                 'Payment flow': 'behavior',
                 'process': 'behavior',
                 'Payment processing': 'behavior',
                 'Compare payment services': 'behavior',
                 'Payment across services': 'behavior',
                 'Payment services together': 'behavior',
                 'Explain the payment workflow': 'behavior'}
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(classify_intent(query), expected)
                self.assertEqual(should_expand(query), (expected in ('dependency', 'behavior'), expected))

    def test_expansion_overrides_and_invalid_policy(self):
        for query in ('only implementation', 'Find charge', 'Trace calls', 'How does checkout work?'):
            with self.subTest(query=query):
                intent = classify_intent(query)
                self.assertEqual(should_expand(query, 'always'), (True, intent))
                self.assertEqual(should_expand(query, 'never'), (False, intent))
        with self.assertRaises(ValueError):
            should_expand('charge', 'sometimes')

    def test_semantic_ranking_filters_classes_before_top_k(self):
        documents = [{'id': 'class', 'kind': 'class', 'name': 'Shop'},
                     {'id': 'method', 'kind': 'method', 'name': 'charge'},
                     {'id': 'ctor', 'kind': 'constructor', 'name': 'Shop()'}]
        matrix = np.array([[1., 0.], [.8, .6], [0., 1.]])
        result = semantic_rank(documents, matrix, [1., 0.], top_k=1, min_similarity=.2)
        self.assertEqual([r['id'] for r in result], ['method'])
        self.assertEqual((result[0]['kind'], result[0]['name']), ('method', 'charge'))
        self.assertAlmostEqual(result[0]['score'], .8)
        result = semantic_rank(documents, matrix, [1., 0.], top_k=1, min_similarity=.2, include_classes=True)
        self.assertEqual([r['id'] for r in result], ['class'])
        self.assertEqual(semantic_rank([], np.empty((0, 2)), [1., 0.], 5, .2), [])
        self.assertEqual(semantic_rank(documents[:1], matrix[:1], [1., 0.], 5, .2), [])


if __name__ == '__main__':
    unittest.main()

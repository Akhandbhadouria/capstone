"""
Test 2 — Dense Embedding Generation
Tests vector shape, semantic separation, and CPU latency.
Requires: sentence-transformers
"""
import time
import unittest


class TestEmbeddingGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sentence_transformers import SentenceTransformer
            cls.model = SentenceTransformer("all-MiniLM-L6-v2")
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("sentence-transformers not installed")

    def test_embedding_shape(self):
        self._skip_if_unavailable()
        vec = self.model.encode("Hello world")
        self.assertEqual(len(vec), 384, "all-MiniLM-L6-v2 should produce 384-dim vectors")

    def test_embedding_is_float_list(self):
        self._skip_if_unavailable()
        vec = self.model.encode("Policy document test")
        self.assertIsInstance(vec.tolist(), list)
        self.assertIsInstance(vec.tolist()[0], float)

    def test_semantic_separation(self):
        """Semantically similar sentences should be closer than unrelated ones."""
        self._skip_if_unavailable()
        import numpy as np
        v1 = self.model.encode("Attendance policy for teachers")
        v2 = self.model.encode("Teacher attendance rules and regulations")
        v3 = self.model.encode("How to cook pasta carbonara")

        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_related = cosine(v1, v2)
        sim_unrelated = cosine(v1, v3)
        self.assertGreater(
            sim_related, sim_unrelated,
            f"Related pair ({sim_related:.3f}) should outscore unrelated ({sim_unrelated:.3f})"
        )

    def test_cpu_latency_acceptable(self):
        """Single-sentence encoding should complete within 2 seconds on CPU."""
        self._skip_if_unavailable()
        sentence = "What is the minimum attendance percentage required for teachers?"
        start = time.time()
        self.model.encode(sentence)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, f"Encoding took too long: {elapsed:.2f}s")

    def test_batch_encoding(self):
        self._skip_if_unavailable()
        texts = [f"Document chunk number {i} with policy content." for i in range(10)]
        embeddings = self.model.encode(texts)
        self.assertEqual(embeddings.shape[0], 10)
        self.assertEqual(embeddings.shape[1], 384)

    def test_different_texts_different_vectors(self):
        self._skip_if_unavailable()
        import numpy as np
        v1 = self.model.encode("Attendance policy")
        v2 = self.model.encode("Leave management system")
        self.assertFalse(
            np.allclose(v1, v2),
            "Different texts must produce different embeddings"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

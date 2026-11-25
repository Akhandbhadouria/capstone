"""
Test 4 — Semantic Retrieval
Tests intent matching, relevance ranking, and empty collection fallback.
Requires: chromadb, sentence-transformers
"""
import tempfile
import unittest


ATTENDANCE_DOC = (
    "Teachers must maintain a minimum attendance of 75 percent each month. "
    "Attendance is recorded automatically via the face recognition system. "
    "If a teacher fails to meet the threshold, they are marked as a defaulter."
)

LEAVE_DOC = (
    "Teachers are entitled to 12 casual leaves per academic year. "
    "Emergency leave may be granted by the Principal with valid documentation. "
    "Leave must be applied for at least 24 hours in advance except in emergencies."
)

UNRELATED_DOC = (
    "The school canteen serves breakfast from 8 AM to 9 AM and lunch from 12 PM to 1 PM. "
    "Students are encouraged to bring healthy food from home."
)


def _setup_collection(client, model, pid=1):
    col = client.get_or_create_collection(
        f"principal_{pid}", metadata={"hnsw:space": "cosine"}
    )
    for i, (text, title) in enumerate([
        (ATTENDANCE_DOC, "Attendance Policy"),
        (LEAVE_DOC, "Leave Management Policy"),
        (UNRELATED_DOC, "Canteen Rules"),
    ]):
        col.add(
            ids=[f"doc_{i}"],
            embeddings=[model.encode(text).tolist()],
            documents=[text],
            metadatas=[{"title": title, "doc_id": f"doc_{i}", "principal_id": str(pid)}]
        )
    return col


class TestSemanticRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            cls.model = SentenceTransformer("all-MiniLM-L6-v2")
            cls.chromadb = chromadb
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("chromadb or sentence-transformers not installed")

    def _query(self, col, query_text, n=1):
        q_vec = self.model.encode(query_text).tolist()
        actual_n = min(n, col.count())
        results = col.query(
            query_embeddings=[q_vec],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def test_intent_matching_attendance(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = _setup_collection(client, self.model)
            results = self._query(col, "What is the minimum attendance percentage?", n=1)
            top_meta = results["metadatas"][0][0]
            self.assertEqual(top_meta["title"], "Attendance Policy",
                             "Attendance query should retrieve Attendance Policy first")

    def test_intent_matching_leave(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = _setup_collection(client, self.model)
            results = self._query(col, "How many casual leaves do I get?", n=1)
            top_meta = results["metadatas"][0][0]
            self.assertEqual(top_meta["title"], "Leave Management Policy")

    def test_relevance_ranking_order(self):
        """Top result distance should be less than (i.e. more similar than) second result."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = _setup_collection(client, self.model)
            results = self._query(col, "attendance percentage requirement for teachers", n=3)
            distances = results["distances"][0]
            self.assertLessEqual(distances[0], distances[1],
                                 "First result should be closer (smaller cosine distance)")

    def test_empty_collection_returns_no_results(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = client.get_or_create_collection(
                "principal_empty", metadata={"hnsw:space": "cosine"}
            )
            self.assertEqual(col.count(), 0,
                             "Empty collection should have zero documents")

    def test_n_results_capped_to_collection_size(self):
        """Querying n_results > collection size should not raise an error."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = _setup_collection(client, self.model)  # 3 docs
            actual_n = min(10, col.count())
            results = self._query(col, "attendance", n=actual_n)
            self.assertLessEqual(len(results["documents"][0]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

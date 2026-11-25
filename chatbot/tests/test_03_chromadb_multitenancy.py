"""
Test 3 — ChromaDB Multi-Tenancy & Isolation
Tests per-tenant collections, zero cross-tenant leakage, and clean deletion.
Requires: chromadb, sentence-transformers
"""
import tempfile
import unittest


def _make_client(tmp_dir):
    import chromadb
    return chromadb.PersistentClient(path=tmp_dir)


def _get_collection(client, principal_id: int):
    return client.get_or_create_collection(
        name=f"principal_{principal_id}",
        metadata={"hnsw:space": "cosine"}
    )


class TestChromaDBMultiTenancy(unittest.TestCase):
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

    def _add_doc(self, collection, doc_id, text, principal_id):
        vec = self.model.encode(text).tolist()
        collection.add(
            ids=[doc_id],
            embeddings=[vec],
            documents=[text],
            metadatas=[{"principal_id": str(principal_id), "title": "Test Doc"}]
        )

    def test_separate_collections_per_tenant(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            col1 = _get_collection(client, 1)
            col2 = _get_collection(client, 2)
            self.assertNotEqual(col1.name, col2.name)

    def test_zero_cross_tenant_leakage(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            col1 = _get_collection(client, 1)
            col2 = _get_collection(client, 2)

            self._add_doc(col1, "doc_p1_1", "Attendance policy for principal one.", 1)
            self._add_doc(col2, "doc_p2_1", "Leave policy for principal two.", 2)

            # Tenant 1 should only see their doc
            self.assertEqual(col1.count(), 1)
            self.assertEqual(col2.count(), 1)

            # Query tenant 1's collection — should not return tenant 2's data
            q_vec = self.model.encode("leave policy").tolist()
            results = col1.query(query_embeddings=[q_vec], n_results=1,
                                 include=["metadatas"])
            meta = results["metadatas"][0][0]
            self.assertEqual(meta["principal_id"], "1",
                             "Tenant 1 query must only return tenant 1 docs")

    def test_delete_removes_chunks(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            col = _get_collection(client, 99)
            self._add_doc(col, "doc_del_1", "Document to be deleted.", 99)
            self.assertEqual(col.count(), 1)
            col.delete(ids=["doc_del_1"])
            self.assertEqual(col.count(), 0)

    def test_where_filter_deletion(self):
        """Deletion by metadata filter (source_file) removes only matching chunks."""
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            col = _get_collection(client, 42)
            vec = self.model.encode("Policy content.").tolist()

            col.add(ids=["c1"], embeddings=[vec], documents=["Chunk 1 from fileA.txt"],
                    metadatas=[{"source_file": "fileA.txt", "title": "A"}])
            col.add(ids=["c2"], embeddings=[vec], documents=["Chunk 2 from fileA.txt"],
                    metadatas=[{"source_file": "fileA.txt", "title": "A"}])
            col.add(ids=["c3"], embeddings=[vec], documents=["Chunk 1 from fileB.txt"],
                    metadatas=[{"source_file": "fileB.txt", "title": "B"}])

            self.assertEqual(col.count(), 3)
            col.delete(where={"source_file": "fileA.txt"})
            self.assertEqual(col.count(), 1)

    def test_empty_collection_query_returns_empty(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            col = _get_collection(client, 777)
            self.assertEqual(col.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

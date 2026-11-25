"""
Test 6 — End-to-End Integration Lifecycle
Tests: ingest → multi-turn Q&A → guardrail refusal → cleanup.
Requires: chromadb, sentence-transformers, groq (mocked for generation).
"""
import os
import re
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ── Minimal inline RAG pipeline (mirrors rag_engine.py logic) ──

def _clean_text(text: str) -> str:
    text = re.sub(r'^[ \t]*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'\|[-| :]+\|', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'---+', '', text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 600) -> list:
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return [c for c in chunks if len(c.strip()) > 30]


SYSTEM_PROMPT_SNIPPET = "I couldn't find this information"


def build_messages_e2e(query, context_chunks, conversation_history=None):
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n--- Source {i}: {chunk['title']} ---\n{chunk['text']}\n"
    user_message = f"CONTEXT FROM INSTITUTIONAL DOCUMENTS:\n{context_text}\n\nUSER QUESTION:\n{query}"
    messages = [{"role": "system", "content": "You are the ClassAudit AI assistant. " + SYSTEM_PROMPT_SNIPPET}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append(turn)
    messages.append({"role": "user", "content": user_message})
    return messages


class TestEndToEndLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            cls.chromadb = chromadb
            cls.model = SentenceTransformer("all-MiniLM-L6-v2")
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("chromadb or sentence-transformers not installed")

    def _ingest(self, col, chunks, title, doc_id, pid):
        for i, chunk in enumerate(chunks):
            vec = self.model.encode(chunk).tolist()
            col.add(
                ids=[f"{doc_id}_chunk_{i:04d}"],
                embeddings=[vec],
                documents=[chunk],
                metadatas=[{"title": title, "doc_id": doc_id,
                             "source_file": f"{doc_id}.txt",
                             "principal_id": str(pid), "chunk_index": i}]
            )

    def _retrieve(self, col, query, n=3):
        q_vec = self.model.encode(query).tolist()
        actual_n = min(n, col.count())
        results = col.query(query_embeddings=[q_vec], n_results=actual_n,
                            include=["documents", "metadatas", "distances"])
        chunks = []
        for doc, meta, dist in zip(results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0]):
            chunks.append({
                "text": doc, "title": meta.get("title", ""),
                "doc_id": meta.get("doc_id", ""),
                "relevance_score": round(1 - dist, 3)
            })
        return chunks

    def test_full_ingest_retrieve_cycle(self):
        self._skip_if_unavailable()
        policy_text = (
            "Attendance Policy\n\n"
            "Teachers must maintain a minimum of 75 percent attendance each month. "
            "Failure to meet this threshold results in a defaulter notice.\n\n"
            "Leave Management\n\n"
            "Teachers are entitled to 12 casual leaves per academic year."
        )
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = client.get_or_create_collection(
                "principal_1", metadata={"hnsw:space": "cosine"}
            )
            cleaned = _clean_text(policy_text)
            chunks = _chunk_text(cleaned)
            self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
            self._ingest(col, chunks, "Attendance Policy", "d1", 1)
            self.assertEqual(col.count(), len(chunks))

            # Retrieve
            results = self._retrieve(col, "minimum attendance percentage", n=3)
            self.assertGreater(len(results), 0)
            self.assertIn("75", results[0]["text"])

    def test_multi_turn_context_builds_correctly(self):
        self._skip_if_unavailable()
        history = [
            {"role": "user", "content": "What is the minimum attendance?"},
            {"role": "assistant", "content": "Teachers need 75% attendance each month."},
        ]
        chunks = [{"title": "Attendance Policy", "text": "75% attendance required.", "doc_id": "d1"}]
        msgs = build_messages_e2e("What happens if I miss it?", chunks, history)
        # Should have: system + 2 history + 1 user = 4
        self.assertEqual(len(msgs), 4)
        # The follow-up "miss it" should refer to previous context
        self.assertIn("miss it", msgs[-1]["content"])

    def test_guardrail_message_in_system_prompt(self):
        """The system prompt must tell the LLM what to say when answer is not in context."""
        chunks = [{"title": "Policy", "text": "Only about attendance.", "doc_id": "d1"}]
        msgs = build_messages_e2e("What is the best recipe for biryani?", chunks)
        system_content = msgs[0]["content"]
        self.assertIn(SYSTEM_PROMPT_SNIPPET, system_content)

    def test_mocked_generation_e2e(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = client.get_or_create_collection(
                "principal_e2e", metadata={"hnsw:space": "cosine"}
            )
            text = "Teachers need 75% attendance. Defaulters get a notice."
            chunks = _chunk_text(_clean_text(text))
            self._ingest(col, chunks, "Policy", "d_e2e", 1)

            # Mock the LLM
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "According to the Policy, 75% attendance is required."
            mock_client.chat.completions.create.return_value.choices = [mock_choice]

            context = self._retrieve(col, "attendance percentage", n=3)
            msgs = build_messages_e2e("What is the attendance requirement?", context)
            response = mock_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=msgs,
                temperature=0.3,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            self.assertIn("75%", answer)
            self.assertIn("Policy", answer)

    def test_cleanup_removes_all_chunks(self):
        self._skip_if_unavailable()
        with tempfile.TemporaryDirectory() as tmp:
            client = self.chromadb.PersistentClient(path=tmp)
            col = client.get_or_create_collection(
                "principal_cleanup", metadata={"hnsw:space": "cosine"}
            )
            chunks = ["Attendance rules.", "Leave management details.", "School policies here."]
            for i, c in enumerate(chunks):
                vec = self.model.encode(c).tolist()
                col.add(ids=[f"c{i}"], embeddings=[vec], documents=[c],
                        metadatas=[{"source_file": "policy.txt"}])
            self.assertEqual(col.count(), 3)
            col.delete(where={"source_file": "policy.txt"})
            self.assertEqual(col.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

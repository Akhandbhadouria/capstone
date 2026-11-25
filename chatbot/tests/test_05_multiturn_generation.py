"""
Test 5 — Multi-Turn Memory & Generation
Tests system prompt guardrails, 6-turn history capping, source formatting,
and error handling using a mocked Groq LLM client.
Runs standalone (no Django required).
"""
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Inline copies of the generation functions so we can run standalone ──

SYSTEM_PROMPT = """You are the ClassAudit AI Institutional Assistant — a professional, helpful, and accurate assistant for teachers and principals at an educational institution.

You answer questions exclusively based on the provided institutional documents: policies, guidelines, handbooks, and system guides.

You are given a conversation history (prior turns) and a new user question. Use the history to resolve follow-up questions and pronouns (e.g. "it", "that policy", "how many?"), but ground every factual claim strictly in the retrieved Context below.

Rules you MUST follow:
1. Answer ONLY from the provided Context. Do not invent policies or rules.
2. If the answer is not in the context, say: "I couldn't find this information in the institutional documents. Please contact your Head of Department or Principal."
3. Keep responses clear, structured, and professional.
4. When citing a rule, reference the document name (e.g., "According to the Attendance Policy...").
5. Use bullet points or numbered steps for procedural answers.
6. Be concise — do not repeat information unnecessarily.
7. If the user's question is a follow-up (e.g. "what about teachers?", "how many?"), interpret it in light of the previous conversation turns before answering."""


def build_messages(query: str, context_chunks: list, conversation_history: list = None):
    """Build the messages list as generate_answer() does, without calling Groq."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n--- Source {i}: {chunk['title']} ---\n{chunk['text']}\n"

    user_message = f"CONTEXT FROM INSTITUTIONAL DOCUMENTS:\n{context_text}\n\nUSER QUESTION:\n{query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer_with_client(client, query: str, context_chunks: list,
                                 conversation_history: list = None) -> str:
    messages = build_messages(query, context_chunks, conversation_history)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


SAMPLE_CHUNKS = [
    {"title": "Attendance Policy", "text": "Teachers must maintain 75% attendance.", "doc_id": "d1"},
    {"title": "Leave Policy", "text": "12 casual leaves per year.", "doc_id": "d2"},
]


class TestSystemPromptGuardrails(unittest.TestCase):
    def test_system_prompt_contains_key_rules(self):
        self.assertIn("ONLY from the provided Context", SYSTEM_PROMPT)
        self.assertIn("I couldn't find this information", SYSTEM_PROMPT)
        self.assertIn("conversation history", SYSTEM_PROMPT)

    def test_system_prompt_is_first_message(self):
        msgs = build_messages("What is the attendance policy?", SAMPLE_CHUNKS)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], SYSTEM_PROMPT)

    def test_user_message_is_last(self):
        msgs = build_messages("Any question", SAMPLE_CHUNKS)
        self.assertEqual(msgs[-1]["role"], "user")

    def test_context_is_embedded_in_user_message(self):
        msgs = build_messages("Query", SAMPLE_CHUNKS)
        user_content = msgs[-1]["content"]
        self.assertIn("Attendance Policy", user_content)
        self.assertIn("Leave Policy", user_content)
        self.assertIn("CONTEXT FROM INSTITUTIONAL DOCUMENTS", user_content)
        self.assertIn("USER QUESTION", user_content)


class TestSixTurnCapping(unittest.TestCase):
    def _make_history(self, n_pairs: int) -> list:
        history = []
        for i in range(n_pairs):
            history.append({"role": "user", "content": f"User question {i}"})
            history.append({"role": "assistant", "content": f"Assistant answer {i}"})
        return history

    def test_six_messages_injected_when_history_has_exactly_six(self):
        history = self._make_history(3)  # 3 pairs = 6 messages
        msgs = build_messages("New question", SAMPLE_CHUNKS, history)
        # system + 6 history + 1 user = 8
        self.assertEqual(len(msgs), 8)

    def test_history_capped_at_six_messages(self):
        history = self._make_history(5)  # 5 pairs = 10 messages
        msgs = build_messages("New question", SAMPLE_CHUNKS, history)
        # system + 6 (capped) + 1 user = 8
        self.assertEqual(len(msgs), 8)

    def test_no_history_produces_system_plus_user(self):
        msgs = build_messages("Question", SAMPLE_CHUNKS, conversation_history=None)
        self.assertEqual(len(msgs), 2)

    def test_empty_history_list_produces_system_plus_user(self):
        msgs = build_messages("Question", SAMPLE_CHUNKS, conversation_history=[])
        self.assertEqual(len(msgs), 2)

    def test_last_six_messages_are_most_recent(self):
        history = self._make_history(5)  # 10 turns, should keep last 6
        msgs = build_messages("Q", SAMPLE_CHUNKS, history)
        history_in_messages = [m for m in msgs if m["role"] in ("user", "assistant")]
        # Last history user message before the final Q should be from pair 4 (index 4)
        last_user_in_history = [m for m in history_in_messages[:-1] if m["role"] == "user"]
        self.assertIn("4", last_user_in_history[-1]["content"])


class TestMockedLLMGeneration(unittest.TestCase):
    def _make_mock_client(self, return_text: str):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = return_text
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        return mock_client

    def test_successful_generation(self):
        client = self._make_mock_client("According to the Attendance Policy, 75% is required.")
        result = generate_answer_with_client(client, "What is attendance?", SAMPLE_CHUNKS)
        self.assertIn("75%", result)
        self.assertIn("Attendance Policy", result)

    def test_error_returns_warning_message(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")
        result = generate_answer_with_client(mock_client, "Any query", SAMPLE_CHUNKS)
        self.assertIn("⚠️", result)
        self.assertIn("API timeout", result)

    def test_history_passed_to_client(self):
        client = self._make_mock_client("Answer.")
        history = [
            {"role": "user", "content": "What is 75%?"},
            {"role": "assistant", "content": "It is the minimum attendance."},
        ]
        generate_answer_with_client(client, "What about leave?", SAMPLE_CHUNKS, history)
        call_args = client.chat.completions.create.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args[1].get("messages")
        roles = [m["role"] for m in messages_sent]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_sources_deduplication(self):
        """Source list should deduplicate by doc_id."""
        chunks = [
            {"title": "Attendance Policy", "doc_id": "d1", "relevance_score": 0.9,
             "text": "75% attendance."},
            {"title": "Attendance Policy", "doc_id": "d1", "relevance_score": 0.8,
             "text": "Attendance is tracked."},
            {"title": "Leave Policy", "doc_id": "d2", "relevance_score": 0.7,
             "text": "12 leaves."},
        ]
        seen = set()
        sources = []
        for chunk in chunks:
            if chunk["doc_id"] not in seen:
                sources.append({"title": chunk["title"], "doc_id": chunk["doc_id"]})
                seen.add(chunk["doc_id"])
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["doc_id"], "d1")
        self.assertEqual(sources[1]["doc_id"], "d2")


if __name__ == "__main__":
    unittest.main(verbosity=2)

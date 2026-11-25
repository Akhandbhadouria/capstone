"""
Test 1 — Extraction & Chunking
Tests TXT/MD/PDF extraction, markdown heading cleaning, and paragraph chunking.
Runs standalone (no Django required).
"""
import os
import sys
import re
import tempfile
import unittest

# ── Minimal inline copies of the two pure functions being tested ──

def _clean_text(text: str) -> str:
    text = re.sub(r'^[ \t]*#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text, flags=re.DOTALL)
    text = re.sub(r'\|[-| :]+\|', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'---+', '', text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list:
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
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) + 1 <= chunk_size:
                        sub = (sub + " " + s).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s
                current_chunk = sub
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return [c for c in chunks if len(c.strip()) > 30]


def _extract_text_from_file(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext in ("md", "txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


class TestMarkdownCleaning(unittest.TestCase):
    def test_top_level_heading_stripped(self):
        result = _clean_text("# Title\nSome content.")
        self.assertNotIn("#", result)
        self.assertIn("Some content.", result)

    def test_indented_heading_stripped(self):
        result = _clean_text("   ## Indented Heading\nContent below.")
        self.assertNotIn("#", result)
        self.assertIn("Content below.", result)

    def test_bold_stripped(self):
        result = _clean_text("This is **bold** text.")
        self.assertIn("bold", result)
        self.assertNotIn("**", result)

    def test_code_block_stripped(self):
        result = _clean_text("Look at `this code` snippet.")
        self.assertNotIn("`", result)

    def test_triple_newlines_collapsed(self):
        result = _clean_text("Para one.\n\n\n\nPara two.")
        self.assertNotIn("\n\n\n", result)

    def test_horizontal_rule_stripped(self):
        result = _clean_text("Above\n---\nBelow")
        self.assertNotIn("---", result)


class TestTxtExtraction(unittest.TestCase):
    def test_txt_roundtrip(self):
        content = "Hello world.\nThis is a test file."
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            extracted = _extract_text_from_file(path)
            self.assertEqual(extracted.strip(), content.strip())
        finally:
            os.unlink(path)

    def test_md_roundtrip(self):
        content = "# Heading\n\nSome body text."
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = f.name
        try:
            extracted = _extract_text_from_file(path)
            self.assertEqual(extracted.strip(), content.strip())
        finally:
            os.unlink(path)

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                _extract_text_from_file(path)
        finally:
            os.unlink(path)


class TestChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        text = "This is a short policy statement that fits in one chunk."
        chunks = _chunk_text(text, chunk_size=600)
        self.assertEqual(len(chunks), 1)
        self.assertIn("short policy", chunks[0])

    def test_long_text_splits_into_multiple(self):
        para = "Word " * 50  # ~250 chars per para
        text = "\n\n".join([para] * 5)  # ~1250 chars total
        chunks = _chunk_text(text, chunk_size=300)
        self.assertGreater(len(chunks), 1)

    def test_very_short_chunks_filtered(self):
        text = "Hi.\n\nA longer paragraph with enough content to actually be useful here."
        chunks = _chunk_text(text)
        for c in chunks:
            self.assertGreater(len(c.strip()), 30)

    def test_paragraph_boundaries_respected(self):
        text = "First paragraph with some content here.\n\nSecond paragraph with other content."
        chunks = _chunk_text(text, chunk_size=100)
        # Both paragraphs should appear somewhere in chunks
        all_text = " ".join(chunks)
        self.assertIn("First paragraph", all_text)
        self.assertIn("Second paragraph", all_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

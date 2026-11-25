"""
Master Test Runner for ClassAudit AI Chatbot RAG Pipeline.

Usage:
    python run_all_tests.py              # run all 6 test modules
    python run_all_tests.py -v           # verbose output
    python run_all_tests.py --failfast   # stop on first failure
"""
import sys
import os
import unittest

# Ensure the tests directory is in the path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

TEST_MODULES = [
    "test_01_extraction_chunking",
    "test_02_embeddings",
    "test_03_chromadb_multitenancy",
    "test_04_retrieval",
    "test_05_multiturn_generation",
    "test_06_e2e_lifecycle",
]

DESCRIPTIONS = {
    "test_01_extraction_chunking":   "1 — Extraction & Chunking",
    "test_02_embeddings":            "2 — Dense Embedding Generation",
    "test_03_chromadb_multitenancy": "3 — ChromaDB Multi-Tenancy & Isolation",
    "test_04_retrieval":             "4 — Semantic Retrieval",
    "test_05_multiturn_generation":  "5 — Multi-Turn Memory & Generation",
    "test_06_e2e_lifecycle":         "6 — End-to-End Integration Lifecycle",
}


def main():
    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1
    failfast = "--failfast" in sys.argv or "-f" in sys.argv

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    print("\n" + "=" * 65)
    print("  ClassAudit AI — RAG Chatbot Test Suite")
    print("=" * 65)

    for mod_name in TEST_MODULES:
        try:
            mod = __import__(mod_name)
            tests = loader.loadTestsFromModule(mod)
            count = tests.countTestCases()
            label = DESCRIPTIONS.get(mod_name, mod_name)
            print(f"  ✓  Loaded {label} ({count} tests)")
            suite.addTests(tests)
        except Exception as exc:
            print(f"  ✗  Failed to load {mod_name}: {exc}")

    total = suite.countTestCases()
    print(f"\n  Running {total} tests total...\n" + "=" * 65 + "\n")

    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    print(f"  Results: {result.testsRun} run | "
          f"{len(result.failures)} failed | "
          f"{len(result.errors)} errors | "
          f"{len(result.skipped)} skipped")
    print("=" * 65 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

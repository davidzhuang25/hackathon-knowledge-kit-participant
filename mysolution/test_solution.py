import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from contract import Page, Snapshot  # noqa: E402
from solution import (  # noqa: E402
    DraftAnswer,
    KnowledgeAssistant,
    MarkdownChunker,
    SnapshotConflictAnalyzer,
    select_snapshot,
)


class FakeVectorStore:
    def __init__(self):
        self.calls = []
        self.added_ids = []
        self.deleted_filters = []

    def add_documents(self, documents, ids):
        self.added_ids.extend(ids)

    def delete(self, **kwargs):
        self.deleted_filters.append(kwargs["where"])

    def similarity_search(self, query, k, filter):
        self.calls.append((query, k, filter))
        return [
            Document(
                page_content="Certificate rate: 3.50% APY",
                metadata={
                    "snapshot_id": filter["snapshot_id"],
                    "snapshot_date": "2026-05-19",
                    "url": "https://example.test/certificates",
                    "title": "Certificate rates",
                    "section": "6-month certificate",
                },
            )
        ]

    def similarity_search_with_score(self, query, k, filter):
        return [(document, 0.0) for document in self.similarity_search(query, k, filter)]

    def get(self, where=None, include=None, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}


class FakeChatModel:
    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: DraftAnswer(text="3.50% APY", confident=True))


class FakeReranker:
    def rerank(self, request):
        return [{"id": passage["id"], "score": 1.0} for passage in request.passages]


class SolutionTests(unittest.TestCase):
    def test_selects_latest_full_snapshot_on_or_before_date(self):
        snapshots = {
            "01_baseline": "2026-05-12",
            "02_update": "2026-05-19",
            "03_latest": "2026-06-09",
        }

        self.assertEqual(select_snapshot(
            snapshots, "2026-05-19"), ("02_update", "2026-05-19"))
        self.assertEqual(select_snapshot(snapshots, None),
                         ("03_latest", "2026-06-09"))
        self.assertIsNone(select_snapshot(snapshots, "2026-05-01"))

    def test_chunker_keeps_page_and_heading_metadata(self):
        page = Page(
            url="https://example.test/rates",
            title="Rates",
            markdown="# Certificate rates\n\n| Term | APY |\n| --- | --- |\n| 6 months | 3.50% |",
            fetched_at="2026-05-12T00:00:00Z",
        )
        snapshot = Snapshot("01_baseline", "2026-05-12", [page])

        documents = MarkdownChunker().split(page, snapshot)

        self.assertEqual(len(documents), 1)
        self.assertIn("6 months", documents[0].page_content)
        self.assertEqual(documents[0].metadata["snapshot_id"], "01_baseline")
        self.assertEqual(documents[0].metadata["section"], "Certificate rates")

    def test_ingest_uses_unique_ids_and_clears_partial_snapshot_rows(self):
        vector_store = FakeVectorStore()
        assistant = KnowledgeAssistant(
            vector_store=vector_store,
            chat_model=FakeChatModel(),
            reranker=FakeReranker(),
        )
        page = Page(
            url="https://example.test/rates",
            title="Rates",
            markdown="# Details\n\nFirst value.\n\n# Details\n\nSecond value.",
            fetched_at="2026-05-12T00:00:00Z",
        )
        snapshot = Snapshot("01_baseline", "2026-05-12", [page])

        with tempfile.TemporaryDirectory() as temp_dir:
            assistant._state_path = Path(temp_dir) / "index_state.json"
            assistant._index_signature = "test"
            assistant.ingest(snapshot)

        self.assertEqual(len(vector_store.added_ids), 2)
        self.assertEqual(len(set(vector_store.added_ids)), 2)
        self.assertEqual(vector_store.deleted_filters, [
                         {"snapshot_id": "01_baseline"}])

    def test_answer_retrieves_from_selected_snapshot_and_returns_source(self):
        vector_store = FakeVectorStore()
        assistant = KnowledgeAssistant(
            vector_store=vector_store,
            chat_model=FakeChatModel(),
            retrieval_k=5,
            reranker=FakeReranker(),
        )
        assistant._snapshots["02_update"] = "2026-05-19"

        answer = assistant.answer("What is the certificate APY?")

        self.assertEqual(vector_store.calls[0][2], {
                         "snapshot_id": "02_update"})
        self.assertEqual(answer.text, "3.50% APY")
        self.assertEqual(answer.sources[0].url,
                         "https://example.test/certificates")

    def test_conflict_analyzer_resolves_changed_rate_to_latest_snapshot(self):
        analyzer = SnapshotConflictAnalyzer()
        for snapshot_id, taken_on, apy in (
            ("01_baseline", "2026-05-12", "2.95%"),
            ("02_update", "2026-05-19", "3.50%"),
        ):
            page = Page(
                url="https://example.test/certificates",
                title="Certificate rates",
                markdown=f"# Certificate rates\n\n| 6-month certificate APY | {apy} |",
                fetched_at=f"{taken_on}T00:00:00Z",
            )
            analyzer.ingest(Snapshot(snapshot_id, taken_on, [page]))

        conflicts = analyzer.conflicts()

        self.assertEqual(len(conflicts), 1)
        self.assertIn("2.95%", conflicts[0].values)
        self.assertIn("3.50%", conflicts[0].values)
        self.assertEqual(conflicts[0].resolution, "3.50%")
        self.assertEqual(len(conflicts[0].sources), 2)


if __name__ == "__main__":
    unittest.main()

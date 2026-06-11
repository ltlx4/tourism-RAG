from dubai_rag.evaluation import EvalCase, evaluate
from dubai_rag.models import Chunk, SearchResult


class StubRetriever:
    def search(self, query: str) -> list[SearchResult]:
        sources = ["wrong", "right"] if query == "second" else ["right", "wrong"]
        return [
            SearchResult(
                Chunk(str(index), source, source, "text", "", "test", 0), score=1 / index
            )
            for index, source in enumerate(sources, 1)
        ]


def test_evaluation_calculates_recall_and_mrr():
    metrics = evaluate(
        StubRetriever(),
        [EvalCase("first", ("right",)), EvalCase("second", ("right",))],
    )

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.75


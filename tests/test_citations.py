from dubai_rag.citations import validate_citations


def test_invalid_citations_are_removed_and_reported():
    answer, metrics = validate_citations(
        "Take the Metro for the main corridor [1]. Avoid an invented route [9].", 2
    )

    assert "[1]" in answer
    assert "[9]" not in answer
    assert metrics["invalid_source_ids"] == [9]
    assert metrics["valid"] is False


def test_missing_citations_are_not_reported_as_valid():
    _, metrics = validate_citations("This answer has no supporting references.", 3)
    assert metrics["valid"] is False
    assert metrics["cited_source_ids"] == []


def test_markdown_bullets_count_toward_coverage_and_reminder_is_repaired():
    answer, metrics = validate_citations(
        "* Take the Metro for the main corridor [1].\n\n"
        "Check changing details at the linked official sources.",
        2,
    )

    assert answer.endswith("[1].")
    assert metrics["sentence_coverage"] == 1.0


def test_unverified_raw_urls_are_removed():
    answer, _ = validate_citations(
        "Check changing details at https://invented.example/path", 1
    )
    assert "http" not in answer

from __future__ import annotations

import re


_CITATION = re.compile(r"\[(\d+)\]")
_URL = re.compile(r"https?://\S+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def validate_citations(answer: str, source_count: int) -> tuple[str, dict[str, object]]:
    cited = [int(value) for value in _CITATION.findall(answer)]
    valid = sorted({value for value in cited if 1 <= value <= source_count})
    invalid = sorted({value for value in cited if value < 1 or value > source_count})
    cleaned = _CITATION.sub(
        lambda match: match.group(0) if int(match.group(1)) in valid else "", answer
    )
    cleaned = _URL.sub("", cleaned).rstrip()
    if valid:
        lines = []
        for line in cleaned.splitlines():
            if (
                "check changing details" in line.lower()
                and not _CITATION.search(line)
                and line.strip()
            ):
                stripped = line.rstrip()
                if re.search(r"[.!?]$", stripped):
                    line = re.sub(r"([.!?])$", rf" [{valid[0]}]\1", stripped)
                else:
                    line = f"{stripped.rstrip(': ')} [{valid[0]}]."
            lines.append(line)
        cleaned = "\n".join(lines)
    factual_sentences = [
        normalized
        for sentence in _SENTENCE.split(cleaned)
        if (normalized := _MARKDOWN_PREFIX.sub("", sentence.strip()))
        and len(normalized.split()) >= 6
        and not normalized.startswith("#")
    ]
    cited_sentences = [sentence for sentence in factual_sentences if _CITATION.search(sentence)]
    coverage = len(cited_sentences) / len(factual_sentences) if factual_sentences else 1.0
    return cleaned, {
        "valid": not invalid and bool(valid),
        "cited_source_ids": valid,
        "invalid_source_ids": invalid,
        "sentence_coverage": round(coverage, 3),
    }

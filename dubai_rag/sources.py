from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "nav", "footer", "svg"}:
            self.ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "svg"} and self.ignored:
            self.ignored -= 1
        if tag in {"p", "h1", "h2", "h3", "li"} and not self.ignored:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored and data.strip():
            self.parts.append(data.strip())


def sync_sources(manifest_path: Path, output_path: Path) -> dict[str, object]:
    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path.mkdir(parents=True, exist_ok=True)
    updated = 0
    failures = []
    for source in sources:
        request = urllib.request.Request(
            source["url"],
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode(
                    response.headers.get_content_charset() or "utf-8"
                )
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append({"id": source["id"], "url": source["url"], "error": str(exc)})
            continue
        parser = _TextExtractor()
        parser.feed(raw)
        text = re.sub(r"[ \t]+", " ", " ".join(parser.parts))
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        snapshot = {
            **source,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "content": text,
        }
        (output_path / f"{source['id']}.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        updated += 1
    report = {
        "attempted": len(sources),
        "updated": updated,
        "failed": len(failures),
        "failures": failures,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    (output_path / "sync-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

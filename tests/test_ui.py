from pathlib import Path

from fastapi.testclient import TestClient

from dubai_rag.api import app


def test_home_serves_accessible_chat_interface():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert 'id="chat-form"' in response.text
    assert 'aria-live="polite"' in response.text
    assert 'aria-label="Ask about Dubai"' in response.text
    assert 'id="new-chat"' in response.text


def test_ui_is_self_contained_and_uses_safe_answer_rendering():
    html = Path("dubai_rag/static/index.html").read_text(encoding="utf-8")

    assert "fonts.googleapis.com" not in html
    assert "appendInlineContent" in html
    assert "container.textContent = \"\"" in html
    assert "data-question" in html
    assert "@media (prefers-reduced-motion: reduce)" in html

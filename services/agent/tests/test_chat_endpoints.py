"""Integration tests for the /chat and /chat/confirm endpoints.

Unlike test_graph.py (pure unit tests, no external services), these hit the
REAL S3 bucket and REAL img-proc-mcp server -- the same infrastructure the
manual curl walkthrough exercised -- and only mock the LLM's ainvoke, so the
tests need the docker-compose stack up (same requirement as img-proc-mcp's
own test suite)."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app as app_module


def make_ai_message(tool_calls=None, content=""):
    return AIMessage(content=content, tool_calls=tool_calls or [])


@pytest.fixture
def test_image_b64():
    with open("beatles.jpeg", "rb") as f:
        return base64.b64encode(f.read()).decode()


def _patch_llm(monkeypatch, responses):
    """responses: list of AIMessage returned on successive ainvoke calls."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=responses)
    monkeypatch.setattr(app_module, "llm_with_tools", mock_llm)


ROTATE_CALL = [{
    "name": "process_region", "id": "tc1",
    "args": {"tool_name": "rotate", "angle": 90, "box": None},
}]


def test_chat_new_image_whole_edit_awaits_confirmation(monkeypatch, test_image_b64):
    _patch_llm(monkeypatch, [make_ai_message(tool_calls=ROTATE_CALL)])
    with TestClient(app_module.app) as client:
        resp = client.post("/chat", json={"message": "rotate this 90 degrees", "image_base64": test_image_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["awaiting_confirmation"] is not None
    assert data["awaiting_confirmation"]["proposed"]["tool_name"] == "rotate"
    assert data["response"] is None
    assert data["thread_id"]


def test_chat_confirm_true_applies_edit(monkeypatch, test_image_b64):
    _patch_llm(monkeypatch, [
        make_ai_message(tool_calls=ROTATE_CALL),
        make_ai_message(content="Done! Rotated."),
    ])
    with TestClient(app_module.app) as client:
        r1 = client.post("/chat", json={"message": "rotate this 90 degrees", "image_base64": test_image_b64})
        thread_id = r1.json()["thread_id"]
        r2 = client.post("/chat/confirm", json={"thread_id": thread_id, "confirmed": True})
    assert r2.status_code == 200
    data = r2.json()
    assert data["awaiting_confirmation"] is None
    assert data["response"] == "Done! Rotated."
    assert data["annotated_image_base64"]  # a real edit really ran against real S3/MCP


def test_chat_confirm_false_declines_edit(monkeypatch, test_image_b64):
    _patch_llm(monkeypatch, [
        make_ai_message(tool_calls=ROTATE_CALL),
        make_ai_message(content="Okay, cancelled."),
    ])
    with TestClient(app_module.app) as client:
        r1 = client.post("/chat", json={"message": "rotate this 90 degrees", "image_base64": test_image_b64})
        thread_id = r1.json()["thread_id"]
        r2 = client.post("/chat/confirm", json={"thread_id": thread_id, "confirmed": False})
    assert r2.status_code == 200
    data = r2.json()
    assert data["awaiting_confirmation"] is None
    assert data["response"] == "Okay, cancelled."


def test_chat_missing_thread_id_and_image_is_400():
    with TestClient(app_module.app) as client:
        resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 400

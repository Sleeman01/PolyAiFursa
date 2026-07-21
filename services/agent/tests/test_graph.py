import asyncio
import json
from unittest.mock import MagicMock

import pytest

from graph import (
    keys_reducer,
    route_after_agent,
    route_after_confirm,
    build_run_detection_node,
    build_run_img_proc_node,
    build_run_undo_node,
    build_graph,
)


def run(coro):
    return asyncio.run(coro)


def make_state(**overrides):
    base = {
        "messages": [],
        "image_key": "orig.png",
        "processed_keys": [],
        "detections": None,
        "detections_for_key": None,
        "pending_edit": None,
        "tools_called": [],
    }
    base.update(overrides)
    return base


def tool_call(name, args=None, call_id="tc1"):
    return {"name": name, "args": args or {}, "id": call_id}


def msg_with_calls(calls):
    m = MagicMock()
    m.tool_calls = calls
    return m


class FakeS3:
    def __init__(self):
        self.store = {}

    def get_object(self, Bucket, Key):
        body = MagicMock()
        body.read.return_value = self.store[Key]
        return {"Body": body}

    def put_object(self, Bucket, Key, Body):
        self.store[Key] = Body


# --- keys_reducer ---

def test_keys_reducer_append():
    assert keys_reducer(["a"], ["b"]) == ["a", "b"]


def test_keys_reducer_replace_shrinks():
    assert keys_reducer(["a", "b", "c"], {"__replace__": ["a", "b"]}) == ["a", "b"]


def test_keys_reducer_replace_to_empty():
    assert keys_reducer(["a"], {"__replace__": []}) == []


# --- route_after_agent ---

def test_route_no_tool_calls_ends():
    state = make_state(messages=[msg_with_calls([])])
    assert route_after_agent(state) == "end"


def test_route_detect_objects_goes_to_detection():
    state = make_state(messages=[msg_with_calls([tool_call("detect_objects")])])
    assert route_after_agent(state) == "run_detection"


def test_route_get_detections_goes_to_detection():
    state = make_state(messages=[msg_with_calls([tool_call("get_detections")])])
    assert route_after_agent(state) == "run_detection"


def test_route_whole_image_edit_skips_detection():
    # box is None -> no detections needed, straight to confirmation
    state = make_state(messages=[msg_with_calls([tool_call("process_region", {"tool_name": "rotate", "box": None})])])
    assert route_after_agent(state) == "await_confirm"


def test_route_object_edit_without_detections_forces_detection_first():
    state = make_state(messages=[msg_with_calls([tool_call("process_region", {"tool_name": "blur", "box": [1, 2, 3, 4]})])])
    assert route_after_agent(state) == "run_detection"


def test_route_object_edit_with_stale_detections_forces_redetection():
    # detections exist but for a DIFFERENT image key -- must not reuse them
    state = make_state(
        messages=[msg_with_calls([tool_call("process_region", {"tool_name": "blur", "box": [1, 2, 3, 4]})])],
        detections={"some": "data"},
        detections_for_key="stale-key",
        processed_keys=["current-key"],
    )
    assert route_after_agent(state) == "run_detection"


def test_route_object_edit_with_valid_detections_goes_to_confirm():
    state = make_state(
        messages=[msg_with_calls([tool_call("process_region", {"tool_name": "blur", "box": [1, 2, 3, 4]})])],
        detections={"some": "data"},
        detections_for_key="current-key",
        processed_keys=["current-key"],
    )
    assert route_after_agent(state) == "await_confirm"


def test_route_undo_edit():
    state = make_state(messages=[msg_with_calls([tool_call("undo_edit")])])
    assert route_after_agent(state) == "run_undo"


def test_route_show_current_image_skips_confirmation():
    state = make_state(messages=[msg_with_calls([tool_call("show_current_image")])])
    assert route_after_agent(state) == "run_img_proc"


# --- route_after_confirm ---

def test_route_after_confirm_pending_edit_runs_img_proc():
    state = make_state(pending_edit={"name": "process_region", "args": {}})
    assert route_after_confirm(state) == "run_img_proc"


def test_route_after_confirm_declined_returns_to_agent():
    state = make_state(pending_edit=None)
    assert route_after_confirm(state) == "agent"


# --- build_run_undo_node ---

def test_undo_node_pops_last_key():
    node = build_run_undo_node()
    state = make_state(
        messages=[msg_with_calls([tool_call("undo_edit", call_id="u1")])],
        processed_keys=["k1", "k2"],
    )
    result = run(node(state))
    assert result["processed_keys"] == {"__replace__": ["k1"]}
    assert result["tools_called"] == ["undo_edit"]
    assert len(result["messages"]) == 1


def test_undo_node_noop_when_nothing_to_undo():
    node = build_run_undo_node()
    state = make_state(
        messages=[msg_with_calls([tool_call("undo_edit", call_id="u1")])],
        processed_keys=[],
    )
    result = run(node(state))
    assert result["processed_keys"] == {"__replace__": []}
    parsed = json.loads(result["messages"][0].content)
    assert "no edit to undo" in parsed["message"]


# --- build_run_img_proc_node (S3 bridging) ---

def test_img_proc_node_uploads_result_and_appends_key():
    s3 = FakeS3()
    s3.store["orig.png"] = b"fake-original-bytes"

    async def fake_ainvoke(tc):
        holder["result_image_b64"] = "ZmFrZS1yZXN1bHQ="  # base64 for b"fake-result"
        return MagicMock(content="ok")

    process_region_tool = MagicMock(name="process_region")
    process_region_tool.name = "process_region"
    process_region_tool.ainvoke = fake_ainvoke

    show_tool = MagicMock(name="show_current_image")
    show_tool.name = "show_current_image"

    holder = {}
    from contextvars import ContextVar
    current_image_var = ContextVar("img", default=None)

    node = build_run_img_proc_node(process_region_tool, show_tool, s3, "bucket", current_image_var, lambda: holder)
    state = make_state(
        messages=[msg_with_calls([tool_call("process_region", call_id="p1")])],
        processed_keys=[],
    )
    result = run(node(state))

    assert len(result["processed_keys"]) == 1
    new_key = result["processed_keys"][0]
    assert s3.store[new_key] == b"fake-result"
    assert current_image_var.get() is None  # reset after the node runs


# --- build_graph wiring ---

def test_build_graph_compiles_with_all_nodes():
    tools = {k: MagicMock(name=k) for k in ["detect_objects", "get_detections", "process_region", "show_current_image"]}
    for key, m in tools.items():
        m.name = key
    from contextvars import ContextVar
    app = build_graph(
        MagicMock(), "system prompt", tools,
        MagicMock(), "bucket", ContextVar("v", default=None), lambda: {},
    )
    nodes = set(app.get_graph().nodes.keys())
    assert nodes == {"__start__", "agent", "run_detection", "run_img_proc", "await_confirm", "run_undo", "__end__"}

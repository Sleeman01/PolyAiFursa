import base64
import json
import uuid
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph.message import add_messages
import operator

MAX_NUDGES = 2


def keys_reducer(current: list, update):
    """Normal edits append (pass a list). Undo needs to shrink the list
    instead, which operator.add can't express -- pass {'__replace__': [...]}
    to fully replace processed_keys rather than concatenate onto it."""
    if isinstance(update, dict) and "__replace__" in update:
        return update["__replace__"]
    return current + update


class VisionState(TypedDict):
    messages: Annotated[list, add_messages]
    image_key: Optional[str]
    processed_keys: Annotated[list, keys_reducer]
    detections: Optional[dict]
    detections_for_key: Optional[str]
    pending_edit: Optional[dict]
    tools_called: Annotated[list, operator.add]
    # Reset to False/0 at the start of each NEW /chat turn by the caller.
    # NOT reset on /chat/confirm resume, since that's mid-turn continuation --
    # a tool call already happened earlier this turn (the one now pending
    # confirmation), so the eventual text answer after confirm/decline is a
    # legitimate completion, not a skipped tool call.
    tool_called_this_turn: bool
    nudge_count: int


def _current_key(state: VisionState) -> str:
    return state["processed_keys"][-1] if state["processed_keys"] else state["image_key"]


def _download_b64(s3_client, bucket, key) -> str:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return base64.b64encode(obj["Body"].read()).decode()


def _upload_b64(s3_client, bucket, key, b64_data):
    s3_client.put_object(Bucket=bucket, Key=key, Body=base64.b64decode(b64_data))


def build_agent_node(llm_with_tools, system_prompt):
    async def agent_node(state: VisionState) -> dict:
        messages = [system_prompt] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        update = {"messages": [response]}
        # Requesting a tool -- even one later gated behind await_confirm --
        # counts as the agent having done its job this turn. This must be
        # set here (not in a downstream node) because a tool-less response
        # never reaches any downstream node at all.
        if getattr(response, "tool_calls", None):
            update["tool_called_this_turn"] = True
        return update
    return agent_node


def build_nudge_node():
    """Structural replacement for the old app.py nudge-loop: instead of the
    HTTP handler manually retrying with a corrective HumanMessage, the graph
    itself routes here when the LLM answers in plain text without having
    called any tool yet this turn, and loops back to agent."""
    async def nudge_node(state: VisionState) -> dict:
        return {
            "messages": [HumanMessage(content=(
                "Call the correct tool now for the previous user message. "
                "Do not explain, apologize, or describe what you are about "
                "to do -- just call the tool."
            ))],
            "nudge_count": state.get("nudge_count", 0) + 1,
        }
    return nudge_node


def build_run_detection_node(detect_objects_tool, get_detections_tool, s3_client, bucket, current_image_var):
    async def run_detection_node(state: VisionState) -> dict:
        last = state["messages"][-1]
        current_key = _current_key(state)
        # No image was ever uploaded this conversation (current_key is None) --
        # don't attempt an S3 download with a None key. Leave the ContextVar
        # unset so the underlying tool's own "No image was provided" check
        # returns a graceful error instead of crashing on a bad S3 call.
        image_b64 = _download_b64(s3_client, bucket, current_key) if current_key else None
        token = current_image_var.set(image_b64)
        results, detections, called = [], None, []
        tool_map = {detect_objects_tool.name: detect_objects_tool, get_detections_tool.name: get_detections_tool}
        try:
            for tc in last.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    continue
                raw = await tool_fn.ainvoke(tc)
                content = raw.content if hasattr(raw, "content") else str(raw)
                results.append(raw)
                called.append(tc["name"])
                try:
                    parsed = json.loads(content)
                    if "detections" in parsed:
                        detections = parsed["detections"]
                    elif "error" not in parsed:
                        detections = parsed
                except Exception:
                    pass
        finally:
            current_image_var.reset(token)
        update = {"messages": results, "tools_called": called}
        if detections is not None:
            update["detections"] = detections
            update["detections_for_key"] = current_key
        return update
    return run_detection_node


def build_run_img_proc_node(process_region_tool, show_current_image_tool, s3_client, bucket, current_image_var, get_holder_fn):
    async def run_img_proc_node(state: VisionState) -> dict:
        last = state["messages"][-1]
        current_key = _current_key(state)
        image_b64 = _download_b64(s3_client, bucket, current_key) if current_key else None
        token = current_image_var.set(image_b64)
        tool_map = {
            process_region_tool.name: process_region_tool,
            show_current_image_tool.name: show_current_image_tool,
        }
        results, called, new_key = [], [], None
        try:
            for tc in last.tool_calls:
                tool_fn = tool_map.get(tc["name"])
                if tool_fn is None:
                    continue
                raw = await tool_fn.ainvoke(tc)
                results.append(raw)
                called.append(tc["name"])
                holder = get_holder_fn()
                result_b64 = holder.get("result_image_b64")
                if result_b64:
                    candidate_key = f"{state['image_key']}.processed.{len(state['processed_keys'])}.{uuid.uuid4().hex[:8]}.png"
                    _upload_b64(s3_client, bucket, candidate_key, result_b64)
                    new_key = candidate_key
        finally:
            current_image_var.reset(token)
        update = {"messages": results, "tools_called": called, "pending_edit": None}
        if new_key:
            update["processed_keys"] = [new_key]
        return update
    return run_img_proc_node


def build_run_undo_node():
    """Non-destructive: pops the last processed_keys entry without deleting
    anything from S3. If there's nothing to undo (still on the original
    image), it's a no-op with an explanatory message."""
    async def run_undo_node(state: VisionState) -> dict:
        last = state["messages"][-1]
        tc = next((t for t in last.tool_calls if t["name"] == "undo_edit"), None)
        keys = state.get("processed_keys") or []
        if keys:
            popped = keys[:-1]
            msg = "Reverted to the previous version."
        else:
            popped = keys
            msg = "There's no edit to undo -- this is already the original image."
        results = []
        if tc:
            results.append(ToolMessage(content=json.dumps({"status": "ok", "message": msg}), tool_call_id=tc["id"]))
        return {
            "messages": results,
            "tools_called": ["undo_edit"],
            "processed_keys": {"__replace__": popped},
        }
    return run_undo_node


def build_await_confirm_node():
    from langgraph.types import interrupt

    async def await_confirm_node(state: VisionState) -> dict:
        last = state["messages"][-1]
        tc = next(t for t in last.tool_calls if t["name"] == "process_region")
        pending = {"tool_call_id": tc["id"], "name": tc["name"], "args": tc["args"]}
        decision = interrupt({
            "type": "confirm_edit",
            "proposed": pending["args"],
            "message": "Apply this edit?",
        })
        return {"pending_edit": pending, "tools_called": ["await_confirm"]} if decision else {"pending_edit": None}
    return await_confirm_node


def route_after_agent(state: VisionState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        if state.get("tool_called_this_turn"):
            return "end"
        if state.get("nudge_count", 0) < MAX_NUDGES:
            return "nudge"
        return "end"  # exhausted nudges -- accept whatever plain-text answer it gave

    requested = {tc["name"] for tc in last.tool_calls}
    current_key = _current_key(state)
    if requested & {"detect_objects", "get_detections"}:
        return "run_detection"
    if "process_region" in requested:
        for tc in last.tool_calls:
            if tc["name"] != "process_region":
                continue
            box = tc.get("args", {}).get("box")
            has_valid_detections = state.get("detections_for_key") == current_key and state.get("detections")
            if box is not None and not has_valid_detections:
                return "run_detection"
        return "await_confirm"
    if "undo_edit" in requested:
        return "run_undo"
    if "show_current_image" in requested:
        return "run_img_proc"
    return "end"


def route_after_confirm(state: VisionState) -> str:
    return "run_img_proc" if state.get("pending_edit") else "agent"


def build_graph(llm_with_tools, system_prompt, tools_by_name, s3_client, bucket, current_image_var, get_holder_fn, checkpointer=None):
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(VisionState)
    g.add_node("agent", build_agent_node(llm_with_tools, system_prompt))
    g.add_node("nudge", build_nudge_node())
    g.add_node("run_detection", build_run_detection_node(
        tools_by_name["detect_objects"], tools_by_name["get_detections"], s3_client, bucket, current_image_var))
    g.add_node("run_img_proc", build_run_img_proc_node(
        tools_by_name["process_region"], tools_by_name["show_current_image"], s3_client, bucket, current_image_var, get_holder_fn))
    g.add_node("await_confirm", build_await_confirm_node())
    g.add_node("run_undo", build_run_undo_node())

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {
        "nudge": "nudge",
        "run_detection": "run_detection",
        "await_confirm": "await_confirm",
        "run_img_proc": "run_img_proc",
        "run_undo": "run_undo",
        "end": END,
    })
    g.add_conditional_edges("await_confirm", route_after_confirm, {
        "run_img_proc": "run_img_proc",
        "agent": "agent",
    })
    g.add_edge("nudge", "agent")
    g.add_edge("run_detection", "agent")
    g.add_edge("run_img_proc", "agent")
    g.add_edge("run_undo", "agent")

    return g.compile(checkpointer=checkpointer)

import base64
import io
import json
import logging
import os
import uuid
import boto3
from contextvars import ContextVar
from typing import Optional
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MCP_SERVICE_URL = os.environ.get("MCP_SERVICE_URL", "http://localhost:9000/mcp")
MODEL = os.environ.get("MODEL")

# S3 configuration from environment variables (never hard-code)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Text-only models
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
    "google_genai:gemini-2.5-flash",
    "google_genai:gemini-2.5-flash-lite",
    "bedrock_converse:amazon.nova-lite-v1:0",
    "bedrock_converse:us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant that analyzes and edits uploaded images. "
    "Every user message describing a new edit (blur, rotate, flip, noise) is a "
    "NEW request that requires a NEW call to process_region — even if a previous "
    "turn already did something similar. Never respond with confirmation text "
    "without first calling the appropriate tool for the CURRENT message. "
    "For whole-image transformations such as blur, rotate, flip, or noise, "
    "call process_region directly and omit the box argument. "
    "Use detect_objects or get_detections only when the user asks about objects "
    "or requests an edit to a specific object or region. "
    "When the user refers to an object by position (e.g. 'leftmost', 'middle', "
    "'second dog', 'the one on the right'), you MUST match their wording against "
    "the 'position' field already computed and returned by get_detections — do "
    "NOT estimate position yourself from the raw box coordinates, the "
    "pre-computed 'position' field is authoritative and correct. "
    "If the user asks to see, view, or show the current image, or says they "
    "don't see an image, call show_current_image — do not just claim you've "
    "shown it in text. "
    "After process_region or show_current_image succeeds for THIS message, the "
    "task is complete. Do not call more tools."
)

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_current_chat_id: ContextVar[Optional[str]] = ContextVar("current_chat_id", default=None)

# Per-request prediction/result state. This used to be a plain module-level dict,
# which is shared by every concurrent request on this process -- if two /chat
# calls overlapped (e.g. a double-submit), one request's cleanup could wipe out
# another in-flight request's result right before it was read. A ContextVar
# gives each request (each asyncio task) its own isolated dict instead.
_prediction_holder_var: ContextVar[Optional[dict]] = ContextVar("prediction_holder", default=None)


def _get_holder() -> dict:
    """Get this request's prediction holder dict, creating one if it doesn't exist yet."""
    holder = _prediction_holder_var.get()
    if holder is None:
        holder = {}
        _prediction_holder_var.set(holder)
    return holder


def _parse_box(box_val):
    """Yolo returns 'box' as a JSON-encoded STRING like '[x1, y1, x2, y2]',
    not an actual list. Sorting/indexing that string directly (e.g. box[0])
    grabs a character, not a number -- which was the root cause of the
    left/middle/right object-selection bug. Parse it into a real list here
    so every downstream consumer (sorting, process_region, the model) gets
    genuine numbers."""
    if isinstance(box_val, str):
        try:
            box_val = json.loads(box_val)
        except Exception:
            import ast as _ast
            try:
                box_val = _ast.literal_eval(box_val)
            except Exception:
                return None
    return box_val


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    if not AWS_S3_BUCKET:
        return json.dumps({"error": "AWS_S3_BUCKET env var is not set."})

    image_bytes = base64.b64decode(image_b64)

    # Build an S3 key organised by chat and prediction id
    chat_id = _current_chat_id.get() or "unknown-chat"
    prediction_id = str(uuid.uuid4())
    original_key = f"{chat_id}/{prediction_id}/original/image.jpg"

    # Upload the original image to S3
    s3_client.upload_fileobj(io.BytesIO(image_bytes), AWS_S3_BUCKET, original_key)

    # Call Yolo with ONLY the S3 key (not the image bytes)
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": original_key},
        )
        response.raise_for_status()
    data = response.json()
    if data.get("prediction_uid"):
        _get_holder()["uid"] = data["prediction_uid"]
    return json.dumps(data)


@tool
def get_detections() -> str:
    """Return the list of detected objects (labels and bounding boxes) for the user's current image,
    using YOLO. Each detection has: index, label, and box [x1, y1, x2, y2] in pixel coordinates.
    Call this before doing object-specific edits so you know where each object is."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})
    if not AWS_S3_BUCKET:
        return json.dumps({"error": "AWS_S3_BUCKET env var is not set."})

    image_bytes = base64.b64decode(image_b64)
    chat_id = _current_chat_id.get() or "unknown-chat"
    prediction_id = str(uuid.uuid4())
    original_key = f"{chat_id}/{prediction_id}/original/image.jpg"
    s3_client.upload_fileobj(io.BytesIO(image_bytes), AWS_S3_BUCKET, original_key)

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{YOLO_SERVICE_URL}/predict", json={"image_s3_key": original_key})
        response.raise_for_status()
    data = response.json()
    uid = data.get("prediction_uid")
    if uid:
        _get_holder()["uid"] = uid

    detections = []
    if uid:
        with httpx.Client(timeout=30.0) as client:
            det = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}")
            if det.status_code == 200:
                objs = det.json().get("detection_objects", [])
                logging.info(f"RAW YOLO DETECTIONS: {json.dumps(objs)}")
                # Sort left-to-right by the box's left edge (x1) so position labels are reliable.
                # _parse_box handles Yolo returning box as a JSON-encoded string.
                objs_sorted = sorted(objs, key=lambda o: (_parse_box(o.get("box")) or [0])[0])
                n = len(objs_sorted)
                for i, o in enumerate(objs_sorted):
                    if n == 1:
                        position = "only"
                    elif i == 0:
                        position = "leftmost"
                    elif i == n - 1:
                        position = "rightmost"
                    elif n == 3 and i == 1:
                        position = "middle"
                    else:
                        position = f"position {i+1} from the left"
                    detections.append({"index": i, "label": o.get("label"), "box": _parse_box(o.get("box")), "position": position})
    return json.dumps({"detections": detections})


async def _apply_mcp_transform(tool_name: str, region_b64: str, params: dict) -> str:
    """Call a single MCP image tool by name on a base64 region, return transformed base64."""
    client = MultiServerMCPClient({
        "img-proc": {"url": MCP_SERVICE_URL, "transport": "streamable_http"}
    })
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}
    if tool_name not in tool_map:
        raise ValueError(f"Unknown image tool: {tool_name}")
    args = {"image_b64": region_b64}
    args.update(params)
    result = await tool_map[tool_name].ainvoke(args)

    # MCP tool results may come back as a plain string, or as a list of
    # content blocks (dicts / objects with a .text). Normalise to a string.
    def _extract_text(r):
        if r is None:
            return ""

        if isinstance(r, str):
            return r

        # LangChain messages commonly store tool output in .content.
        if hasattr(r, "content"):
            return _extract_text(r.content)

        if isinstance(r, (list, tuple)):
            return "".join(_extract_text(block) for block in r)

        if isinstance(r, dict):
            if "text" in r:
                return _extract_text(r["text"])
            if "content" in r:
                return _extract_text(r["content"])

        if hasattr(r, "text"):
            return _extract_text(r.text)

        return str(r)

    transformed = _extract_text(result).strip()

    # Support base64 data URLs if a tool ever returns one.
    if transformed.startswith("data:image/") and "," in transformed:
        transformed = transformed.split(",", 1)[1]

    return transformed


@tool
async def process_region(tool_name: str, box: list | str | None = None, blur_radius: float = 3.0, angle: float = 90.0, direction: str = "horizontal") -> str:
    """Apply an image transformation to a specific region of the user's image and composite it back.
    - tool_name: one of "blur", "rotate", "flip", "add_noise"
    - box: optional [x1, y1, x2, y2] coordinates; omit it to process the entire image
    - blur_radius: used when tool_name is "blur" (default 3.0)
    - angle: used when tool_name is "rotate". Positive angle = counter-clockwise
      ("rotate left"), negative angle = clockwise ("rotate right"). For example,
      "rotate 90 degrees left" means angle=90, and "rotate 90 degrees right" means
      angle=-90. To undo a previous rotation, use the exact opposite sign of the
      angle that was originally applied.
    - direction: used when tool_name is "flip", either "horizontal" or "vertical"
    Returns a short status; the edited image is sent back to the user automatically."""
    import base64 as _b64
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})
    # Build params for the chosen tool
    if tool_name == "blur":
        params = {"radius": blur_radius}
    elif tool_name == "rotate":
        params = {"angle": angle}
    elif tool_name == "flip":
        params = {"direction": direction}
    else:
        params = {}

    from PIL import Image

    full = Image.open(io.BytesIO(_b64.b64decode(image_b64))).convert("RGB")

    # No box means apply the transformation to the entire image.
    if box is None:
        x1, y1 = 0, 0
        x2, y2 = full.size
    else:
        # box may arrive as a list or as a string like "[145, 88, 210, 300]"
        if isinstance(box, str):
            import ast as _ast
            try:
                box = _ast.literal_eval(box)
            except Exception:
                box = [float(x) for x in box.strip("[]() ").split(",")]

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return json.dumps({
                "error": "box must contain four values: [x1, y1, x2, y2]"
            })

        x1, y1, x2, y2 = [int(float(v)) for v in box]

        # Keep the region inside the image boundaries.
        width, height = full.size
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return json.dumps({"error": "The provided box is invalid."})

    region = full.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    region.save(buf, format="PNG")
    region_b64 = _b64.b64encode(buf.getvalue()).decode()

    transformed_b64 = await _apply_mcp_transform(tool_name, region_b64, params)
    transformed = Image.open(io.BytesIO(_b64.b64decode(transformed_b64))).convert("RGB")

    if box is None:
        # Whole-image edit: the transformed image IS the new full image.
        # Don't force it back into the old (pre-rotation) canvas size --
        # that's what was squashing/distorting every whole-image rotate.
        full = transformed
    else:
        # Region edit: the surrounding canvas must stay the same size,
        # so resize the transformed region back to fit exactly where it came from.
        transformed = transformed.resize((x2 - x1, y2 - y1))
        full.paste(transformed, (x1, y1))

    out = io.BytesIO()
    full.save(out, format="PNG")
    result_b64 = _b64.b64encode(out.getvalue()).decode()
    _get_holder()["result_image_b64"] = result_b64
    # Do NOT return the base64 image to the LLM (it blows the token limit).
    # The image is stored and returned to the user by the /chat handler.
    return json.dumps({"status": "ok", "message": f"Applied {tool_name} to the region and produced the edited image."})

@tool
def show_current_image() -> str:
    """Return the user's current image as-is, with no transformation applied.
    Call this when the user asks to see, view, or show the current image
    (e.g. "show me the image", "I don't see it", "can I see the result")
    rather than requesting a new edit."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})
    _get_holder()["result_image_b64"] = image_b64
    return json.dumps({"status": "ok", "message": "Displaying the current image."})


# Local tools the agent always has
_LOCAL_TOOLS = [detect_objects, get_detections, process_region, show_current_image]

# Load the image-processing tools from the MCP server once at startup
def _load_mcp_tools():
    try:
        client = MultiServerMCPClient({
            "img-proc": {"url": MCP_SERVICE_URL, "transport": "streamable_http"}
        })
        return asyncio.run(client.get_tools())
    except Exception as e:
        logging.warning(f"Could not load MCP tools from {MCP_SERVICE_URL}: {e}")
        return []

_mcp_tools = _load_mcp_tools()

# MCP tools are used internally by process_region.
# Do not expose the low-level tools directly to the language model.
_all_tools = _LOCAL_TOOLS

# Registry: map tool name -> tool object (for execution in the loop)
TOOLS = {t.name: t for t in _all_tools}
logging.info(f"Agent tools available: {list(TOOLS.keys())}")

_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.16,
    check_every_n_seconds=0.1,
    max_bucket_size=3,
)

llm = init_chat_model(MODEL, temperature=0, rate_limiter=_rate_limiter)
llm_with_tools = llm.bind_tools(_all_tools)

async def run_agent(history: list, max_iterations: int = 10) -> str:
    """
    Simple ReAct loop (async, so it can call async MCP tools):
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    nudge_count = 0
    max_nudges = 2

    for _ in range(max_iterations):
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if nudge_count < max_nudges:
                # Known failure mode: a small model, deep in a repetitive
                # conversation, sometimes just repeats a prior confirmation
                # phrase in plain text instead of actually calling a tool.
                # Give it an explicit corrective nudge before accepting
                # a tool-less answer.
                nudge_count += 1
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=(
                    "You responded without calling any tool. If the previous "
                    "user message requested an image edit (blur, rotate, flip, "
                    "noise) or asked to see/view the image, you must call "
                    "process_region or show_current_image now instead of just "
                    "repeating a confirmation in text."
                )))
                continue
            return content

        # Execute every tool the model requested (ainvoke works for both sync and async tools)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_fn = TOOLS.get(tool_name)

            if tool_fn is None:
                return f"Tool {tool_name} is not available."

            tool_result = await tool_fn.ainvoke(tool_call)
            messages.append(tool_result)

            # The edited image is already stored for the /chat response.
            # Do not ask the model to perform more unnecessary steps.
            if (
                tool_name in ("process_region", "show_current_image")
                and _get_holder().get("result_image_b64")
            ):
                return "Here you go."

    # Hit the iteration cap without a final answer
    return "Sorry, I couldn't complete the request within the allowed number of steps."


app = FastAPI(title="Vision Agent")

_cors_origins = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    annotated_image_base64: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64          # saved for detect_objects tool
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    chat_id = str(uuid.uuid4())
    token = _current_image_b64.set(latest_image)
    chat_token = _current_chat_id.set(chat_id)
    # Give this request its own fresh, isolated prediction holder -- this is what
    # actually fixes the race: even if another request is in flight concurrently,
    # it has its own separate dict via its own ContextVar token, so neither can
    # clobber the other's result.
    holder_token = _prediction_holder_var.set({})
    try:
        answer = await run_agent(lc_messages)
        annotated = None
        # If an image edit (process_region) produced a result, return that.
        edited = _get_holder().get("result_image_b64")
        if edited:
            annotated = edited
        else:
            uid = _get_holder().get("uid")
            if uid:
                try:
                    with httpx.Client(timeout=30.0) as client:
                        img_resp = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}/image")
                        img_resp.raise_for_status()
                    annotated = base64.b64encode(img_resp.content).decode("utf-8")
                except Exception:
                    annotated = None
        return ChatResponse(response=answer, annotated_image_base64=annotated)
    finally:
        _current_image_b64.reset(token)
        _current_chat_id.reset(chat_token)
        _prediction_holder_var.reset(holder_token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
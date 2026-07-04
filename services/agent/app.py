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
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_current_chat_id: ContextVar[Optional[str]] = ContextVar("current_chat_id", default=None)
_prediction_holder: dict = {}
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
        _prediction_holder["uid"] = data["prediction_uid"]
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
        _prediction_holder["uid"] = uid

    detections = []
    if uid:
        with httpx.Client(timeout=30.0) as client:
            det = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}")
            if det.status_code == 200:
                objs = det.json().get("detection_objects", [])
                for i, o in enumerate(objs):
                    detections.append({"index": i, "label": o.get("label"), "box": o.get("box")})
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
        if isinstance(r, str):
            return r
        if isinstance(r, list):
            parts = []
            for block in r:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "".join(parts)
        if isinstance(r, dict):
            return r.get("text", "")
        if hasattr(r, "text"):
            return r.text
        return str(r)

    return _extract_text(result)


@tool
async def process_region(tool_name: str, box: list, params: dict = None) -> str:
    """Apply an image transformation to a specific region of the user's current image and composite it back.
    - tool_name: one of rotate, flip, blur, resize, crop, add_noise
    - box: [x1, y1, x2, y2] pixel coordinates of the region (e.g. from get_detections)
    - params: extra parameters for the tool (e.g. {"radius": 5} for blur, {"angle": 90} for rotate)
    Returns a JSON object with the resulting full image as base64 under 'image_b64'."""
    import base64 as _b64
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})
    params = params or {}

    from PIL import Image

    # box may arrive as a real list, or as a string like "[145, 88, 210, 300]"
    if isinstance(box, str):
        import ast as _ast
        try:
            box = _ast.literal_eval(box)
        except Exception:
            box = [float(x) for x in box.strip("[]() ").split(",")]

    full = Image.open(io.BytesIO(_b64.b64decode(image_b64))).convert("RGB")
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    region = full.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    region.save(buf, format="PNG")
    region_b64 = _b64.b64encode(buf.getvalue()).decode()

    transformed_b64 = await _apply_mcp_transform(tool_name, region_b64, params)
    transformed = Image.open(io.BytesIO(_b64.b64decode(transformed_b64))).convert("RGB")

    transformed = transformed.resize((x2 - x1, y2 - y1))
    full.paste(transformed, (x1, y1))

    out = io.BytesIO()
    full.save(out, format="PNG")
    result_b64 = _b64.b64encode(out.getvalue()).decode()
    _prediction_holder["result_image_b64"] = result_b64
    return json.dumps({"status": "ok", "image_b64": result_b64})


# Local tools the agent always has
_LOCAL_TOOLS = [detect_objects, get_detections, process_region]

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
_all_tools = _LOCAL_TOOLS + _mcp_tools

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
            return content

        # Execute every tool the model requested (ainvoke works for both sync and async tools)
        for tool_call in response.tool_calls:
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = await tool_fn.ainvoke(tool_call)   # returns a ToolMessage
            messages.append(tool_result)

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
    _prediction_holder.pop("uid", None)
    try:
        answer = await run_agent(lc_messages)
        annotated = None
        uid = _prediction_holder.get("uid")
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
        _prediction_holder.pop("uid", None)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

import base64
import io
import json
import logging
import os
import uuid
import boto3
from contextvars import ContextVar
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MODEL = os.environ.get("MODEL")

# S3 configuration from environment variables (never hard-code)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Text-only models
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
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
    image_name = "image.jpg"
    original_key = f"{chat_id}/{prediction_id}/original/{image_name}"

    # Upload the original image to S3
    s3_client.upload_fileobj(io.BytesIO(image_bytes), AWS_S3_BUCKET, original_key)

    # Call Yolo with ONLY the S3 key (not the image bytes)
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": original_key},
        )
        response.raise_for_status()
    return json.dumps(response.json())


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects
}

llm = init_chat_model(MODEL, temperature=0)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

def run_agent(history: list) -> str:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    while True:
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            # Some Bedrock models (e.g. Nova) return content as a list of
            # blocks rather than a plain string. Normalise to text.
            content = response.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" or "text" in block:
                            parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return content

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = tool_fn.invoke(tool_call)          # returns a ToolMessage
            messages.append(tool_result)


app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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
    try:
        return ChatResponse(response=run_agent(lc_messages))
    finally:
        _current_image_b64.reset(token)
        _current_chat_id.reset(chat_token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

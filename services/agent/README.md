# Vision Agent

A LangChain-powered AI vision agent with a manual ReAct loop. Accepts text and base64-encoded images, and can call tools (e.g. YOLO object detection) to answer questions.

## Prerequisites

- Python 3.10+
- A running YOLO service (optional - only needed for `detect_objects`)


## Setup

Install dependencies (from `services/agent/`):

```bash
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Edit .env and set at least OPENAI_API_KEY (or another provider key) and MODEL
```

`.env` variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | - | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | - | Required for Anthropic models |
| `GOOGLE_API_KEY` | - | Required for Google models |
| `MODEL` | `claude-sonnet-4-6` | Any model string supported by `init_chat_model` |
| `YOLO_SERVICE_URL` | `http://localhost:8080` | URL of the YOLO microservice |

## Running

```bash
cd services/agent
python app.py
```

The server starts at `http://localhost:8000`.

## Testing with curl

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### Plain text message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you do?"}'
```

### Send a message with an image

```bash
echo "{\"message\": \"What objects are in this image?\", \"image_base64\": \"$(base64 -w0 beatles.jpeg)\"}" \
  | curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d @-
```

## API Reference

### `POST /chat`

Request body:

```json
{
  "message": "string (optional, defaults to 'What's in this image?')",
  "image_base64": "string (optional, base64-encoded JPEG or PNG)"
}
```

Response:

```json
{
  "response": "string"
}
```

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
import sqlite3
import logging
import os
import uuid
import shutil
import time
import boto3
import tempfile

# S3 configuration from environment variables (never hard-code)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:  # pragma: no cover
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)")
else:
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")


def save_prediction_session(uid, original_image, predicted_image):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict")
def predict(request: dict):
    start_time = time.time()

    image_s3_key = request.get("image_s3_key")
    if not image_s3_key:
        raise HTTPException(status_code=400, detail="image_s3_key is required")
    if not AWS_S3_BUCKET:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET env var is not set")

    uid = str(uuid.uuid4())
    ext = os.path.splitext(image_s3_key)[1] or ".jpg"
    image_name = os.path.basename(image_s3_key)

    # Download the original image from S3
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    s3_client.download_file(AWS_S3_BUCKET, image_s3_key, original_path)

    results = model(original_path, device="cpu")

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
    annotated_image.save(predicted_path)

    # Upload the predicted image to S3, mirroring the original key path
    if "/original/" in image_s3_key:
        predicted_s3_key = image_s3_key.replace("/original/", "/predicted/", 1)
    else:
        predicted_s3_key = f"predicted/{uid}{ext}"
    s3_client.upload_file(predicted_path, AWS_S3_BUCKET, predicted_s3_key)

    save_prediction_session(uid, image_s3_key, predicted_s3_key)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return {
         "prediction_uid": uid,
         "detection_count": len(results[0].boxes),
         "labels": detected_labels,
         "time_took": processing_time,
         "image_s3_key": image_s3_key,
         "predicted_s3_key": predicted_s3_key
     }
@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)
        ).fetchone()
    if not row or not os.path.exists(row[0]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])



@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):

    if not label.strip():
        raise HTTPException(
            status_code=400,
            detail="Label cannot be empty"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        sessions = conn.execute(
            """
            SELECT DISTINCT ps.*
            FROM prediction_sessions ps
            JOIN detection_objects d
            ON ps.uid = d.prediction_uid
            WHERE d.label = ?
            """,
            (label,)
        ).fetchall()

        results = []

        for session in sessions:

            objects = conn.execute(
                """
                SELECT *
                FROM detection_objects
                WHERE prediction_uid = ? AND label = ?
                """,
                (session["uid"], label)
            ).fetchall()

            results.append({
                "uid": session["uid"],
                "timestamp": session["timestamp"],
                "detection_objects": [
                    {
                        "id": obj["id"],
                        "label": obj["label"],
                        "score": obj["score"],
                        "box": obj["box"]
                    }
                    for obj in objects
                ]
            })

        return results
@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):

    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        objects = conn.execute(
            """
            SELECT *
            FROM detection_objects
            WHERE score >= ?
            """,
            (min_score,)
        ).fetchall()

        return [
            {
                "id": obj["id"],
                "prediction_uid": obj["prediction_uid"],
                "label": obj["label"],
                "score": obj["score"],
                "box": obj["box"]
            }
            for obj in objects
        ]



@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    init_db()
    
    uvicorn.run(app, host="0.0.0.0", port=8080)

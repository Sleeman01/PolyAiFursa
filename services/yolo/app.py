from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
from sqlalchemy.orm import Session
import logging
import os
import uuid
import shutil
import time
from datetime import datetime

from db import get_db, init_db
from models import PredictionSession, DetectionObject
import boto3

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

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")


@app.post("/predict")
def predict(request: dict, db: Session = Depends(get_db)):
    start_time = time.time()

    image_s3_key = request.get("image_s3_key")
    if not image_s3_key:
        raise HTTPException(status_code=400, detail="image_s3_key is required")
    if not AWS_S3_BUCKET:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET env var is not set")

    uid = str(uuid.uuid4())
    ext = os.path.splitext(image_s3_key)[1] or ".jpg"

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

    db.add(PredictionSession(uid=uid, original_image=image_s3_key, predicted_image=predicted_s3_key))

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        db.add(DetectionObject(prediction_uid=uid, label=label, score=score, box=str(bbox)))
        detected_labels.append(label)

    db.commit()

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
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = db.query(DetectionObject).filter_by(prediction_uid=uid).all()

    return {
        "uid": session.uid,
        "timestamp": session.timestamp,
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box
            } for obj in objects
        ]
    }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Image not found")

    # predicted_image is now an S3 key; download it and stream the bytes back
    if not AWS_S3_BUCKET:
        raise HTTPException(status_code=500, detail="AWS_S3_BUCKET env var is not set")
    try:
        local_path = os.path.join(PREDICTED_DIR, f"dl_{uid}.jpg")
        s3_client.download_file(AWS_S3_BUCKET, session.predicted_image, local_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found in S3")
    return FileResponse(local_path)


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):

    if not label.strip():
        raise HTTPException(
            status_code=400,
            detail="Label cannot be empty"
        )

    sessions = (
        db.query(PredictionSession)
        .join(DetectionObject, PredictionSession.uid == DetectionObject.prediction_uid)
        .filter(DetectionObject.label == label)
        .distinct()
        .all()
    )

    results = []
    for session in sessions:
        objects = (
            db.query(DetectionObject)
            .filter_by(prediction_uid=session.uid, label=label)
            .all()
        )
        results.append({
            "uid": session.uid,
            "timestamp": session.timestamp,
            "detection_objects": [
                {
                    "id": o.id,
                    "label": o.label,
                    "score": o.score,
                    "box": o.box
                } for o in objects
            ],
        })

    return results

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float, db: Session = Depends(get_db)):

    if min_score < 0.0 or min_score > 1.0:
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0"
        )

    objects = db.query(DetectionObject).filter(DetectionObject.score >= min_score).all()

    return [
        {
            "id": obj.id,
            "prediction_uid": obj.prediction_uid,
            "label": obj.label,
            "score": obj.score,
            "box": obj.box
        }
        for obj in objects
    ]

@app.on_event("shutdown")
def graceful_shutdown():
    logging.info("Received SIGTERM, shutting down YOLO service gracefully...")

@app.on_event("shutdown")
def graceful_shutdown():
    logging.info("Received SIGTERM, shutting down YOLO service gracefully...")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/welcoming")
def welcoming():
    return {"message": "welcome"}

@app.get("/morning")
def morning():
    return {"message": "Good morning"}


@app.get("/onePlusTwo")
def onePlusTwo():
    return {"OnePlusTwo Equals to" : "Three"}




@app.get("/timen")
def timen():
    return datetime.now()



if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8080)

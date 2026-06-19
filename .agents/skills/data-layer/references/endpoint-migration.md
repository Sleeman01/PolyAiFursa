# Endpoint migration reference

The precise ORM form of every database-touching endpoint in `services/yolo/app.py`,
with the exact response shape each one must still return.

Every endpoint below gains `db: Session = Depends(get_db)` in its signature.

---

## `POST /predict`

```python
db.add(PredictionSession(uid=uid, original_image=original_path, predicted_image=predicted_path))
detected_labels = []
for box in results[0].boxes:
    label_idx = int(box.cls[0].item())
    label = model.names[label_idx]
    score = float(box.conf[0])
    bbox = box.xyxy[0].tolist()
    db.add(DetectionObject(prediction_uid=uid, label=label, score=score, box=str(bbox)))
    detected_labels.append(label)
db.commit()
```

**Response (unchanged):**
```json
{"prediction_uid": "<uid>", "detection_count": <int>, "labels": ["..."], "time_took": <float>}
```

---

## `GET /prediction/{uid}`

```python
session = db.query(PredictionSession).filter_by(uid=uid).first()
if not session:
    raise HTTPException(status_code=404, detail="Prediction not found")
objects = db.query(DetectionObject).filter_by(prediction_uid=uid).all()
```

**Response (unchanged):**
```json
{
  "uid": "...", "timestamp": "...", "original_image": "...", "predicted_image": "...",
  "detection_objects": [{"id": <int>, "label": "...", "score": <float>, "box": "..."}]
}
```

---

## `GET /prediction/{uid}/image`

```python
session = db.query(PredictionSession).filter_by(uid=uid).first()
if not session or not os.path.exists(session.predicted_image):
    raise HTTPException(status_code=404, detail="Image not found")
return FileResponse(session.predicted_image)
```

---

## `GET /predictions/label/{label}`

```python
if not label.strip():
    raise HTTPException(status_code=400, detail="Label cannot be empty")

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
            {"id": o.id, "label": o.label, "score": o.score, "box": o.box} for o in objects
        ],
    })
return results
```

**Response:** list of `{"uid", "timestamp", "detection_objects": [...]}`.
No `original_image`/`predicted_image` here. Unknown label returns `[]` with 200.

---

## `GET /predictions/score/{min_score}`

```python
if min_score < 0.0 or min_score > 1.0:
    raise HTTPException(status_code=400, detail="min_score must be between 0.0 and 1.0")

objects = db.query(DetectionObject).filter(DetectionObject.score >= min_score).all()
return [
    {"id": o.id, "prediction_uid": o.prediction_uid, "label": o.label, "score": o.score, "box": o.box}
    for o in objects
]
```

**Response:** flat list of detection objects, each **including `prediction_uid`**.

---

## Endpoints with no database access

`/health`, `/welcoming`, `/morning`, `/onePlusTwo`, `/metrics`, and the
`shutdown` handler don't touch the DB — don't add a `db` dependency to them.

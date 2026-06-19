# Extending the data layer

Recipes for data-layer tasks beyond the initial refactor. All assume the
SQLAlchemy architecture is already in place.

## Add a column (e.g. `processing_time_ms`)

1. Add to the model in `models.py`:
```python
   processing_time_ms = Column(Integer, nullable=True)
```
2. Populate it in `POST /predict` before `db.commit()`.
3. Note: `Base.metadata.create_all` does not alter existing tables. In dev,
   delete `predictions.db` and it recreates. In production Postgres, you need
   a real `ALTER TABLE` migration.

## Add a table (e.g. `UserFeedback`)

1. New model in `models.py`:
```python
   class UserFeedback(Base):
       __tablename__ = "user_feedback"
       id = Column(Integer, primary_key=True, autoincrement=True)
       prediction_uid = Column(String)
       rating = Column(Integer)
       comment = Column(String, nullable=True)
       created_at = Column(DateTime, default=datetime.utcnow)
```
2. `init_db()` already calls `create_all` so the table is created on next boot.
3. Add endpoints with `db: Session = Depends(get_db)`. Return 404 if the
   referenced prediction does not exist.
4. Add tests covering create, read, and not-found.

## Add an endpoint (e.g. `GET /predictions/recent`)

```python
@app.get("/predictions/recent")
def get_recent_predictions(db: Session = Depends(get_db)):
    sessions = (
        db.query(PredictionSession)
        .order_by(PredictionSession.timestamp.desc())
        .limit(10)
        .all()
    )
    return [
        {
            "uid": s.uid,
            "timestamp": s.timestamp,
            "original_image": s.original_image,
            "predicted_image": s.predicted_image,
        }
        for s in sessions
    ]
```

## Delete a session and its detection objects

```python
@app.delete("/prediction/{uid}")
def delete_prediction(uid: str, db: Session = Depends(get_db)):
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.query(DetectionObject).filter_by(prediction_uid=uid).delete()
    db.delete(session)
    db.commit()
    return {"detail": "Prediction deleted"}
```

## Make the backend configurable (SQLite dev / Postgres prod)

In `db.py`:
```python
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "predictions")

if DB_BACKEND == "postgres":
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
else:
    DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
```

Add `psycopg2-binary` to `requirements.txt` for the Postgres driver.

To verify locally with Docker:
```bash
docker run --rm -e POSTGRES_USER=user -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=predictions -p 5432:5432 postgres
export DB_BACKEND=postgres DB_USER=user DB_PASSWORD=pass
python app.py
```

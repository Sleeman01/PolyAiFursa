# Testing the data layer

The existing tests are welded to the old raw-SQLite implementation and must be
migrated. Keep every assertion identical — same statuses, same JSON — change
only how the test database is wired up.

## The pattern: test engine + dependency override

```python
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import app
from db import get_db
from models import Base, PredictionSession, DetectionObject

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def db_session(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)
```

## Seeding fixture rows via the ORM

Replace the old `sqlite3.connect(...)` inserts with model instances:

```python
def insert_sample_data(db_session):
    db = db_session()
    db.add(PredictionSession(uid="abc-123", original_image="original.jpg", predicted_image="predicted.jpg"))
    db.add(DetectionObject(prediction_uid="abc-123", label="person", score=0.91, box="[10, 20, 100, 200]"))
    db.add(DetectionObject(prediction_uid="abc-123", label="car", score=0.45, box="[1, 2, 3, 4]"))
    db.commit()
    db.close()
```

Tests that need seed data take the `db_session` fixture and call
`insert_sample_data(db_session)`.

## test_prediction_time.py

Same idea: in `setUp`, build a test engine, `Base.metadata.create_all`, and set
`app.dependency_overrides[get_db] = override_get_db`. In `tearDown`, clear the
override and drop all tables.

## What no regression in coverage means

The demo endpoints (`/welcoming`, `/morning`, `/onePlusTwo`) were uncovered
before and may stay uncovered. A real regression is a previously-tested branch
(a 404 path, a 400 guard) dropping out of coverage.

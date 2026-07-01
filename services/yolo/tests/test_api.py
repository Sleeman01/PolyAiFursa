import os
import pytest
from unittest.mock import patch
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


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def insert_sample_data(db_session):
    db = db_session()
    db.add(PredictionSession(uid="abc-123", original_image="original.jpg", predicted_image="predicted.jpg"))
    db.add(DetectionObject(prediction_uid="abc-123", label="person", score=0.91, box="[10, 20, 100, 200]"))
    db.add(DetectionObject(prediction_uid="abc-123", label="car", score=0.45, box="[1, 2, 3, 4]"))
    db.commit()
    db.close()


def test_get_predictions_by_label_found(client, db_session):
    insert_sample_data(db_session)

    response = client.get("/predictions/label/person")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["uid"] == "abc-123"
    assert data[0]["detection_objects"][0]["label"] == "person"


def test_get_predictions_by_label_not_found(client):
    response = client.get("/predictions/label/dog")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_score_found(client, db_session):
    insert_sample_data(db_session)

    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["label"] == "person"
    assert data[0]["score"] == 0.91


def test_get_predictions_by_score_not_found(client, db_session):
    insert_sample_data(db_session)

    response = client.get("/predictions/score/0.99")

    assert response.status_code == 200
    assert response.json() == []


def test_get_predictions_by_score_invalid_low(client):
    response = client.get("/predictions/score/-0.1")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


def test_get_predictions_by_score_invalid_high(client):
    response = client.get("/predictions/score/1.1")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


def test_get_predictions_by_label_empty(client):
    response = client.get("/predictions/label/%20")

    assert response.status_code == 400
    assert response.json()["detail"] == "Label cannot be empty"


def test_get_prediction_by_uid_found(client, db_session):
    insert_sample_data(db_session)

    response = client.get("/prediction/abc-123")

    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "abc-123"
    assert len(data["detection_objects"]) == 2


def test_get_prediction_by_uid_not_found(client):
    response = client.get("/prediction/not-found")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_get_prediction_image_found(client, db_session, tmp_path):
    db = db_session()
    db.add(PredictionSession(uid="img-123", original_image="some/original/image.jpg", predicted_image="some/predicted/image.jpg"))
    db.commit()
    db.close()

    def fake_download(bucket, key, dest):
        with open(dest, "wb") as f:
            f.write(b"fake image content")

    with patch("app.AWS_S3_BUCKET", "test-bucket"), \
         patch("app.s3_client.download_file", side_effect=fake_download):
        response = client.get("/prediction/img-123/image")
    assert response.status_code == 200
    assert response.content == b"fake image content"
def test_get_prediction_image_not_found(client):
    response = client.get("/prediction/missing/image")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"

import os
import pytest
from fastapi.testclient import TestClient
import sqlite3


from app import app, init_db

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}




def insert_sample_data():
    import app as app_module

    with sqlite3.connect(app_module.DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, ("abc-123", "original.jpg", "predicted.jpg"))

        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, ("abc-123", "person", 0.91, "[10, 20, 100, 200]"))

        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, ("abc-123", "car", 0.45, "[1, 2, 3, 4]"))


def test_get_predictions_by_label_found(client):
    insert_sample_data()

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


def test_get_predictions_by_score_found(client):
    insert_sample_data()

    response = client.get("/predictions/score/0.5")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["label"] == "person"
    assert data[0]["score"] == 0.91


def test_get_predictions_by_score_not_found(client):
    insert_sample_data()

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
def test_get_prediction_by_uid_found(client):
    insert_sample_data()

    response = client.get("/prediction/abc-123")

    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "abc-123"
    assert len(data["detection_objects"]) == 2


def test_get_prediction_by_uid_not_found(client):
    response = client.get("/prediction/not-found")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_get_prediction_image_found(client, tmp_path):
    import app as app_module
    import sqlite3

    image_file = tmp_path / "predicted.jpg"
    image_file.write_bytes(b"fake image content")

    with sqlite3.connect(app_module.DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, ("img-123", "original.jpg", str(image_file)))

    response = client.get("/prediction/img-123/image")

    assert response.status_code == 200
    assert response.content == b"fake image content"


def test_get_prediction_image_not_found(client):
    response = client.get("/prediction/missing/image")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"
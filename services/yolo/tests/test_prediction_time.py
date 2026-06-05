import os
import unittest
from fastapi.testclient import TestClient


from app import app, init_db


TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


class TestPredictionTime(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = TestClient(app)

    def test_predict_includes_processing_time(self):
        with open(TEST_IMAGE, "rb") as f:
            response = self.client.post(
                "/predict",
                files={"file": ("beatles.jpeg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()

        assert "time_took" in data
        assert isinstance(data["time_took"], float)
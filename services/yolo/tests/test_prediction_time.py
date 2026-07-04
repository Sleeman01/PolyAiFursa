import os
import unittest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import shutil
from unittest.mock import patch

from app import app
from db import get_db
from models import Base


TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


class TestPredictionTime(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        db_file = os.path.join(self.tmp_dir, "test_predictions.db")
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
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_predict_includes_processing_time(self):
        def fake_download(bucket, key, dest):
            shutil.copy(TEST_IMAGE, dest)

        with patch("app.AWS_S3_BUCKET", "test-bucket"), \
             patch("app.s3_client.download_file", side_effect=fake_download), \
             patch("app.s3_client.upload_file", return_value=None):
            response = self.client.post(
                "/predict",
                json={"image_s3_key": "some-chat/some-pred/original/beatles.jpeg"}
            )
        assert response.status_code == 200
        data = response.json()
        assert "time_took" in data
        assert isinstance(data["time_took"], float)

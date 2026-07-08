import base64
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import app


def _make_image_b64(width=100, height=60, color=(120, 200, 40)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_b64(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_blur_returns_valid_image():
    out = app.blur(_make_image_b64(), radius=2.0)
    img = _decode_b64(out)
    assert img.size == (100, 60)


def test_rotate_90_swaps_dimensions():
    out = app.rotate(_make_image_b64(100, 60), angle=90)
    img = _decode_b64(out)
    # rotating 90 with expand swaps width/height
    assert img.size == (60, 100)


def test_flip_horizontal_keeps_size():
    out = app.flip(_make_image_b64(100, 60), direction="horizontal")
    img = _decode_b64(out)
    assert img.size == (100, 60)


def test_flip_vertical_keeps_size():
    out = app.flip(_make_image_b64(100, 60), direction="vertical")
    img = _decode_b64(out)
    assert img.size == (100, 60)


def test_flip_invalid_direction_raises():
    try:
        app.flip(_make_image_b64(), direction="diagonal")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resize_changes_dimensions():
    out = app.resize(_make_image_b64(100, 60), width=40, height=30)
    img = _decode_b64(out)
    assert img.size == (40, 30)


def test_crop_region():
    out = app.crop(_make_image_b64(100, 60), left=10, upper=10, right=50, lower=40)
    img = _decode_b64(out)
    assert img.size == (40, 30)


def test_add_noise_returns_same_size():
    out = app.add_noise(_make_image_b64(100, 60), amount=0.1)
    img = _decode_b64(out)
    assert img.size == (100, 60)


def test_add_noise_actually_changes_pixels():
    original_b64 = _make_image_b64(100, 60, color=(120, 200, 40))
    out = app.add_noise(original_b64, amount=0.2)
    original = _decode_b64(original_b64).convert("RGB")
    noised = _decode_b64(out).convert("RGB")
    # at least some pixels should differ
    assert list(original.getdata()) != list(noised.getdata())

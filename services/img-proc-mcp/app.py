import base64
import io
import random

from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageFilter

mcp = FastMCP("img-proc")


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@mcp.tool()
def rotate(image_b64: str, angle: float) -> str:
    """Rotate the image by a given angle (degrees, counter-clockwise). Returns base64-encoded PNG."""
    img = _decode(image_b64).rotate(angle, expand=True)
    return _encode(img)


@mcp.tool()
def flip(image_b64: str, direction: str = "horizontal") -> str:
    """Flip the image horizontally or vertically. direction: 'horizontal' or 'vertical'. Returns base64-encoded PNG."""
    img = _decode(image_b64)
    if direction == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    return _encode(img)


@mcp.tool()
def blur(image_b64: str, radius: float = 2.0) -> str:
    """Apply Gaussian blur to an image with a given radius. Returns base64-encoded PNG."""
    img = _decode(image_b64).filter(ImageFilter.GaussianBlur(radius))
    return _encode(img)


@mcp.tool()
def resize(image_b64: str, width: int, height: int) -> str:
    """Resize the image to the given width x height (pixels). Returns base64-encoded PNG."""
    img = _decode(image_b64).resize((width, height))
    return _encode(img)


@mcp.tool()
def crop(image_b64: str, left: int, upper: int, right: int, lower: int) -> str:
    """Crop a region defined by the bounding box (left, upper, right, lower). Returns base64-encoded PNG."""
    img = _decode(image_b64).crop((left, upper, right, lower))
    return _encode(img)


@mcp.tool()
def add_noise(image_b64: str, amount: float = 0.05) -> str:
    """Add salt-and-pepper noise to the image. amount is the fraction of pixels affected (0.0-1.0). Returns base64-encoded PNG."""
    img = _decode(image_b64).convert("RGB")
    pixels = img.load()
    w, h = img.size
    num = int(amount * w * h)
    for _ in range(num):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pixels[x, y] = (255, 255, 255) if random.random() < 0.5 else (0, 0, 0)
    return _encode(img)


if __name__ == "__main__":
    import os
    # Run as a streamable HTTP server so other services (the agent) can reach it
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("MCP_PORT", "9000"))
    mcp.run(transport="streamable-http")

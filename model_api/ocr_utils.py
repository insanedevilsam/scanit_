# ocr_utils.py

import pytesseract
from PIL import Image
import io

def extract_text_from_bytes(image_bytes: bytes) -> str:
    """
    Extract text from image bytes.
    Used for images uploaded via FastAPI.
    """
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text
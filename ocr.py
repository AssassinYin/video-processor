"""OCR module for text extraction from video frames."""
import cv2
import re
import pytesseract
from PIL import Image
import Levenshtein

# Global variable for EasyOCR reader (lazy loading)
reader_easyocr = None


def extract_text_from_frame(frame, region='all', lang='eng', ocr_engine='tesseract'):
    """
    Extract text from a frame using OCR, optionally focusing on a region and language.

    Args:
        frame: frame to extract text from
        region: 'all', 'top', or 'bottom'
        lang: Tesseract language code
        ocr_engine: 'tesseract' or 'easyocr'
    """
    # Crop frame if needed
    if region == 'top':
        frame = frame[:frame.shape[0] // 3, :, :]
    elif region == 'bottom':
        frame = frame[-frame.shape[0] // 3:, :, :]

    if ocr_engine == 'tesseract':
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get better text contrast
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(thresh)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
    else:
        global reader_easyocr
        if reader_easyocr is None:
            import easyocr
            reader_easyocr = easyocr.Reader([lang], gpu=True)
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = reader_easyocr.readtext(rgb, detail=0)
        text = ' '.join(result)

    # Clean up the text
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()


def normalize_text(text):
    """Normalize text for comparison purposes."""
    if not text or len(text.strip()) < 2:  # More lenient - allow shorter text
        return ''
    # Remove some punctuation but keep more characters for Chinese
    normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text).strip().lower()
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    # Much more lenient - only filter extremely short results
    if len(normalized) < 2:
        return ''
    return normalized


def is_text_significantly_different(a, b, threshold=0.85):
    """Check if two texts are significantly different based on Levenshtein distance."""
    # If both are empty, they're the same
    if not a and not b:
        return False
    # If one is empty and the other isn't, they're different
    if not a or not b:
        return True
    # Calculate similarity ratio - much more lenient threshold
    ratio = Levenshtein.ratio(a, b)
    return ratio < threshold  # True if less than 85% similar


def has_meaningful_text(text):
    """Check if a text contains meaningful content - much more lenient"""
    if not text or len(text.strip()) < 2:  # Much more permissive
        return False

    normalized = normalize_text(text)
    if not normalized:
        return False

    # Much more lenient - any text with 2+ characters is considered meaningful
    return len(normalized) >= 2

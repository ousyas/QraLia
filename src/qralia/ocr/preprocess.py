"""Image preprocessing for OCR: perspective correction, binarization, contrast."""

import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Clean up a photo of an invoice for better OCR results.

    Steps: grayscale → detect document contour → perspective correction →
    adaptive binarization → denoise.
    """
    # PIL → OpenCV (BGR)
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1. Grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 2. Try to detect document contour and correct perspective
    warped = _detect_and_warp(gray, cv_img)
    if warped is not None:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive binarization (handles shadows and uneven lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10
    )

    # 4. Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # OpenCV → PIL
    return Image.fromarray(denoised)


def _detect_and_warp(gray, original):
    """Detect the largest rectangular contour and warp to a flat rectangle."""
    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort by area, take the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If 4 corners found → it's the document
        if len(approx) == 4:
            return _four_point_transform(original, approx.reshape(4, 2))

    return None


def _four_point_transform(image, pts):
    """Warp a quadrilateral defined by 4 points into a flat rectangle."""
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = _order_points(pts.astype(np.float32))
    tl, tr, br, bl = rect

    # Compute new width and height
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    # Destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    # Perspective transform
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
    return rect
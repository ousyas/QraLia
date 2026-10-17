"""Prétraitement optionnel d'une photo avant OCR."""

import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(cv2.fastNlMeansDenoising(binary, h=10))


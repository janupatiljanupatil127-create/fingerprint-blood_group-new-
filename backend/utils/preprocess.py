import cv2
import numpy as np


def preprocess_fingerprint(image_path_or_array):
    """
    Preprocesses a fingerprint image for machine learning.

    Steps:
    1. Read image
    2. Convert to grayscale
    3. Resize (128x128)
    4. Gaussian Blur
    5. Histogram Equalization
    6. Adaptive Threshold
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_path_or_array}")
    else:
        img = image_path_or_array

    if img is None:
        raise ValueError("Input image is None.")

    if len(img.shape) == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img.copy()
    else:
        gray = img.copy()

    # cv2 functions require uint8 input; enforce it in case a float array is passed in
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    resized = cv2.resize(gray, (128, 128))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    processed_img = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

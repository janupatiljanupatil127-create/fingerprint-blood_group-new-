import numpy as np
from skimage.feature import hog


def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extracts HOG (Histogram of Oriented Gradients) features from a
    preprocessed grayscale fingerprint image.

    Expects a 2D grayscale numpy array (e.g. the output of
    preprocess_fingerprint in utils/preprocess.py). Returns a 1D
    feature vector ready to be fed into scaler.transform() and the
    trained classifier.

    NOTE: pixels_per_cell / cells_per_block / orientations must exactly
    match whatever values were used when the model (voting_classifier.pkl)
    and scaler (scaler.pkl) were originally trained, otherwise the
    resulting feature vector length/shape won't match what the model
    expects. Update these defaults if your training script used different
    parameters.
    """
    image = np.asarray(image)

    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features

# External libraries
import joblib

# Own libraries
from unsupervised.python.dimensionality_reduction import PCA, SVD, TSNE

from metadata.path import Path


def get_model():
    return joblib.load(Path.model)


"""def get_svd():
    return joblib.load(Path.svd)"""

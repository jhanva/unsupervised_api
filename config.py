# External libraries
import joblib

# Own libraries
from metadata.path import Path


def get_model():
    return joblib.load(Path.model)

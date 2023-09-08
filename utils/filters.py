from io import BytesIO

import numpy as np
from PIL import Image

from config import get_model


def read_image_file(file) -> Image.Image:
    """Reads an image file from binary data and returns it as a PIL Image.

    Args:
        file: Binary image data.

    Returns:
        Image.Image: A PIL Image object.

    """
    image = Image.open(BytesIO(file))
    return image


def predict(image: Image.Image) -> dict:
    """Predicts a value from an input image using a trained model.

    Args:
        image (Image.Image): The input image.

    Returns:
        dict: A dictionary containing the prediction result.

    """
    # Resize and convert the image
    image = image.resize((28, 28))
    image = image.convert('L')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Reshape the array to 1x(28*28) and normalize pixel values
    image_array = image_array.reshape(1, -1)
    image_array = image_array.astype(float) / 255.0

    # Perform prediction using a trained model
    # Replace 'get_model()' with the actual function that retrieves your model
    prediction = get_model().predict(image_array)[0]

    return {'predict': str(prediction)}

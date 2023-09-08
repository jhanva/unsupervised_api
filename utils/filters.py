import numpy as np
from io import BytesIO
from PIL import Image

from config import get_model


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def predict(image: Image.Image):
    model = get_model()

    image = image.resize((28, 28))

    image = image.convert('L')

    image_array = np.array(image)

    image_array = image_array.astype(float) / 255.0

    prediction = model.predict([image_array.flatten()])[0]

    return prediction





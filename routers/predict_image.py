from fastapi import File, UploadFile, APIRouter

from utils.filters import read_image_file
from utils.filters import predict

predict_image_router = APIRouter(prefix='/predict_image', tags=['predict'])


@predict_image_router.post('/predict', status_code=200)
async def predict_api(file: UploadFile = File(...)):
    image = read_image_file(await file.read())
    prediction = predict(image)

    return prediction


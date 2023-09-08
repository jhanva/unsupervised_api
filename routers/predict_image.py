from fastapi import APIRouter, File, UploadFile

from utils.filters import predict, read_image_file

predict_image_router = APIRouter(prefix='/predict_image', tags=['predict'])


@predict_image_router.post('/predict', status_code=200)
async def predict_api(file: UploadFile = File(...)):
    try:
        image = read_image_file(await file.read())
        prediction = predict(image)
        return prediction
    except Exception as e:
        return {'error': str(e)}

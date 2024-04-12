from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from challenge3.src.api.schemas.predict_request import PredictRequest, PredictRequestFile


router = APIRouter()


@router.post('/csv/predict', response_class=JSONResponse)
async def api_csv_predict(request: PredictRequestFile = Depends()):
    """
    TODO: Support ch2 model
    """


@router.post('/predict', response_class=JSONResponse)
async def api_predict(request: PredictRequest):
    """
    TODO: Support ch2 model
    """

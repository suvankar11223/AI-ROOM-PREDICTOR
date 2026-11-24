from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_listing

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.get("/health")
def health(request: Request):
    registry = request.app.state.model_registry
    return {
        "status": "ok" if registry.ready else "not_ready",
        "models_loaded": registry.ready,
    }


@router.get("/models")
def models(request: Request):
    registry = request.app.state.model_registry
    if not registry.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not loaded.")
    return {"available_models": registry.status()}


@router.post("/predict", response_model=PredictionResponse)
def predict(request_body: PredictionRequest, request: Request):
    registry = request.app.state.model_registry
    if not registry.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not loaded.")
    try:
        return predict_listing(request_body, registry)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

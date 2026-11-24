from app.ml.inference import predict
from app.schemas.prediction import PredictionRequest
from app.services.model_registry import ModelRegistry

DISCLAIMER = (
    "This is a research model trained on a 2019 NYC Airbnb listing snapshot. "
    "It classifies room type from listing attributes; it does not estimate price, "
    "booking demand, profitability, or future performance. It has not been approved "
    "for production decision-making."
)


def predict_listing(request: PredictionRequest, registry: ModelRegistry) -> dict:
    bundle = registry.get(request.mode)
    listing = request.listing.model_dump(exclude_none=False)
    feature_columns = bundle["feature_columns"]
    model_input = {name: listing[name] for name in feature_columns}
    result = predict(bundle, model_input).iloc[0]

    probabilities = {
        class_name: float(result[f"probability_{class_name}"])
        for class_name in bundle["classes"]
    }
    return {
        "predicted_room_type": str(result["predicted_room_type"]),
        "predicted_class_probability": float(result["predicted_class_probability"]),
        "probabilities": probabilities,
        "mode": request.mode,
        "model_version": str(bundle.get("model_version", "unknown")),
        "disclaimer": DISCLAIMER,
    }

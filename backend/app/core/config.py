from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_FILES = {
    "without_reviews": ARTIFACT_DIR / "airbnb_room_type_without_reviews.joblib",
    "with_reviews": ARTIFACT_DIR / "airbnb_room_type_with_reviews.joblib",
}
API_TITLE = "NYC Airbnb Room Type Predictor API"
API_VERSION = "1.0.0"

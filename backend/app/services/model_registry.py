from app.core.config import MODEL_FILES
from app.ml.inference import load_model


class ModelRegistry:
    """Loads trusted, version-pinned model bundles once per API process."""

    def __init__(self) -> None:
        self._models: dict[str, dict] = {}

    def load_all(self) -> None:
        loaded: dict[str, dict] = {}
        for mode, path in MODEL_FILES.items():
            if not path.is_file():
                raise FileNotFoundError(f"Required model artifact not found: {path}")
            loaded[mode] = load_model(path)
        self._models = loaded

    def get(self, mode: str) -> dict:
        try:
            return self._models[mode]
        except KeyError as exc:
            raise RuntimeError(f"Model '{mode}' is not loaded.") from exc

    @property
    def ready(self) -> bool:
        return set(self._models) == set(MODEL_FILES)

    def status(self) -> dict[str, dict[str, str]]:
        return {
            mode: {
                "model_version": str(bundle.get("model_version", "unknown")),
                "bundle_format": str(bundle.get("bundle_format", "unknown")),
            }
            for mode, bundle in self._models.items()
        }


model_registry = ModelRegistry()

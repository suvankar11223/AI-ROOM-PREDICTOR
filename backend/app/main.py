from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import API_TITLE, API_VERSION
from app.services.model_registry import model_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_registry.load_all()
    app.state.model_registry = model_registry
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Prediction API for the NYC Airbnb room type research models.",
    lifespan=lifespan,
)

# Local development origins. Replace with the deployed frontend origin before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.common.logger import setup_logging, get_logger
from src.inference.model_loader import ModelLoader
from src.inference.predict import Predictor

# ------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Abusive Text Detection API",
    version="1.0.0",
    description="Production-ready NLP inference service"
)
# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # <- Allows requests from file:// and localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------------------------------------
# Global objects (loaded once)
# ------------------------------------------------------------------
model_loader: Optional[ModelLoader] = None
predictor: Optional[Predictor] = None


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., example="You are a terrible person")


class PredictResponse(BaseModel):
    label: str
    probability: float
    latency_ms: float


# ------------------------------------------------------------------
# Startup event
# ------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model_loader, predictor

    try:
        logger.info("Starting model initialization")

        registry_path = os.getenv(
            "MODEL_REGISTRY_PATH",
            "models/registry.json"
        )

        model_loader = ModelLoader(registry_path=registry_path)

        predictor = Predictor(
            model=model_loader.get_model(),
            tokenizer=model_loader.get_tokenizer(),
            threshold=0.5
        )

        logger.info(
            f"Model loaded successfully | "
            f"version={model_loader.model_version}"
        )

    except Exception as e:
        logger.exception("Failed to load model at startup")
        raise e


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/health")
def health():
    if predictor is None:
        logger.warning("Health check failed: model not loaded")
        return {"status": "unhealthy"}

    return {"status": "healthy"}


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    logger.info("Received /predict request")

    try:
        result = predictor.predict(request.text)
        return result

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction error"
        )

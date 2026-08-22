from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

MODEL_PATH = Path("models/iris_model.pkl")
CLASS_NAMES = load_iris().target_names.tolist()

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Iris Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


class PredictResponse(BaseModel):
    prediction: int
    class_name: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prediction = int(model.predict([request.features])[0])
    return PredictResponse(
        prediction=prediction, class_name=CLASS_NAMES[prediction]
    )
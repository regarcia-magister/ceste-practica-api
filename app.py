from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


APP_VERSION = "1.0.0"
ALGORITHM_NAME = "LogisticRegression"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
CLASSES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., json_schema_extra={"example": 5.1})
    sepal_width: float = Field(..., json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., json_schema_extra={"example": 1.4})
    petal_width: float = Field(..., json_schema_extra={"example": 0.2})


app = FastAPI(
    title="Iris Classification API",
    version=APP_VERSION,
    description="API para clasificar flores Iris usando un modelo de Machine Learning.",
)

model = joblib.load(MODEL_PATH)


@app.get("/")
def home() -> dict:
    return {
        "api_name": "Iris Classification API",
        "version": APP_VERSION,
        "algorithm": ALGORITHM_NAME,
        "expected_features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
        "example_request": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    }


@app.post("/predict")
def predict(features: IrisFeatures) -> dict:
    values = np.array(
        [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]
    )

    prediction_index = int(model.predict(values)[0])
    probabilities = model.predict_proba(values)[0]

    probability_map = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(CLASSES, probabilities)
    }
    confidence = round(float(np.max(probabilities)), 4)

    return {
        "prediction": CLASSES[prediction_index],
        "prediction_index": prediction_index,
        "probabilities": probability_map,
        "confidence": confidence,
        "status": "success",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

app = FastAPI(title="Diabetes Surgery Risk API", description="API for diabetes-related surgery risk prediction.")

# Load models and encoders at startup
models_path = Path(__file__).parent / "models"

try:
    model = joblib.load(models_path / "best_model_xgboost_gpu.pkl")
    scaler = joblib.load(models_path / "scaler.pkl")
    label_encoder = joblib.load(models_path / "label_encoder.pkl")
    with open(models_path / "feature_names.txt", 'r') as f:
        content = f.read().strip()
        if '\\n' in content:
            feature_names = [name.strip() for name in content.split('\\n') if name.strip()]
        else:
            feature_names = [line.strip() for line in content.split('\n') if line.strip()]
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

class PatientData(BaseModel):
    # Dynamically create fields for all features
    __annotations__ = {name: float for name in feature_names}
    # You can adjust types as needed (e.g., int for categorical)

@app.post("/predict", summary="Predict surgery risk", response_model=Dict[str, Any])
def predict_risk(data: PatientData):
    try:
        # Convert input to correct order
        input_array = np.array([getattr(data, f) for f in feature_names]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        risk_label = label_encoder.inverse_transform([pred])[0]
        class_names = label_encoder.classes_
        prob_dict = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
        return {
            "risk_level": risk_label,
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}

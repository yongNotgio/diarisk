
# Model Deployment Instructions

## Best Model: XGBoost_GPU
- F1 Score: 1.0000
- Accuracy: 1.0000

## Files Required for Deployment:
- best_model_xgboost_gpu.pkl
- model_lightgbm_gpu.pkl
- model_catboost_gpu.pkl
- model_randomforest_cpu.pkl
- model_extratrees_cpu.pkl
- scaler.pkl
- label_encoder.pkl
- feature_names.txt
- model_evaluation_results.json

## Usage Example:
```python
import joblib
import torch
import numpy as np

# Load preprocessing components
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Load best model
# For other models:
model = joblib.load('models/best_model_xgboost_gpu.pkl')

# Make predictions on new data
# new_data = ... (your new data)
# scaled_data = scaler.transform(new_data)
# predictions = model.predict(scaled_data)
# predicted_classes = label_encoder.inverse_transform(predictions)
```

## Class Labels:
{0: 'High', 1: 'Low', 2: 'Moderate'}

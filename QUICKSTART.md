# Quick Start Guide for Surgical Risk Assessment API

## Installation Steps

### 1. Install Python dependencies
pip install fastapi uvicorn[standard] pydantic numpy scikit-fuzzy python-multipart requests

### 2. Verify model file exists
# Check that surgical_risk_fuzzy_system_complete.pkl is in the directory

### 3. Start the API server
python main.py

### 4. Test the API (in a new terminal)
python test_api.py

### 5. Access interactive documentation
# Open your browser and go to:
# http://localhost:8000/docs

## Quick Commands

# Start API with auto-reload (development)
uvicorn main:app --reload

# Start API for production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Test health endpoint
curl http://localhost:8000/health

# Test risk assessment
curl -X POST http://localhost:8000/api/v1/assess-risk -H "Content-Type: application/json" -d "{\"age\": 55, \"glucose\": 8.5, \"systolic_bp\": 130, \"bmi\": 28.0, \"comorbidities\": 2}"

## Available Endpoints

- GET  /               - API information
- GET  /health         - Health check
- POST /api/v1/assess-risk     - Single patient assessment
- POST /api/v1/batch-assess    - Batch assessment
- GET  /api/v1/reference-ranges - Reference ranges

## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Full README: README_API.md

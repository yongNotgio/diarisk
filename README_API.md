# Surgical Risk Assessment FastAPI

A FastAPI-based REST API for surgical risk assessment using fuzzy logic system.

## Features

- 🚀 Fast and efficient REST API
- 📊 Fuzzy logic-based risk calculation
- 🔍 Comprehensive clinical recommendations
- 📝 Interactive API documentation (Swagger UI)
- 🔄 Batch processing support
- ✅ Input validation with Pydantic

## Installation

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

Or install manually:

```powershell
pip install fastapi uvicorn[standard] pydantic numpy scikit-fuzzy python-multipart
```

### 2. Ensure Model File Exists

Make sure `surgical_risk_fuzzy_system_complete.pkl` is in the same directory as `main.py`.

## Running the API

### Development Mode (with auto-reload)

```powershell
python main.py
```

Or using uvicorn directly:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```http
GET /health
```
Check if the API and model are loaded correctly.

### 3. Assess Surgical Risk (Single Patient)
```http
POST /api/v1/assess-risk
```

**Request Body:**
```json
{
  "age": 55,
  "glucose": 8.5,
  "systolic_bp": 130,
  "bmi": 28.0,
  "comorbidities": 2,
  "family_diabetes": true,
  "hypertensive": true,
  "cardiovascular": false,
  "stroke": false,
  "family_hypertension": false
}
```

**Response:**
```json
{
  "risk_score": 52.3,
  "risk_category": {
    "level": "Moderate Risk",
    "color": "orange"
  },
  "patient_data": {
    "age": 55,
    "glucose": 8.5,
    "systolic_bp": 130,
    "bmi": 28.0,
    "comorbidities": 2,
    "family_diabetes": true,
    "hypertensive": true,
    "cardiovascular": false,
    "stroke": false,
    "family_hypertension": false
  },
  "recommendations": {
    "category": "Moderate Risk",
    "preoperative": [
      "Enhanced preoperative evaluation",
      "Additional cardiac assessment if indicated",
      "Target glucose <8.0 mmol/L (144 mg/dL)",
      "Optimize blood pressure <140/90 mmHg",
      "Review and optimize medications"
    ],
    "perioperative": [
      "Enhanced monitoring protocols",
      "Frequent glucose checks (q2-4h)",
      "Tight blood pressure control",
      "Consider arterial line for major surgery",
      "Insulin protocol if indicated"
    ],
    "postoperative": [
      "Enhanced recovery monitoring",
      "Frequent vital signs (q2-4h initially)",
      "Glucose monitoring q4-6h for 24-48h",
      "Early physiotherapy",
      "Monitor for complications"
    ]
  },
  "timestamp": "2025-11-09T10:30:00.123456",
  "glucose_mgdl": 153.0
}
```

### 4. Batch Assessment
```http
POST /api/v1/batch-assess
```

**Request Body:**
```json
[
  {
    "age": 55,
    "glucose": 8.5,
    "systolic_bp": 130,
    "bmi": 28.0,
    "comorbidities": 2
  },
  {
    "age": 65,
    "glucose": 12.0,
    "systolic_bp": 160,
    "bmi": 32.0,
    "comorbidities": 4
  }
]
```

Maximum: 100 patients per request.

### 5. Reference Ranges
```http
GET /api/v1/reference-ranges
```
Returns reference ranges for all clinical parameters.

## Usage Examples

### Using cURL (PowerShell)

```powershell
# Health check
curl http://localhost:8000/health

# Single assessment
curl -X POST http://localhost:8000/api/v1/assess-risk `
  -H "Content-Type: application/json" `
  -d '{\"age\": 55, \"glucose\": 8.5, \"systolic_bp\": 130, \"bmi\": 28.0, \"comorbidities\": 2}'
```

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/api/v1/assess-risk"

# Patient data
patient_data = {
    "age": 55,
    "glucose": 8.5,
    "systolic_bp": 130,
    "bmi": 28.0,
    "comorbidities": 2,
    "family_diabetes": True,
    "hypertensive": True
}

# Make request
response = requests.post(url, json=patient_data)

# Print result
if response.status_code == 200:
    result = response.json()
    print(f"Risk Score: {result['risk_score']}")
    print(f"Risk Category: {result['risk_category']['level']}")
else:
    print(f"Error: {response.status_code}")
```

### Using JavaScript (fetch)

```javascript
const assessRisk = async (patientData) => {
  const response = await fetch('http://localhost:8000/api/v1/assess-risk', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData)
  });
  
  const result = await response.json();
  console.log('Risk Score:', result.risk_score);
  console.log('Risk Category:', result.risk_category.level);
  return result;
};

// Example usage
const patient = {
  age: 55,
  glucose: 8.5,
  systolic_bp: 130,
  bmi: 28.0,
  comorbidities: 2
};

assessRisk(patient);
```

## Input Parameters

| Parameter | Type | Range | Unit | Description |
|-----------|------|-------|------|-------------|
| `age` | float | 20-100 | years | Patient age |
| `glucose` | float | 3.0-25.0 | mmol/L | Fasting glucose level |
| `systolic_bp` | float | 80-250 | mmHg | Systolic blood pressure |
| `bmi` | float | 15.0-50.0 | kg/m² | Body Mass Index |
| `comorbidities` | int | 0-5 | count | Number of comorbidities |

### Optional Parameters (for detailed tracking)
- `family_diabetes`: boolean
- `hypertensive`: boolean
- `cardiovascular`: boolean
- `stroke`: boolean
- `family_hypertension`: boolean

## Risk Categories

| Category | Risk Score | Color | Description |
|----------|------------|-------|-------------|
| Low Risk | 0-35 | 🟢 Green | Standard surgical protocols |
| Moderate Risk | 36-65 | 🟡 Orange | Enhanced perioperative care |
| High Risk | 66-100 | 🔴 Red | Intensive monitoring required |

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `422`: Validation Error (parameter out of range)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Testing the API

### Using the Interactive Documentation

1. Start the API
2. Navigate to http://localhost:8000/docs
3. Click on any endpoint to expand it
4. Click "Try it out"
5. Fill in the parameters
6. Click "Execute"

### Quick Test

```powershell
# Start the server
python main.py

# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Test the assessment endpoint
curl -X POST http://localhost:8000/api/v1/assess-risk `
  -H "Content-Type: application/json" `
  -d '{\"age\": 60, \"glucose\": 10.0, \"systolic_bp\": 150, \"bmi\": 30.0, \"comorbidities\": 3}'
```

## Deployment Considerations

### Production Deployment

For production, consider:

1. **Use multiple workers**:
   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Add authentication** (JWT, OAuth2, etc.)

3. **Use HTTPS** with a reverse proxy (nginx, traefik)

4. **Rate limiting** to prevent abuse

5. **Monitoring and logging** (Sentry, CloudWatch, etc.)

6. **Update CORS settings** in `main.py` to restrict origins

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY surgical_risk_fuzzy_system_complete.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```powershell
docker build -t surgical-risk-api .
docker run -p 8000:8000 surgical-risk-api
```

## License

MIT

## Support

For issues or questions, please open an issue on the repository.

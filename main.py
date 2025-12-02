"""
FastAPI application for Surgical Risk Assessment using Fuzzy Logic
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pickle
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Surgical Risk Assessment API",
    description="Fuzzy Logic-Based Clinical Decision Support System for Surgical Risk Assessment",
    version="1.0.0",
    contact={
        "name": "Surgical Risk Assessment System",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD FUZZY MODEL
# ============================================
simulation = None

@app.on_event("startup")
async def load_model():
    """Load the fuzzy logic model on startup"""
    global simulation
    try:
        with open('surgical_risk_fuzzy_system_complete.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Handle both single simulation and package formats
        if isinstance(model, dict):
            simulation = model.get('simulation', model)
        else:
            simulation = model
        
        print("✓ Fuzzy logic model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

# ============================================
# PYDANTIC MODELS
# ============================================
class PatientInput(BaseModel):
    """Patient clinical parameters for surgical risk assessment"""
    
    age: float = Field(
        ..., 
        ge=20, 
        le=100, 
        description="Patient age in years",
        examples=[55]
    )
    glucose: float = Field(
        ..., 
        ge=3.0, 
        le=25.0, 
        description="Fasting glucose level in mmol/L",
        examples=[8.5]
    )
    systolic_bp: float = Field(
        ..., 
        ge=80, 
        le=250, 
        description="Systolic blood pressure in mmHg",
        examples=[130]
    )
    bmi: float = Field(
        ..., 
        ge=15.0, 
        le=50.0, 
        description="Body Mass Index in kg/m²",
        examples=[28.0]
    )
    comorbidities: int = Field(
        ..., 
        ge=0, 
        le=5, 
        description="Number of comorbidities (0-5)",
        examples=[2]
    )
    
    # Optional fields for detailed comorbidity tracking
    family_diabetes: Optional[bool] = Field(default=False, description="Family history of diabetes")
    hypertensive: Optional[bool] = Field(default=False, description="Hypertensive condition")
    cardiovascular: Optional[bool] = Field(default=False, description="Cardiovascular disease")
    stroke: Optional[bool] = Field(default=False, description="Stroke history")
    family_hypertension: Optional[bool] = Field(default=False, description="Family history of hypertension")
    
    @field_validator('age', 'glucose', 'systolic_bp', 'bmi')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v


class RiskCategory(BaseModel):
    """Risk category classification"""
    level: str = Field(..., description="Risk level: Low Risk, Moderate Risk, or High Risk")
    color: str = Field(..., description="Color indicator: green, orange, or red")


class ClinicalRecommendations(BaseModel):
    """Clinical recommendations based on risk level"""
    category: str
    preoperative: list[str]
    perioperative: list[str]
    postoperative: list[str]


class RiskAssessmentResult(BaseModel):
    """Complete surgical risk assessment result"""
    
    risk_score: float = Field(..., description="Surgical risk score (0-100)")
    risk_category: RiskCategory
    patient_data: PatientInput
    recommendations: ClinicalRecommendations
    timestamp: str = Field(..., description="Assessment timestamp")
    glucose_mgdl: float = Field(..., description="Glucose in mg/dL")


# ============================================
# HELPER FUNCTIONS
# ============================================
def calculate_surgical_risk(
    age: float, 
    glucose: float, 
    systolic_bp: float, 
    bmi: float, 
    comorbidities: int
) -> tuple[float, str, str, bool, Optional[str]]:
    """
    Calculate surgical risk using fuzzy logic system
    
    Returns:
        tuple: (risk_score, category, color, success, error_msg)
    """
    try:
        # Input validation and clamping
        age = max(20, min(100, age))
        glucose = max(3, min(25, glucose))
        systolic_bp = max(80, min(250, systolic_bp))
        bmi = max(15, min(50, bmi))
        comorbidities = max(0, min(5, comorbidities))
        
        # Set inputs
        simulation.input['age'] = age
        simulation.input['glucose'] = glucose
        simulation.input['systolic_bp'] = systolic_bp
        simulation.input['bmi'] = bmi
        simulation.input['comorbidities'] = comorbidities
        
        # Compute
        simulation.compute()
        risk_score = simulation.output['surgical_risk']
        
        # Categorize
        if risk_score <= 35:
            category = 'Low Risk'
            color = 'green'
        elif risk_score <= 65:
            category = 'Moderate Risk'
            color = 'orange'
        else:
            category = 'High Risk'
            color = 'red'
        
        return risk_score, category, color, True, None
        
    except Exception as e:
        return 50.0, 'Error', 'gray', False, str(e)


def get_clinical_recommendations(category: str) -> ClinicalRecommendations:
    """Get clinical recommendations based on risk category"""
    
    recommendations = {
        'Low Risk': {
            'preoperative': [
                "Standard preoperative assessment",
                "Routine laboratory tests",
                "Continue regular medications",
                "Standard fasting protocols"
            ],
            'perioperative': [
                "Standard anesthesia protocol",
                "Routine glucose monitoring",
                "Standard vital signs monitoring",
                "Normal recovery protocols"
            ],
            'postoperative': [
                "Standard postoperative care",
                "Routine vital signs monitoring",
                "Early mobilization as tolerated",
                "Standard discharge criteria"
            ]
        },
        'Moderate Risk': {
            'preoperative': [
                "Enhanced preoperative evaluation",
                "Additional cardiac assessment if indicated",
                "Target glucose <8.0 mmol/L (144 mg/dL)",
                "Optimize blood pressure <140/90 mmHg",
                "Review and optimize medications"
            ],
            'perioperative': [
                "Enhanced monitoring protocols",
                "Frequent glucose checks (q2-4h)",
                "Tight blood pressure control",
                "Consider arterial line for major surgery",
                "Insulin protocol if indicated"
            ],
            'postoperative': [
                "Enhanced recovery monitoring",
                "Frequent vital signs (q2-4h initially)",
                "Glucose monitoring q4-6h for 24-48h",
                "Early physiotherapy",
                "Monitor for complications"
            ]
        },
        'High Risk': {
            'preoperative': [
                "Comprehensive preoperative evaluation REQUIRED",
                "Cardiology consultation for cardiac risk stratification",
                "Target glucose <7.0 mmol/L (126 mg/dL) preoperatively",
                "Blood pressure optimization to <140/90 mmHg",
                "Consider postponing elective surgery until optimization",
                "Multidisciplinary team consultation"
            ],
            'perioperative': [
                "Intensive monitoring (arterial line, possible CVP)",
                "Continuous glucose monitoring or q1-2h checks",
                "Insulin infusion protocol",
                "ICU/HDU bed reservation",
                "Enhanced anesthesia care",
                "Consider regional techniques to reduce stress"
            ],
            'postoperative': [
                "ICU/HDU admission for 24-48h minimum",
                "Continuous monitoring of vital signs",
                "Hourly glucose monitoring initially",
                "Aggressive DVT prophylaxis",
                "Early mobility with physiotherapy support",
                "Extended hospital stay planning",
                "Close follow-up arrangement"
            ]
        }
    }
    
    rec = recommendations.get(category, recommendations['Moderate Risk'])
    return ClinicalRecommendations(
        category=category,
        preoperative=rec['preoperative'],
        perioperative=rec['perioperative'],
        postoperative=rec['postoperative']
    )


# ============================================
# API ENDPOINTS
# ============================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Surgical Risk Assessment API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": simulation is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "assess": "/api/v1/assess-risk"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if simulation is not None else "unhealthy",
        "model_loaded": simulation is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/api/v1/assess-risk",
    response_model=RiskAssessmentResult,
    tags=["Risk Assessment"],
    summary="Assess surgical risk",
    description="Calculate surgical risk score based on patient clinical parameters using fuzzy logic"
)
async def assess_surgical_risk(patient: PatientInput):
    """
    Assess surgical risk for a patient
    
    - **age**: Patient age in years (20-100)
    - **glucose**: Fasting glucose in mmol/L (3.0-25.0)
    - **systolic_bp**: Systolic blood pressure in mmHg (80-250)
    - **bmi**: Body Mass Index in kg/m² (15.0-50.0)
    - **comorbidities**: Number of comorbidities (0-5)
    """
    
    if simulation is None:
        raise HTTPException(
            status_code=503,
            detail="Fuzzy logic model not loaded. Service unavailable."
        )
    
    try:
        # Calculate risk
        risk_score, category, color, success, error_msg = calculate_surgical_risk(
            patient.age,
            patient.glucose,
            patient.systolic_bp,
            patient.bmi,
            patient.comorbidities
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating risk: {error_msg}"
            )
        
        # Get recommendations
        recommendations = get_clinical_recommendations(category)
        
        # Convert glucose to mg/dL
        glucose_mgdl = patient.glucose * 18.0
        
        # Build result
        result = RiskAssessmentResult(
            risk_score=round(risk_score, 2),
            risk_category=RiskCategory(level=category, color=color),
            patient_data=patient,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            glucose_mgdl=round(glucose_mgdl, 1)
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/api/v1/batch-assess",
    response_model=list[RiskAssessmentResult],
    tags=["Risk Assessment"],
    summary="Batch assess surgical risk",
    description="Calculate surgical risk scores for multiple patients"
)
async def batch_assess_surgical_risk(patients: list[PatientInput]):
    """
    Assess surgical risk for multiple patients in batch
    """
    
    if simulation is None:
        raise HTTPException(
            status_code=503,
            detail="Fuzzy logic model not loaded. Service unavailable."
        )
    
    if len(patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 patients per batch request"
        )
    
    results = []
    for patient in patients:
        try:
            result = await assess_surgical_risk(patient)
            results.append(result)
        except Exception as e:
            # Continue processing other patients even if one fails
            print(f"Error processing patient: {e}")
            continue
    
    return results


@app.get(
    "/api/v1/reference-ranges",
    tags=["Reference"],
    summary="Get reference ranges",
    description="Get reference ranges for all clinical parameters"
)
async def get_reference_ranges():
    """Get reference ranges for clinical parameters"""
    return {
        "age": {
            "min": 20,
            "max": 100,
            "unit": "years"
        },
        "glucose": {
            "min": 3.0,
            "max": 25.0,
            "unit": "mmol/L",
            "conversion": "mg/dL = mmol/L × 18",
            "normal_range": "3.9-5.6 mmol/L (70-100 mg/dL) fasting"
        },
        "systolic_bp": {
            "min": 80,
            "max": 250,
            "unit": "mmHg",
            "normal_range": "<120 mmHg"
        },
        "bmi": {
            "min": 15.0,
            "max": 50.0,
            "unit": "kg/m²",
            "categories": {
                "underweight": "<18.5",
                "normal": "18.5-24.9",
                "overweight": "25.0-29.9",
                "obese": "≥30.0"
            }
        },
        "comorbidities": {
            "min": 0,
            "max": 5,
            "unit": "count",
            "types": [
                "Family history of diabetes",
                "Hypertensive",
                "Cardiovascular disease",
                "Stroke history",
                "Family history of hypertension"
            ]
        }
    }


# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

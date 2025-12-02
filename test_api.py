"""
Test client for Surgical Risk Assessment API
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_assessment(patient_data: Dict[str, Any]):
    """Test single patient risk assessment"""
    print("\n" + "="*60)
    print("Testing Single Patient Assessment")
    print("="*60)
    print(f"Input: {json.dumps(patient_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/assess-risk",
        json=patient_data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n{'─'*60}")
        print(f"Risk Score: {result['risk_score']:.2f}/100")
        print(f"Risk Category: {result['risk_category']['level']} ({result['risk_category']['color']})")
        print(f"Glucose: {result['glucose_mgdl']:.1f} mg/dL")
        print(f"Timestamp: {result['timestamp']}")
        
        print(f"\n{'─'*60}")
        print("CLINICAL RECOMMENDATIONS")
        print(f"{'─'*60}")
        
        print("\nPreoperative:")
        for rec in result['recommendations']['preoperative']:
            print(f"  • {rec}")
        
        print("\nPerioperative:")
        for rec in result['recommendations']['perioperative']:
            print(f"  • {rec}")
        
        print("\nPostoperative:")
        for rec in result['recommendations']['postoperative']:
            print(f"  • {rec}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_batch_assessment(patients: list):
    """Test batch patient assessment"""
    print("\n" + "="*60)
    print(f"Testing Batch Assessment ({len(patients)} patients)")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/batch-assess",
        json=patients
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nProcessed {len(results)} patients:")
        print(f"\n{'No.':<5} {'Age':<5} {'Glucose':<8} {'BP':<6} {'BMI':<6} {'Comorbid':<9} {'Risk Score':<11} {'Category':<15}")
        print("─" * 80)
        
        for i, result in enumerate(results, 1):
            patient = result['patient_data']
            print(f"{i:<5} {patient['age']:<5.0f} {patient['glucose']:<8.1f} "
                  f"{patient['systolic_bp']:<6.0f} {patient['bmi']:<6.1f} "
                  f"{patient['comorbidities']:<9} {result['risk_score']:<11.2f} "
                  f"{result['risk_category']['level']:<15}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_reference_ranges():
    """Test reference ranges endpoint"""
    print("\n" + "="*60)
    print("Testing Reference Ranges Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/v1/reference-ranges")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        ranges = response.json()
        print(json.dumps(ranges, indent=2))
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SURGICAL RISK ASSESSMENT API TEST CLIENT")
    print("="*60)
    print(f"API Base URL: {BASE_URL}")
    
    # Test 1: Health check
    try:
        if not test_health():
            print("\n⚠️  Health check failed. Is the API running?")
            return
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {BASE_URL}")
        print("Please ensure the API is running with: python main.py")
        return
    
    # Test 2: Single assessment - Low Risk Patient
    low_risk_patient = {
        "age": 35,
        "glucose": 5.5,
        "systolic_bp": 110,
        "bmi": 22.0,
        "comorbidities": 0
    }
    test_single_assessment(low_risk_patient)
    
    # Test 3: Single assessment - Moderate Risk Patient
    moderate_risk_patient = {
        "age": 55,
        "glucose": 8.5,
        "systolic_bp": 130,
        "bmi": 28.0,
        "comorbidities": 2,
        "family_diabetes": True,
        "hypertensive": True
    }
    test_single_assessment(moderate_risk_patient)
    
    # Test 4: Single assessment - High Risk Patient
    high_risk_patient = {
        "age": 70,
        "glucose": 15.0,
        "systolic_bp": 180,
        "bmi": 35.0,
        "comorbidities": 4,
        "family_diabetes": True,
        "hypertensive": True,
        "cardiovascular": True,
        "stroke": True
    }
    test_single_assessment(high_risk_patient)
    
    # Test 5: Batch assessment
    batch_patients = [
        {"age": 30, "glucose": 5.0, "systolic_bp": 100, "bmi": 20.0, "comorbidities": 0},
        {"age": 45, "glucose": 7.0, "systolic_bp": 120, "bmi": 25.0, "comorbidities": 1},
        {"age": 60, "glucose": 10.0, "systolic_bp": 150, "bmi": 30.0, "comorbidities": 3},
        {"age": 75, "glucose": 14.0, "systolic_bp": 170, "bmi": 33.0, "comorbidities": 4},
    ]
    test_batch_assessment(batch_patients)
    
    # Test 6: Reference ranges
    test_reference_ranges()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Visit http://localhost:8000/docs for interactive API documentation")
    print("  2. Integrate the API into your application")
    print("  3. Customize the clinical recommendations as needed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

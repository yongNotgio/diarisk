"""
Simple example script to test the Surgical Risk Assessment API
Run this while the API server is running (python main.py)
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def main():
    print("\n" + "="*70)
    print(" SURGICAL RISK ASSESSMENT API - Quick Example")
    print("="*70)
    
    # Example patient data
    patient = {
        "age": 60,
        "glucose": 10.0,  # mmol/L
        "systolic_bp": 150,  # mmHg
        "bmi": 30.0,  # kg/m²
        "comorbidities": 3,
        "family_diabetes": True,
        "hypertensive": True,
        "cardiovascular": True
    }
    
    print("\n📋 Patient Data:")
    print(f"   Age: {patient['age']} years")
    print(f"   Glucose: {patient['glucose']} mmol/L ({patient['glucose']*18:.0f} mg/dL)")
    print(f"   Systolic BP: {patient['systolic_bp']} mmHg")
    print(f"   BMI: {patient['bmi']} kg/m²")
    print(f"   Comorbidities: {patient['comorbidities']}")
    
    print("\n🔄 Sending request to API...")
    
    try:
        # Make API request
        response = requests.post(
            f"{BASE_URL}/api/v1/assess-risk",
            json=patient,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print("\n" + "="*70)
            print(" ✅ RISK ASSESSMENT RESULTS")
            print("="*70)
            
            print(f"\n🎯 Risk Score: {result['risk_score']:.1f}/100")
            
            category = result['risk_category']['level']
            color = result['risk_category']['color']
            
            # Color emoji mapping
            emoji_map = {
                'green': '🟢',
                'orange': '🟡',
                'red': '🔴'
            }
            
            print(f"   Category: {emoji_map.get(color, '⚪')} {category}")
            print(f"   Timestamp: {result['timestamp']}")
            
            print("\n" + "-"*70)
            print(" 📋 CLINICAL RECOMMENDATIONS")
            print("-"*70)
            
            print("\n✓ Preoperative Management:")
            for i, rec in enumerate(result['recommendations']['preoperative'][:3], 1):
                print(f"   {i}. {rec}")
            
            print("\n✓ Perioperative Care:")
            for i, rec in enumerate(result['recommendations']['perioperative'][:3], 1):
                print(f"   {i}. {rec}")
            
            print("\n✓ Postoperative Monitoring:")
            for i, rec in enumerate(result['recommendations']['postoperative'][:3], 1):
                print(f"   {i}. {rec}")
            
            print("\n" + "="*70)
            print(" 🌐 For full details, visit: http://localhost:8000/docs")
            print("="*70 + "\n")
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API!")
        print("   Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

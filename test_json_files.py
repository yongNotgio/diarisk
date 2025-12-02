"""
Test the API using JSON files
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_with_json_file(filename, description):
    """Test API with a JSON file"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"File: {filename}")
    print('='*70)
    
    try:
        # Load JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"\n📄 Input Data:")
        print(json.dumps(data, indent=2))
        
        # Make API request
        response = requests.post(
            f"{BASE_URL}/api/v1/assess-risk",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS!")
            print(f"   Risk Score: {result['risk_score']:.1f}/100")
            print(f"   Category: {result['risk_category']['level']}")
            print(f"   Color: {result['risk_category']['color']}")
            
            return True
        else:
            print(f"\n❌ Error {response.status_code}:")
            print(f"   {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch_json(filename):
    """Test batch endpoint with JSON file"""
    print(f"\n{'='*70}")
    print(f"Testing: Batch Assessment")
    print(f"File: {filename}")
    print('='*70)
    
    try:
        # Load JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"\n📄 Testing {len(data)} patients...")
        
        # Make API request
        response = requests.post(
            f"{BASE_URL}/api/v1/batch-assess",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"\n✅ SUCCESS! Processed {len(results)} patients:\n")
            print(f"{'No.':<5} {'Age':<5} {'Glucose':<8} {'BP':<6} {'BMI':<6} {'Score':<8} {'Category':<15}")
            print("-" * 70)
            
            for i, result in enumerate(results, 1):
                p = result['patient_data']
                print(f"{i:<5} {p['age']:<5.0f} {p['glucose']:<8.1f} "
                      f"{p['systolic_bp']:<6.0f} {p['bmi']:<6.1f} "
                      f"{result['risk_score']:<8.1f} {result['risk_category']['level']:<15}")
            
            return True
        else:
            print(f"\n❌ Error {response.status_code}:")
            print(f"   {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" TESTING API WITH JSON FILES")
    print("="*70)
    
    # Test single assessments
    test_with_json_file(
        "test_data/low_risk_patient.json",
        "Low Risk Patient"
    )
    
    test_with_json_file(
        "test_data/moderate_risk_patient.json",
        "Moderate Risk Patient"
    )
    
    test_with_json_file(
        "test_data/high_risk_patient.json",
        "High Risk Patient"
    )
    
    # Test batch assessment
    test_batch_json("test_data/batch_patients.json")
    
    print("\n" + "="*70)
    print(" ✅ All tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

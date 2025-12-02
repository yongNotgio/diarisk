import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Surgical Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD THE FUZZY MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load the fuzzy logic surgical risk model"""
    try:
        with open('surgical_risk_fuzzy_system_complete.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Handle both single simulation and package formats
        if isinstance(model, dict):
            return model['simulation'], model.get('metadata', {})
        else:
            return model, {}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, {}

# Load model
simulation, metadata = load_model()

# ============================================
# RISK CALCULATION FUNCTION
# ============================================
def calculate_surgical_risk(age, glucose, systolic_bp, bmi, comorbidities):
    """Calculate surgical risk using fuzzy logic system"""
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

# ============================================
# HEADER
# ============================================
st.title("🏥 Surgical Risk Assessment System")
st.markdown("### Fuzzy Logic-Based Clinical Decision Support")
st.markdown("---")

# ============================================
# SIDEBAR - PATIENT INPUT
# ============================================
st.sidebar.header("📋 Patient Information")

with st.sidebar:
    st.subheader("Demographics")
    age = st.slider("Age (years)", 20, 100, 55, help="Patient age in years")
    
    st.subheader("Clinical Parameters")
    glucose = st.number_input(
        "Glucose (mmol/L)", 
        min_value=3.0, 
        max_value=25.0, 
        value=8.5,
        step=0.1,
        help="Fasting glucose level"
    )
    
    glucose_mgdl = glucose * 18.0
    st.caption(f"≈ {glucose_mgdl:.0f} mg/dL")
    
    systolic_bp = st.slider(
        "Systolic BP (mmHg)", 
        80, 250, 130,
        help="Systolic blood pressure"
    )
    
    diastolic_bp = st.slider(
        "Diastolic BP (mmHg)", 
        40, 150, 80,
        help="Diastolic blood pressure"
    )
    
    bmi = st.number_input(
        "BMI (kg/m²)", 
        min_value=15.0, 
        max_value=50.0, 
        value=28.0,
        step=0.1,
        help="Body Mass Index"
    )
    
    st.subheader("Comorbidities")
    family_diabetes = st.checkbox("Family History of Diabetes")
    hypertensive = st.checkbox("Hypertensive")
    cardiovascular = st.checkbox("Cardiovascular Disease")
    stroke = st.checkbox("Stroke History")
    family_hypertension = st.checkbox("Family History of Hypertension")
    
    comorbidity_score = sum([
        family_diabetes, 
        hypertensive, 
        cardiovascular, 
        stroke, 
        family_hypertension
    ])
    
    st.info(f"**Comorbidity Score:** {comorbidity_score}/5")
    
    assess_button = st.button("🔍 Assess Surgical Risk", type="primary", use_container_width=True)

# ============================================
# MAIN CONTENT
# ============================================

if assess_button:
    if simulation is None:
        st.error("⚠️ Model not loaded. Please check the model file.")
    else:
        # Calculate risk
        risk_score, category, color, success, error_msg = calculate_surgical_risk(
            age, glucose, systolic_bp, bmi, comorbidity_score
        )
        
        if success:
            # Display results
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("## 📊 Risk Assessment Results")
                
                # Risk score gauge
                st.metric(
                    label="Surgical Risk Score",
                    value=f"{risk_score:.1f}/100",
                    delta=category
                )
                
                # Progress bar
                st.progress(risk_score / 100)
                
                # Risk category badge
                if color == 'red':
                    st.error(f"### 🔴 {category}")
                elif color == 'orange':
                    st.warning(f"### 🟡 {category}")
                else:
                    st.success(f"### 🟢 {category}")
            
            st.markdown("---")
            
            # Clinical recommendations
            st.subheader("📋 Clinical Recommendations")
            
            if category == 'High Risk':
                st.error("**⚠️ HIGH RISK PATIENT - Enhanced Perioperative Care Required**")
                st.markdown("""
                **Preoperative Optimization:**
                - ✓ Comprehensive preoperative evaluation REQUIRED
                - ✓ Cardiology consultation for cardiac risk stratification
                - ✓ Target glucose <7.0 mmol/L (126 mg/dL) preoperatively
                - ✓ Blood pressure optimization to <140/90 mmHg
                - ✓ Consider postponing elective surgery until optimization
                
                **Perioperative Management:**
                - ✓ Enhanced intraoperative monitoring
                - ✓ Continuous glucose monitoring
                - ✓ ICU bed availability required
                
                **Postoperative Care:**
                - ✓ Intensive monitoring (hourly vitals for 24h)
                - ✓ Serial glucose checks every 1-2 hours
                - ✓ Early complication detection protocols
                """)
                
            elif category == 'Moderate Risk':
                st.warning("**⚠️ MODERATE RISK PATIENT - Standard Enhanced Care**")
                st.markdown("""
                **Preoperative Evaluation:**
                - ✓ Anesthesiology evaluation recommended
                - ✓ Optimize glucose control: <8.5 mmol/L (153 mg/dL)
                - ✓ Blood pressure monitoring and control
                - ✓ Baseline ECG if >50 years
                
                **Perioperative Management:**
                - ✓ Standard enhanced monitoring
                - ✓ Glucose checks every 2-4 hours
                - ✓ Target perioperative glucose: 7.8-11.1 mmol/L
                
                **Postoperative Care:**
                - ✓ Vital signs every 4 hours for 24h
                - ✓ Standard wound care and infection precautions
                - ✓ Early mobilization encouraged
                """)
                
            else:
                st.success("**✓ LOW RISK PATIENT - Standard Perioperative Care**")
                st.markdown("""
                **Preoperative Preparation:**
                - ✓ Standard preoperative evaluation sufficient
                - ✓ Maintain current diabetes management
                - ✓ Target glucose <8.5 mmol/L
                
                **Perioperative Management:**
                - ✓ Routine monitoring protocols
                - ✓ Standard glucose checks
                
                **Postoperative Care:**
                - ✓ Standard post-operative care
                - ✓ Low probability of major complications
                """)
            
            st.markdown("---")
            
            # Patient summary
            with st.expander("📄 Patient Summary"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Demographics & Vitals:**")
                    st.write(f"- Age: {age} years")
                    st.write(f"- BMI: {bmi:.1f} kg/m²")
                    st.write(f"- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg")
                
                with col2:
                    st.markdown("**Metabolic Parameters:**")
                    st.write(f"- Glucose: {glucose:.1f} mmol/L ({glucose_mgdl:.0f} mg/dL)")
                    st.write(f"- Comorbidity Score: {comorbidity_score}/5")
                    st.write(f"- Risk Score: {risk_score:.1f}/100")
            
            # Export report button
            report_text = f"""
SURGICAL RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT PARAMETERS:
- Age: {age} years
- Glucose: {glucose:.1f} mmol/L ({glucose_mgdl:.0f} mg/dL)
- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
- BMI: {bmi:.1f} kg/m²
- Comorbidities: {comorbidity_score}/5

RISK ASSESSMENT:
- Risk Score: {risk_score:.1f}/100
- Risk Category: {category}

System: Fuzzy Logic Decision Support System v1.0
            """
            
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        else:
            st.error(f"⚠️ Error calculating risk: {error_msg}")

else:
    # Welcome message
    st.info("""
    👋 **Welcome to the Surgical Risk Assessment System**
    
    This clinical decision support tool uses **fuzzy logic** to assess surgical risk 
    in patients based on:
    - Age
    - Glucose control
    - Blood pressure
    - BMI
    - Comorbidity burden
    
    **To get started:**
    1. Enter patient information in the sidebar
    2. Click "Assess Surgical Risk"
    3. Review the detailed risk assessment and recommendations
    
    **Clinical Guidelines Applied:**
    - ADA Standards of Medical Care in Diabetes 2025
    - ACC/AHA Perioperative Cardiovascular Evaluation Guidelines
    - CPOC Perioperative Care Guidelines
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is a decision support tool. Clinical decisions should be made by qualified healthcare professionals.")
st.caption("📊 **System:** Fuzzy Logic Surgical Risk Assessment v1.0 | 70 Clinical Rules")
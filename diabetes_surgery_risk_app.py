"""
🏥 Diabetes Surgery Risk Assessment Tool
A comprehensive, patient-centric Streamlit application for assessing diabetes-related surgery risk.

This application uses state-of-the-art machine learning models to provide personalized 
risk assessments based on patient health indicators and lifestyle factors.

Author: AI Assistant
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import time
from io import BytesIO
from datetime import datetime

# Import required scikit-learn components
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.stop()

# Import PDF generation components
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
except ImportError as e:
    st.error(f"Missing PDF generation dependency: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="🏥 Diabetes Surgery Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_models():
    """Load all required models and preprocessors"""
    try:
        # Path to models directory in the current repository
        models_path = Path(__file__).parent / "models"
        
        # Verify the models directory exists
        if not models_path.exists():
            st.error(f"Models directory not found at {models_path}")
            st.error("Please ensure you have cloned the repository with all files.")
            return None, None, None, None, None
        
        # Check if all required files exist
        required_files = [
            "best_model_xgboost_gpu.pkl",
            "scaler.pkl", 
            "label_encoder.pkl",
            "feature_names.txt",
            "model_evaluation_results.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (models_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            st.error(f"Missing required model files: {', '.join(missing_files)}")
            return None, None, None, None, None
        
        # Load the best model (XGBoost GPU)
        try:
            model = joblib.load(models_path / "best_model_xgboost_gpu.pkl")
        except Exception as e:
            st.error(f"Error loading XGBoost model: {e}")
            return None, None, None, None, None
        
        # Load preprocessors
        try:
            scaler = joblib.load(models_path / "scaler.pkl")
            label_encoder = joblib.load(models_path / "label_encoder.pkl")
        except Exception as e:
            st.error(f"Error loading preprocessors: {e}")
            return None, None, None, None, None
        
        # Load feature names
        try:
            with open(models_path / "feature_names.txt", 'r') as f:
                content = f.read().strip()
                # Split by actual newlines or escaped newlines
                if '\\n' in content:
                    feature_names = [name.strip() for name in content.split('\\n') if name.strip()]
                else:
                    feature_names = [line.strip() for line in content.split('\n') if line.strip()]
            
            if len(feature_names) == 0:
                raise ValueError("No feature names found in feature_names.txt")
                
        except Exception as e:
            st.error(f"Error loading feature names: {e}")
            return None, None, None, None, None
        
        # Load model evaluation results
        try:
            with open(models_path / "model_evaluation_results.json", 'r') as f:
                model_results = json.load(f)
        except Exception as e:
            st.error(f"Error loading model evaluation results: {e}")
            return None, None, None, None, None
        
        # Verify model compatibility
        try:
            # Test prediction with dummy data
            dummy_data = np.zeros((1, len(feature_names)))
            scaled_dummy = scaler.transform(dummy_data)
            test_pred = model.predict(scaled_dummy)
            test_proba = model.predict_proba(scaled_dummy)
        except Exception as e:
            st.error(f"Model compatibility test failed: {e}")
            return None, None, None, None, None
        
        st.success("✅ All models loaded successfully!")
        return model, scaler, label_encoder, feature_names, model_results
        
    except Exception as e:
        st.error(f"Unexpected error loading models: {str(e)}")
        st.error("Please ensure all required dependencies are installed and model files are accessible.")
        return None, None, None, None, None

def get_feature_info():
    """Return comprehensive information about each feature for patient-friendly display"""
    return {
        'HighBP': {
            'question': 'Do you have high blood pressure?',
            'help': 'High blood pressure (hypertension) is when your blood pressure is consistently 140/90 mmHg or higher.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'HighChol': {
            'question': 'Do you have high cholesterol?',
            'help': 'High cholesterol means your total cholesterol level is 240 mg/dL or higher.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'CholCheck': {
            'question': 'Have you had a cholesterol check in the past 5 years?',
            'help': 'Regular cholesterol screening helps monitor cardiovascular health.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'BMI': {
            'question': 'What is your Body Mass Index (BMI)?',
            'help': 'BMI is calculated as weight (kg) ÷ height (m)². Normal range: 18.5-24.9.',
            'type': 'continuous',
            'range': (10, 60),
            'default': 25
        },
        'Smoker': {
            'question': 'Have you smoked at least 100 cigarettes in your entire life?',
            'help': '100 cigarettes equals about 5 packs. This includes any form of tobacco smoking.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'Stroke': {
            'question': 'Have you ever been told you had a stroke?',
            'help': 'A stroke occurs when blood flow to part of the brain is blocked or reduced.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'HeartDiseaseorAttack': {
            'question': 'Have you ever been told you had coronary heart disease or a heart attack?',
            'help': 'This includes any heart condition or myocardial infarction (heart attack).',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'PhysActivity': {
            'question': 'Have you had any physical activity in the past 30 days (not including job)?',
            'help': 'Any recreational physical activity, exercise, or sports in your free time.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'Fruits': {
            'question': 'Do you consume fruit 1 or more times per day?',
            'help': 'Fresh, frozen, or canned fruits (not including fruit juices).',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'Veggies': {
            'question': 'Do you consume vegetables 1 or more times per day?',
            'help': 'Fresh, frozen, or canned vegetables (not including French fries).',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'HvyAlcoholConsump': {
            'question': 'Are you a heavy alcohol consumer?',
            'help': 'Heavy drinking: More than 14 drinks/week for men, more than 7 drinks/week for women.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'AnyHealthcare': {
            'question': 'Do you have any kind of healthcare coverage?',
            'help': 'Including health insurance, prepaid plans such as HMO, government plans, etc.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'NoDocbcCost': {
            'question': 'In the past 12 months, was there a time you needed to see a doctor but could not because of cost?',
            'help': 'Financial barriers to accessing healthcare services.',
            'type': 'binary',
            'options': ['Yes', 'No']  # Note: inverted logic for this question
        },
        'GenHlth': {
            'question': 'How would you rate your general health?',
            'help': 'Your overall perception of your current health status.',
            'type': 'scale',
            'options': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'],
            'values': [1, 2, 3, 4, 5]
        },
        'MentHlth': {
            'question': 'For how many days in the past 30 days was your mental health not good?',
            'help': 'Including stress, depression, and problems with emotions. Enter 0 if none.',
            'type': 'continuous',
            'range': (0, 30),
            'default': 0
        },
        'PhysHlth': {
            'question': 'For how many days in the past 30 days was your physical health not good?',
            'help': 'Including physical illness and injury. Enter 0 if none.',
            'type': 'continuous',
            'range': (0, 30),
            'default': 0
        },
        'DiffWalk': {
            'question': 'Do you have serious difficulty walking or climbing stairs?',
            'help': 'Any mobility issues that significantly impact your daily activities.',
            'type': 'binary',
            'options': ['No', 'Yes']
        },
        'Sex': {
            'question': 'What is your biological sex?',
            'help': 'Biological sex as recorded in medical records.',
            'type': 'binary',
            'options': ['Female', 'Male']
        },
        'Age': {
            'question': 'What is your age group?',
            'help': 'Select the age range that includes your current age.',
            'type': 'scale',
            'options': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                       '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        },
        'Education': {
            'question': 'What is your highest level of education?',
            'help': 'Your highest completed level of formal education.',
            'type': 'scale',
            'options': ['Never attended school', 'Elementary', 'Some high school', 
                       'High school graduate', 'Some college', 'College graduate'],
            'values': [1, 2, 3, 4, 5, 6]
        },
        'Income': {
            'question': 'What is your household income level?',
            'help': 'Your total household income before taxes.',
            'type': 'scale',
            'options': ['Less than $10,000', '$10,000-$15,000', '$15,000-$20,000',
                       '$20,000-$25,000', '$25,000-$35,000', '$35,000-$50,000',
                       '$50,000-$75,000', '$75,000+'],
            'values': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        'Diabetes_binary': {
            'question': 'Do you have diabetes or prediabetes?',
            'help': 'Have you been diagnosed with diabetes or prediabetes by a healthcare provider?',
            'type': 'binary',
            'options': ['No', 'Yes']
        }
    }

def collect_patient_data():
    """Collect patient data through user-friendly interface"""
    st.markdown("### 📋 Patient Health Assessment")
    st.markdown("Please answer the following questions honestly. All information is confidential and used only for risk assessment.")
    
    feature_info = get_feature_info()
    patient_data = {}
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Medical History", "💪 Lifestyle", "🧠 Mental & Physical Health", "👤 Demographics"])
    
    with tab1:
        st.markdown("#### Medical History & Conditions")
        col1, col2 = st.columns(2)
        
        with col1:
            # Diabetes
            diabetes_options = feature_info['Diabetes_binary']['options']
            diabetes_val = st.selectbox(
                feature_info['Diabetes_binary']['question'],
                diabetes_options,
                help=feature_info['Diabetes_binary']['help'],
                key='diabetes'
            )
            patient_data['Diabetes_binary'] = diabetes_options.index(diabetes_val)
            
            # High BP
            bp_options = feature_info['HighBP']['options']
            bp_val = st.selectbox(
                feature_info['HighBP']['question'],
                bp_options,
                help=feature_info['HighBP']['help'],
                key='bp'
            )
            patient_data['HighBP'] = bp_options.index(bp_val)
            
            # High Cholesterol
            chol_options = feature_info['HighChol']['options']
            chol_val = st.selectbox(
                feature_info['HighChol']['question'],
                chol_options,
                help=feature_info['HighChol']['help'],
                key='chol'
            )
            patient_data['HighChol'] = chol_options.index(chol_val)
            
            # Cholesterol Check
            cholcheck_options = feature_info['CholCheck']['options']
            cholcheck_val = st.selectbox(
                feature_info['CholCheck']['question'],
                cholcheck_options,
                help=feature_info['CholCheck']['help'],
                key='cholcheck'
            )
            patient_data['CholCheck'] = cholcheck_options.index(cholcheck_val)
        
        with col2:
            # Stroke
            stroke_options = feature_info['Stroke']['options']
            stroke_val = st.selectbox(
                feature_info['Stroke']['question'],
                stroke_options,
                help=feature_info['Stroke']['help'],
                key='stroke'
            )
            patient_data['Stroke'] = stroke_options.index(stroke_val)
            
            # Heart Disease
            heart_options = feature_info['HeartDiseaseorAttack']['options']
            heart_val = st.selectbox(
                feature_info['HeartDiseaseorAttack']['question'],
                heart_options,
                help=feature_info['HeartDiseaseorAttack']['help'],
                key='heart'
            )
            patient_data['HeartDiseaseorAttack'] = heart_options.index(heart_val)
            
            # Healthcare Coverage
            healthcare_options = feature_info['AnyHealthcare']['options']
            healthcare_val = st.selectbox(
                feature_info['AnyHealthcare']['question'],
                healthcare_options,
                help=feature_info['AnyHealthcare']['help'],
                key='healthcare'
            )
            patient_data['AnyHealthcare'] = healthcare_options.index(healthcare_val)
            
            # Doctor Cost Barrier
            cost_options = feature_info['NoDocbcCost']['options']
            cost_val = st.selectbox(
                feature_info['NoDocbcCost']['question'],
                cost_options,
                help=feature_info['NoDocbcCost']['help'],
                key='cost'
            )
            patient_data['NoDocbcCost'] = cost_options.index(cost_val)
    
    with tab2:
        st.markdown("#### Lifestyle & Habits")
        col1, col2 = st.columns(2)
        
        with col1:
            # BMI
            bmi_val = st.slider(
                feature_info['BMI']['question'],
                min_value=feature_info['BMI']['range'][0],
                max_value=feature_info['BMI']['range'][1],
                value=feature_info['BMI']['default'],
                help=feature_info['BMI']['help'],
                key='bmi'
            )
            patient_data['BMI'] = bmi_val
            
            # Calculate BMI category
            if bmi_val < 18.5:
                bmi_category = "Underweight"
                bmi_color = "blue"
            elif bmi_val < 25:
                bmi_category = "Normal weight"
                bmi_color = "green"
            elif bmi_val < 30:
                bmi_category = "Overweight"
                bmi_color = "orange"
            else:
                bmi_category = "Obese"
                bmi_color = "red"
            
            st.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")
            
            # Smoking
            smoke_options = feature_info['Smoker']['options']
            smoke_val = st.selectbox(
                feature_info['Smoker']['question'],
                smoke_options,
                help=feature_info['Smoker']['help'],
                key='smoke'
            )
            patient_data['Smoker'] = smoke_options.index(smoke_val)
            
            # Physical Activity
            activity_options = feature_info['PhysActivity']['options']
            activity_val = st.selectbox(
                feature_info['PhysActivity']['question'],
                activity_options,
                help=feature_info['PhysActivity']['help'],
                key='activity'
            )
            patient_data['PhysActivity'] = activity_options.index(activity_val)
        
        with col2:
            # Fruits
            fruit_options = feature_info['Fruits']['options']
            fruit_val = st.selectbox(
                feature_info['Fruits']['question'],
                fruit_options,
                help=feature_info['Fruits']['help'],
                key='fruit'
            )
            patient_data['Fruits'] = fruit_options.index(fruit_val)
            
            # Vegetables
            veg_options = feature_info['Veggies']['options']
            veg_val = st.selectbox(
                feature_info['Veggies']['question'],
                veg_options,
                help=feature_info['Veggies']['help'],
                key='veg'
            )
            patient_data['Veggies'] = veg_options.index(veg_val)
            
            # Heavy Alcohol
            alcohol_options = feature_info['HvyAlcoholConsump']['options']
            alcohol_val = st.selectbox(
                feature_info['HvyAlcoholConsump']['question'],
                alcohol_options,
                help=feature_info['HvyAlcoholConsump']['help'],
                key='alcohol'
            )
            patient_data['HvyAlcoholConsump'] = alcohol_options.index(alcohol_val)
    
    with tab3:
        st.markdown("#### Mental & Physical Health")
        col1, col2 = st.columns(2)
        
        with col1:
            # General Health
            genhealth_options = feature_info['GenHlth']['options']
            genhealth_val = st.selectbox(
                feature_info['GenHlth']['question'],
                genhealth_options,
                help=feature_info['GenHlth']['help'],
                key='genhealth'
            )
            patient_data['GenHlth'] = feature_info['GenHlth']['values'][genhealth_options.index(genhealth_val)]
            
            # Mental Health Days
            mental_val = st.slider(
                feature_info['MentHlth']['question'],
                min_value=feature_info['MentHlth']['range'][0],
                max_value=feature_info['MentHlth']['range'][1],
                value=feature_info['MentHlth']['default'],
                help=feature_info['MentHlth']['help'],
                key='mental'
            )
            patient_data['MentHlth'] = mental_val
        
        with col2:
            # Physical Health Days
            physical_val = st.slider(
                feature_info['PhysHlth']['question'],
                min_value=feature_info['PhysHlth']['range'][0],
                max_value=feature_info['PhysHlth']['range'][1],
                value=feature_info['PhysHlth']['default'],
                help=feature_info['PhysHlth']['help'],
                key='physical'
            )
            patient_data['PhysHlth'] = physical_val
            
            # Difficulty Walking
            walk_options = feature_info['DiffWalk']['options']
            walk_val = st.selectbox(
                feature_info['DiffWalk']['question'],
                walk_options,
                help=feature_info['DiffWalk']['help'],
                key='walk'
            )
            patient_data['DiffWalk'] = walk_options.index(walk_val)
    
    with tab4:
        st.markdown("#### Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sex
            sex_options = feature_info['Sex']['options']
            sex_val = st.selectbox(
                feature_info['Sex']['question'],
                sex_options,
                help=feature_info['Sex']['help'],
                key='sex'
            )
            patient_data['Sex'] = sex_options.index(sex_val)
            
            # Age
            age_options = feature_info['Age']['options']
            age_val = st.selectbox(
                feature_info['Age']['question'],
                age_options,
                help=feature_info['Age']['help'],
                key='age'
            )
            patient_data['Age'] = feature_info['Age']['values'][age_options.index(age_val)]
        
        with col2:
            # Education
            edu_options = feature_info['Education']['options']
            edu_val = st.selectbox(
                feature_info['Education']['question'],
                edu_options,
                help=feature_info['Education']['help'],
                key='edu'
            )
            patient_data['Education'] = feature_info['Education']['values'][edu_options.index(edu_val)]
            
            # Income
            income_options = feature_info['Income']['options']
            income_val = st.selectbox(
                feature_info['Income']['question'],
                income_options,
                help=feature_info['Income']['help'],
                key='income'
            )
            patient_data['Income'] = feature_info['Income']['values'][income_options.index(income_val)]
    
    return patient_data

def predict_risk(patient_data, model, scaler, label_encoder, feature_names):
    """Make risk prediction for patient"""
    try:
        # Create feature array in correct order
        feature_array = np.array([patient_data[feature] for feature in feature_names]).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Convert prediction back to risk level
        risk_level = label_encoder.inverse_transform([prediction])[0]
        
        # Get class probabilities
        class_names = label_encoder.classes_
        prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        
        return risk_level, prob_dict, probabilities
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def generate_pdf_report(patient_data, risk_level, prob_dict):
    """Generate a mobile-friendly PDF report"""
    buffer = BytesIO()
    
    # Create PDF document with portrait orientation (mobile-friendly)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.blue,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue,
        spaceBefore=15,
        spaceAfter=10
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    # Build story (content)
    story = []
    
    # Title
    story.append(Paragraph("🏥 Diabetes Surgery Risk Assessment Report", title_style))
    story.append(Spacer(1, 20))
    
    # Assessment date
    assessment_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    story.append(Paragraph(f"<b>Assessment Date:</b> {assessment_date}", normal_style))
    story.append(Spacer(1, 15))
    
    # Risk level section
    story.append(Paragraph("Risk Assessment Results", heading_style))
    
    # Risk level with appropriate styling
    if risk_level == "High":
        risk_color = colors.red
        risk_text = "⚠️ HIGH RISK"
        risk_description = "Your assessment indicates a HIGH risk for diabetes-related surgery complications. We strongly recommend immediate consultation with your healthcare provider to discuss risk management strategies."
    elif risk_level == "Moderate":
        risk_color = colors.orange
        risk_text = "⚠️ MODERATE RISK"
        risk_description = "Your assessment indicates a MODERATE risk for diabetes-related surgery complications. Consider discussing your health status with your healthcare provider before any planned surgeries."
    else:
        risk_color = colors.green
        risk_text = "✅ LOW RISK"
        risk_description = "Your assessment indicates a LOW risk for diabetes-related surgery complications. Continue maintaining your current health practices and regular medical check-ups."
    
    risk_style = ParagraphStyle(
        'RiskStyle',
        parent=normal_style,
        fontSize=14,
        textColor=risk_color,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=10
    )
    
    story.append(Paragraph(risk_text, risk_style))
    story.append(Paragraph(risk_description, normal_style))
    story.append(Spacer(1, 15))
    
    # Risk probabilities
    story.append(Paragraph("Risk Probability Breakdown", heading_style))
    prob_data = [['Risk Level', 'Probability']]
    for risk, prob in prob_dict.items():
        prob_data.append([risk, f"{prob*100:.1f}%"])
    
    prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # Health profile summary
    story.append(Paragraph("Your Health Profile", heading_style))
    
    # Get feature info for readable labels
    feature_info = get_feature_info()
    
    # Create a summary of key health indicators
    health_data = [['Health Indicator', 'Your Response']]
    
    key_features = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Smoker', 
                   'PhysActivity', 'GenHlth', 'Age', 'Sex']
    
    for feature in key_features:
        if feature in patient_data and feature in feature_info:
            info = feature_info[feature]
            value = patient_data[feature]
            
            # Format the value for display
            if info['type'] == 'binary':
                display_value = info['options'][value] if value < len(info['options']) else str(value)
            elif info['type'] == 'scale' and 'options' in info:
                try:
                    idx = info['values'].index(value)
                    display_value = info['options'][idx]
                except (ValueError, IndexError):
                    display_value = str(value)
            else:
                display_value = str(value)
            
            # Clean up the question for display
            question = info['question'].replace('Do you have', '').replace('Have you', '').replace('?', '').strip()
            if feature == 'BMI':
                question = 'BMI'
            elif feature == 'GenHlth':
                question = 'General Health'
            elif feature == 'PhysActivity':
                question = 'Physical Activity'
            elif feature == 'Diabetes_binary':
                question = 'Diabetes/Prediabetes'
            
            health_data.append([question, display_value])
    
    health_table = Table(health_data, colWidths=[2.5*inch, 2*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    
    story.append(health_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This assessment is for educational purposes only and does not replace professional medical advice. 
    Always consult with your healthcare provider before making any medical decisions. This assessment 
    is based on general population patterns and may not reflect your individual medical situation.
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Generated by Diabetes Surgery Risk Assessment Tool", footer_style))
    story.append(Paragraph("Powered by Advanced Machine Learning", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_risk_assessment(risk_level, prob_dict, probabilities):
    """Display risk assessment results"""
    st.markdown("## 🎯 Your Surgery Risk Assessment")
    
    # Main risk display using Streamlit components
    if risk_level == "High":
        st.error("⚠️ **HIGH RISK** - Your assessment indicates a HIGH risk for diabetes-related surgery complications. We strongly recommend immediate consultation with your healthcare provider to discuss risk management strategies.")
    elif risk_level == "Moderate":
        st.warning("⚠️ **MODERATE RISK** - Your assessment indicates a MODERATE risk for diabetes-related surgery complications. Consider discussing your health status with your healthcare provider before any planned surgeries.")
    else:  # Low
        st.success("✅ **LOW RISK** - Your assessment indicates a LOW risk for diabetes-related surgery complications. Continue maintaining your current health practices and regular medical check-ups.")
    
    # Risk probability visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Risk Breakdown")
        for risk, prob in prob_dict.items():
            confidence = prob * 100
            if risk == risk_level:
                st.metric(
                    label=f"{risk} Risk",
                    value=f"{confidence:.1f}%",
                    delta=None
                )
            else:
                st.write(f"**{risk}:** {confidence:.1f}%")
    
    with col2:
        # Create probability chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(prob_dict.keys()),
                y=list(prob_dict.values()),
                marker_color=['red' if k=='High' else 'orange' if k=='Moderate' else 'green' for k in prob_dict.keys()],
                text=[f"{v*100:.1f}%" for v in prob_dict.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Risk Probability Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Probability",
            yaxis=dict(tickformat='.0%'),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def generate_recommendations(patient_data, risk_level):
    """Generate personalized health recommendations"""
    st.markdown("## 💡 Personalized Health Recommendations")
    
    recommendations = []
    
    # BMI recommendations
    bmi = patient_data.get('BMI', 25)
    if bmi >= 30:
        recommendations.append({
            'category': '⚖️ Weight Management',
            'recommendation': 'Consider working with a nutritionist and fitness trainer to develop a safe weight loss plan.',
            'priority': 'High'
        })
    elif bmi >= 25:
        recommendations.append({
            'category': '⚖️ Weight Management', 
            'recommendation': 'Maintain current weight and consider healthy lifestyle adjustments.',
            'priority': 'Medium'
        })
    
    # Blood pressure
    if patient_data.get('HighBP', 0) == 1:
        recommendations.append({
            'category': '💓 Cardiovascular Health',
            'recommendation': 'Continue monitoring blood pressure regularly and follow your prescribed treatment plan.',
            'priority': 'High'
        })
    
    # Physical activity
    if patient_data.get('PhysActivity', 0) == 0:
        recommendations.append({
            'category': '🏃‍♂️ Physical Activity',
            'recommendation': 'Start with light physical activities like walking for 30 minutes daily.',
            'priority': 'Medium'
        })
    
    # Diet
    if patient_data.get('Fruits', 0) == 0 or patient_data.get('Veggies', 0) == 0:
        recommendations.append({
            'category': '🥗 Nutrition',
            'recommendation': 'Increase daily intake of fruits and vegetables to at least 5 servings per day.',
            'priority': 'Medium'
        })
    
    # Smoking
    if patient_data.get('Smoker', 0) == 1:
        recommendations.append({
            'category': '🚭 Smoking Cessation',
            'recommendation': 'Consider joining a smoking cessation program to reduce surgical and overall health risks.',
            'priority': 'High'
        })
    
    # Mental health
    mental_days = patient_data.get('MentHlth', 0)
    if mental_days > 14:
        recommendations.append({
            'category': '🧠 Mental Health',
            'recommendation': 'Consider speaking with a mental health professional about stress management techniques.',
            'priority': 'Medium'
        })
    
    # Healthcare access
    if patient_data.get('AnyHealthcare', 0) == 0:
        recommendations.append({
            'category': '🏥 Healthcare Access',
            'recommendation': 'Explore healthcare coverage options to ensure regular medical monitoring.',
            'priority': 'High'
        })
    
    # Display recommendations using Streamlit containers
    if recommendations:
        for rec in recommendations:
            with st.container():
                if rec['priority'] == 'High':
                    st.error(f"**{rec['category']}** - High Priority")
                elif rec['priority'] == 'Medium':
                    st.warning(f"**{rec['category']}** - Medium Priority")
                else:
                    st.info(f"**{rec['category']}** - Low Priority")
                st.write(rec['recommendation'])
                st.write("")  # Add spacing
    else:
        st.success("🎉 Great job! You're maintaining excellent health habits.")

def display_model_info(model_results):
    """Display information about the AI model"""
    st.markdown("## 🤖 About This Assessment Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How It Works
        This tool uses advanced machine learning algorithms trained on diabetes health indicators to assess surgery risk. The model analyzes your health profile and compares it to patterns found in medical data.
        
        ### Model Performance
        Our AI model has been rigorously tested and shows excellent performance:
        """)
        
        # Display model metrics
        best_model = "XGBoost_GPU"
        if best_model in model_results:
            metrics = model_results[best_model]
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            st.metric("Precision", f"{metrics['precision']*100:.1f}%")
            st.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")
    
    with col2:
        st.markdown("""
        ### Important Disclaimers
        
        ⚠️ **This tool is for educational purposes only and does not replace professional medical advice.**
        
        ✅ **Always consult with your healthcare provider before making any medical decisions.**
        
        ✅ **This assessment is based on general population patterns and may not reflect your individual medical situation.**
        
        ✅ **Keep your healthcare provider informed about any changes in your health status.**
        """)

def main():
    """Main application function"""
    
    # Header
    st.title("🏥 Diabetes Surgery Risk Assessment Tool")
    
    st.markdown("""
    Welcome to our comprehensive diabetes surgery risk assessment tool. This application uses state-of-the-art 
    machine learning to help you understand your potential risk factors for diabetes-related surgical complications.
    
    **Please complete the health assessment below to receive your personalized risk evaluation.**
    """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        model, scaler, label_encoder, feature_names, model_results = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## ℹ️ Information")
        st.markdown("""
        ### What is assessed?
        - Medical conditions
        - Lifestyle factors  
        - Mental & physical health
        - Demographics
        
        ### Risk levels:
        - 🟢 **Low Risk**: Minimal complications expected
        - 🟡 **Moderate Risk**: Some precautions needed
        - 🔴 **High Risk**: Careful evaluation required
        
        ### Data Privacy
        Your information is processed locally and not stored or shared.
        """)
        
        st.markdown("---")
        st.markdown("**Need help?** Contact your healthcare provider.")
    
    # Collect patient data
    patient_data = collect_patient_data()
    
    # Assessment button
    if st.button("🎯 Generate Risk Assessment", type="primary", use_container_width=True):
        if len(patient_data) == len(feature_names):
            with st.spinner("Analyzing your health profile..."):
                # Add a brief delay for better UX
                time.sleep(1)
                
                # Make prediction
                risk_level, prob_dict, probabilities = predict_risk(
                    patient_data, model, scaler, label_encoder, feature_names
                )
                
                if risk_level:
                    # Display results
                    display_risk_assessment(risk_level, prob_dict, probabilities)
                    
                    # Generate recommendations
                    generate_recommendations(patient_data, risk_level)
                    
                    # Model information
                    display_model_info(model_results)
                    
                    # Download report option
                    st.markdown("---")
                    st.markdown("### 📄 Save Your Assessment")
                    st.markdown("Download a comprehensive PDF report optimized for mobile viewing:")
                    
                    # Generate PDF report
                    try:
                        pdf_buffer = generate_pdf_report(patient_data, risk_level, prob_dict)
                        
                        st.download_button(
                            label="📱 Download PDF Report (Mobile Optimized)",
                            data=pdf_buffer,
                            file_name=f"diabetes_surgery_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.info("💡 **Tip:** This PDF is optimized for mobile viewing and can be easily shared with your healthcare provider.")
                        
                    except Exception as e:
                        st.error(f"Error generating PDF report: {e}")
                        st.info("PDF generation is temporarily unavailable. Please screenshot your results if needed.")
                    
        else:
            st.error("Please complete all questions before generating assessment.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🏥 Diabetes Surgery Risk Assessment Tool | Powered by Advanced Machine Learning**
    
    *For educational purposes only. Always consult with healthcare professionals.*
    """)

if __name__ == "__main__":
    main()

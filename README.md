# 🏥 Diabetes Surgery Risk Assessment Tool

A comprehensive, patient-centric Streamlit application for assessing diabetes-related surgery risk using advanced machine learning models.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- All model files in the `../models/` directory

### Installation

1. Navigate to the application directory:
```bash
cd diabetes_risk_app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run diabetes_surgery_risk_app.py
```

4. Open your web browser and go to: `http://localhost:8501`

## 📁 Required Files Structure

Ensure the following files exist in the `../models/` directory:
```
models/
├── best_model_xgboost_gpu.pkl      # Main ML model
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl               # Target encoder
├── feature_names.txt               # Feature names list
├── model_evaluation_results.json   # Model performance metrics
└── DEPLOYMENT_INSTRUCTIONS.md      # Deployment guide
```

## 🎯 Features

### Patient-Centric Interface
- **Intuitive Questionnaire**: Easy-to-understand health questions
- **Organized Tabs**: Medical history, lifestyle, mental health, demographics
- **Interactive Elements**: Sliders, dropdowns, and helpful tooltips
- **Real-time Validation**: Immediate feedback on inputs

### Advanced Risk Assessment
- **AI-Powered Predictions**: Uses state-of-the-art XGBoost model
- **Risk Categorization**: High, Moderate, and Low risk levels
- **Confidence Scores**: Probability breakdown for each risk level
- **Visual Analytics**: Interactive charts and risk visualizations

### Personalized Recommendations
- **Targeted Advice**: Based on individual health profile
- **Priority Levels**: High, Medium, Low priority recommendations
- **Multiple Categories**: Weight, cardiovascular, nutrition, mental health
- **Actionable Steps**: Specific, implementable suggestions

### Professional Features
- **Model Transparency**: Performance metrics and methodology
- **Data Privacy**: Local processing, no data storage
- **Report Generation**: Downloadable CSV assessment reports
- **Medical Disclaimers**: Clear limitations and professional advice

## 📊 Health Assessment Categories

### 🏥 Medical History
- Diabetes status
- Blood pressure conditions
- Cholesterol levels and monitoring
- Stroke and heart disease history
- Healthcare access and barriers

### 💪 Lifestyle Factors
- Body Mass Index (BMI) with automatic categorization
- Smoking history
- Physical activity levels
- Dietary habits (fruits and vegetables)
- Alcohol consumption patterns

### 🧠 Mental & Physical Health
- General health perception
- Mental health days in past month
- Physical health limitations
- Mobility and walking difficulties

### 👤 Demographics
- Age group classification
- Education level
- Income bracket
- Biological sex

## 🤖 AI Model Information

### Model Performance
- **Accuracy**: 100%
- **Precision**: 100%
- **F1-Score**: 100%
- **Model Type**: XGBoost with GPU acceleration

### Risk Categories
- **🟢 Low Risk**: Minimal surgical complications expected
- **🟡 Moderate Risk**: Some precautions and monitoring needed
- **🔴 High Risk**: Careful evaluation and risk management required

## 🛡️ Data Privacy & Security

- **Local Processing**: All calculations performed locally
- **No Data Storage**: Information not saved or transmitted
- **Session-Based**: Data cleared when browser closes
- **HIPAA Considerations**: Designed with healthcare privacy in mind

## ⚠️ Important Disclaimers

- **Educational Tool Only**: Not a substitute for professional medical advice
- **Clinical Consultation Required**: Always consult healthcare providers
- **Individual Variation**: Results based on population patterns
- **Medical Decisions**: Should not be used for treatment decisions

## 🚀 Deployment Options

### Local Development
```bash
streamlit run diabetes_surgery_risk_app.py
```

### Production Deployment
```bash
streamlit run diabetes_surgery_risk_app.py --server.port 8501 --server.address 0.0.0.0
```

### Cloud Deployment
The application is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS/Azure/GCP
- Docker containers

## 🔧 Customization

### Adding New Features
1. Modify `collect_patient_data()` for new questions
2. Update `get_feature_info()` for new feature definitions
3. Adjust `generate_recommendations()` for new advice categories

### Styling Customization
- Modify CSS in the `st.markdown()` sections
- Adjust color schemes in risk display functions
- Update chart styling in Plotly visualizations

## 📞 Support

For technical support or questions about the application:
1. Check the console for error messages
2. Verify all model files are present
3. Ensure Python dependencies are installed
4. Contact your healthcare IT administrator

## 📈 Future Enhancements

Potential improvements for future versions:
- Multi-language support
- Integration with electronic health records (EHR)
- Batch patient processing
- Advanced visualization dashboards
- Mobile-responsive design optimization
- Patient progress tracking over time

---

**Version**: 1.0.0  
**Last Updated**: July 2025  
**Compatible With**: Python 3.8+, Streamlit 1.28+

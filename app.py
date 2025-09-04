# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# ------------------------------------------------
# Load saved models
# ------------------------------------------------
reg_loaded = load('linear_regression_model.joblib')
reg_model, reg_features = reg_loaded[0], reg_loaded[1]

clf_loaded = load('rf_classifier.joblib')
clf_model, clf_features = clf_loaded[0], clf_loaded[1]

# Parental Education columns (one-hot)
parental_cols = [col for col in reg_features if col.startswith('Parental_Education_Level_')]

# Validation ranges
valid_ranges = {
    "Study_Hours_per_Week": (0, 100),
    "Attendance_Rate": (0, 100),
    "Past_Exam_Scores": (0, 100)
}

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def grade(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"

# ------------------------------------------------
# Streamlit App Layout
# ------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction")
st.markdown("Predict **Final Exam Score** or **Pass/Fail Outcome** using AI models!")

# Sidebar: choose prediction type
prediction_type = st.sidebar.radio("Select Prediction Type", ("Final Exam Score", "Pass/Fail Outcome"))

# ------------------------------------------------
# User Input Form
# ------------------------------------------------
with st.form(key='student_input_form'):
    st.subheader("Enter Student Details")
    
    # Binary inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    internet = st.selectbox("Internet Access at Home", ["No", "Yes"])
    extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    
    # Numeric inputs
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=100, value=10)
    attendance = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, value=90)
    past_scores = st.number_input("Past Exam Scores", min_value=0, max_value=100, value=75)
    
    # Parental education
    st.markdown("**Parental Education Level**")
    parental_option = st.selectbox("Choose Level", [col.replace("Parental_Education_Level_", "") for col in parental_cols])
    
    # Submit button
    submit_btn = st.form_submit_button(label="Predict")

# ------------------------------------------------
# Make Prediction
# ------------------------------------------------
if submit_btn:
    # Prepare input dataframe
    input_dict = {}
    for col in reg_features:
        if col == "Gender":
            input_dict[col] = 1 if gender == "Female" else 0
        elif col == "Internet_Access_at_Home":
            input_dict[col] = 1 if internet == "Yes" else 0
        elif col == "Extracurricular_Activities":
            input_dict[col] = 1 if extra == "Yes" else 0
        elif col in parental_cols:
            input_dict[col] = 1 if col == f"Parental_Education_Level_{parental_option}" else 0
        elif col == "Study_Hours_per_Week":
            input_dict[col] = study_hours
        elif col == "Attendance_Rate":
            input_dict[col] = attendance
        elif col == "Past_Exam_Scores":
            input_dict[col] = past_scores
    
    input_df = pd.DataFrame([input_dict], columns=reg_features)
    
    if prediction_type == "Final Exam Score":
        predicted_score = int(round(reg_model.predict(input_df.values)[0]))
        student_grade = grade(predicted_score)
        
        st.success(f"**Predicted Final Exam Score:** {predicted_score}")
        st.info(f"**Predicted Grade:** {student_grade}")
        
        st.markdown("""
        **Grade Ranges:**
        - A: 90-100
        - B: 80-89
        - C: 70-79
        - D: 60-69
        - F: 0-59
        """)
    
    else:  # Pass/Fail
        input_df_class = input_df[clf_features]
        predicted_class = clf_model.predict(input_df_class.values)[0]
        proba = clf_model.predict_proba(input_df_class.values)[0]
        predicted_label = "Pass" if predicted_class == 1 else "Fail"
        
        st.success(f"**Predicted Student Outcome:** {predicted_label}")
        st.info(f"Probability to Pass: {proba[1]*100:.2f}%")
        st.warning(f"Probability to Fail: {proba[0]*100:.2f}%")

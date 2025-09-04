# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -------------------------
# Load Models
# -------------------------
reg_loaded = load('linear_regression_model.joblib')
reg_model, reg_features = reg_loaded[0], reg_loaded[1]

clf_loaded = load('rf_classifier.joblib')
clf_model, clf_features = clf_loaded[0], clf_loaded[1]

# Parental columns (for one-hot encoding)
parental_cols = [col for col in reg_features if col.startswith('Parental_Education_Level_')]

# Validation ranges for numeric inputs
valid_ranges = {
    "Study_Hours_per_Week": (0, 100),
    "Attendance_Rate": (0, 100),
    "Past_Exam_Scores": (0, 100)
}

# -------------------------
# Helper Functions
# -------------------------
def grade(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"

def get_input():
    user_input_values = {}

    user_input_values["Gender"] = st.selectbox("Gender", options=["Male", "Female"])
    user_input_values["Gender"] = 0 if user_input_values["Gender"]=="Male" else 1

    user_input_values["Internet_Access_at_Home"] = st.selectbox("Internet Access at Home", options=["No", "Yes"])
    user_input_values["Internet_Access_at_Home"] = 0 if user_input_values["Internet_Access_at_Home"]=="No" else 1

    user_input_values["Extracurricular_Activities"] = st.selectbox("Extracurricular Activities", options=["No", "Yes"])
    user_input_values["Extracurricular_Activities"] = 0 if user_input_values["Extracurricular_Activities"]=="No" else 1

    # Numeric inputs
    for col, (low, high) in valid_ranges.items():
        user_input_values[col] = st.number_input(f"{col} ({low}-{high})", min_value=low, max_value=high, value=(low+high)//2)

    # Parental Education Level
    options = [col.replace("Parental_Education_Level_", "") for col in parental_cols]
    chosen_level = st.selectbox("Parental Education Level", options)
    for col in parental_cols:
        user_input_values[col] = 1 if col.endswith(chosen_level) else 0

    return pd.DataFrame([user_input_values], columns=reg_features)

# -------------------------
# Streamlit App UI
# -------------------------
st.title("🎓 Student Performance Prediction")

prediction_type = st.radio("Choose Prediction Type", ["Final Exam Score", "Pass/Fail Outcome"])

input_df = get_input()

if st.button("Predict"):
    if prediction_type == "Final Exam Score":
        predicted_score = int(round(reg_model.predict(input_df.values)[0]))
        student_grade = grade(predicted_score)
        st.success(f"Predicted Final Exam Score: {predicted_score}")
        st.info(f"Predicted Grade: {student_grade}")

        grade_ranges = {
            "A": (90, 100),
            "B": (80, 89),
            "C": (70, 79),
            "D": (60, 69),
            "F": (0, 59)
        }

        st.write("Grade Ranges:")
        for g, (low, high) in grade_ranges.items():
            st.write(f"{g}: {low}-{high}")

    else:
        # Classification
        input_df_class = input_df[clf_features]
        predicted_class = clf_model.predict(input_df_class.values)[0]
        proba = clf_model.predict_proba(input_df_class.values)[0]
        predicted_label = "Pass" if predicted_class == 1 else "Fail"

        st.success(f"Predicted Student Outcome: {predicted_label}")
        st.info(f"Probability to Pass: {proba[1]*100:.2f}%")
        st.warning(f"Probability to Fail: {proba[0]*100:.2f}%")

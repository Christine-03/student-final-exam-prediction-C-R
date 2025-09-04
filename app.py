# app.py
import streamlit as st
import pandas as pd
from joblib import load

# ----------------------------
# Load saved models
# ----------------------------
reg_loaded = load('linear_regression_model.joblib')
reg_model, reg_features = reg_loaded[0], reg_loaded[1]

clf_loaded = load('rf_classifier.joblib')
clf_model, clf_features = clf_loaded[0], clf_loaded[1]

# Parental Education Level columns
parental_cols = [col for col in reg_features if col.startswith('Parental_Education_Level_')]

# Validation ranges
valid_ranges = {
    "Study_Hours_per_Week": (0, 100),
    "Attendance_Rate": (0, 100),
    "Past_Exam_Scores": (0, 100)
}

# ----------------------------
# Grade function
# ----------------------------
def grade(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    else: return "F"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon=":mortar_board:", layout="centered")
st.title("🎓 Student Performance Predictor")

st.markdown("""
Predict **Final Exam Score** or **Pass/Fail Outcome** using our trained A

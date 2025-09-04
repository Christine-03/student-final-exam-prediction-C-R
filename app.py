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
Predict **Final Exam Score** or **Pass/Fail Outcome** using our trained AI models.
""")

# ----------------------------
# User Inputs
# ----------------------------
st.header("Student Information")

user_input_values = {}

# Gender
user_input_values['Gender'] = st.radio("Gender:", options=[0, 1], format_func=lambda x: "Male" if x==0 else "Female")

# Internet Access at Home
user_input_values['Internet_Access_at_Home'] = st.radio("Internet Access at Home:", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

# Extracurricular Activities
user_input_values['Extracurricular_Activities'] = st.radio("Extracurricular Activities:", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

# Numeric inputs
for col in ['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores']:
    min_val, max_val = valid_ranges[col]
    user_input_values[col] = st.number_input(f"{col} (between {min_val} and {max_val})", min_value=min_val, max_value=max_val, value=(min_val+max_val)//2)

# Parental Education Level (One-hot)
st.subheader("Parental Education Level")
options = [col.replace("Parental_Education_Level_", "") for col in parental_cols]
chosen_level = st.selectbox("Select Parental Education Level:", options)
for col in parental_cols:
    user_input_values[col] = 0
one_hot_col = f"Parental_Education_Level_{chosen_level}"
user_input_values[one_hot_col] = 1

# Convert to DataFrame
input_df = pd.DataFrame([user_input_values], columns=reg_features)

# ----------------------------
# Prediction Type
# ----------------------------
st.header("Choose Prediction Type")
prediction_type = st.radio("Select prediction type:", ("Final Exam Score", "Pass/Fail Outcome"))

if st.button("Predict"):
    if prediction_type == "Final Exam Score":
        # Regression Prediction
        predicted_score = int(round(reg_model.predict(input_df.values)[0]))
        student_grade = grade(predicted_score)
        
        st.success(f"Predicted Final Exam Score: {predicted_score}")
        st.info(f"Predicted Grade: {student_grade}")

        # Show grade ranges
        grade_ranges = {
            "A": (90, 100),
            "B": (80, 89),
            "C": (70, 79),
            "D": (60, 69),
            "F": (0, 59)
        }
        st.subheader("Grade Ranges")
        for g, (low, high) in grade_ranges.items():
            st.write(f"{g}: {low} - {high}")

    elif prediction_type == "Pass/Fail Outcome":
        # Classification Prediction
        input_df_class = input_df[clf_features]
        predicted_class = clf_model.predict(input_df_class.values)[0]
        proba = clf_model.predict_proba(input_df_class.values)[0]
        predicted_label = "Pass" if predicted_class == 1 else "Fail"

        st.success(f"Predicted Student Outcome: {predicted_label}")
        st.info(f"Probability to Pass: {proba[1]*100:.2f}% | Probability to Fail: {proba[0]*100:.2f}%")

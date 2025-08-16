import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained pipeline
model = joblib.load("best_student_performance_pipeline.joblib")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("📘 Student Performance Prediction")
st.write("Enter student details below to predict performance:")

# Example input fields (adjust to your dataset features!)
gender = st.selectbox("Gender", ["male", "female"])
parent_edu = st.selectbox(
    "Parental Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation", ["none", "completed"])
math_score = st.number_input("Math Score", 0, 100, 50)
reading_score = st.number_input("Reading Score", 0, 100, 50)
writing_score = st.number_input("Writing Score", 0, 100, 50)

# Prepare input
input_data = pd.DataFrame([{
    "gender": gender,
    "parental level of education": parent_edu,
    "lunch": lunch,
    "test preparation course": test_prep,
    "math score": math_score,
    "reading score": reading_score,
    "writing score": writing_score
}])

if st.button("Predict Performance"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Performance: **{prediction[0]}**")

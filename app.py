import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("best_student_performance_pipeline.joblib")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student data to predict performance (Pass/Fail).")

# Collect user inputs
Attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
Assignment_Score = st.number_input("Assignment Score", min_value=0, max_value=100, value=75)
Quiz_Score = st.number_input("Quiz Score", min_value=0, max_value=100, value=70)
Study_Hours_Per_Week = st.number_input("Study Hours per Week", min_value=0, max_value=80, value=10)
Internal_Assessment = st.number_input("Internal Assessment Score", min_value=0, max_value=100, value=78)
Participation_Score = st.number_input("Participation Score", min_value=0, max_value=100, value=60)
Project_Score = st.number_input("Project Score", min_value=0, max_value=100, value=80)
Exam_Anxiety_Level = st.number_input("Exam Anxiety Level (1-10)", min_value=1, max_value=10, value=5)

if st.button("Predict Performance"):
    # Match the exact order of features
    input_data = pd.DataFrame([[
        Attendance,
        Assignment_Score,
        Quiz_Score,
        Study_Hours_Per_Week,
        Internal_Assessment,
        Participation_Score,
        Project_Score,
        Exam_Anxiety_Level
    ]], columns=[
        'Attendance',
        'Assignment_Score',
        'Quiz_Score',
        'Study_Hours_Per_Week',
        'Internal_Assessment',
        'Participation_Score',
        'Project_Score',
        'Exam_Anxiety_Level'
    ])

    prediction = model.predict(input_data)[0]

    result = "Pass" if prediction == 1 else "Fail"
    st.success(f"Predicted Performance: **{result}**")






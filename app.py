from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model (pipeline with preprocessing + classifier)
model = joblib.load("best_student_performance_pipeline.joblib")

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Student Performance Prediction API is running."}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON input example:
        {
          "Attendance": 85,
          "Assignment_Score": 50,
          "Quiz_Score": 70,
          "Study_Hours_Per_Week": 8,
          "Internal_Assessment": 60,
          "Participation_Score": 5,
          "Project_Score": 75,
          "Exam_Anxiety_Level": 3
        }

        data = request.get_json()

        # Convert input dict to DataFrame (1 row)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Map 0/1 to Fail/Pass if needed
        result = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "prediction": result,
            "probability_pass": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

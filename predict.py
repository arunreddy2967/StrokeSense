import pandas as pd
import joblib
import json

MODEL_PATH = "models/best_model.pkl"
INFO_PATH = "models/model_info.json"

model = joblib.load(MODEL_PATH)

with open(INFO_PATH, "r") as file:
    model_info = json.load(file)

threshold = model_info["threshold"]

patient = pd.DataFrame([{
    "gender": "Male",
    "age": 65,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 180.0,
    "bmi": 32.5,
    "smoking_status": "formerly smoked"
}])

probability = model.predict_proba(patient)[0][1]

if probability >= threshold:
    prediction = "HIGH RISK"
else:
    prediction = "LOW RISK"

print("=" * 50)
print("STROKESENSE PREDICTION")
print("=" * 50)

print("\nPatient Information:")
print(patient.to_string(index=False))

print("\nPrediction Result:")
print(f"Risk Probability : {probability * 100:.2f}%")
print(f"Risk Category    : {prediction}")
print(f"Decision Threshold: {threshold:.2f}")

print("\nNote:")
print("This result is a machine-learning risk estimate")
print("and is not a medical diagnosis.")

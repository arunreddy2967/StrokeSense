
import pandas as pd
import joblib
import json
import os

# --------------------------------
# FILE PATHS
# --------------------------------

MODEL_PATH = "models/best_model.pkl"
INFO_PATH = "models/model_info.json"

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(INFO_PATH):
    raise FileNotFoundError(f"Model information file not found: {INFO_PATH}")

model = joblib.load(MODEL_PATH)

# --------------------------------
# COMPATIBILITY CHECK
# --------------------------------

classifier = model.named_steps["model"]

if not hasattr(classifier, "multi_class"):
    classifier.multi_class = "auto"

# --------------------------------
# LOAD MODEL INFORMATION
# --------------------------------

with open(INFO_PATH, "r") as file:
    model_info = json.load(file)

threshold = float(model_info["threshold"])

# --------------------------------
# PATIENT INPUT
# --------------------------------

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

# --------------------------------
# PREDICTION
# --------------------------------

probability = model.predict_proba(patient)[0][1]

if probability >= threshold:
    prediction = "HIGH RISK"
else:
    prediction = "LOW RISK"

# --------------------------------
# DISPLAY RESULTS
# --------------------------------

print("=" * 50)
print("          STROKESENSE PREDICTION")
print("=" * 50)

print("\nPatient Information:")
print(patient.to_string(index=False))

print("\nPrediction Result:")
print(f"Risk Probability  : {probability * 100:.2f}%")
print(f"Risk Category     : {prediction}")
print(f"Decision Threshold: {threshold:.2f}")

print("\nNote:")
print("This result is a machine-learning risk estimate")
print("and is not a medical diagnosis.")

print("=" * 50)

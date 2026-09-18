import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
MODEL_PATH = "models/best_model.pkl"
INFO_PATH = "models/model_info.json"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["id"])

X = df.drop(columns=["stroke"])
y = df["stroke"]

numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = LogisticRegression(
    C=0.01,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline,
    MODEL_PATH
)

model_info = {
    "model": "Logistic Regression",
    "C": 0.01,
    "class_weight": "balanced",
    "threshold": 0.60,
    "accuracy": 0.819961,
    "precision": 0.186916,
    "recall": 0.80,
    "f1_score": 0.303030,
    "roc_auc": 0.840679,
    "features": [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status"
    ]
}

with open(INFO_PATH, "w") as file:
    json.dump(model_info, file, indent=4)

print("=" * 50)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print("\nModel:")
print("Logistic Regression")

print("\nThreshold:")
print("0.60")

print("\nModel file:")
print(MODEL_PATH)

print("\nModel information:")
print(INFO_PATH)

print("\nFiles created successfully.")

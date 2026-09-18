import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": ["balanced", None]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=42
        ),
        {
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
            "model__class_weight": ["balanced", None]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, 15],
            "model__min_samples_leaf": [1, 2, 5],
            "model__class_weight": ["balanced", None]
        }
    )
}

tuning_results = []

for name, (model, parameters) in models.items():

    print("\n" + "=" * 60)
    print("TUNING:", name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        parameters,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nTest Results:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    tuning_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "True Negative": cm[0, 0],
        "False Positive": cm[0, 1],
        "False Negative": cm[1, 0],
        "True Positive": cm[1, 1],
        "Best Parameters": str(grid.best_params_)
    })

results_df = pd.DataFrame(tuning_results)

results_df.to_csv(
    "results/tuned_model_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("TUNING COMPLETED")
print("=" * 60)

print("\nSaved:")
print("results/tuned_model_results.csv")

print("\nTUNED MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

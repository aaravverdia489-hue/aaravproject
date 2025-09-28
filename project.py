"""
Student Performance Analysis Pipeline
--------------------------------------
This script performs:
1. Data exploration (EDA)
2. Preprocessing (Imputation, Scaling, Encoding)
3. Model training & evaluation (Classification or Regression auto-detected)

Dataset: student_performance.csv (ensure it is in /data/ folder before running)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)

# ✅ Paths
DATA_PATH = "data/student_performance.csv"
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)
print("\n✅ Dataset Loaded Successfully!")
print(df.head())

# ✅ EDA Summary
print("\n🔍 Data Shape:", df.shape)
print("\n📌 Column Types:\n", df.dtypes)
print("\n❌ Missing Values:\n", df.isna().sum())
print("\n📊 Summary Statistics:\n", df.describe())

df.to_csv(f"{OUTPUT_DIR}/dataset_preview.csv", index=False)

# ✅ Correlation Heatmap (if numeric data exists)
num_df = df.select_dtypes(include=["number"])
if num_df.shape[1] > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

# ✅ Infer Target Column Automatically (last column)
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

print(f"\n🎯 Target Column Selected Automatically: {target}")

# ✅ Infer Problem Type
def infer_problem_type(y):
    if y.dtype in ["int64", "float64"] and y.nunique() > 10:
        return "regression"
    else:
        return "classification"

problem_type = infer_problem_type(y)
print(f"\n🧠 Detected Problem Type: {problem_type.upper()}")

# ✅ Preprocessing
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
    ]
)

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Model Selection
if problem_type == "classification":
    model = RandomForestClassifier(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ✅ Train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ✅ Evaluation
if problem_type == "classification":
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
else:
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

print("\n✅ Model Performance:")
print(metrics)

# ✅ Save Results
pd.DataFrame([metrics]).to_csv(f"{OUTPUT_DIR}/results.csv", index=False)
joblib.dump(pipeline, f"{OUTPUT_DIR}/trained_model.joblib")

print("\n🎉 Pipeline Completed! Outputs saved in /reports/")

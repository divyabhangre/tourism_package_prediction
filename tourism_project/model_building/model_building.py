from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score)
from huggingface_hub import HfApi
import joblib
# ── 1. Load train and test data from Hugging Face ──
api = HfApi(token=os.getenv("HF_TOKEN"))

train_X = load_dataset("divyabhangre/tourism-pkg-prediction-data",data_files="Xtrain.csv", split="train").to_pandas()
test_X  = load_dataset("divyabhangre/tourism-pkg-prediction-data",data_files="Xtest.csv",  split="train").to_pandas()
train_y = load_dataset("divyabhangre/tourism-pkg-prediction-data",data_files="ytrain.csv", split="train").to_pandas()
test_y  = load_dataset("divyabhangre/tourism-pkg-prediction-data",data_files="ytest.csv",  split="train").to_pandas()

print("✅ Data loaded successfully from Hugging Face!")
print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")
# ── 2. Define models and parameters ──
models = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.7, 1.0]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42,
                               eval_metric="logloss",
                               use_label_encoder=False),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    }
}
# ── 3. Tune, Log, and Evaluate all models ──
best_model = None
best_score = 0
best_model_name = ""

for model_name, model_info in models.items():
    model = model_info["model"]
    params = model_info["params"]

    # Try all parameter combinations
    from itertools import product
    param_keys = list(params.keys())
    param_values = list(params.values())

    for combo in product(*param_values):
        param_combo = dict(zip(param_keys, combo))

        with mlflow.start_run(run_name=f"{model_name}_{param_combo}"):

            # Set parameters on model
            model.set_params(**param_combo)

            # Train the model
            model.fit(train_X, train_y.values.ravel())

            # Predict
            predictions = model.predict(test_X)

            # Evaluate
            acc  = accuracy_score(test_y, predictions)
            prec = precision_score(test_y, predictions, average="weighted")
            rec  = recall_score(test_y, predictions, average="weighted")
            f1   = f1_score(test_y, predictions, average="weighted")

            # Log parameters to MLflow
            mlflow.log_param("model_name", model_name)
            for k, v in param_combo.items():
                mlflow.log_param(k, v)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("f1_score",  f1)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, model_name)

            print(f"✅ {model_name} | Params: {param_combo} | Accuracy: {acc:.4f}")

            # Track best model
            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = model_name

print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_score:.4f}")
# ── 4. Save and Register best model to HF ──
os.makedirs("tourism_project/model_building/model", exist_ok=True)
model_path = "tourism_project/model_building/model/best_model.pkl"

# Save best model locally
joblib.dump(best_model, model_path)
print(f"✅ Best model saved locally at {model_path}")

# Upload best model to Hugging Face
api.upload_file(path_or_fileobj=model_path,path_in_repo="best_model.pkl",repo_id="divyabhangre/tourism-package-predict",repo_type="space",)
print(f"✅ Best model uploaded to Hugging Face Space!")

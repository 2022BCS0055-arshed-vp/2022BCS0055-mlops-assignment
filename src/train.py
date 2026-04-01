import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json


NAME = "Arshed V P"
ROLL_NO = "2022BCS0055"

mlflow.set_experiment("2022BCS0055_experiment")

# CHANGE THESE VALUES FOR DIFFERENT RUNS
RUN_TYPE = "run5"   # change: run1, run2, run3, run4, run5

# Load dataset
data = load_iris()
X = data.data
y = data.target

# DATASET VERSION CHANGE (Run 3)
if RUN_TYPE == "run3":
    X = X[:100]
    y = y[:100]

# FEATURE SELECTION (Run 4)
if RUN_TYPE == "run4":
    X = X[:, :2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL + HYPERPARAMETER CHANGE
if RUN_TYPE == "run5":
    model = LogisticRegression(max_iter=200)
else:
    n_estimators = 100
    if RUN_TYPE == "run2":
        n_estimators = 50
    model = RandomForestClassifier(n_estimators=n_estimators)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

metrics_dict = {
    "accuracy": accuracy,
    "f1_score": f1,
    "Name": NAME,
    "Roll No": ROLL_NO
}

with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f)

with mlflow.start_run():
    mlflow.log_param("run_type", RUN_TYPE)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_param("Name", NAME)
    mlflow.log_param("Roll_No", ROLL_NO)

    mlflow.sklearn.log_model(model, "model")

print(f"{RUN_TYPE} completed:", metrics_dict)
from fastapi import FastAPI
import random

app = FastAPI()

NAME = "Arshed V P"
ROLL_NO = "2022BCS0055"

# Health endpoint (MANDATORY)
@app.get("/")
def health():
    return {
        "Name": NAME,
        "Roll No": ROLL_NO
    }

# Prediction endpoint (MANDATORY)
@app.post("/predict")
def predict():
    prediction = random.choice([0, 1])

    return {
        "prediction": prediction,
        "Name": NAME,
        "Roll No": ROLL_NO
    }

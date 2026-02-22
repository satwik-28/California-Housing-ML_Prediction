from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models safely
regression_model = joblib.load(os.path.join(MODELS_DIR, "regression_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "classifier_model.pkl"))
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

labels = {0: "Low", 1: "Medium", 2: "High"}


@app.route("/", methods=["GET", "POST"])
def home():

    results = None

    if request.method == "POST":

        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"])
        ]

        data = scaler.transform([features])

        # Predictions
        reg_pred = regression_model.predict(data)[0]
        rf_pred = labels[rf_model.predict(data)[0]]
        svm_pred = labels[svm_model.predict(data)[0]]

        results = {
            "regression": f"{reg_pred:.3f}",
            "rf": rf_pred,
            "svm": svm_pred
        }

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
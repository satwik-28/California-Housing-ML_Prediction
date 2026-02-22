from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load models
regression_model = joblib.load("models/regression_model.pkl")
rf_model = joblib.load("models/classifier_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

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
    app.run(debug=True)

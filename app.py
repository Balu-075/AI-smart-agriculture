from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Model
model = pickle.load(open("model/crop_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["Nitrogen"])
    P = float(request.form["Phosphorus"])
    K = float(request.form["Potassium"])
    temperature = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])
    ph = float(request.form["Ph"])
    rainfall = float(request.form["Rainfall"])

    features = np.array([[N, P, K,
                          temperature,
                          humidity,
                          ph,
                          rainfall]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Recommended Crop: {prediction[0]}"
    )


if __name__ == "__main__":
    app.run(debug=True)
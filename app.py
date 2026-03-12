from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load ML model and encoders
model = pickle.load(open("model.pkl", "rb"))
soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
fert_encoder = pickle.load(open("fert_encoder.pkl", "rb"))

# Get values directly from encoders (for dropdown)
soil_types = list(soil_encoder.classes_)
crop_types = list(crop_encoder.classes_)


@app.route("/")
def home():
    return render_template(
        "index.html",
        soils=soil_types,
        crops=crop_types
    )


# Soil Health Analysis Function
def soil_health(n, p, k):

    report = []

    if n < 40:
        report.append("Nitrogen Deficient")

    if p < 40:
        report.append("Phosphorous Deficient")

    if k < 40:
        report.append("Potassium Deficient")

    if not report:
        return "Soil Nutrients Balanced"

    return ", ".join(report) + " Soil"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # User Inputs
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])

        soil = request.form["soil"]
        crop = request.form["crop"]

        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        # Encode categorical values
        soil_encoded = soil_encoder.transform([soil])[0]
        crop_encoded = crop_encoder.transform([crop])[0]

        # Prepare feature array
        features = np.array([[temp, humidity, moisture,
                              soil_encoded, crop_encoded,
                              nitrogen, potassium, phosphorous]])

        # Predict fertilizer
        prediction = model.predict(features)
        fertilizer = fert_encoder.inverse_transform(prediction)[0]

        fertilizer_display = f"NPK Fertilizer {fertilizer}"

        # Soil health report
        soil_report = soil_health(nitrogen, phosphorous, potassium)

        # AI Explanation
        explanation = (
            f"The AI system analyzed the soil nutrient levels and environmental conditions. "
            f"Based on Nitrogen ({nitrogen}), Phosphorous ({phosphorous}), "
            f"and Potassium ({potassium}), the recommended fertilizer is "
            f"{fertilizer_display}. This fertilizer helps restore soil nutrients "
            f"and improve crop productivity."
        )

        # Create NPK graph
        labels = ["Nitrogen", "Phosphorous", "Potassium"]
        values = [nitrogen, phosphorous, potassium]

        if not os.path.exists("static"):
            os.makedirs("static")

        plt.clf()
        plt.bar(labels, values)
        plt.title("Soil Nutrient Levels (NPK)")
        plt.ylabel("Nutrient Value")
        plt.savefig("static/npk_graph.png")

        return render_template(
            "index.html",
            prediction_text=f"Recommended Fertilizer: {fertilizer_display}",
            soil_report=soil_report,
            explanation=explanation,
            graph="static/npk_graph.png",
            soils=soil_types,
            crops=crop_types
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="Error: " + str(e),
            soils=soil_types,
            crops=crop_types
        )


if __name__ == "__main__":
    app.run(debug=True)
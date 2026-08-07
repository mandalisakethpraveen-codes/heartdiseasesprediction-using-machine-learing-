print("Step 1")
from flask import Flask, render_template, request
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

app = Flask(__name__)

# ==========================
# Load Dataset
# ==========================

dataset = pd.read_csv("HeartData.csv")

label_encoder = LabelEncoder()
dataset["Label"] = label_encoder.fit_transform(dataset["Label"])

dataset.fillna(0, inplace=True)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2)
    )
    # ==========================
# Predict Route
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return "Please select a CSV file."

    test_data = pd.read_csv(uploaded_file)

    test_data.fillna(0, inplace=True)

    # Remove label column if it exists
    if "Label" in test_data.columns:
        test_data = test_data.drop(columns=["Label"])

    test_data = scaler.transform(test_data)

    predictions = model.predict(test_data)

    results = []

    for prediction in predictions:
        disease = label_encoder.inverse_transform([int(prediction)])[0]
        results.append(disease)

    return render_template(
        "result.html",
        result=results,
        accuracy=round(accuracy * 100, 2)
    )
    # -----------------------------
# Run Flask
# ==========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

app = Flask(__name__)

# -----------------------------
# Train Model
# -----------------------------

dataset = pd.read_csv("HeartData.csv")

label_encoder = LabelEncoder()
dataset["Label"] = label_encoder.fit_transform(dataset["Label"])

dataset.fillna(0, inplace=True)

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100,2)
    )

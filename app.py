from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
linear_model = pickle.load(open("linear_model.pkl", "rb"))
logistic_model = pickle.load(open("logistic_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['sepal_length']),
        float(request.form['sepal_width']),
        float(request.form['petal_length']),
        float(request.form['petal_width'])
    ]

    features = np.array([features])

    linear_pred = linear_model.predict(features)[0]
    logistic_pred = int(logistic_model.predict(features)[0])
    
    # Safety boundary for linear regression prediction
    linear_idx = int(round(linear_pred))
    linear_idx = max(0, min(2, linear_idx))
    
    # Calculate confidence score
    proba = logistic_model.predict_proba(features)[0]
    confidence = round(max(proba) * 100, 2)

    flowers = ["Setosa", "Versicolor", "Virginica"]
    images = [
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg", 
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    ]

    return render_template(
        "index.html",
        linear_result=flowers[linear_idx],
        logistic_result=flowers[logistic_pred],
        confidence=confidence,
        flower_image=images[logistic_pred]
    )

if __name__ == "__main__":
    app.run(debug=True)
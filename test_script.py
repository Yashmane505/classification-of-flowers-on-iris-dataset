import pickle
import numpy as np

# Load the saved models
try:
    linear_model = pickle.load(open("linear_model.pkl", "rb"))
    logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model files not found. Please run model.py first!")
    exit()

# Define the species names (from app.py logic)
flowers = ["Setosa", "Versicolor", "Virginica"]

# Test cases: [Sepal Length, Sepal Width, Petal Length, Petal Width]
test_cases = [
    # --- Standard Cases ---
    {"name": "Setosa (Standard)", "data": [5.1, 3.5, 1.4, 0.2]},
    {"name": "Versicolor (Standard)", "data": [5.9, 3.0, 4.2, 1.5]},
    {"name": "Virginica (Standard)", "data": [6.9, 3.1, 5.4, 2.1]},
    
    # --- Edge Cases (Boundary / High Ambiguity) ---
    {"name": "Versi/Virgi Boundary 1", "data": [6.0, 2.5, 4.8, 1.6]},
    {"name": "Versi/Virgi Boundary 2", "data": [5.0, 2.0, 3.3, 1.0]},
    {"name": "Large Setosa (Confusing)", "data": [5.5, 3.8, 2.2, 0.6]},

    # --- "Likely to Fail" / Extreme Cases ---
    {"name": "Giant Flower (Out of Bounds)", "data": [15.0, 10.0, 20.0, 10.0]},
    {"name": "Micro Flower (Near Zero)", "data": [0.1, 0.1, 0.1, 0.1]},
    {"name": "Contradictory (Long Sepal, Tiny Petal)", "data": [8.0, 4.0, 0.5, 0.1]},
    {"name": "Contradictory (Short Sepal, Huge Petal)", "data": [3.0, 2.0, 7.0, 2.5]}
]

# Run tests and save to output.txt
with open("output.txt", "w") as f:
    f.write("=== Iris Model Test Results ===\n\n")
    
    for case in test_cases:
        features = np.array([case["data"]])
        
        # Linear Regression prediction
        lin_pred = int(round(linear_model.predict(features)[0]))
        # Clip to ensure index stays within range [0, 2]
        lin_pred = max(0, min(2, lin_pred))
        
        # Logistic Regression prediction
        log_pred = int(logistic_model.predict(features)[0])
        
        result_text = (
            f"Test Case: {case['name']}\n"
            f"Inputs: {case['data']}\n"
            f"Linear Model Prediction: {flowers[lin_pred]}\n"
            f"Logistic Model Prediction: {flowers[log_pred]}\n"
            "-----------------------------------\n"
        )
        
        print(result_text)  # Show in terminal
        f.write(result_text) # Save to file

print("\nAll test cases completed. Results saved to 'output.txt'.")

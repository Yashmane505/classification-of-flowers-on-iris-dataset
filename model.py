from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train models
linear_model = LinearRegression()
linear_model.fit(X, y)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X, y)

# Save models
pickle.dump(linear_model, open("linear_model.pkl", "wb"))
pickle.dump(logistic_model, open("logistic_model.pkl", "wb"))

print("Models trained and saved!")
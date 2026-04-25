import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle

# Sample dataset (you can replace with real dataset)
data = pd.DataFrame({
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [200000, 300000, 400000, 500000, 600000]
})

X = data[["area", "bedrooms"]]
y = data["price"]

# Better model using pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

pipeline.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved!")
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Request body schema
class InputData(BaseModel):
    area: float
    bedrooms: int

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": float(prediction)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

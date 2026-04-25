from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Request schema
class InputData(BaseModel):
    area: float
    bedrooms: int

# API endpoint
@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": float(prediction)}

# Serve React build
app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("build/index.html")

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

app = FastAPI()

class CarFeatures(BaseModel):
    year: int
    km_driven: int
    power_per_liter: float
    mileage: float
    torque: float

class CarFeaturesBatch(BaseModel):
    objects: List[CarFeatures]

model = joblib.load("model.pkl") 
scaler = joblib.load("scaler.pkl")

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    return data

@app.post("/predict_item")
async def predict_item(item: CarFeatures) -> float:
    input_df = pd.DataFrame([item.dict()])
    input_df = preprocess(input_df)
    
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)
    return prediction[0]

@app.post("/predict_items")
async def predict_items(items: CarFeaturesBatch) -> List[float]:
    input_df = pd.DataFrame([item.dict() for item in items.objects])
    input_df = preprocess(input_df)
    
    scaled_input = scaler.transform(input_df)
    
    predictions = model.predict(scaled_input)
    return predictions.tolist()

@app.post("/upload_csv")
async def predict_csv(file: UploadFile):
    input_df = pd.read_csv(file.file)
    input_df = preprocess(input_df)
    
    scaled_input = scaler.transform(input_df)
    
    predictions = model.predict(scaled_input)
    
    input_df["predictions"] = predictions
    

    output_file = "predictions.csv"
    input_df.to_csv(output_file, index=False)
    
    return {"message": "Predictions saved", "file_path": output_file}

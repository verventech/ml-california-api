import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="California Housing ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BOTH models into memory
classifier = joblib.load("classifier_model.joblib")
regressor = joblib.load("regressor_model.joblib")

# Ensure this exactly matches the 8 features in scikit-learn's dataset
class HouseFeatures(BaseModel):
    MedInc: float      # Median income in block group (in tens of thousands)
    HouseAge: float    # Median house age in block group
    AveRooms: float    # Average number of rooms per household
    AveBedrms: float   # Average number of bedrooms per household
    Population: float  # Block group population
    AveOccup: float    # Average number of household members
    Latitude: float    # Block group latitude
    Longitude: float   # Block group longitude

def format_input(data: HouseFeatures):
    return [[
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms, 
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]]

@app.post("/predict/price")
def predict_price(data: HouseFeatures):
    # Dataset target is in $100,000s, so multiply by 100k
    predicted_price_scaled = regressor.predict(format_input(data))[0]
    actual_usd = predicted_price_scaled * 100000
    
    return {"estimated_price": round(actual_usd, 2)}

@app.post("/predict/category")
def predict_category(data: HouseFeatures):
    prediction = classifier.predict(format_input(data))[0]
    probabilities = classifier.predict_proba(format_input(data))[0].tolist()
    
    result = "Above California Median" if prediction == 1 else "Below California Median"
    return {
        "category": result,
        "confidence": round(max(probabilities) * 100, 2)
    }
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionInput(BaseModel):
    model_name: str = "xgboost"  # Valor por defecto
    temporada: int
    anio: int
    mes: int
    hora: int
    feriado: int
    dia_trabajo: int
    clima: int
    temperatura: float
    sensacion_termica: float
    humedad: float
    velocidad_viento: float
    dia_semana: int

class PredictionOutput(BaseModel):
    total_alquileres_predicho: float

models = {
    "xgboost": joblib.load("models/ml_pipeline.joblib"),
    "random_forest": joblib.load("models/rf_pipeline.joblib") 
}

@app.get("/")
async def root():
    return {"message": "¡La API de predicción de alquileres está funcionando con múltiples modelos!"}

@app.post("/predict")
async def predict_rentals(data: PredictionInput):
    model_to_use = data.model_name.lower()
    if model_to_use not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_to_use}' no encontrado. Los modelos disponibles son: {list(models.keys())}")

    try:
        input_data = pd.DataFrame([data.dict(exclude={"model_name"})]) # Excluimos model_name de las características
        prediction = models[model_to_use].predict(input_data)[0]
        return PredictionOutput(total_alquileres_predicho=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción con {model_to_use}: {e}")
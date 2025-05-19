from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo entrenado
try:
    model = joblib.load("models/ml_pipeline.joblib")
except FileNotFoundError:
    raise Exception("El archivo del modelo no se encontró. Asegúrate de que 'models/ml_pipeline.joblib' exista.")

app = FastAPI()

class PredictionInput(BaseModel):
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

@app.post("/predict")
async def predict_rentals(data: PredictionInput):
    try:
        # Convertir los datos de entrada a un DataFrame de pandas
        input_df = pd.DataFrame([data.dict()])

        # Realizar la predicción utilizando el pipeline cargado
        prediction = model.predict(input_df)[0]

        return PredictionOutput(total_alquileres_predicho=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "¡La API de predicción de alquileres está funcionando! :D"}
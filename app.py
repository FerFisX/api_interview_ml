from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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

try:
    model = joblib.load("models/ml_pipeline.joblib")
except Exception as e:
    print(f"Error AL CARGAR el modelo: {e}")
    # Considerar levantar una excepción aquí para que Render lo registre como un fallo de inicio

@app.get("/")
async def root():
    return {"message": "¡La API de predicción de alquileres está funcionando!"}

@app.post("/predict")
async def predict_rentals(data: PredictionInput):
    print("Entrando a la función predict_rentals")  # Añadimos este print al inicio
    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)[0]
        return PredictionOutput(total_alquileres_predicho=prediction)
    except Exception as e:
        error_message = f"Error DURANTE la predicción: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)
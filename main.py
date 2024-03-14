from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Deploy Depression Tanisha Miranda",
    version="0.0.1"
)

# Cargar el modelo entrenado
model_rf = joblib.load('model/modelo_depresion_rf_v01.pkl')

# Endpoint para realizar predicciones con parámetros de consulta
@app.post("/api/v1/predict-depression")
async def predict(
    sex: float,
    age: float,
    married: float,
    number_children: float,
    total_members: float,
    incoming_salary: float
):
    data = {
        'sex': sex,
        'age': age,
        'married': married,
        'number_children': number_children,
        'total_members': total_members,
        'incoming_salary': incoming_salary
    }
    df_new_data = pd.DataFrame([data])
    prediction = model_rf.predict(df_new_data)
    return {"prediction": prediction.tolist()}

# 4. Deploy do Modelo (Criação da API: FastAPI) 

# Importando as bibliotecas importantes
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from prometheus_fastapi_instrumentator import Instrumentator
import os
import tensorflow as tf

# Caminho para o modelo pré-treinado
MODEL_PATH = os.getenv("MODEL_PATH", "models/modelo_lstm_wege3.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f" Modelo carregado com sucesso de {MODEL_PATH}")
except Exception as e:
    print(f" Erro ao carregar modelo: {e}")
    raise

scaler = MinMaxScaler(feature_range=(0, 1))

app = FastAPI(title="API de Previsão de Preços da ação WEGE3.SA com LSTM")

Instrumentator().instrument(app).expose(app)

class PriceData(BaseModel):
    prices: list[float] 

# Função para criar sequência de entrada para o modelo(Deixei o seq_length dinâmico para evitar erros caso a sequência seja menor que 60)
def create_sequence(data, seq_length=60):
    seq_length = min(len(data), seq_length)
    return np.array(data[-seq_length:]).reshape(1, seq_length, 1)


@app.post("/predict")
def predict(data: PriceData):
    try:
        if len(data.prices) < 60:
            return {"error": "É necessário fornecer pelo menos 60 preços."}
        seq = create_sequence(data.prices, seq_length=60)
        prediction = model.predict(seq)
        return {"predicted_price": float(prediction[0][0])}
    except Exception as e:
        return {"error": str(e)}

    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    print(f" Iniciando API na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


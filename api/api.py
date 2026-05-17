
# 4. Deploy do Modelo (Criação da API: FastAPI) 

# Importando as bibliotecas importantes
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from prometheus_fastapi_instrumentator import Instrumentator
import os
import tensorflow as tf

# Caminho para o modelo pré-treinado
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_lstm_wege3_dir")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f" Modelo carregado com sucesso de {MODEL_PATH}")
except Exception as e:
    print(f" Erro ao carregar modelo: {e}")
    raise

scaler = MinMaxScaler(feature_range=(0, 1))





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
    df = pd.DataFrame(data.prices, columns=["Close"])

    scaled = scaler.fit_transform(df)

    seq = create_sequence(scaled)

    prediction = model.predict(seq)
    prediction_rescaled = scaler.inverse_transform(prediction)

    return {"predicted_price": round(float(prediction_rescaled[0][0]), 2)}
    
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    print(f" Iniciando API na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


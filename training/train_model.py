
# Importando as bibliotecas importantes
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
# Métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# 1. Coleta e Pré-processamento dos Dados 

# Realizando o Download dos dados
symbol = 'WEGE3.SA'
start_date = '2015-01-01'
end_date = '2025-12-31'

print("Download dos dados...")
df = yf.download(symbol, start=start_date, end=end_date)

# Realizando a limpeza
df = df[['Close']].copy()
df.dropna(inplace=True)

# Realizando a normalização
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']])

# Criando as sequências para o modelo LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)


#2. Desenvolvimento do Modelo LSTM 

# Divisão em treino e teste (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ajuste de dimensões para LSTM 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Modelo LSTM
model = Sequential()
model.add(Input(shape=(seq_length, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Realizando o treinamento
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# Realizando as previsões
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Métricas de avaliação
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evita divisão por zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test_rescaled, predicted) # MAPE (Erro Percentual Absoluto Médio) 
mae = mean_absolute_error(y_test_rescaled, predicted) # MAE (Mean Absolute Error)
rmse = math.sqrt(mean_squared_error(y_test_rescaled, predicted)) # RMSE (Root Mean Square Error)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Apresentando gráfico de treinamento
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Loss (treino)')
plt.plot(history.history['val_loss'], label='Loss (validação)')
plt.title('Evolução do treinamento')
plt.xlabel('Épocas')
plt.ylabel('Erro (MSE)')
plt.legend()
plt.show()

# Apresentando gráfico de comparação (previsão vs. realidade)
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label='Real')
plt.plot(predicted, label='Previsto')
plt.title(f'Previsão com LSTM - {symbol}')
plt.xlabel('Dias')
plt.ylabel('Preço')
plt.legend()
plt.show()


# 3. Salvamento e Exportação do Modelo
model.save("modelo_lstm_wege3.keras")  # Salva no formato Keras HDF5
model.export("modelo_lstm_wege3_dir") # Salva no formato TensorFlow SavedModel para servir em produção.
print("Seu Modelo foi salvo!")
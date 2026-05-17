# 📘 Documentação do Projeto


# 📈 Previsão de Preços da Ação WEGE3.SA com LSTM
Este projeto utiliza **Deep Learning (LSTM)** para prever preços da ação **WEGE3.SA**.  
O fluxo inclui coleta de dados históricos, pré-processamento, treinamento do modelo e deploy em uma API com **FastAPI**.


## 🚀 Estrutura do Projeto
Fase 4 - Deep Learning e IA/


├── training/                   # Scripts de treino

└── train_model.py

└── modelo_lstm_wege3.keras

└── modelo_lstm_wege3_dir

├── models/                     # Modelos salvos

└── modelo_lstm_wege3.keras
    
└── modelo_lstm_wege3_dir
    
├── api/                        # Código da API

└── api.py

└── modelo_lstm_wege3_dir

└── prometheus.yml

└── requirements.txt

├── requirements.txt            # Dependências

└── README.md                   # Documentação


## 🧑‍💻 Treinamento do Modelo (train_model.py)

O script de treinamento segue essas etapas:

🔹 Coleta e Pré-processamento
    ° Usa Yahoo Finance (yfinance) para baixar os preços históricos da ação WEGE3.SA entre 2015 e 2025.
    ° Seleciona apenas a coluna Close (preço de fechamento).
    ° Remove valores nulos.
    ° Normaliza os preços para o intervalo [0, 1] usando MinMaxScaler.

🔹 Criação das Sequências
    ° Define janelas de 60 dias para alimentar o LSTM.
    ° Cada sequência de 60 valores é usada para prever o próximo preço.
    ° Função create_sequences gera os pares (X, y) para treino e teste.

🔹 Desenvolvimento do Modelo LSTM
    ° Divide os dados em treino (80%) e teste (20%).
    ° Ajusta dimensões para o formato esperado pelo LSTM (amostras, seq_length, 1).
    ° Modelo:
        . Camada de entrada com 60 timesteps.
        . Duas camadas LSTM com 50 neurônios cada.
        . Camada densa final para saída de um valor.

    ° Compila com otimizador Adam e loss MSE.
    ° Treina por 20 épocas com batch size 32.

🔹 Avaliação
    ° Faz previsões no conjunto de teste.
    ° Reverte a normalização para comparar com valores reais.
    °  Avalia com métricas:
        . MAE (Mean Absolute Error)
        . RMSE (Root Mean Squared Error)  
        . MAPE (Mean Absolute Percentage Error)

    ° Exibe gráficos:
        . Evolução do treinamento (loss vs. val_loss).
        . Comparação entre preços reais e previstos.

🔹 Salvamento
    ° Salva o modelo em dois formatos:
        . modelo_lstm_wege3.keras (formato Keras).
        . modelo_lstm_wege3_dir (formato TensorFlow SavedModel).

# 🛠️ Instalação e Configuração

## 1. Clone o repositório: 
Bash:
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

## 2. Crie e ative o ambiente virtual
Bash:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

## 3. Instale as dependências
Bash:
pip install -r requirements.txt


# ▶️ Instruções para Execução

## 1. Execute o servidor local
Bash:
uvicorn main:app --reload

## 2. Acesse a interface Swagger gerada pelo FastAPI
Abra no navegador:
http://127.0.0.1:8000/docs

## 3. Acesse para leitura e navegação detalhada
Abra no navegador:
http://127.0.0.1:8000/redoc


Como executar:
Após instalar as bibliotecas necessárias:
Execute no terminal:
python train_model.py




## 🌐  Deploy da API (api.py)

° Carrega o modelo salvo (models/modelo_lstm_wege3.keras).
° Cria uma API com FastAPI.

> Endpoint /predict:
    Recebe uma lista de preços históricos (prices).
    Normaliza os dados.
    Cria sequência de entrada (até 60 valores).
    Faz previsão com o modelo LSTM.
    Retorna o preço previsto.



▶️ Executando a API

No terminal:
uvicorn api:app --reload

Acesse a documentação interativa:
http://127.0.0.1:8000/docs


📤 Exemplo de Requisição

JSON de entrada:

{
  "prices": [
    10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4,
    11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4,
    12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4,
    13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4,
    14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 15.2, 15.3, 15.4,
    15.5, 15.6, 15.7, 15.8, 15.9, 16.0, 16.1, 16.2, 16.3, 16.4
  ]
}

Resposta da API:

{
  "predicted_price": 16.17
}



## 📌 Observação Importante
> O modelo foi treinado com janelas de 60 dias. Portanto, o endpoint /predict deve receber pelo menos 60 valores para funcionar corretamente.
> Para previsões reais, utilize os últimos 60 preços da ação WEGE3.SA.




import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import joblib

# Configurações globais
symbol = "US500"
timeframe = mt5.TIMEFRAME_M1
data_folder = "dados_candles"
model_folder = "modelos"
input_window = 200
output_window = 20
features = ["open", "high", "low", "close", "volume", "rsi"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, output_window, 5)


# Função 1: Coleta de dados e salvamento por dia
def coletar_dados_mt5(dias=7):

    if not mt5.initialize():
        print("Erro ao conectar ao MT5")
        return

    os.makedirs(data_folder, exist_ok=True)

    # Limpa o diretório data_folder antes de continuar
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=dias)
    current_date = start_date

    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        rates = mt5.copy_rates_range(symbol, timeframe, current_date, next_date)

        if rates is None or len(rates) == 0:
            current_date = next_date
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        filename = f"{symbol}_{current_date.date()}.csv"
        df.to_csv(os.path.join(data_folder, filename), index=False)
        current_date = next_date

    mt5.shutdown()


# Função 2: Treinamento do modelo
def treinar_modelo(epochs=30):
    all_data = []
    for file in sorted(os.listdir(data_folder)):
        df = pd.read_csv(os.path.join(data_folder, file))
        df["rsi"] = ta.rsi(df["close"], length=14)
        df.dropna(inplace=True)
        all_data.append(df)

    full_df = pd.concat(all_data).reset_index(drop=True)
    scaler = MinMaxScaler()
    full_df[features] = scaler.fit_transform(full_df[features])
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_folder, "scaler.pkl"))

    X, y = [], []
    for i in range(len(full_df) - input_window - output_window):
        x_window = full_df[features].iloc[i : i + input_window].values
        y_window = (
            full_df[["open", "high", "low", "close", "volume"]]
            .iloc[i + input_window : i + input_window + output_window]
            .values
        )
        X.append(x_window)
        y.append(y_window)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMModel(input_size=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(model_folder, "modelo_lstm.pth"))


# Função 3: Previsão e re-treinamento
def executar_previsao(csv_file):
    df = pd.read_csv(csv_file)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df.dropna(inplace=True)

    scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
    df[features] = scaler.transform(df[features])

    latest_window = df[features].iloc[-input_window:].values
    X = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0).to(device)

    model = LSTMModel(input_size=len(features)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(model_folder, "modelo_lstm.pth"), map_location=device)
    )
    model.eval()

    with torch.no_grad():
        prediction = model(X).cpu().numpy().squeeze()

    print("Previsão dos próximos 20 candles (OHLCV):")
    print(prediction)

    # Re-treinamento com os dados mais recentes
    X_retrain, y_retrain = [], []
    for i in range(len(df) - input_window - output_window):
        x_window = df[features].iloc[i : i + input_window].values
        y_window = (
            df[["open", "high", "low", "close", "volume"]]
            .iloc[i + input_window : i + input_window + output_window]
            .values
        )
        X_retrain.append(x_window)
        y_retrain.append(y_window)

    X_retrain = torch.tensor(np.array(X_retrain), dtype=torch.float32).to(device)
    y_retrain = torch.tensor(np.array(y_retrain), dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(5):
        pred = model(X_retrain)
        loss = criterion(pred, y_retrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Re-treino Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(model_folder, "modelo_lstm.pth"))


if __name__ == "__main__":
    coletar_dados_mt5(dias=100)

    treinar_modelo(epochs=100)

    # executar_previsao("dados_candles/WIN$N_2023-07-29.csv")

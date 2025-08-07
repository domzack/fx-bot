import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configurações globais
symbol = "US500"
timeframe = mt5.TIMEFRAME_M1
data_folder = "dados_candles"
model_folder = "modelos"
input_window = 1000
output_window = 100
features = ["open", "high", "low", "close", "volume"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, output_window, 5)


# Função 1: Coleta de dados e salvamento por dia
def coletar_dados_mt5(dias=30):
    print(f"Iniciando coleta de dados do MT5 para {dias} dias...")
    if not mt5.initialize():
        print("Erro ao conectar ao MT5")
        return

    os.makedirs(data_folder, exist_ok=True)
    # Limpa o diretório data_folder antes de continuar
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Diretório '{data_folder}' limpo.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=dias)
    current_date = start_date

    total_candles = 0
    while current_date < end_date:
        next_date = current_date + timedelta(days=1)
        print(f"Coletando candles de {current_date.date()}...")
        rates = mt5.copy_rates_range(symbol, timeframe, current_date, next_date)

        if rates is None or len(rates) == 0:
            print(f"Nenhum dado encontrado para {current_date.date()}.")
            current_date = next_date
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        filename = f"{symbol}_{current_date.date()}.csv"
        df.to_csv(os.path.join(data_folder, filename), index=False)
        print(f"Salvo {len(df)} candles em '{filename}'.")
        total_candles += len(df)
        current_date = next_date

    mt5.shutdown()
    print(f"Coleta finalizada. Total de candles coletados: {total_candles}")


# Função 2: Preparar e normalizar dados para treinamento
def preparar_dados():
    all_data = []
    for file in sorted(os.listdir(data_folder)):
        df = pd.read_csv(os.path.join(data_folder, file))
        df["time"] = pd.to_datetime(df["time"])
        all_data.append(df)
    full_df = pd.concat(all_data).reset_index(drop=True)
    scaler = MinMaxScaler()
    full_df[features] = scaler.fit_transform(full_df[features])
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_folder, "scaler.pkl"))
    return full_df


# Função 3: Treinamento do modelo
def treinar_modelo(epochs=30):
    full_df = preparar_dados()
    X, y, times_in, times_out = [], [], [], []
    for i in range(len(full_df) - input_window - output_window):
        x_window = full_df[features].iloc[i : i + input_window].values
        y_window = (
            full_df[features]
            .iloc[i + input_window : i + input_window + output_window]
            .values
        )
        time_in = full_df["time"].iloc[i : i + input_window].values
        time_out = (
            full_df["time"]
            .iloc[i + input_window : i + input_window + output_window]
            .values
        )
        X.append(x_window)
        y.append(y_window)
        times_in.append(time_in)
        times_out.append(time_out)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

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


# Função 4: Executar predição
def executar_predicao_tempo_real():
    if not mt5.initialize():
        print("Erro ao conectar ao MT5")
        return

    # Obtém 1000 candles mais recentes
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, input_window)
    mt5.shutdown()
    if rates is None or len(rates) < input_window:
        print("Dados insuficientes para previsão.")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
    df[features] = scaler.transform(df[features])

    X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(0).to(device)
    model = LSTMModel(input_size=len(features)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(model_folder, "modelo_lstm.pth"), map_location=device)
    )
    model.eval()

    with torch.no_grad():
        prediction = model(X).cpu().numpy().squeeze()

    # Desnormaliza os dados previstos
    pred_desnorm = scaler.inverse_transform(prediction)
    # Gera datas sequenciais para os candles previstos
    last_date = df["time"].iloc[-1]
    dates_pred = [last_date + timedelta(minutes=i + 1) for i in range(output_window)]
    df_pred = pd.DataFrame(pred_desnorm, columns=features)
    df_pred["time"] = dates_pred

    print("Previsão dos próximos 100 candles (OHLCV e data):")
    print(df_pred)
    return df_pred


# coletar_dados_mt5(dias=20)

# treinar_modelo(epochs=30)
if __name__ == "__main__":
    coletar_dados_mt5(dias=1000)
    # treinar_modelo(epochs=50)
    # pred_df = executar_predicao_tempo_real()
    # Aqui você pode plotar ou salvar pred_df conforme desejar

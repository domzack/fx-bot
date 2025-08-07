import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import mplfinance as mpf
from trainer import LSTMTrainer  # Importa a classe de treinamento

# Configurações
input_window = 1000
output_window = 100
features = ["open", "high", "low", "close", "volume"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 64


# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)
        self.output_size = output_size  # Store output_size as an instance variable

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, output_window, self.output_size)


# Função para preparar os dados
def preparar_dados(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df[features].copy()

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    joblib.dump(scaler, "scaler.pkl")

    X, y = [], []
    for i in range(len(df) - input_window - output_window):
        x_window = df.iloc[i : i + input_window].values
        y_window = df.iloc[i + input_window : i + input_window + output_window].values
        X.append(x_window)
        y.append(y_window)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(features)


# Função para previsão
def executar_previsao(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df[features].copy()
    scaler = joblib.load("scaler.pkl")
    df[features] = scaler.transform(df[features])

    latest_window = df.iloc[-input_window:].values
    X = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0).to(device)

    model = LSTMModel(input_size=len(features)).to(device)
    model.load_state_dict(torch.load("modelo_lstm_ohlcv.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        prediction = model(X).cpu().numpy().squeeze()

    predicted_real = scaler.inverse_transform(prediction.reshape(-1, len(features)))
    print("Previsão dos próximos 100 candles (OHLCV):")
    print(predicted_real)


def executar_previsao_mt5(symbol="US500"):
    import MetaTrader5 as mt5

    if not mt5.initialize():
        print("Erro ao conectar ao MT5")
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1001)
    mt5.shutdown()

    if rates is None or len(rates) < 1001:
        print("Não foi possível obter 1001 candles do MT5.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[features].copy()
    scaler = joblib.load("scaler.pkl")
    df[features] = scaler.transform(df[features])

    # Ignora o candle mais atual
    latest_window = df.iloc[-1001:-1].values

    X = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0).to(device)

    model = LSTMModel(input_size=len(features)).to(device)
    model.load_state_dict(torch.load("modelo_lstm_ohlcv.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        prediction = model(X).cpu().numpy().squeeze()

    predicted_real = scaler.inverse_transform(prediction.reshape(-1, len(features)))
    return predicted_real


def Deseja_retreinar_modelo():
    resposta = input("Deseja re-treinar o modelo com os novos dados? (s/n): ")
    return resposta.lower() == "s"


def desnormalizar_dados(dados_norm, scaler):
    """
    Desnormaliza os dados usando o scaler salvo.
    """
    return scaler.inverse_transform(dados_norm.reshape(-1, len(features)))


def obter_ultimos_candles_mt5(symbol="US500", n=100):
    import MetaTrader5 as mt5

    if not mt5.initialize():
        print("Erro ao conectar ao MT5")
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    mt5.shutdown()

    if rates is None or len(rates) < n:
        print(f"Não foi possível obter {n} candles do MT5.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


def plotar_ohlc_com_previsao(symbol="US500"):
    scaler = joblib.load("scaler.pkl")
    # Obtém últimos 100 candles reais do MT5
    df_real = obter_ultimos_candles_mt5(symbol=symbol, n=100)
    if df_real is None:
        return

    # Obtém dados previstos
    dados_preditos_norm = executar_previsao_mt5(symbol=symbol)
    if dados_preditos_norm is None:
        return
    dados_preditos = desnormalizar_dados(dados_preditos_norm, scaler)

    # Cria DataFrame para os dados previstos
    last_time = df_real["time"].iloc[-1]
    times_pred = [
        last_time + pd.Timedelta(minutes=i + 1) for i in range(len(dados_preditos))
    ]
    df_pred = pd.DataFrame(dados_preditos, columns=features)
    df_pred["time"] = times_pred
    df_pred["volume"] = df_pred["volume"].astype(int)

    # Mescla os dados reais e previstos
    df_plot = pd.concat([df_real, df_pred], ignore_index=True)
    df_plot.set_index("time", inplace=True)

    # Plota o gráfico OHLC
    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        volume=True,
        title="OHLC US500 - Real + Previsão",
        ylabel="Preço",
        ylabel_lower="Volume",
        show_nontrading=True,
    )


csv_path = "history/US500_all.csv"  # Caminho para o arquivo CSV com os dados OHLCV
if __name__ == "__main__":

    trainer = LSTMTrainer(csv_path=csv_path, device=device, features=features)
    if not os.path.exists("modelo_lstm_ohlcv.pth"):
        trainer.train()
    elif Deseja_retreinar_modelo():
        print("Re-treinando o modelo com os dados mais recentes...")
        trainer.retrain()

    # executar_previsao(csv_path)  # Executa previsão com os dados mais recentes
    predict = executar_previsao_mt5(symbol="US500")
    plotar_ohlc_com_previsao(symbol="US500")

import sys
import subprocess

required = ["pandas", "numpy", "torch", "scikit-learn", "pandas_ta", "joblib"]
for pkg in required:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import joblib

# Configurações globais
data_folder = "dados_candles"
model_folder = "modelos"
input_window = 200
output_window = 20
features = ["open", "high", "low", "close", "volume", "rsi"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, output_window, 5)


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


if __name__ == "__main__":
    treinar_modelo(epochs=100)
    print("Modelo treinado e salvo com sucesso.")
    while True:
        pass

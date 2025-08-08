# Este arquivo é compatível com python3.
# Para rodar: python3 trainer.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import time
import psutil


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5, output_window=100):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)
        self.output_size = output_size
        self.output_window = output_window

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.view(-1, self.output_window, self.output_size)


class LSTMTrainer:
    def __init__(
        self,
        csv_path,
        device,
        features,
        input_window=1000,
        output_window=100,
        batch_size=64,
        epochs=100,
        artifacts_folder="models",
    ):
        self.csv_path = csv_path
        self.device = device
        self.features = features
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.epochs = epochs
        self.artifacts_folder = artifacts_folder

    def preparar_dados(self):
        print("[LSTMTrainer] Preparando dados para treinamento...")
        df = pd.read_csv(self.csv_path, parse_dates=["time"])
        print(
            f"[LSTMTrainer] Dados carregados de {self.csv_path}. Total de registros: {len(df)}"
        )
        df = df[self.features].copy()

        scaler = MinMaxScaler()
        df[self.features] = scaler.fit_transform(df[self.features])
        os.makedirs(self.artifacts_folder, exist_ok=True)
        joblib.dump(scaler, os.path.join(self.artifacts_folder, "scaler.pkl"))
        print(
            f"[LSTMTrainer] Dados normalizados e scaler salvo em {self.artifacts_folder}/scaler.pkl."
        )

        X, y = [], []
        for i in range(len(df) - self.input_window - self.output_window):
            x_window = df.iloc[i : i + self.input_window].values
            y_window = df.iloc[
                i + self.input_window : i + self.input_window + self.output_window
            ].values
            X.append(x_window)
            y.append(y_window)

        print(f"[LSTMTrainer] Total de amostras para treinamento: {len(X)}")
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
            print(
                f"[LSTMTrainer] Memória antes de converter para tensor: {mem_before:.2f} MB"
            )
            print("[LSTMTrainer] Convertendo X para tensor...")
            X = torch.tensor(np.array(X), dtype=torch.float32)
            mem_after_X = process.memory_info().rss / (1024 * 1024)
            print(
                f"[LSTMTrainer] X convertido para tensor. Memória atual: {mem_after_X:.2f} MB"
            )
            print("[LSTMTrainer] Convertendo y para tensor...")
            y = torch.tensor(np.array(y), dtype=torch.float32)
            mem_after_y = process.memory_info().rss / (1024 * 1024)
            print(
                f"[LSTMTrainer] y convertido para tensor. Memória atual: {mem_after_y:.2f} MB"
            )
        except Exception as e:
            print(f"[LSTMTrainer] Erro ao converter para tensor: {e}")
            import traceback

            print(traceback.format_exc())
            raise
        print("[LSTMTrainer] Dados convertidos para tensores PyTorch.")
        dataset = TensorDataset(X, y)
        print("[LSTMTrainer] Dataset criado com tensores de entrada e saída.")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print(f"[LSTMTrainer] DataLoader criado com batch_size={self.batch_size}")
        return dataloader, len(self.features)

    def train(self):
        print("[LSTMTrainer] Iniciando treinamento do modelo LSTM...")
        dataloader, input_size = self.preparar_dados()
        print(f"[LSTMTrainer] Tamanho da entrada do modelo: {input_size}")
        model = LSTMModel(
            input_size=input_size,
            output_size=len(self.features),
            output_window=self.output_window,
        ).to(self.device)
        print("[LSTMTrainer] Modelo LSTM criado e movido para o dispositivo.")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("[LSTMTrainer] Otimizador Adam criado com taxa de aprendizado 0.001.")
        criterion = nn.MSELoss()
        print("[LSTMTrainer] Função de perda MSELoss definida.")
        last_epoch_time = time.time()
        print("[LSTMTrainer] Iniciando o loop de treinamento...")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            now = time.time()
            elapsed = now - last_epoch_time
            print(
                f"[LSTMTrainer] Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f} - Tempo desde última época: {elapsed:.2f}s"
            )
            last_epoch_time = now

        os.makedirs(self.artifacts_folder, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth"),
        )
        print(
            f"[LSTMTrainer] Modelo salvo em {self.artifacts_folder}/modelo_lstm_ohlcv.pth"
        )

    def retrain(self):
        print("[LSTMTrainer] Iniciando re-treinamento do modelo LSTM...")
        dataloader, input_size = self.preparar_dados()
        model = LSTMModel(
            input_size=input_size,
            output_size=len(self.features),
            output_window=self.output_window,
        ).to(self.device)
        model.load_state_dict(
            torch.load(
                os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth"),
                map_location=self.device,
            )
        )
        print(
            f"[LSTMTrainer] Pesos do modelo carregados de {self.artifacts_folder}/modelo_lstm_ohlcv.pth."
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        last_epoch_time = time.time()
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            now = time.time()
            elapsed = now - last_epoch_time
            print(
                f"[LSTMTrainer] Re-treino Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f} - Tempo desde última época: {elapsed:.2f}s"
            )
            last_epoch_time = now

        torch.save(
            model.state_dict(),
            os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth"),
        )
        print(
            f"[LSTMTrainer] Modelo atualizado e salvo em {self.artifacts_folder}/modelo_lstm_ohlcv.pth"
        )

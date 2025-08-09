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
import glob
from datetime import datetime, timedelta
import shutil
from class_OHLCVNormalizer import OHLCVNormalizer


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

    def train(self):
        """
        Lê o arquivo ./dados_candles/historico.csv, treina em blocos de 1000 a 2000 candles não treinados,
        marca os registros treinados na coluna [treinado], salva o modelo e emite logs.
        """

        def log(msg):
            n_threads = torch.get_num_threads()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CPUs: {n_threads} | {msg}"
            )

        csv_path = "./dados_candles/US500_M1.csv"
        artifacts_folder = self.artifacts_folder
        model_path = os.path.join(artifacts_folder, "modelo_lstm_ohlcv.pth")
        scaler_path = os.path.join(artifacts_folder, "scaler.pkl")

        # Garante que a coluna 'treinado' existe
        if not os.path.exists(csv_path):
            log(f"Arquivo {csv_path} não encontrado.")
            return

        # Carrega todo o histórico
        df = pd.read_csv(csv_path)
        if "treinado" not in df.columns:
            df["treinado"] = 0

        normalizer = OHLCVNormalizer()

        # Treina enquanto houver blocos não treinados
        while True:
            # Seleciona bloco de 1000 a 2000 candles não treinados
            bloco = df[df["treinado"] == 0].head(2000)
            if len(bloco) < self.input_window + self.output_window:
                log(
                    "Não há dados suficientes para formar um bloco. Fim do treinamento."
                )
                break

            log(f"Iniciando treinamento de bloco com {len(bloco)} dados não treinados.")

            # Prepara dados do bloco
            # Monta lista de listas [OHLCV] para cada linha
            bloco_ohlcv = bloco[self.features].values.tolist()
            # Normaliza usando OHLCVNormalizer
            bloco_normalizado = normalizer.normalize(bloco_ohlcv)
            # Converte para DataFrame para facilitar manipulação
            bloco_features = pd.DataFrame(
                bloco_normalizado, columns=["var", "max", "min", "vol"]
            )

            # Não precisa salvar scaler, mas mantém compatibilidade
            joblib.dump(normalizer, scaler_path)
            log(f"Normalizer salvo em {scaler_path}.")

            X, y = [], []
            for i in range(
                len(bloco_features) - self.input_window - self.output_window
            ):
                x_window = bloco_features.iloc[i : i + self.input_window].values
                y_window = bloco_features.iloc[
                    i + self.input_window : i + self.input_window + self.output_window
                ].values
                X.append(x_window)
                y.append(y_window)

            if len(X) == 0:
                log("Nenhuma amostra suficiente para treinamento neste bloco. Pulando.")
                # Não marca como treinado, apenas ignora e aguarda mais dados
                continue

            X = torch.tensor(np.array(X), dtype=torch.float32)
            y = torch.tensor(np.array(y), dtype=torch.float32)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Carrega ou cria modelo
            if os.path.exists(model_path):
                model = LSTMModel(
                    input_size=4,  # agora são 4 colunas: var, max, min, vol
                    output_size=4,
                    output_window=self.output_window,
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                log("Modelo carregado do treinamento anterior.")
            else:
                model = LSTMModel(
                    input_size=4,
                    output_size=4,
                    output_window=self.output_window,
                ).to(self.device)
                log("Novo modelo criado.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            for epoch in range(self.epochs):

                # Ajusta recursos conforme horário
                now = datetime.now()
                is_sunday = now.weekday() == 6
                hora = now.hour + now.minute / 60.0
                if not (is_sunday or hora >= 22 or hora < 7.5):
                    torch.set_num_threads(1)
                    log("Restringindo recursos.")
                else:
                    n_threads = max(1, os.cpu_count() - 1)
                    torch.set_num_threads(n_threads)
                    log(f"Recursos liberados.")

                inicio_epoca = time.time()
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
                fim_epoca = time.time()
                tempo_epoca = fim_epoca - inicio_epoca
                epocas_restantes = self.epochs - (epoch + 1)
                tempo_estimado = tempo_epoca * epocas_restantes
                log(
                    f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f} - Tempo: {tempo_epoca:.2f}s - Estimativa restante: {tempo_estimado/60:.2f} min"
                )

            torch.save(model.state_dict(), model_path)
            log(f"Modelo salvo em {model_path}")

            # Marca como treinado apenas após sucesso do treinamento
            idxs = bloco.index
            df.loc[idxs, "treinado"] = 1
            df.to_csv(csv_path, index=False)
            log(f"Bloco de {len(idxs)} candles marcado e salvo em {csv_path}")

            # Executa git pop após o sucesso do treinamento
            os.system(f"git pop 'bloco_{idxs[0]}_{idxs[-1]} train success'")

        log("Treinamento de todos os blocos concluído.")

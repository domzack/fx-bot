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
    def __init__(
        self,
        input_size,
        hidden_size=256,
        output_size=4,
        output_window=100,
        num_layers=2,
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, output_size)  # Corrigido para Teacher Forcing
        self.output_size = output_size
        self.output_window = output_window

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # Pega todos os passos da sequência
        out = self.fc(out)
        return out  # [batch_size, seq_len, output_size]

    def forward_step(self, x, hidden=None):
        # x: [batch_size, 1, input_size]
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out.view(-1, 1, self.output_size), hidden


class LSTMTrainer:
    def __init__(
        self,
        csv_path,
        device,
        features,
        input_window=100,  # janela de entrada agora é 100
        output_window=50,
        batch_size=128,  # batch maior
        epochs=100,
        artifacts_folder="models",
        hidden_size=256,  # novo parâmetro
        num_layers=2,  # novo parâmetro
    ):
        self.csv_path = csv_path
        self.device = device
        self.features = features
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.epochs = epochs
        self.artifacts_folder = artifacts_folder
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

        normalizer = OHLCVNormalizer()

        # Carrega todo o histórico
        df = pd.read_csv(csv_path, parse_dates=["time"])
        if "treinado" not in df.columns:
            df["treinado"] = 0
        normalized_df = normalizer.normalizeV2(df)
        print(normalized_df.head())

        # Treina enquanto houver blocos não treinados
        while True:
            # Seleciona bloco de 1000 a 2000 candles não treinados
            bloco = df[df["treinado"] == 0].head(10000)
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
            bloco_normalizado = np.array(bloco_normalizado)
            if bloco_normalizado.ndim == 1 or bloco_normalizado.shape[1] != 4:
                # Tenta reshape para (N, 4)
                bloco_normalizado = bloco_normalizado.reshape(-1, 4)
            bloco_normalizado = np.round(bloco_normalizado, 2)  # 2 casas decimais

            # print(bloco_normalizado)

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
            model = LSTMModel(
                input_size=4,
                output_size=4,
                output_window=self.output_window,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            ).to(self.device)
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(
                        torch.load(
                            model_path, map_location=self.device, weights_only=True
                        )
                    )
                    log("Modelo carregado do treinamento anterior.")
                except Exception as e:
                    log(f"Falha ao carregar modelo antigo: {e}.")
                    log("Criando novo modelo do zero")
            else:
                log("Novo modelo criado.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            n_threads = max(1, os.cpu_count() - 3)
            torch.set_num_threads(n_threads)

            for epoch in range(self.epochs):

                # Ajusta recursos conforme horário
                now = datetime.now()
                is_sunday = now.weekday() == 6
                hora = now.hour + now.minute / 60.0
                if not (is_sunday or hora >= 22 or hora < 7.5):
                    if n_threads > 1:
                        n_threads = 1
                        torch.set_num_threads(n_threads)
                        log("Restringindo recursos.")
                else:
                    if n_threads == 1:
                        n_threads = max(1, os.cpu_count() - 3)
                        torch.set_num_threads(n_threads)
                        log(f"Recursos liberados.")

                inicio_epoca = time.time()
                model.train()
                total_loss = 0
                for xb, yb in dataloader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    batch_size = xb.size(0)
                    input_seq = xb  # [batch_size, input_window, features]
                    target_seq = yb  # [batch_size, output_window, features]
                    # Inicializa o hidden state
                    hidden = None
                    decoder_input = input_seq[:, -1, :].unsqueeze(
                        1
                    )  # [batch_size, 1, features]
                    # Inicializa outputs corretamente
                    outputs = torch.zeros(
                        batch_size,
                        self.output_window,
                        input_seq.size(2),
                        device=self.device,
                    )
                    for t in range(self.output_window):
                        out, hidden = model.forward_step(
                            decoder_input, hidden
                        )  # [batch_size, 1, features]
                        outputs[:, t, :] = out[
                            :, 0, :
                        ]  # atribui diretamente na posição correta
                        decoder_input = target_seq[:, t, :].unsqueeze(1)
                    loss = criterion(outputs, target_seq)
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
            # os.system(f"git pop 'bloco_{idxs[0]}_{idxs[-1]} train success'")

        log("Treinamento de todos os blocos concluído.")

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

    def train_incremental(self, csv_parts):
        """
        Treina o modelo incrementalmente, processando cada parte dos dados separadamente.
        csv_parts: lista de caminhos para arquivos CSV menores.
        """
        print("[LSTMTrainer] Iniciando treinamento incremental...")
        model_path = os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth")
        for idx, csv_part in enumerate(csv_parts):
            print(f"[LSTMTrainer] Treinando parte {idx+1}/{len(csv_parts)}: {csv_part}")
            self.csv_path = csv_part
            dataloader, input_size = self.preparar_dados()
            if os.path.exists(model_path):
                model = LSTMModel(
                    input_size=input_size,
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("[LSTMTrainer] Modelo carregado do treinamento anterior.")
            else:
                model = LSTMModel(
                    input_size=input_size,
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                print("[LSTMTrainer] Novo modelo criado.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

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
                print(
                    f"[LSTMTrainer] Parte {idx+1} Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}"
                )

            torch.save(model.state_dict(), model_path)
            print(f"[LSTMTrainer] Modelo salvo após parte {idx+1} em {model_path}")

        print("[LSTMTrainer] Treinamento incremental concluído.")

    def train_csv_in_chunks(self):
        """
        Carrega o CSV em grupos de 10 mil linhas e executa o treinamento sucessivamente em cada grupo.
        O modelo é atualizado a cada grupo e salvo ao final de cada chunk.
        """
        print("[LSTMTrainer] Iniciando treinamento em chunks de 10 mil linhas...")
        chunk_size = 10000
        model_path = os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth")
        scaler_path = os.path.join(self.artifacts_folder, "scaler.pkl")
        chunk_iter = pd.read_csv(
            self.csv_path, parse_dates=["time"], chunksize=chunk_size
        )
        chunk_count = 0

        for chunk in chunk_iter:
            chunk_count += 1
            print(f"[LSTMTrainer] Processando chunk {chunk_count}...")
            df = chunk[self.features].copy()

            scaler = MinMaxScaler()
            df[self.features] = scaler.fit_transform(df[self.features])
            joblib.dump(scaler, scaler_path)
            print(f"[LSTMTrainer] Dados normalizados e scaler salvo em {scaler_path}.")

            X, y = [], []
            for i in range(len(df) - self.input_window - self.output_window):
                x_window = df.iloc[i : i + self.input_window].values
                y_window = df.iloc[
                    i + self.input_window : i + self.input_window + self.output_window
                ].values
                X.append(x_window)
                y.append(y_window)

            print(
                f"[LSTMTrainer] Chunk {chunk_count}: Total de amostras para treinamento: {len(X)}"
            )
            if len(X) == 0:
                print(
                    f"[LSTMTrainer] Chunk {chunk_count}: Nenhuma amostra suficiente para treinamento, pulando chunk."
                )
                continue

            X = torch.tensor(np.array(X), dtype=torch.float32)
            y = torch.tensor(np.array(y), dtype=torch.float32)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            if os.path.exists(model_path):
                model = LSTMModel(
                    input_size=len(self.features),
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(
                    f"[LSTMTrainer] Chunk {chunk_count}: Modelo carregado do treinamento anterior."
                )
            else:
                model = LSTMModel(
                    input_size=len(self.features),
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                print(f"[LSTMTrainer] Chunk {chunk_count}: Novo modelo criado.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

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
                print(
                    f"[LSTMTrainer] Chunk {chunk_count} Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}"
                )

            torch.save(model.state_dict(), model_path)
            print(
                f"[LSTMTrainer] Modelo salvo após chunk {chunk_count} em {model_path}"
            )

        print("[LSTMTrainer] Treinamento em chunks concluído.")

    def train_all_history_csvs_scheduled(self):
        """
        Treina todos os arquivos CSV da pasta ./history de forma incremental, seguindo regras de horário e uso de recursos.
        """

        def log(msg):
            n_threads = torch.get_num_threads()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CPUs: {n_threads} | {msg}"
            )

        history_folder = "./history"
        csv_files = sorted(
            glob.glob(os.path.join(history_folder, "US500-M1-*.csv")), key=lambda x: x
        )

        if not csv_files:
            log("Nenhum arquivo CSV encontrado em ./history.")
            return

        log(f"Encontrados {len(csv_files)} arquivos CSV para treinamento.")

        # Verifica se é domingo
        now = datetime.now()
        is_sunday = now.weekday() == 6

        # Função para verificar se está dentro do horário permitido
        def pode_treinar():
            now = datetime.now()
            if is_sunday:
                return True
            hora = now.hour + now.minute / 60.0
            return hora >= 22 or hora < 7.5

        # Função para limitar uso de CPU (apenas reduz threads do pytorch)
        def limitar_recursos():
            torch.set_num_threads(1)
            log("Recursos limitados: set_num_threads(1)")

        def liberar_recursos():
            n_threads = max(1, os.cpu_count() - 1)
            torch.set_num_threads(n_threads)
            log(f"Recursos liberados: set_num_threads({n_threads})")

        model_path = os.path.join(self.artifacts_folder, "modelo_lstm_ohlcv.pth")
        scaler_path = os.path.join(self.artifacts_folder, "scaler.pkl")

        for csv_file in csv_files:
            log(f"Iniciando treinamento do arquivo: {csv_file}")

            # Ajusta recursos conforme horário
            if pode_treinar():
                liberar_recursos()
            else:
                limitar_recursos()

            # Carrega dados
            df = pd.read_csv(csv_file, header=None)
            df.columns = self.features
            scaler = MinMaxScaler()
            df[self.features] = scaler.fit_transform(df[self.features])
            joblib.dump(scaler, scaler_path)
            log(f"Scaler salvo em {scaler_path}.")

            X, y = [], []
            for i in range(len(df) - self.input_window - self.output_window):
                x_window = df.iloc[i : i + self.input_window].values
                y_window = df.iloc[
                    i + self.input_window : i + self.input_window + self.output_window
                ].values
                X.append(x_window)
                y.append(y_window)

            if len(X) == 0:
                log(
                    "Nenhuma amostra suficiente para treinamento neste arquivo. Pulando."
                )
                # Renomeia para indicar que foi processado, mesmo sem treinar
                novo_nome = csv_file + ".x"
                shutil.move(csv_file, novo_nome)
                log(f"Arquivo renomeado para {novo_nome}")
                continue

            X = torch.tensor(np.array(X), dtype=torch.float32)
            y = torch.tensor(np.array(y), dtype=torch.float32)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Carrega ou cria modelo
            if os.path.exists(model_path):
                model = LSTMModel(
                    input_size=len(self.features),
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                log("Modelo carregado do treinamento anterior.")
            else:
                model = LSTMModel(
                    input_size=len(self.features),
                    output_size=len(self.features),
                    output_window=self.output_window,
                ).to(self.device)
                log("Novo modelo criado.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            epoch_times = []
            for epoch in range(self.epochs):
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
                epoch_times.append(tempo_epoca)
                tempo_medio = np.mean(epoch_times)
                tempo_estimado = tempo_medio * (self.epochs - (epoch + 1))
                log(
                    f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f} - Tempo: {tempo_epoca:.2f}s - Estimativa restante: {tempo_estimado/60:.2f} min"
                )

                # Após cada época, verifica se está em horário restrito e ajusta recursos dinamicamente
                if not pode_treinar():
                    torch.set_num_threads(1)
                    log(
                        "Restrição de recursos aplicada após época (set_num_threads(1))."
                    )
                else:
                    n_threads = max(1, os.cpu_count() - 1)
                    torch.set_num_threads(n_threads)
                    log(
                        f"Recursos liberados após época (set_num_threads({n_threads}))."
                    )

            torch.save(model.state_dict(), model_path)
            log(f"Modelo salvo em {model_path}")

            # Renomeia arquivo para indicar que foi treinado
            novo_nome = csv_file + ".x"
            shutil.move(csv_file, novo_nome)
            log(f"Arquivo renomeado para {novo_nome}")

            # Executa git pop após o sucesso do treinamento
            os.system(f"git pop '{novo_nome} train success'")

        log("Treinamento de todos arquivos CSV concluído.")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib


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
    ):
        self.csv_path = csv_path
        self.device = device
        self.features = features
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.epochs = epochs

    def preparar_dados(self):
        df = pd.read_csv(self.csv_path, parse_dates=["time"])
        df = df[self.features].copy()

        scaler = MinMaxScaler()
        df[self.features] = scaler.fit_transform(df[self.features])
        joblib.dump(scaler, "scaler.pkl")

        X, y = [], []
        for i in range(len(df) - self.input_window - self.output_window):
            x_window = df.iloc[i : i + self.input_window].values
            y_window = df.iloc[
                i + self.input_window : i + self.input_window + self.output_window
            ].values
            X.append(x_window)
            y.append(y_window)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader, len(self.features)

    def train(self):
        dataloader, input_size = self.preparar_dados()
        model = LSTMModel(
            input_size=input_size,
            output_size=len(self.features),
            output_window=self.output_window,
        ).to(self.device)
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
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), "modelo_lstm_ohlcv.pth")
        print("Modelo salvo como modelo_lstm_ohlcv.pth")

    def retrain(self):
        dataloader, input_size = self.preparar_dados()
        model = LSTMModel(
            input_size=input_size,
            output_size=len(self.features),
            output_window=self.output_window,
        ).to(self.device)
        model.load_state_dict(
            torch.load("modelo_lstm_ohlcv.pth", map_location=self.device)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
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
            print(f"Re-treino Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), "modelo_lstm_ohlcv.pth")
        print("Modelo atualizado com re-treinamento.")

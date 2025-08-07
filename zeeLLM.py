from trainer import LSTMTrainer  # Importa a classe de treinamento
import torch
import os


if __name__ == "__main__":
    try:
        features = ["open", "high", "low", "close", "volume"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        csv_path = (
            "history/US500_all.csv"  # Caminho para o arquivo CSV com os dados OHLCV
        )
        trainer = LSTMTrainer(csv_path=csv_path, device=device, features=features)
        trainer.train()  # Executa o treinamento

        # Executa o comando git pop 'model' após o treinamento
        os.system("git pop 'model'")
        print("[LSTMTrainer] Treinamento concluído com sucesso.")
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")

# pm2 start zeeLLM.py --interpreter python --no-autorestart --name zeeLLM --output zeeLLM_out.log --error zeeLLM_err.log

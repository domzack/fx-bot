from trainer import LSTMTrainer  # Importa a classe de treinamento
import torch
import os
import traceback


if __name__ == "__main__":
    try:
        features = ["open", "high", "low", "close", "volume"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        csv_path = "history/US500_all.csv"
        artifacts_folder = "training_artifacts"
        os.makedirs(artifacts_folder, exist_ok=True)

        print("[zeeLLM] Instanciando LSTMTrainer...")
        trainer = LSTMTrainer(
            csv_path=csv_path,
            device=device,
            features=features,
            artifacts_folder=artifacts_folder,  # novo parâmetro
        )
        print("[zeeLLM] Chamando trainer.train_all_history_csvs_scheduled()...")
        trainer.train_all_history_csvs_scheduled()
        print("[zeeLLM] Treinamento finalizado, executando git pop 'model'...")

        # Executa o comando git pop 'model' após o treinamento
        os.system("git pop 'model'")
        print("[LSTMTrainer] Treinamento concluído com sucesso.")
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        with open(os.path.join(artifacts_folder, "zeeLLM_error.log"), "a") as f:
            f.write(traceback_str + "\n")

# pm2 start zeeLLM.py --interpreter python --no-autorestart --name zeeLLM --output zeeLLM_out.log --error zeeLLM_err.log

# c:\mygits\fx-bot\src\split_ohlcv_by_day.py

import os
import csv
from collections import defaultdict


def split_ohlcv_by_day(input_path):
    print(f"Lendo arquivo: {input_path}")
    day_data = defaultdict(list)
    with open(input_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            time = row[0]
            date = time.split(" ")[0]  # YYYY-MM-DD
            year, month, day = date.split("-")
            filename = f"./history/US500-M1-{year}-{month}-{day}.csv"
            ohlcv = row[1:]  # [open, high, low, close, volume]
            day_data[filename].append(ohlcv)

    print(f"Encontrados {len(day_data)} dias diferentes.")
    for filename, rows in day_data.items():
        print(f"Salvando {len(rows)} candles em {filename}")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    print("Processo concluído.")


# Exemplo de uso:
split_ohlcv_by_day("./history/US500_all.csv")

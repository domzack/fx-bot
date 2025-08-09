def Normalize_OHLCV(OHLCV):

    normalized_data = []

    try:
        for i in range(len(OHLCV)):
            # candle = O[0], H[1], L[2], C[3], V[4]

            # Se não houver candle anterior, cria um genérico igual ao atual
            if i == 0:
                before_open = OHLCV[i][0]
                before_close = OHLCV[i][3]
                before_volume = OHLCV[i][4]
            elif i == 1:
                before_open = OHLCV[i - 1][0]
                before_close = OHLCV[i - 1][3]
                before_volume = OHLCV[i - 1][4]
                pass
            else:
                before_open = OHLCV[i - 2][3]  # O = C do candle anterior
                before_close = OHLCV[i - 1][3]
                before_volume = OHLCV[i - 1][4]

            # Define os valores do candle atual
            now_open = before_close
            now_high = max(before_close, OHLCV[i][1], OHLCV[i][2], OHLCV[i][3])
            now_low = min(before_close, OHLCV[i][1], OHLCV[i][2], OHLCV[i][3])
            now_close = OHLCV[i][3]
            now_volume = OHLCV[i][4]

            # calcula o "corpo" (spread)
            spread = now_close - now_open
            before_spread = before_close - before_open

            # calcula os "pavios" (ghosts)
            ghost_high = max(now_open, now_close) - now_high
            ghost_low = min(now_open, now_close) - now_low
            # normaliza valor percentual
            norm_spread = spread / before_spread if before_spread != 0 else 0
            norm_maxima = ghost_high / spread if spread != 0 else 0
            norm_minima = ghost_low / spread if spread != 0 else 0
            norm_volume = now_volume / before_volume if before_volume != 0 else 0

            if spread >= 0 and norm_spread < 0:
                norm_spread = norm_spread * -1
            if spread < 0 and norm_spread > 0:
                norm_spread = norm_spread * -1

            norm_maxima = norm_maxima if norm_maxima >= 0 else norm_maxima * -1
            norm_minima = norm_minima if norm_minima < 0 else norm_minima * -1

            txt = f"{i:<3} {round(norm_spread,2):>8} {round(norm_maxima,2):>8} {round(norm_minima,2):>8} {round(norm_volume,2):>8} {round(spread,2):>8}"
            print(txt)

            normalized_data.append(
                [
                    round(norm_spread, 2),
                    round(norm_maxima, 2),
                    round(norm_minima, 2),
                    round(norm_volume, 2),
                ]
            )

        return True, normalized_data, "Dados normalizados com sucesso."
    except Exception as e:
        print(f"Erro na normalização: {e}")
        return False, normalized_data, f"Erro na normalização: {e}"


data = [
    [5480.32, 5481.12, 5476.72, 5478.42, 271],
    [5478.42, 5479.42, 5476.32, 5478.12, 280],
    [5478.12, 5478.22, 5475.92, 5476.82, 208],
    [5476.82, 5477.72, 5475.42, 5477.42, 229],
    [5477.42, 5479.32, 5476.62, 5478.02, 256],
    [5478.02, 5480.32, 5477.42, 5478.62, 298],
    [5478.62, 5483.02, 5477.92, 5483.02, 296],
    [5483.02, 5483.29, 5479.69, 5479.79, 295],
    [5479.79, 5480.52, 5474.52, 5475.12, 302],
    [5475.12, 5475.12, 5471.52, 5472.22, 252],
    [5472.22, 5473.42, 5470.32, 5470.82, 307],
    [5470.82, 5473.42, 5469.22, 5469.72, 326],
    [5469.72, 5471.82, 5469.72, 5471.62, 280],
    [5471.62, 5472.53, 5471.03, 5471.53, 254],
    [5471.53, 5475.23, 5470.53, 5474.63, 293],
    [5474.63, 5475.31, 5472.11, 5473.01, 284],
    [5473.01, 5475.83, 5471.23, 5475.73, 342],
    [5475.73, 5475.92, 5472.72, 5475.72, 270],
    [5475.72, 5478.62, 5475.52, 5478.32, 267],
    [5478.32, 5480.32, 5477.62, 5479.72, 188],
]

(success, normalized, message) = Normalize_OHLCV(data)
print(f"Success: {success}, Message: {message}")


import csv


def ler_csv_e_normalizar(input_path, output_path):
    # Lê os dados do CSV
    dados = []
    linhas = []
    with open(input_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            # Converte para float os campos OHLCV
            dados.append(
                [
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                ]
            )
            linhas.append(row)

    # Normaliza os dados
    success, normalized, message = Normalize_OHLCV(dados)
    # print(f"Normalização: {message}")

    # Adiciona a coluna 'normalized[]' ao cabeçalho
    header.append("normalized[]")

    # Salva novo arquivo CSV com coluna extra
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row, norm in zip(linhas, normalized):
            row.append(str(norm))
            writer.writerow(row)


ler_csv_e_normalizar(
    "./dados_candles/US500_M1.csv", "./dados_candles/US500_M1_normalized.csv"
)

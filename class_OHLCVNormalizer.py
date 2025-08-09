import csv
import pandas as pd


class OHLCVNormalizer:
    def __init__(self, csv_path=None):
        self.data = []
        self.raw_rows = []
        if csv_path:
            self.load_csv(csv_path)

    def load_csv(self, csv_path):
        self.data = []
        self.raw_rows = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            self.header = next(reader)
            for row in reader:
                self.data.append(
                    [
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ]
                )
                self.raw_rows.append(row)

    def normalizeV2(self, df):
        """
        Normaliza candles com base no candle anterior.
        Retorna um DataFrame com spread, upper_wick, lower_wick e volume_pct.
        """
        normalized = []

        for i in range(1, len(df)):
            prev_open = df.loc[i - 1, "open"]
            prev_volume = df.loc[i - 1, "volume"]

            open_ = df.loc[i, "open"]
            high = df.loc[i, "high"]
            low = df.loc[i, "low"]
            close = df.loc[i, "close"]
            volume = df.loc[i, "volume"]

            spread = (close - open_) / prev_open
            upper_wick = (high - max(open_, close)) / prev_open
            lower_wick = (min(open_, close) - low) / prev_open
            volume_pct = volume / prev_volume if prev_volume != 0 else 0

            normalized.append(
                {
                    "spread": spread,
                    "upper_wick": upper_wick,
                    "lower_wick": lower_wick,
                    "volume_pct": volume_pct,
                }
            )

        return pd.DataFrame(normalized)

    def normalize(self, OHLCV=None):
        if OHLCV is None:
            OHLCV = self.data
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

                normalized_data.append(
                    [
                        norm_spread,
                        norm_maxima,
                        norm_minima,
                        norm_volume,
                    ]
                )

            return normalized_data
        except Exception as e:
            raise Exception(f"Erro na normalização: {e}")

    def denormalize(self, normalized, OHLCV=None):
        # Exemplo simples: retorna os dados originais para cada linha normalizada
        # (A desnormalização real depende da lógica usada na normalização)
        if OHLCV is None:
            OHLCV = self.data
        desnormalized_data = []
        try:
            for i, norm in enumerate(normalized):
                # Aqui você pode implementar a lógica inversa da normalização
                # Exemplo: retorna o OHLCV original
                desnormalized_data.append(OHLCV[i])
            return True, desnormalized_data, "Dados desnormalizados com sucesso."
        except Exception as e:
            print(f"Erro na desnormalização: {e}")
            return False, desnormalized_data, f"Erro na desnormalização: {e}"

    def save_with_normalized(self, output_path, normalized):
        header = self.header + ["normalized[]"]
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row, norm in zip(self.raw_rows, normalized):
                row.append(str(norm))
                writer.writerow(row)


# Exemplo de uso:
# normalizer = OHLCVNormalizer("./dados_candles/US500_M1.csv")
# success, normalized, msg = normalizer.normalize()
# normalizer.save_with_normalized("./dados_candles/US500_M1_normalized.csv", normalized)

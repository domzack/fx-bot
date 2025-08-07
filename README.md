# FX-Bot - Predição de Candles OHLCV com LSTM

Este projeto utiliza uma rede neural LSTM para prever candles futuros (OHLCV) de ativos financeiros usando dados do MetaTrader 5.

## Funcionalidades

- **Coleta de dados**: Baixa candles históricos do MetaTrader 5 e salva em arquivos CSV.
- **Preparação e normalização**: Junta os dados, normaliza os valores de OHLCV e prepara para o treinamento.
- **Treinamento**: Treina uma rede LSTM para prever os próximos 100 candles a partir de uma janela de 1000 candles.
- **Predição**: Executa a predição dos próximos 100 candles (OHLCV) e gera as datas sequenciais.
- **Salvamento de modelo**: Salva o modelo treinado e o scaler para uso futuro.

## Como usar

1. **Coletar dados do MT5**
   - Edite o símbolo desejado em `symbol`.
   - Execute:
     ```python
     coletar_dados_mt5(dias=30)
     ```
   - Os arquivos CSV serão salvos em `dados_candles/`.

2. **Treinar o modelo**
   - Execute:
     ```python
     treinar_modelo(epochs=50)
     ```
   - O modelo será salvo em `modelos/modelo_lstm.pth`.

3. **Executar predição**
   - Execute:
     ```python
     pred_df = executar_predicao_tempo_real()
     ```
   - O resultado será um DataFrame com os próximos 100 candles previstos (OHLCV e data).

## Estrutura dos dados

- **Entrada**: 1000 candles (OHLCV) normalizados.
- **Saída**: 100 candles previstos (OHLCV) desnormalizados e datas sequenciais.

## Requisitos

- Python 3.8+
- MetaTrader5 (`pip install MetaTrader5`)
- pandas, numpy, torch, scikit-learn, joblib

## Observações

- O código não utiliza indicadores técnicos, apenas os valores OHLCV.
- As datas dos candles previstos são geradas sequencialmente após o último candle real.
- O modelo pode ser ajustado para outros ativos e timeframes.

## Exemplo de uso

```python
if __name__ == "__main__":
    coletar_dados_mt5(dias=30)
    treinar_modelo(epochs=50)
    pred_df = executar_predicao_tempo_real()
    print(pred_df)
```

---
GitHub Copilot

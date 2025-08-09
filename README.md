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
# Documentação da função `train_all_history_csvs_scheduled`

## Objetivo

A função `train_all_history_csvs_scheduled` foi desenvolvida para realizar o treinamento incremental de um modelo LSTM utilizando dados históricos de candles (OHLCV) de forma eficiente, segura e auditável. Ela garante que cada bloco de dados seja treinado apenas uma vez, controla o uso de recursos do sistema conforme o horário e registra o progresso do treinamento.

---

## Lógica Geral

1. **Leitura do Histórico**
   - O arquivo de dados deve estar em `./dados_candles/US500_M1.csv`.
   - O arquivo contém os candles históricos e uma coluna adicional chamada `treinado` (0 = não treinado, 1 = treinado).
   - Se a coluna `treinado` não existir, ela é criada automaticamente.

2. **Seleção de Blocos para Treinamento**
   - A função seleciona blocos de até 2000 candles consecutivos que ainda não foram treinados (`treinado == 0`).
   - O treinamento só ocorre se houver pelo menos `input_window + output_window` registros (padrão: 1100 candles).

3. **Preparação dos Dados**
   - Os dados do bloco são normalizados usando `MinMaxScaler`.
   - São criadas amostras de entrada (`X`) e saída (`y`) para o modelo, usando janelas de 1000 candles para entrada e 100 para saída.

4. **Treinamento do Modelo**
   - O modelo LSTM é carregado (se já existir) ou criado do zero.
   - Para cada época:
     - O modelo é treinado com o bloco de dados.
     - É calculado o erro (`loss`), o tempo gasto na época e a estimativa de tempo restante para concluir as épocas.
     - O log exibe o tempo de treinamento da época.
     - O uso de CPUs é ajustado dinamicamente conforme o horário:
       - Horário restrito (exceto domingo, das 7:30 às 22h): usa apenas 1 CPU.
       - Horário liberado (domingo ou das 22h às 7:30): usa quase todos os CPUs disponíveis.

5. **Finalização do Bloco**
   - Após o treinamento do bloco, o modelo é salvo.
   - Os registros do bloco são marcados como treinados (`treinado = 1`) e o CSV é atualizado.
   - Um comando `git pop` é executado para registrar o sucesso do treinamento do bloco.

6. **Finalização Geral**
   - O processo repete até não haver mais blocos suficientes para treinar.
   - Ao final, um log indica que todos os blocos foram treinados.

---

## Logs

Durante o processo, são emitidos logs detalhados contendo:
- Data e hora da operação.
- Quantidade de CPUs utilizadas.
- Progresso do treinamento (época, erro, tempo, estimativa restante).
- Tempo de treinamento de cada época.
- Status de recursos (restrição/liberação de CPUs).
- Salvatagem do modelo e marcação dos dados como treinados.
- Execução do comando `git pop` para cada bloco treinado.

---

## Vantagens da Lógica

- **Segurança:** Apenas dados realmente treinados são marcados como treinados.
- **Auditabilidade:** Todo o histórico é mantido e pode ser revisitado.
- **Eficiência:** O uso de recursos é ajustado conforme o horário, evitando sobrecarga do sistema.
- **Incremental:** Permite adicionar novos dados ao arquivo e continuar o treinamento sem retrabalho.
- **Automação:** O processo é totalmente automatizado e pode ser executado em horários programados.

---

## Observações

- O arquivo CSV deve conter as colunas: `open`, `high`, `low`, `close`, `volume` e `treinado`.
- O treinamento só ocorre quando há dados suficientes para formar um bloco.
- O modelo e o scaler são salvos na pasta definida por `artifacts_folder`.
- O comando `git pop` é usado para registrar o sucesso do treinamento de cada bloco.

---

## Exemplo de Uso

```python
trainer = LSTMTrainer(
    csv_path="./dados_candles/US500_M1.csv",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    features=["open", "high", "low", "close", "volume"],
    artifacts_folder="models"
)
trainer.train_all_history_csvs_scheduled()
```

## O que é MinMaxScaler?

O `MinMaxScaler` é uma ferramenta de normalização de dados disponível na biblioteca scikit-learn.  
Ele transforma os valores das colunas para um intervalo definido (geralmente entre 0 e 1), mantendo a proporção entre os valores originais.

**Como funciona:**  
Para cada valor de uma coluna, o MinMaxScaler aplica a seguinte fórmula:

```
X_norm = (X - X_min) / (X_max - X_min)
```

Onde:
- `X` é o valor original,
- `X_min` é o menor valor da coluna,
- `X_max` é o maior valor da coluna,
- `X_norm` é o valor normalizado (entre 0 e 1).

**Para que serve:**  
- Facilita o treinamento de modelos de machine learning, pois evita que variáveis com escalas diferentes prejudiquem o aprendizado.
- Ajuda a acelerar a convergência do treinamento.
- Garante que todos os dados estejam no mesmo intervalo, tornando o modelo mais estável.

---
GitHub Copilot

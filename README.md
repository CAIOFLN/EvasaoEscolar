# Evas-o-escolar-

Projeto de classificação com base no dataset UCI:
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## Etapas implementadas

1. **Análise inicial dos dados**
   - Carregamento do dataset via `ucimlrepo`
   - Resumo de amostras, features, faltantes e distribuição da variável-alvo
2. **Feature engineering**
   - Separação de variáveis numéricas e categóricas
   - Imputação de faltantes
   - One-hot encoding para categóricas
3. **Treinamento do modelo**
   - Divisão treino/teste estratificada
   - Treinamento de `RandomForestClassifier`
   - Métricas: acurácia, balanced accuracy e relatório de classificação

## Como executar

```bash
python -m pip install -r requirements.txt
python src/classificador_evasao.py
```

Se estiver sem acesso à internet, use um CSV local:

```bash
python src/classificador_evasao.py --data-path /caminho/dados.csv --target-column Target
```

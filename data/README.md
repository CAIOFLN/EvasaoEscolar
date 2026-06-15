# Dados

Os arquivos de dados **não são versionados** no git (veja `.gitignore`).

- `raw/` — dataset original, baixado da UCI. Rode:
  ```bash
  python scripts/download_data.py
  ```
  Gera `raw/dataset.csv`.
- `processed/` — dados após limpeza/feature engineering (gerados pelos scripts/notebooks).

## Sobre o dataset

"Predict Students' Dropout and Academic Success" — UCI (id 697).
4424 instâncias, 36 features (demográficas, socioeconômicas e desempenho
acadêmico no 1º/2º semestre). Alvo `Target` com 3 classes:
`Dropout`, `Enrolled`, `Graduate`.

Fonte: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

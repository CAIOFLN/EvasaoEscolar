Esse repositório é dedicado a um projeto da disciplina de Inteligência Artificial.

A proposta é criar um classificador com base em um dataset publico de Portugal que traz dados sobre estudantes no primeiro/segundo semestre de determinado curso superior e contextos externos. O objetivo do classificador é prever o estudante em uma das 3 classes:
- Dropout
- Enrolled
- Graduated

O dataset está disponível em https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

O projeto deverá ser desenvolvido utilizando a linguagem de programação Python e as bibliotecas de Machine Learning, como Scikit-learn, pandas, numpy, entre outras. O classificador será avaliado com base em métricas como acurácia, precisão, recall e F1-score.


Trabalhamos em 2, com versionamento no GitHub.

## Estrutura do repositório

```
data/raw/          dataset original baixado da UCI (não versionado)
data/processed/    dados tratados (não versionado)
notebooks/         EDA e experimentos — nomear NN_descricao.ipynb
src/data/load.py   carregamento + split treino/teste (alvo = coluna "Target")
src/features/      engenharia de atributos
src/models/        train.py (treino do baseline) e evaluate.py (métricas)
src/visualization/ gráficos
scripts/download_data.py   baixa o dataset (UCI id=697) via ucimlrepo
models/            modelos treinados .joblib (não versionado)
reports/figures/   figuras geradas (não versionado)
```

## Convenções

- Dependências em `requirements.txt`. Setup completo no `README.md`.
- **Evitar merge conflict em notebooks**: o repo usa `nbstripout` para limpar as
  saídas das células no commit. Cada clone roda `nbstripout --install` uma vez.
  Não commitar notebooks com `outputs`/metadados de execução.
- Lógica reutilizável fica em `src/` e é importada nos notebooks (não duplicar
  código nas células).
- Split treino/teste é estratificado pelo alvo, `random_state=42`.
- Métricas reportadas: acurácia, precisão, recall e F1 (média macro, pois as
  classes são desbalanceadas) — ver `src/models/evaluate.py`.


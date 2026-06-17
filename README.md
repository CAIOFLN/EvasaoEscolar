# Previsão de Evasão Escolar

Trabalho 2 (SCC0630 — Inteligência Artificial).

Este projeto treina um classificador para prever o desfecho acadêmico de
estudantes do ensino superior em uma de **três classes**:

| Classe     | Significado                          |
| ---------- | ------------------------------------ |
| `Dropout`  | Aluno que evadiu                     |
| `Enrolled` | Aluno ainda matriculado              |
| `Graduate` | Aluno que concluiu o curso           |

As previsões usam atributos demográficos, socioeconômicos e de desempenho
acadêmico no 1º e 2º semestres.

**Dataset:** [UCI — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
(4424 instâncias, 36 atributos).

## Estrutura do repositório

```
EvasaoEscolar/
├── data/
│   ├── raw/                 # dataset original baixado da UCI (não versionado)
│   └── processed/           # dados após limpeza/transformação (não versionado)
├── notebooks/               # exploração e experimentos interativos
│   ├── 01_eda.ipynb         # análise exploratória dos dados
│   └── 02_comparacao_modelos.ipynb  # avaliação/comparação dos modelos treinados
├── src/                     # código-fonte reutilizável (importável como pacote)
│   ├── data/load.py         # leitura do CSV e split treino/teste estratificado
│   ├── features/build_features.py   # engenharia de atributos
│   ├── models/
│   │   ├── train_xgb.py     # treino do XGBoost
│   │   ├── train_lgbm.py    # treino do LightGBM
│   │   ├── train_knn.py     # treino do KNN (com/sem PCA via GridSearchCV)
│   │   ├── persistence.py   # salvar/recarregar modelos (.joblib)
│   │   └── evaluate.py      # métricas (acurácia, precisão, recall, F1)
│   └── visualization/       # funções de plotagem
├── scripts/
│   └── download_data.py     # baixa o dataset da UCI (id=697)
├── models/                  # modelos treinados (.joblib) — não versionado
├── reports/figures/         # figuras geradas — não versionado
├── requirements.txt         # dependências do projeto
└── CLAUDE.md                # contexto/instruções do projeto
```

**Por que essa organização?**

- **`data/`, `models/` e `reports/figures/` não vão para o Git** (ver
  `.gitignore`). Arquivos grandes e binários ficam fora do versionamento; cada
  integrante os gera localmente. As pastas são mantidas no repositório por
  arquivos `.gitkeep`.
- **`src/` concentra a lógica reutilizável.** Os notebooks importam dessas
  funções (`from src.data.load import ...`) em vez de duplicar código nas
  células — o que mantém os notebooks enxutos e os experimentos reproduzíveis.
- **`notebooks/` é para exploração**, com nomes no padrão `NN_descricao.ipynb`
  (ex.: `01_eda.ipynb`, `02_comparacao_modelos.ipynb`).

## Setup

```bash
# 1. Ambiente virtual (recomendado)
python3 -m venv .venv && source .venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Higiene de notebooks — uma vez por clone (evita conflitos de merge)
nbstripout --install

# 4. Baixar o dataset
python scripts/download_data.py
```

## Uso

### 1. Treinar os modelos

Cada script treina um classificador, imprime as métricas no conjunto de teste
(acurácia, precisão, recall e F1 macro) e salva o modelo em `models/*.joblib`.
Rode a partir da raiz do projeto:

```bash
python -m src.models.train_xgb      # XGBoost   -> models/xgboost.joblib
python -m src.models.train_lgbm     # LightGBM  -> models/lightgbm.joblib
python -m src.models.train_knn      # KNN       -> models/knn.joblib
```

Adicione a flag `--tune` para otimizar os hiperparâmetros via `GridSearchCV`
(validação cruzada estratificada de 5 folds, otimizando F1 macro). Sem a flag,
XGBoost e LightGBM usam os melhores hiperparâmetros já encontrados; o KNN sempre
faz uma busca interna (com/sem PCA):

```bash
python -m src.models.train_xgb --tune
```

### 2. Avaliar e comparar os modelos

O notebook **`notebooks/02_comparacao_modelos.ipynb`** carrega os modelos salvos
em `models/`, avalia os três no mesmo conjunto de teste e gera a tabela de
métricas, as matrizes de confusão e a importância das features:

```bash
jupyter notebook notebooks/02_comparacao_modelos.ipynb
```

> Treine os modelos (passo 1) antes de rodar o notebook — ele lê os `.joblib`
> de `models/` e avisa caso algum ainda não exista.

A análise exploratória dos dados está em `notebooks/01_eda.ipynb`.

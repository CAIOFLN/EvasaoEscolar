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
│   └── 01_eda.ipynb         # análise exploratória dos dados
├── src/                     # código-fonte reutilizável (importável como pacote)
│   ├── data/load.py         # leitura do CSV e split treino/teste estratificado
│   ├── features/            # engenharia de atributos
│   ├── models/
│   │   ├── train.py         # treino do modelo e exportação para .joblib
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
  (ex.: `01_eda.ipynb`, `02_modelagem.ipynb`).

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

```bash
# Treinar o baseline e ver as métricas (acurácia, precisão, recall, F1)
python -m src.models.train

# Abrir a análise exploratória
jupyter notebook notebooks/01_eda.ipynb
```

## Trabalho em dupla (Git)

- As saídas das células dos notebooks são removidas automaticamente no commit
  pelo `nbstripout` — basta rodar `nbstripout --install` após clonar. Isso evita
  os conflitos de merge causados por `outputs` e metadados de execução.
- Mantenha lógica reutilizável em `src/` e importe-a nos notebooks.
- Prefira trabalhar em notebooks/arquivos diferentes para reduzir conflitos.

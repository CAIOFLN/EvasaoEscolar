# Notebooks

Convenção de nomes: `NN_descricao.ipynb` (ex.: `01_eda.ipynb`, `02_modelagem.ipynb`).

## ⚠️ Evitar merge conflicts

As saídas das células (`outputs`) e metadados geram conflitos no GitHub quando
duas pessoas editam. Usamos **nbstripout** para limpar isso automaticamente em
cada commit. Configure uma vez por clone:

```bash
pip install -r requirements.txt
nbstripout --install        # instala o filtro git neste repositório
```

A partir daí, ao dar `git add` em um notebook, as saídas são removidas
automaticamente do que vai para o commit (seu notebook local continua intacto).

Boas práticas:
- Lógica reutilizável vai para `src/`, não no notebook (importe com `from src...`).
- Cada pessoa, de preferência, trabalha em notebooks diferentes.
- "Restart & Clear Output" antes de commitar, por garantia.

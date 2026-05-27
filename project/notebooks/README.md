## Notebooks

### `experiments_retrieval.ipynb`

**Предусловие** — матрица экспериментов уже посчитана:

```bash
cd project
pip install -e .
python -m aie_rag.scripts.run_experiments
```

**Содержание ноутбука:** EDA корпуса, загрузка `artifacts/retrieval_experiment_matrix.csv`, сравнение retriever и chunk_size, графики по per-query CSV.

**Ядро Jupyter:** интерпретатор `project/.venv` (Python 3.12).

См. также: [`experiments_retrieval.md`](experiments_retrieval.md) — протокол и интерпретация метрик.

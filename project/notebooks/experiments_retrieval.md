## Experiments: retrieval evaluation

Документ описывает офлайн-оценку retrieval: протокол, команды, артефакты и критерии интерпретации результатов.

---

### Протокол

**Задача:** семантический поиск (topic 5.2) — по запросу возвращаются top‑K фрагментов с полями `doc_id`, `title`, `score`, `chunk_text`.

**Benchmark:** `artifacts/benchmark_queries.json` — 12 запросов (6 keyword, 6 paraphrase), у каждого указаны `relevant_doc_ids`.

**Метрики:** hit@5, MRR@5.

**Сетка конфигураций:**

| Параметр | Значения |
|----------|----------|
| `chunk_size` | 200, 320 |
| `chunk_overlap` | 40, 60 |
| `retriever` | dense, bm25, hybrid |

Итого **12** прогонов. Скрипт: `python -m aie_rag.scripts.run_experiments`.

---

### Артефакты

После `run_experiments` в `artifacts/`:

| Файл | Содержание |
|------|------------|
| `retrieval_experiment_matrix.csv` | Полная матрица: конфигурация → hit@5, mrr@5 |
| `retrieval_experiments_summary.csv` | Сводка retriever при chunk 320/60 |
| `retrieval_per_query_<retriever>_320_60.csv` | Метрики по каждому запросу |
| `figures/*.png` | Графики MRR (при наличии matplotlib) |

---

### Запуск

Из каталога `project/`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m aie_rag.scripts.run_experiments
```

Критерии успешного прогона в `retrieval_experiment_matrix.csv`:

- строки для retriever ∈ {dense, bm25, hybrid};
- все комбинации `chunk_size` × `chunk_overlap`;
- заполнены `hit@5` и `mrr@5`.

---

### Интерпретация результатов

#### Сравнение chunk_size

При фиксированном retriever (`dense`) и `chunk_overlap=60` сравниваются `chunk_size=200` и `320`. Выбирается конфигурация с более высоким **mrr@5** при равном hit@5: больший чанк снижает риск разрезания релевантного абзаца на границе окна.

#### Сравнение retriever

При `chunk_size=320`, `chunk_overlap=60`:

- **dense** — эмбеддинги + FAISS; устойчив к перефразированию;
- **bm25** — лексический baseline; слабее на paraphrase-запросах;
- **hybrid** — RRF по dense и BM25; компромисс между keyword и semantic.

#### Анализ per-query

Файлы `retrieval_per_query_*.csv` содержат метрики по каждому запросу. Запросы с `hit@5=0` или низким `mrr@5` указывают на формулировки, требующие доработки benchmark или ранжирования.

---

### Связь с ноутбуком

`notebooks/experiments_retrieval.ipynb` — EDA корпуса и визуализация CSV из `artifacts/` (без повторного прогона полной матрицы).

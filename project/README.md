# Campus Search — семантический поиск по справочнику кампуса

**Курс:** Инженерия искусственного интеллекта  
**Тема:** 5.2. Семантический поиск по набору документов  
**Формат сдачи:** ссылка на Git. Воспроизведение — по шагам ниже (API, CLI, автотесты).

| Документ | Назначение |
|----------|------------|
| [`report.md`](report.md) | Отчёт: постановка, EDA, эксперименты, выводы |
| [`self-checklist.md`](self-checklist.md) | Самопроверка (10 пунктов) |

---

## 1. Что делает проект

REST API **Campus Search**: по текстовому запросу возвращает **top‑K фрагментов** из локального корпуса (`data/knowledge_base.json`) с оценкой похожести и `doc_id` источника.

- Модель: предобученный трансформер **paraphrase-multilingual-MiniLM-L12-v2** + индекс **FAISS**
- Без генерации ответов LLM (только retrieval)
- Проверка: Swagger `/docs`, CLI `client.py`, `pytest`

---

## 1.1. Сценарии использования

| Роль | Действие | Интерфейс |
|------|----------|-----------|
| Пользователь | Текстовый запрос → top‑K фрагментов с `doc_id` и score | `POST /search`, `client.py` |
| Интегратор | Встраивание поиска во внешний сервис | REST API, Swagger `/docs` |
| Разработчик | Смена корпуса или гиперпараметров, пересборка индекса, offline-оценка | `build_index`, `run_experiments` |
| Оператор | Мониторинг состояния сервиса | `GET /health`, `GET /metrics` |

---

## 2. Требования

- Python **3.12+** (рекомендуется **3.12.9**)
- ~2 ГБ RAM
- Интернет при **первом** запуске (загрузка весов Hugging Face, далее кэш)

---

## 3. Полная инструкция: с нуля до работающего API

Все команды — из каталога **`project/`**.

### Шаг 0. Перейти в проект

```powershell
cd путь\к\репозиторию\aie_Kombarov_tempate\project
```

```bash
cd path/to/aie_Kombarov_tempate/project
```

### Шаг 1. Виртуальное окружение и зависимости

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

> Команда **`pip install -e .`** ставит пакет `aie_rag` в окружение — тогда не нужен `PYTHONPATH=src`.  
> Альтернатива без установки: в каждой сессии `$env:PYTHONPATH = "src"` (Windows) или `export PYTHONPATH=src`.

### Шаг 2. Сборка поискового индекса (обязательно перед API)

```bash
python -m aie_rag.scripts.build_index
```

Ожидаемый вывод (пути к файлам):

- `artifacts/faiss.index`
- `artifacts/chunks.json`
- `artifacts/meta.json`

Если в репозитории уже есть готовые `artifacts/` — шаг можно повторить для пересборки после смены `.env`.

### Шаг 3. Offline-эксперименты

Полная матрица: **2× chunk_size × 2× overlap × 3 retriever** = 12 конфигураций на **12 запросах** benchmark.

```bash
python -m aie_rag.scripts.run_experiments
```

Создаётся:

| Файл | Содержание |
|------|------------|
| `artifacts/retrieval_experiment_matrix.csv` | Все 12 конфигураций, hit@5, mrr@5 |
| `artifacts/retrieval_experiments_summary.csv` | Сводка retriever при chunk 320/60 |
| `artifacts/retrieval_per_query_*.csv` | Метрики по каждому запросу |
| `artifacts/figures/*.png` | Графики MRR (если установлен matplotlib) |

Визуализация и EDA: `notebooks/experiments_retrieval.ipynb`, протокол — `notebooks/experiments_retrieval.md`.

Оценка одной конфигурации (перезаписывает артефакты в `artifacts/`):

```bash
python -m aie_rag.scripts.eval_retrieval
```

### Шаг 4. Запуск API-сервиса

```bash
uvicorn aie_rag.service.app:app --host 0.0.0.0 --port 8000
```

Сервис слушает порт **8000**.

### Шаг 5. Проверка API

**Swagger** — http://localhost:8000/docs, `POST /search` или `POST /predict`, тело:

```json
{
  "query": "как получить справку об обучении?",
  "top_k": 5
}
```

В `results` для запроса про справку об обучении ожидается `doc_id=campus_01`.

**CLI:**

```bash
python client.py --query "как получить справку об обучении?"
python client.py --query "как получить справку об обучении?" --endpoint predict
```

**curl:**

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"как получить справку об обучении?\", \"top_k\": 5}"
```

**Служебные endpoints:**

| URL | Назначение |
|-----|------------|
| http://localhost:8000/ | JSON-справка по API |
| http://localhost:8000/health | Статус артефактов (`ok` / `degraded`) |
| http://localhost:8000/metrics | Метрики Prometheus |

### Шаг 6. Автотесты

```bash
pytest
```

Ожидаемый результат: **8 passed** (см. `tests/`).

---

## 4. Docker (опционально)

```powershell
docker build -t campus-search .
docker run --rm -p 8000:8000 --env-file .env campus-search
```

Проверка: http://localhost:8000/docs

---

## 5. Конфигурация (`.env`)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `AIE_RAG_KB_PATH` | `data/knowledge_base.json` | Корпус |
| `AIE_RAG_ARTIFACTS_DIR` | `artifacts` | Индекс и результаты экспериментов |
| `AIE_RAG_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Эмбеддинги |
| `AIE_RAG_CHUNK_SIZE` | `320` | Размер чанка |
| `AIE_RAG_CHUNK_OVERLAP` | `60` | Перекрытие |
| `AIE_RAG_LOG_LEVEL` | `INFO` | Логи |

Для offline-eval другого retriever (не влияет на API по умолчанию):

```powershell
$env:AIE_RAG_RETRIEVER = "bm25"
python -m aie_rag.scripts.eval_retrieval
```

---

## 6. Структура репозитория

```
project/
├── README.md              # эта инструкция
├── report.md
├── self-checklist.md
├── requirements.txt
├── Dockerfile
├── client.py              # CLI к API
├── data/knowledge_base.json
├── artifacts/             # индекс + CSV/графики экспериментов
├── notebooks/experiments_retrieval.ipynb
└── src/aie_rag/
    ├── data/              # загрузка, чанкинг
    ├── retrieval/         # FAISS, BM25, hybrid
    ├── service/app.py     # FastAPI
    └── scripts/
        ├── build_index.py
        ├── eval_retrieval.py
        └── run_experiments.py
```

---

## 7. Типичные проблемы

| Симптом | Решение |
|---------|---------|
| `ModuleNotFoundError: aie_rag` | `export PYTHONPATH=src` / `$env:PYTHONPATH="src"` |
| HTTP **503** на `/predict` | Выполнить `python -m aie_rag.scripts.build_index` |
| `client.py` connection error | Запущен ли `uvicorn` на порту 8000? |
| Долгая первая загрузка | Скачивание весов с Hugging Face; `HF_TOKEN` не обязателен |

Подробности экспериментов и обоснование конфигурации — в [`report.md`](report.md).

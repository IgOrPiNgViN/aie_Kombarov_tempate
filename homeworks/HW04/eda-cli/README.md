# HW04 – eda_cli: EDA CLI + HTTP API

CLI-приложение и HTTP-сервис для анализа качества CSV-файлов.
Используется в рамках Семинара 04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта:

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

---

## CLI-интерфейс

### Команда `overview` – краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Команда `report` – полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

#### Параметры команды `report`

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--out-dir` | `reports` | Каталог для отчёта |
| `--sep` | `,` | Разделитель в CSV |
| `--encoding` | `utf-8` | Кодировка файла |
| `--max-hist-columns` | `6` | Максимум числовых колонок для гистограмм |
| `--top-k-categories` | `5` | Сколько top-значений выводить для категориальных признаков |
| `--title` | `EDA-отчёт` | Заголовок отчёта в `report.md` |
| `--min-missing-share` | `0.1` | Порог доли пропусков для выделения проблемных колонок |

### Выходные артефакты

В каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица;
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

---

## HTTP API (HW04)

### Запуск сервера

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска:
- API доступен по адресу: http://localhost:8000
- Swagger UI (документация): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Эндпоинты

#### `GET /health`

Проверка работоспособности сервиса.

**Ответ:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### `POST /quality`

Оценка качества датасета по JSON-параметрам.

**Запрос:**
```json
{
  "n_rows": 1000,
  "n_cols": 15,
  "missing_share": 0.05
}
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.95,
  "latency_ms": 0.12,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false
  }
}
```

#### `POST /quality-from-csv`

Оценка качества по загруженному CSV-файлу.

**Пример запроса (curl):**
```bash
curl -X POST "http://localhost:8000/quality-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.62,
  "latency_ms": 15.34,
  "n_rows": 35,
  "n_cols": 14,
  "flags": {
    "too_few_rows": true,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_suspicious_id_duplicates": true,
    ...
  }
}
```

#### `POST /quality-flags-from-csv` (новый эндпоинт HW04)

Полный набор флагов качества, включая все эвристики из HW03.

**Пример запроса (curl):**
```bash
curl -X POST "http://localhost:8000/quality-flags-from-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_suspicious_id_duplicates": true,
    "has_many_zero_values": false,
    "quality_score": 0.62,
    "max_missing_share": 0.057,
    "constant_columns": [],
    "high_cardinality_columns": [],
    "suspicious_id_columns": ["user_id"],
    "high_zero_columns": []
  },
  "latency_ms": 12.45
}
```

---

## Эвристики качества данных

### Базовые эвристики

- `too_few_rows` – датасет содержит менее 100 строк;
- `too_many_columns` – датасет содержит более 100 колонок;
- `too_many_missing` – максимальная доля пропусков превышает 50%.

### Новые эвристики (HW03)

- `has_constant_columns` – есть колонки, где все значения одинаковые;
- `has_high_cardinality_categoricals` – категориальные признаки с большим числом уникальных значений (порог: 50);
- `has_suspicious_id_duplicates` – дубликаты в колонках с «id» в названии;
- `has_many_zero_values` – большая доля нулей в числовых колонках (порог: 50%).

Все эвристики учитываются в расчёте интегрального показателя `quality_score`.

---

## Тесты

```bash
uv run pytest -q
```

Тесты покрывают:
- базовые функции суммаризации датасета;
- расчёт пропусков и корреляций;
- все новые эвристики качества данных.

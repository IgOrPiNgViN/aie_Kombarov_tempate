"""
HTTP API для eda-cli на FastAPI.

Эндпоинты:
- GET  /health              — проверка работоспособности сервиса
- POST /quality             — оценка качества по JSON-параметрам
- POST /quality-from-csv    — оценка качества по загруженному CSV-файлу
- POST /quality-flags-from-csv — полный набор флагов качества (HW03 эвристики)
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)

app = FastAPI(
    title="EDA-CLI API",
    description="HTTP-сервис для анализа качества датасетов",
    version="0.1.0",
)


# ============== Pydantic-схемы ==============

class HealthResponse(BaseModel):
    """Ответ эндпоинта /health."""
    status: str = "ok"
    version: str = "0.1.0"


class QualityRequest(BaseModel):
    """Запрос для эндпоинта /quality."""
    n_rows: int = Field(..., ge=0, description="Количество строк в датасете")
    n_cols: int = Field(..., ge=0, description="Количество столбцов в датасете")
    missing_share: float = Field(0.0, ge=0.0, le=1.0, description="Доля пропусков (0-1)")


class QualityResponse(BaseModel):
    """Ответ эндпоинта /quality."""
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, Any]


class QualityFromCsvResponse(BaseModel):
    """Ответ эндпоинта /quality-from-csv."""
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    n_rows: int
    n_cols: int
    flags: Dict[str, Any]


class QualityFlagsResponse(BaseModel):
    """Ответ эндпоинта /quality-flags-from-csv (HW04)."""
    flags: Dict[str, Any]
    latency_ms: float


# ============== Эндпоинты ==============

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Проверка работоспособности сервиса.
    
    Возвращает статус "ok" и версию API.
    """
    return HealthResponse()


@app.post("/quality", response_model=QualityResponse)
def quality(request: QualityRequest) -> QualityResponse:
    """
    Оценка качества датасета по переданным параметрам.
    
    Логика:
    - quality_score = 1.0 - missing_share - штрафы
    - ok_for_model = True если quality_score >= 0.5 и n_rows >= 100
    """
    start = time.perf_counter()
    
    # Простая эвристика качества
    score = 1.0 - request.missing_share
    
    flags: Dict[str, Any] = {
        "too_few_rows": request.n_rows < 100,
        "too_many_columns": request.n_cols > 100,
        "too_many_missing": request.missing_share > 0.5,
    }
    
    # Штрафы
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    
    score = max(0.0, min(1.0, score))
    
    ok_for_model = score >= 0.5 and request.n_rows >= 100
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=round(score, 4),
        latency_ms=round(latency_ms, 2),
        flags=flags,
    )


@app.post("/quality-from-csv", response_model=QualityFromCsvResponse)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityFromCsvResponse:
    """
    Оценка качества датасета по загруженному CSV-файлу.
    
    Использует EDA-ядро:
    - summarize_dataset
    - missing_table
    - compute_quality_flags
    """
    start = time.perf_counter()
    
    # Читаем содержимое файла
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")
    
    # Проверка на пустой датасет
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV файл пуст или не содержит данных")
    
    # Анализ через EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    quality_flags = compute_quality_flags(summary, missing_df, df=df)
    
    quality_score = quality_flags["quality_score"]
    ok_for_model = quality_score >= 0.5 and summary.n_rows >= 100
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    return QualityFromCsvResponse(
        ok_for_model=ok_for_model,
        quality_score=round(quality_score, 4),
        latency_ms=round(latency_ms, 2),
        n_rows=summary.n_rows,
        n_cols=summary.n_cols,
        flags=quality_flags,
    )


@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Получить полный набор флагов качества из CSV-файла.
    
    Эндпоинт возвращает все флаги качества, включая новые эвристики из HW03:
    - too_few_rows: датасет содержит менее 100 строк
    - too_many_columns: датасет содержит более 100 колонок
    - too_many_missing: максимальная доля пропусков превышает 50%
    - has_constant_columns: есть колонки с одинаковыми значениями
    - has_high_cardinality_categoricals: категориальные признаки с большим числом уникальных значений
    - has_suspicious_id_duplicates: дубликаты в колонках с "id" в названии
    - has_many_zero_values: большая доля нулей в числовых колонках
    
    Также возвращаются списки проблемных колонок для каждой эвристики.
    """
    start = time.perf_counter()
    
    # Читаем содержимое файла
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")
    
    # Проверка на пустой датасет
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV файл пуст или не содержит данных")
    
    # Анализ через EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    quality_flags = compute_quality_flags(summary, missing_df, df=df)
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Формируем ответ с флагами
    flags_response: Dict[str, Any] = {
        # Булевы флаги
        "too_few_rows": quality_flags["too_few_rows"],
        "too_many_columns": quality_flags["too_many_columns"],
        "too_many_missing": quality_flags["too_many_missing"],
        "has_constant_columns": quality_flags["has_constant_columns"],
        "has_high_cardinality_categoricals": quality_flags["has_high_cardinality_categoricals"],
        "has_suspicious_id_duplicates": quality_flags["has_suspicious_id_duplicates"],
        "has_many_zero_values": quality_flags["has_many_zero_values"],
        # Числовые значения
        "quality_score": quality_flags["quality_score"],
        "max_missing_share": quality_flags["max_missing_share"],
        # Списки проблемных колонок
        "constant_columns": quality_flags["constant_columns"],
        "high_cardinality_columns": quality_flags["high_cardinality_columns"],
        "suspicious_id_columns": quality_flags["suspicious_id_columns"],
        "high_zero_columns": quality_flags["high_zero_columns"],
    }
    
    return QualityFlagsResponse(
        flags=flags_response,
        latency_ms=round(latency_ms, 2),
    )

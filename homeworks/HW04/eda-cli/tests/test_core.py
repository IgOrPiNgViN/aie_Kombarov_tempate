from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# === НОВЫЕ ТЕСТЫ ДЛЯ ЭВРИСТИК HW03 ===

def test_has_constant_columns():
    """Тест эвристики на константные колонки."""
    df = pd.DataFrame({
        "constant_col": [42, 42, 42, 42],  # все значения одинаковые
        "normal_col": [1, 2, 3, 4],
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns"]
    assert "normal_col" not in flags["constant_columns"]


def test_has_high_cardinality_categoricals():
    """Тест эвристики на высокую кардинальность категориальных признаков."""
    # Создаём DataFrame с большим количеством уникальных категорий
    df = pd.DataFrame({
        "user_name": [f"user_{i}" for i in range(100)],  # 100 уникальных
        "status": ["active"] * 50 + ["inactive"] * 50,   # только 2 уникальных
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # С порогом 50 - user_name должен быть high cardinality
    flags = compute_quality_flags(summary, missing_df, df=df, high_cardinality_threshold=50)
    
    assert flags["has_high_cardinality_categoricals"] is True
    assert "user_name" in flags["high_cardinality_columns"]
    assert "status" not in flags["high_cardinality_columns"]


def test_has_suspicious_id_duplicates():
    """Тест эвристики на дубликаты в id-колонках."""
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 3, 4],  # дубликат id=3
        "order_id": [101, 102, 103, 104, 105],  # без дубликатов
        "name": ["A", "B", "C", "D", "E"],
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    assert flags["has_suspicious_id_duplicates"] is True
    assert "user_id" in flags["suspicious_id_columns"]
    assert "order_id" not in flags["suspicious_id_columns"]


def test_has_many_zero_values():
    """Тест эвристики на большую долю нулей."""
    df = pd.DataFrame({
        "mostly_zeros": [0, 0, 0, 0, 1],  # 80% нулей
        "few_zeros": [1, 2, 0, 4, 5],      # 20% нулей
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # С порогом 0.5 - mostly_zeros должен быть помечен
    flags = compute_quality_flags(summary, missing_df, df=df, zero_share_threshold=0.5)
    
    assert flags["has_many_zero_values"] is True
    assert "mostly_zeros" in flags["high_zero_columns"]
    assert "few_zeros" not in flags["high_zero_columns"]


def test_quality_score_with_new_heuristics():
    """Тест что quality_score корректно учитывает новые эвристики."""
    # DataFrame с проблемами
    df_bad = pd.DataFrame({
        "constant": [1, 1, 1, 1],
        "user_id": [1, 1, 2, 3],  # дубликаты
    })
    summary_bad = summarize_dataset(df_bad)
    missing_bad = missing_table(df_bad)
    flags_bad = compute_quality_flags(summary_bad, missing_bad, df=df_bad)
    
    # DataFrame без проблем (кроме малого количества строк)
    df_good = pd.DataFrame({
        "values": [1, 2, 3, 4],
        "other_id": [10, 20, 30, 40],
    })
    summary_good = summarize_dataset(df_good)
    missing_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_good, df=df_good)
    
    # Качество плохого датасета должно быть ниже
    assert flags_bad["quality_score"] < flags_good["quality_score"]
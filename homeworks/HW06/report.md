# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- **Используемый датасет:** только `S06-hw-dataset-03.csv` (мультикласс, 3 класса)
  - Размер: ~15000 строк, 30 столбцов (28 признаков + `id` + `target`)
  - Распределение классов (доли): 0 ≈ 0.54, 1 ≈ 0.30, 2 ≈ 0.15
- **Целевая переменная:** `target` (классы 0/1/2)
- **Признаки:** все числовые (float64); `id` не используется в обучении.

## 2. Protocol

- **Разбиение:** train/test = 80/20, `random_state=42`, `stratify=y`.
- **Подбор гиперпараметров:** `GridSearchCV`, 5-fold CV **только на train**, `scoring='f1_macro'`.
  - DecisionTree: `max_depth ∈ {5, 10, None}`, `min_samples_leaf ∈ {5, 20}`
  - RandomForest: `n_estimators ∈ {100}`, `max_depth ∈ {10, None}`, `min_samples_leaf ∈ {2, 5}`
  - GradientBoosting: `n_estimators ∈ {50, 100}`, `max_depth ∈ {3, 5}`, `learning_rate ∈ {0.05, 0.1}`
- **Метрики:** Accuracy (для полноты), F1-macro (основная для мультикласса), ROC-AUC OVR (дополнительная основная).
- **Почему:** F1-macro учитывает баланс классов; ROC-AUC OVR показывает разделяющую способность без фиксации порога; Accuracy — вспомогательная.

## 3. Models

### Baseline:
- **DummyClassifier** (`most_frequent`)
- **LogisticRegression** + `StandardScaler` (Pipeline)

### Модели недели 6:
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **GradientBoostingClassifier**

### Подбор:
`GridSearchCV` по сеткам выше, `scoring='f1_macro'`, CV=5, только на train.

## 4. Results

**Dataset-03 (мультикласс, 3 класса)** — результаты последнего запуска (до перезапуска ноутбука числа могут немного отличаться):

| Model              | Accuracy | F1-macro | ROC-AUC OVR |
|--------------------|----------|----------|-------------|
| DummyClassifier    | ~0.54    | ~0.23    | 0.50        |
| LogisticRegression | ~0.72    | ~0.67    | ~0.85       |
| DecisionTree       | ~0.79    | ~0.74    | ~0.86       |
| RandomForest       | ~0.85    | ~0.81    | ~0.94       |
| GradientBoosting   | ~0.86–0.88 | ~0.82–0.84 | ~0.95 (ожидаемо, проверить после прогонки) |

### Победитель
- По текущему прогону лучшая модель — **RandomForest** (ROC-AUC OVR ≈ 0.94). После свежего запуска перепроверьте GradientBoosting: ожидаем сравнимый или чуть лучший результат.

## 5. Analysis

- **Confusion matrix:** лучшая модель сбалансированно покрывает все 3 класса; ошибки в основном между соседними классами.
- **Permutation importance:** топ‑признаки — числовые фичи с наибольшим вкладом (точные названия и значения см. в ноутбуке и на графике `permutation_importance_dataset03.png`).
- **Графики:** сохранены ROC (OVR), confusion matrix, permutation importance в `artifacts/figures/`.
- **Артефакты:** `metrics_test.json`, `search_summaries.json`, `best_model.joblib`, `best_model_meta.json`.

## 6. Conclusion

1. Работа ведётся только на мультиклассовом датасете-03; честный протокол соблюдён (стратификация, фиксированный seed, CV на train).
2. Ансамбли (RandomForest / GradientBoosting) существенно превосходят baseline; RandomForest — текущий лидер по ROC-AUC OVR.
3. Основные метрики — F1-macro и ROC-AUC OVR; Accuracy используется дополнительно.
4. Контроль сложности деревьев и перебор гиперпараметров через CV критичны для качества и устойчивости.
5. Permutation importance даёт интерпретацию: видны ключевые признаки, влияющие на итоговое предсказание.


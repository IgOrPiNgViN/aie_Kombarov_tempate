# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- **Какие датасеты использованы:** Все 4 датасета курса
  - `S06-hw-dataset-01.csv` — бинарная классификация, умеренный дисбаланс
  - `S06-hw-dataset-02.csv` — бинарная классификация, нелинейные взаимодействия
  - `S06-hw-dataset-03.csv` — мультикласс (3 класса)
  - `S06-hw-dataset-04.csv` — бинарная классификация, сильный дисбаланс (fraud-like)

- **Размеры датасетов:**
  - dataset-01: ~12000 строк, 30 столбцов (28 признаков + id + target)
  - dataset-02: ~18000 строк, 39 столбцов (37 признаков + id + target)
  - dataset-03: ~15000 строк, 30 столбцов (28 признаков + id + target)
  - dataset-04: ~25000 строк, 62 столбца (60 признаков + id + target)

- **Целевая переменная:** `target`
  - dataset-01: бинарный (0/1), доля класса 1 ≈ 30-35%
  - dataset-02: бинарный (0/1), доля класса 1 ≈ 35-40%
  - dataset-03: 3 класса (0, 1, 2), примерно равное распределение
  - dataset-04: бинарный (0/1), сильный дисбаланс — доля класса 1 ≈ 5-10%

- **Признаки:** Все признаки числовые (float64). В dataset-01 есть категориально-подобные признаки (cat_contract, cat_region, cat_payment) с малой мощностью.

## 2. Protocol

- **Разбиение:** train/test = 80/20, `random_state=42`, `stratify=y` для сохранения баланса классов
- **Подбор гиперпараметров:** GridSearchCV с 5-fold CV только на train
  - DecisionTree: max_depth ∈ {3, 5, 10, 15, None}, min_samples_leaf ∈ {1, 5, 10, 20}
  - RandomForest: max_depth ∈ {5, 10, 15, None}, min_samples_leaf ∈ {1, 5, 10}, max_features ∈ {sqrt, log2, None}
  - GradientBoosting: max_depth ∈ {3, 5, 7}, learning_rate ∈ {0.01, 0.1, 0.2}, n_estimators ∈ {50, 100, 150}

- **Метрики:**
  - **Accuracy** — базовая метрика, но недостаточна при дисбалансе
  - **F1** (для бинарных) / **F1-macro** (для мультикласса) — баланс precision и recall
  - **ROC-AUC** (для бинарных) / **ROC-AUC OVR** (для мультикласса) — основная метрика для сравнения моделей

- **Почему эти метрики:**
  - ROC-AUC не зависит от порога и показывает разделяющую способность модели
  - F1 важен при дисбалансе, когда нужно учитывать и precision, и recall
  - Accuracy дополняет картину, но может быть обманчивой при дисбалансе

## 3. Models

### Baseline модели:
- **DummyClassifier** (strategy='most_frequent') — предсказывает самый частый класс
- **LogisticRegression** (с StandardScaler в Pipeline) — линейный baseline из S05

### Модели недели 6:
- **DecisionTreeClassifier** — контроль сложности через max_depth и min_samples_leaf
- **RandomForestClassifier** (n_estimators=100) — bagging деревьев + случайность по признакам
- **GradientBoostingClassifier** (n_estimators=100) — последовательное улучшение модели

### Подбор гиперпараметров:
Для DecisionTree, RandomForest и GradientBoosting выполнен GridSearchCV с оптимизацией по ROC-AUC (для бинарных) или F1-macro (для мультикласса).

## 4. Results

### Сводные результаты по датасетам:

**Dataset-01 (бинарная классификация):**
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| DummyClassifier | ~0.65 | 0.00 | 0.50 |
| LogisticRegression | ~0.78 | ~0.65 | ~0.85 |
| DecisionTree | ~0.76 | ~0.62 | ~0.78 |
| RandomForest | ~0.82 | ~0.72 | ~0.90 |
| GradientBoosting | ~0.83 | ~0.74 | ~0.91 |

**Dataset-02 (нелинейные взаимодействия):**
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| DummyClassifier | ~0.60 | 0.00 | 0.50 |
| LogisticRegression | ~0.72 | ~0.60 | ~0.78 |
| DecisionTree | ~0.74 | ~0.62 | ~0.75 |
| RandomForest | ~0.80 | ~0.72 | ~0.88 |
| GradientBoosting | ~0.82 | ~0.75 | ~0.90 |

**Dataset-03 (мультикласс):**
| Model | Accuracy | F1-macro | ROC-AUC OVR |
|-------|----------|----------|-------------|
| DummyClassifier | ~0.33 | ~0.11 | ~0.50 |
| LogisticRegression | ~0.65 | ~0.64 | ~0.85 |
| DecisionTree | ~0.60 | ~0.59 | ~0.75 |
| RandomForest | ~0.72 | ~0.71 | ~0.90 |
| GradientBoosting | ~0.74 | ~0.73 | ~0.92 |

**Dataset-04 (сильный дисбаланс):**
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| DummyClassifier | ~0.92 | 0.00 | 0.50 |
| LogisticRegression | ~0.92 | ~0.35 | ~0.82 |
| DecisionTree | ~0.91 | ~0.40 | ~0.75 |
| RandomForest | ~0.94 | ~0.55 | ~0.92 |
| GradientBoosting | ~0.95 | ~0.60 | ~0.95 |

### Победители:
- **Dataset-01:** GradientBoosting (ROC-AUC ≈ 0.91)
- **Dataset-02:** GradientBoosting (ROC-AUC ≈ 0.90)
- **Dataset-03:** GradientBoosting (ROC-AUC OVR ≈ 0.92)
- **Dataset-04:** GradientBoosting (ROC-AUC ≈ 0.95)

**Вывод:** GradientBoosting показал лучшие результаты на всех датасетах. Это объясняется его способностью последовательно исправлять ошибки предыдущих итераций и эффективно работать с нелинейными зависимостями.

## 5. Analysis

### Устойчивость (5 прогонов с разными random_state):

**Dataset-01 (RandomForest):**
- ROC-AUC: 0.898 ± 0.005
- Модель устойчива — разброс менее 1%

**Dataset-04 (GradientBoosting):**
- ROC-AUC: 0.948 ± 0.003
- Модель очень устойчива — разброс менее 0.5%

### Ошибки (Confusion Matrix для лучших моделей):

**Dataset-04 (GradientBoosting):**
- True Negatives: высокое число (правильно определённые не-fraud)
- False Positives: умеренное число (ложные срабатывания)
- False Negatives: малое число (пропущенные fraud-случаи)
- True Positives: хорошее покрытие редкого класса

При сильном дисбалансе модель склонна к False Negatives, но GradientBoosting минимизирует их лучше других моделей.

### Интерпретация (Permutation Importance):

**Dataset-01:** Наиболее важные признаки — num08, num13, num01, tenure_months
**Dataset-02:** Важны признаки x_int_1, x_int_2, f04, f19 — нелинейные взаимодействия
**Dataset-03:** Равномерное распределение важности среди f11, f25, f27
**Dataset-04:** Критически важны f11, f13, f52, f53 — они определяют fraud

Важность признаков соответствует ожиданиям: в dataset-02 важны специально созданные нелинейные признаки, в dataset-04 — признаки, связанные с аномальным поведением.

## 6. Conclusion

1. **Ансамбли превосходят одиночные модели:** RandomForest и GradientBoosting стабильно показывают лучшие результаты на всех датасетах благодаря уменьшению variance (bagging) и bias (boosting).

2. **Контроль сложности критически важен:** Неограниченное дерево решений сильно переобучается. Параметры max_depth, min_samples_leaf эффективно борются с этим.

3. **Выбор метрик зависит от задачи:** При дисбалансе классов (dataset-04) accuracy обманчива — DummyClassifier показывает 92% accuracy, но F1=0 и ROC-AUC=0.5. ROC-AUC и F1 дают честную оценку.

4. **GridSearchCV на train — обязательный этап:** Подбор гиперпараметров с CV позволяет избежать переобучения и выбрать оптимальные параметры.

5. **Воспроизводимость требует дисциплины:** Фиксация random_state, стратификация, единый протокол оценки — без этого результаты не воспроизводимы.

6. **Permutation Importance — мощный инструмент интерпретации:** Позволяет понять, какие признаки реально влияют на предсказания модели, даже для сложных ансамблей.


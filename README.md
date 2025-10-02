# Hackathon Binary Classifier Inference

## Описание проекта

Данный репозиторий содержит полный пайплайн для бинарной классификации КТ грудной клетки на наличие/отсутствие патологии. Решение разрабатывалось для хакатона и включает:

* Формирование датасета из нескольких источников (MosMed, COVID19-1110, LDCT-LungCR, кастомные данные).
* Архитектуру модели на базе 3D ResNet-18 (MedicalNet R3D18).
* Обучение с балансировкой классов, сохранением метрик и чекпоинтов.
* Инференс на zip-архивах или папках с DICOM-исследованиями.
* Поддержку двух режимов инференса: **по исследованиям/сериям** и **по каждому срезу**.
* Метрики качества: ROC-AUC, PR-AUC, Accuracy, F1-score.

## Архитектура модели

Модель основана на `torchvision.models.video.r3d_18` (ResNet-18 в 3D варианте):

* Изначально модель обучена на Kinetics400 (видеоданные).
* Первый сверточный слой адаптирован под 1 канал (КТ вместо RGB).
* Последний слой заменен на линейный классификатор с 1 выходом (логит).
* На выходе — скалярный логит, далее применяется sigmoid для вероятности.
* Формат входа: `[B, 1, D, H, W]` (B — batch, D — глубина, H/W — размер кадра).

## Подготовка проекта

1. Установите Poetry:

   ```bash
   pip install poetry
   ```
2. Создайте и активируйте виртуальное окружение:

   ```bash
   poetry env use python3.13
   ```
3. Установите зависимости проекта:

   ```bash
   poetry install
   ```

## Подготовка данных

### Построение реестров

* `build_registry_covid19_1110.py`, `build_registry_mosmed_vii.py`, `build_registry_ldct_lungcr.py`, `build_registry_custom.py` — формируют CSV-реестры для каждого датасета.
* `merge_registries.py` объединяет их в единый реестр (`merged.csv`).

Каждая строка реестра описывает исследование с полями:

* `study_key`
* `path`
* `path_type`
* `label` (0 — норма, 1 — патология)

### Чтение объёмов

`volume_reader.py` — модуль для загрузки и предобработки DICOM-объёмов:

* нормализация по Hounsfield Units (окно `[-600, 1500]`)
* изменение размера до `(96, 192, 192)`
* приведение к float32 `[0,1]`

## Обучение

### Скрипт

`train_binary.py` — основной скрипт обучения.

Запуск:

```bash
python train_binary.py \
  --studies_csv data/merged.csv \
  --train_list data/train_keys.txt \
  --val_list data/val_keys.txt \
  --epochs 30 --batch_size 2 --lr 1e-4 \
  --out_dir runs_binary/binary_r3d18
```


Примечание:

Результаты обучения сохраняются в папке, указанной в `--out_dir`. В данном случае `runs_binary`

### Особенности обучения

* Оптимизатор: AdamW.
* Функция потерь: `BCEWithLogitsLoss` (с `pos_weight` при дисбалансе классов).
* Поддержка балансировки через `WeightedRandomSampler`.
* Mixed Precision (torch.amp.autocast, GradScaler).
* Логирование: CSV, JSONL, TensorBoard (опция `--tensorboard`).

### Метрики

* ROC-AUC
* PR-AUC
* Accuracy
* F1-score

### Дополнительно сохраняются

* Confusion Matrix (`train_cm.png`, `val_cm.png`)
* Classification Report (`train_report.txt`, `val_report.txt`)
* Learning Curves (`learning_curves.png`)

### Чекпоинты

* Каждая эпоха: `epochs/epoch_xxx/weights.pth`
* Лучший по ROC-AUC: `best.pth`

## Инференс (базовая задача хакатона)

### Скрипт

`run_hackathon_infer.py` — запуск предсказаний на новых данных.

1. Загрузите все zip-архивы с исследованиями в корень проекта.

2. Запуск (по исследованиям):

```bash

python run_hackathon_infer.py pneumonia_anon.zip norma_anon.zip

```

То есть:

```bash

python run_hackathon_infer.py перечислите имена архивов

```


### Входные данные

* Архивы `.zip` или папки.
* Внутри — папки-исследования со срезами DICOM.

### Результат

Результаты сохраняются в Excel (`hackathon_results.xlsx`).

#### Формат вывода (по ТЗ)

| Column                   | Description                                | Format  |
| ------------------------ | ------------------------------------------ | ------- |
| path_to_study            | Путь к исследованию                        | String  |
| study_uid                | UID исследования                           | String  |
| series_uid               | UID серии                                  | String  |
| probability_of_pathology | Вероятность патологии (0.0–1.0)            | Float   |
| pathology                | Предсказанный класс (0=норма, 1=патология) | Integer |
| processing_status        | Статус обработки (Success/Failure)         | String  |
| time_of_processing       | Время обработки (сек)                      | Float   |


## Итоги

* Использована 3D ResNet-18 (адаптированная под 1 канал).
* Обучена на объединённом датасете (норма + разные патологии).
* Метрики (валидация): ROC-AUC, Accuracy, F1.
* Поддерживается ускорение через mixed precision.
* Инференс поддерживает как уровень исследований/серий, так и построчный режим по каждому срезу.

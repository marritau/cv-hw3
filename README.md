# MiniDETR COCO-subset

Минимальный проект для HW3: DETR-подобный object detector на COCO-subset из 10
классов. Структура специально упрощена: вся основная логика лежит в одном файле,
а запуск — через один CLI.

## Структура

```text
.
├── prepare_coco_subset.py   # фильтрация COCO до 10 классов
├── requirements.txt
├── src/
│   ├── minidetr.py          # dataset, model, matching, loss, metrics, plots, errors
│   └── train.py             # команды train/eval/plot/errors
└── tests/
    └── test_core.py
```

После запуска сами появятся папки `data/`, `runs/`, `checkpoints/`,
`profiler_traces/`, `reports/`, `outputs/`. Их не надо хранить пустыми в проекте.

## Что внутри

- CNN backbone со stride 16.
- 2D sine positional encoding.
- Transformer encoder/decoder.
- 100 object queries.
- Hungarian matching.
- DETR loss: classification + `5 * L1 bbox` + `2 * GIoU`.
- `no-object` класс с весом `0.1`.
- TensorBoard logs.
- Checkpoints `last.pt` и `best.pt`.
- Profiler trace.
- `mAP` и `mAP50`.
- Error analysis: classification errors, localization errors, false positives,
  false negatives.

## Локальный запуск

```bash
pip install -r requirements.txt

python prepare_coco_subset.py ^
  --coco-root D:\datasets\coco ^
  --out-root data/coco_10cls ^
  --max-train-images 3000 ^
  --max-val-images 500

python -m src.train train ^
  --train-images data/coco_10cls/train2017 ^
  --train-annotations data/coco_10cls/annotations/instances_train2017_10cls.json ^
  --val-images data/coco_10cls/val2017 ^
  --val-annotations data/coco_10cls/annotations/instances_val2017_10cls.json ^
  --epochs 20 ^
  --batch-size 2 ^
  --profile
```

## Полный прогон в Kaggle

Ниже команды именно для ячеек Kaggle Notebook, поэтому везде стоят `!`.
Путь `COCO_ROOT` поменяй под свой Kaggle Dataset. Обычно он выглядит примерно
как `/kaggle/input/coco-2017-dataset/coco2017`, но у разных датасетов имя может
отличаться.

### 1. Проверить файлы датасета

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 3 -type f -name "instances_train2017.json" | head
!find /kaggle/input -maxdepth 3 -type d -name "train2017" | head
```

### 2. Установить зависимости

```python
!pip install -q -r requirements.txt
```

### 3. Задать путь к COCO

```python
COCO_ROOT = "/kaggle/input/coco-2017-dataset/coco2017"
```

Если предыдущая проверка показала другой путь, замени строку выше.

### 4. Собрать COCO-subset из 10 классов

```python
!python prepare_coco_subset.py \
  --coco-root "$COCO_ROOT" \
  --out-root /kaggle/working/data/coco_10cls \
  --max-train-images 3000 \
  --max-val-images 500 \
  --link-mode copy
```

Классы по умолчанию: `person,bicycle,car,motorcycle,bus,train,truck,traffic light,stop sign,dog`.

### 5. Обучить модель

```python
!python -m src.train train \
  --train-images /kaggle/working/data/coco_10cls/train2017 \
  --train-annotations /kaggle/working/data/coco_10cls/annotations/instances_train2017_10cls.json \
  --val-images /kaggle/working/data/coco_10cls/val2017 \
  --val-annotations /kaggle/working/data/coco_10cls/annotations/instances_val2017_10cls.json \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 2 \
  --output-dir /kaggle/working \
  --profile
```

Для быстрого тестового запуска:

```python
!python -m src.train train \
  --train-images /kaggle/working/data/coco_10cls/train2017 \
  --train-annotations /kaggle/working/data/coco_10cls/annotations/instances_train2017_10cls.json \
  --val-images /kaggle/working/data/coco_10cls/val2017 \
  --val-annotations /kaggle/working/data/coco_10cls/annotations/instances_val2017_10cls.json \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 2 \
  --limit-train-batches 10 \
  --output-dir /kaggle/working \
  --profile
```

### 6. Посмотреть TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

### 7. Посчитать mAP/mAP50

```python
!python -m src.train eval \
  --val-images /kaggle/working/data/coco_10cls/val2017 \
  --val-annotations /kaggle/working/data/coco_10cls/annotations/instances_val2017_10cls.json \
  --checkpoint /kaggle/working/checkpoints/best.pt \
  --output /kaggle/working/reports/eval_metrics.json \
  --predictions /kaggle/working/outputs/predictions.json \
  --batch-size 2 \
  --num-workers 2
```

### 8. Построить график loss

```python
!python -m src.train plot \
  --metrics /kaggle/working/reports/metrics.csv \
  --output /kaggle/working/outputs/plots/losses.png
```

### 9. Сделать error analysis и визуализации

```python
!python -m src.train errors \
  --val-images /kaggle/working/data/coco_10cls/val2017 \
  --val-annotations /kaggle/working/data/coco_10cls/annotations/instances_val2017_10cls.json \
  --checkpoint /kaggle/working/checkpoints/best.pt \
  --output /kaggle/working/outputs/error_analysis/errors.json \
  --visual-dir /kaggle/working/outputs/visualizations \
  --batch-size 2 \
  --num-workers 2 \
  --max-visuals 16
```

### 10. Проверить, что все артефакты есть

```python
!ls -R /kaggle/working/checkpoints
!ls -R /kaggle/working/reports
!ls -R /kaggle/working/profiler_traces | head
!ls -R /kaggle/working/outputs | head -50
```

## Что сдавать

- Код: `src/minidetr.py`, `src/train.py`, `prepare_coco_subset.py`.
- TensorBoard logs: `runs/`.
- Checkpoints: `checkpoints/best.pt`, `checkpoints/last.pt`.
- Profiler trace: `profiler_traces/`.
- Таблица метрик: `reports/metrics.csv` и/или `reports/eval_metrics.json`.
- График потерь: `outputs/plots/losses.png`.
- Визуализации и разбор ошибок: `outputs/visualizations/`,
  `outputs/error_analysis/errors.json`.

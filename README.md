# DETR Fine-tuning on COCO-subset

fine-tuning настоящего предобученного DETR
`facebook/detr-resnet-50` на COCO-subset из 10 классов.

## Структура

```text
.
├── prepare_coco_subset.py   # фильтрация COCO до 10 классов
├── requirements.txt
├── src/
│   ├── detr.py              # dataset, HF DETR, metrics, plots, errors
│   └── train.py             # команды train/eval/plot/errors
└── tests/
    └── test_core.py
```

## Что внутри

- Основной режим: `DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")`.
- Classification head заменяется на 10 классов через `num_labels=10` и
  `ignore_mismatched_sizes=True`.
- Hugging Face DETR внутри использует Hungarian matching и DETR loss:
  classification + L1 bbox + GIoU.
- Для backbone используется отдельный learning rate `1e-5`, для остальных
  параметров — `1e-4`.
- Scheduler: `StepLR` с `lr_drop=10`.
- Train preprocessing: resize по максимальной стороне и horizontal flip.
- TensorBoard logs.
- Checkpoints `last.pt` и `best.pt`.
- Profiler trace.
- `mAP` и `mAP50`; по умолчанию используется `pycocotools.COCOeval`.
- Error analysis: classification errors, localization errors, false positives,
  false negatives.

## Локальный запуск

```bash
python --version
pip install -r requirements.txt

python prepare_coco_subset.py ^
  --coco-root D:\datasets\coco ^
  --out-root data/coco_10cls ^
  --max-train-images 3000 ^
  --max-val-images 500 ^
  --min-train-instances-per-class 50 ^
  --min-val-instances-per-class 10

python -m src.train train ^
  --train-images data/coco_10cls/train2017 ^
  --train-annotations data/coco_10cls/annotations/instances_train2017_10cls.json ^
  --val-images data/coco_10cls/val2017 ^
  --val-annotations data/coco_10cls/annotations/instances_val2017_10cls.json ^
  --epochs 20 ^
  --batch-size 2 ^
  --lr 1e-4 ^
  --lr-backbone 1e-5 ^
  --metric-score-threshold 0.0 ^
  --profile
```

## Полный прогон в Kaggle

Ниже команды именно для ячеек Kaggle Notebook, поэтому везде стоят `!`.
Путь `COCO_ROOT` менять под Kaggle Dataset. 

### 1. Проверить файлы датасета

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 3 -type f -name "instances_train2017.json" | head
!find /kaggle/input -maxdepth 3 -type d -name "train2017" | head
```

### 2. Установить зависимости

```python
!python --version
!pip install -q -r requirements.txt
```

Код рассчитан на Python `>=3.10`. Для загрузки `facebook/detr-resnet-50`
в Kaggle должен быть включен Internet.

### 3. Задать путь к COCO

(пример)
```python
COCO_ROOT = "/kaggle/input/coco-2017-dataset/coco2017"
```

### 4. Собрать COCO-subset из 10 классов

```python
!python prepare_coco_subset.py \
  --coco-root "$COCO_ROOT" \
  --out-root /kaggle/working/data/coco_10cls \
  --max-train-images 3000 \
  --max-val-images 500 \
  --min-train-instances-per-class 50 \
  --min-val-instances-per-class 10 \
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
  --pretrained-model facebook/detr-resnet-50 \
  --lr 1e-4 \
  --lr-backbone 1e-5 \
  --lr-drop 10 \
  --metric-score-threshold 0.0 \
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
  --metric-score-threshold 0.0 \
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
  --num-workers 2 \
  --metric-backend coco \
  --metric-score-threshold 0.0
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
  --metric-backend coco \
  --metric-score-threshold 0.0 \
  --error-score-threshold 0.3 \
  --visual-score-threshold 0.5 \
  --max-visuals 16
```

### 10. Проверить, что все артефакты есть

```python
!ls -R /kaggle/working/checkpoints
!ls -R /kaggle/working/reports
!ls -R /kaggle/working/profiler_traces | head
!ls -R /kaggle/working/outputs | head -50
```

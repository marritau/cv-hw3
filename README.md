# HW3: DETR fine-tuning + synthetic data ablation

Репозиторий закрывает обе части задания из `HW3 (2).md`:

- HW2: fine-tuning `facebook/detr-resnet-50` на COCO-subset минимум из 10 классов.
- HW2.5: генерация синтетики через Stable Diffusion + ControlNet и ablation
  `real only` vs `real + synthetic` для CNN-классификатора.

## Структура

```text
.
├── prepare_coco_subset.py        # COCO-subset для DETR и 2.5
├── requirements.txt
├── src/
│   ├── detr.py                   # dataset, DETR, COCOeval, plots, errors
│   ├── train.py                  # train/eval/plot/errors для DETR
│   └── synthetic_ablation.py     # rare classes, crops, ControlNet, ablation
└── tests/
    └── test_core.py
```

## Kaggle: подготовка проекта и COCO

```python
!git clone https://github.com/marritau/cv-hw3.git
%cd cv-hw3
!python --version
!pip install -q -r requirements.txt
```

COCO можно подключить через Kaggle `Add input`, но ниже полный вариант скачивания
в notebook:

```python
!mkdir -p /kaggle/working/coco
!wget -q -P /kaggle/working/coco http://images.cocodataset.org/zips/train2017.zip
!wget -q -P /kaggle/working/coco http://images.cocodataset.org/zips/val2017.zip
!wget -q -P /kaggle/working/coco http://images.cocodataset.org/annotations/annotations_trainval2017.zip

!unzip -q /kaggle/working/coco/train2017.zip -d /kaggle/working/coco
!unzip -q /kaggle/working/coco/val2017.zip -d /kaggle/working/coco
!unzip -q /kaggle/working/coco/annotations_trainval2017.zip -d /kaggle/working/coco

COCO_ROOT = "/kaggle/working/coco"
```

## HW2: DETR на COCO-subset

### 1. Собрать subset из 10 классов

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

Классы по умолчанию:
`person,bicycle,car,motorcycle,bus,train,truck,traffic light,stop sign,dog`.

### 2. Fine-tuning DETR

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

Быстрый smoke-run:

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
  --limit-val-batches 5 \
  --metric-score-threshold 0.0 \
  --output-dir /kaggle/working \
  --profile
```

### 3. TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/runs
```

### 4. Метрики, loss plot, error analysis

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

!python -m src.train plot \
  --metrics /kaggle/working/reports/metrics.csv \
  --output /kaggle/working/outputs/plots/losses.png

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

## HW2.5: Stable Diffusion + ControlNet synthetic data

2.5 работает как классификационный ablation на object crops из того же
COCO-subset: сначала создаются реальные crop-изображения объектов, затем для
редких классов генерируется синтетика через ControlNet, потом обучается ResNet18
без синтетики и с синтетикой.

### 1. Выбрать редкие классы

```python
!python -m src.synthetic_ablation select-rare \
  --train-annotations /kaggle/working/data/coco_10cls/annotations/instances_train2017_10cls.json \
  --num-classes 3 \
  --output /kaggle/working/reports/rare_classes.json
```

### 2. Сделать crop dataset для CNN

```python
!python -m src.synthetic_ablation make-crops \
  --train-images /kaggle/working/data/coco_10cls/train2017 \
  --train-annotations /kaggle/working/data/coco_10cls/annotations/instances_train2017_10cls.json \
  --val-images /kaggle/working/data/coco_10cls/val2017 \
  --val-annotations /kaggle/working/data/coco_10cls/annotations/instances_val2017_10cls.json \
  --output-dir /kaggle/working/data/classification_crops \
  --min-crop-size 24
```

### 3. Сгенерировать синтетику Stable Diffusion + ControlNet

Для этой ячейки нужен включенный Internet и GPU. Если модель требует Hugging Face
доступ, перед запуском выполняется login через токен.

```python
!python -m src.synthetic_ablation generate \
  --rare-classes /kaggle/working/reports/rare_classes.json \
  --crops-dir /kaggle/working/data/classification_crops/train \
  --output-dir /kaggle/working/data/synthetic_controlnet \
  --images-per-class 50 \
  --resolution 512 \
  --steps 25 \
  --guidance-scale 7.5 \
  --base-model runwayml/stable-diffusion-v1-5 \
  --controlnet-model lllyasviel/sd-controlnet-canny \
  --device cuda
```

Результаты генерации:

- `/kaggle/working/data/synthetic_controlnet/manifest.csv`
- `/kaggle/working/data/synthetic_controlnet/synthetic_examples.png`
- class folders с синтетическими изображениями.

### 4. Ablation: CNN без синтетики и с синтетикой

```python
!python -m src.synthetic_ablation ablation \
  --real-train-dir /kaggle/working/data/classification_crops/train \
  --val-dir /kaggle/working/data/classification_crops/val \
  --synthetic-dir /kaggle/working/data/synthetic_controlnet \
  --output-dir /kaggle/working/reports \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 2 \
  --pretrained \
  --device cuda
```

Результаты ablation:

- `/kaggle/working/reports/synthetic_ablation.csv`
- `/kaggle/working/reports/synthetic_ablation.json`

## Артефакты для сдачи

DETR:

- `runs/` — TensorBoard logs.
- `checkpoints/best.pt`, `checkpoints/last.pt`.
- `profiler_traces/`.
- `reports/metrics.csv`, `reports/eval_metrics.json`.
- `outputs/plots/losses.png`.
- `outputs/visualizations/`.
- `outputs/error_analysis/errors.json`.

Synthetic data:

- `data/synthetic_controlnet/manifest.csv`.
- `data/synthetic_controlnet/synthetic_examples.png`.
- `reports/synthetic_ablation.csv`.
- `reports/synthetic_ablation.json`.

После полного запуска в README добавляются фактические значения `mAP/mAP50`,
таблица ablation и краткие наблюдения по loss/error analysis.

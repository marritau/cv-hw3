import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_class_names(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def save_json(path: str | Path, data: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_metrics_row(path: str | Path, row: dict[str, Any]):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class JsonlSummaryWriter:
    def __init__(self, log_dir: str | Path):
        self.path = ensure_dir(log_dir) / "scalars.jsonl"

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": global_step, "tag": tag, "value": float(scalar_value)}) + "\n")

    def close(self):
        return None


def make_summary_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return JsonlSummaryWriter(log_dir)


class CocoDetectionSubset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        class_names: list[str],
        max_size: int | None = 640,
        train: bool = False,
        hflip_prob: float = 0.0,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.class_names = class_names
        self.max_size = max_size
        self.train = train
        self.hflip_prob = hflip_prob

        with self.annotation_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
        name_to_cat_id = {name: cat_id for cat_id, name in categories.items()}
        missing = [name for name in class_names if name not in name_to_cat_id]
        if missing:
            raise ValueError(f"Classes not found in annotations: {missing}")
        self.cat_id_to_label = {name_to_cat_id[name]: idx for idx, name in enumerate(class_names)}

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_label or ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0 and ann.get("area", w * h) > 0:
                anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        self.images = [img for img in coco["images"] if int(img["id"]) in anns_by_image]
        self.images.sort(key=lambda img: int(img["id"]))
        self.anns_by_image = anns_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        image_id = int(img_info["id"])
        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        orig_w, orig_h = image.size

        boxes, labels = [], []
        for ann in self.anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(orig_w), float(x + w))
            y2 = min(float(orig_h), float(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([((x1 + x2) * 0.5) / orig_w, ((y1 + y2) * 0.5) / orig_h, (x2 - x1) / orig_w, (y2 - y1) / orig_h])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for box in boxes:
                box[0] = 1.0 - box[0]

        if self.max_size is not None and max(orig_h, orig_w) > self.max_size:
            scale = self.max_size / max(orig_h, orig_w)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            new_w, new_h = orig_w, orig_h

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([new_h, new_w], dtype=torch.long),
            "file_name": img_info["file_name"],
        }
        return self._image_to_tensor(image), target

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def collate_fn(batch):
    images, targets = zip(*batch)
    max_h = max(image.shape[1] for image in images)
    max_w = max(image.shape[2] for image in images)
    padded = images[0].new_zeros((len(images), 3, max_h, max_w))
    masks = torch.ones((len(images), max_h, max_w), dtype=torch.bool)
    for idx, image in enumerate(images):
        _, h, w = image.shape
        padded[idx, :, :h, :w] = image
        masks[idx, :h, :w] = False
    return {"images": padded, "masks": masks}, list(targets)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    return inter / (union + eps)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    iou = inter / (union + eps)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    return iou - (area_c - union) / (area_c + eps)


class HFDetr(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        pretrained_model: str = "facebook/detr-resnet-50",
        local_pretrained_dir: str | None = None,
    ):
        super().__init__()
        from transformers import DetrForObjectDetection

        id2label = {idx: name for idx, name in enumerate(class_names)}
        label2id = {name: idx for idx, name in id2label.items()}
        source = local_pretrained_dir if local_pretrained_dir else pretrained_model
        self.model = DetrForObjectDetection.from_pretrained(
            source,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id

    def forward(self, samples: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        labels = None
        if targets is not None:
            labels = [{"class_labels": target["labels"], "boxes": target["boxes"]} for target in targets]
        pixel_mask = (~samples["masks"].bool()).long()
        outputs = self.model(pixel_values=samples["images"], pixel_mask=pixel_mask, labels=labels)
        result = {"logits": outputs.logits, "boxes": outputs.pred_boxes}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
            result["loss_dict"] = outputs.loss_dict or {}
        return result


def loss_from_outputs(outputs: dict[str, torch.Tensor]):
    loss_dict = outputs.get("loss_dict", {})
    zero = outputs["logits"].sum() * 0.0
    loss_total = outputs.get("loss", zero)
    return {
        "loss_total": loss_total,
        "loss_ce": loss_dict.get("loss_ce", zero).detach(),
        "loss_bbox": loss_dict.get("loss_bbox", zero).detach(),
        "loss_giou": loss_dict.get("loss_giou", zero).detach(),
    }


@torch.no_grad()
def batch_to_predictions(outputs: dict[str, torch.Tensor], targets: list[dict], score_threshold: float = 0.0, top_k: int = 100):
    logits = outputs["logits"].detach().cpu()
    boxes = box_cxcywh_to_xyxy(outputs["boxes"].detach().cpu()).clamp(0.0, 1.0)
    probs = logits.softmax(-1)[..., :-1]
    scores, labels = probs.max(dim=-1)
    predictions, ground_truths = [], []

    for batch_idx, target in enumerate(targets):
        image_id = int(target["image_id"].detach().cpu())
        keep = scores[batch_idx] >= score_threshold
        kept_scores = scores[batch_idx][keep]
        kept_labels = labels[batch_idx][keep]
        kept_boxes = boxes[batch_idx][keep]
        if kept_scores.numel() > top_k:
            order = torch.argsort(kept_scores, descending=True)[:top_k]
            kept_scores, kept_labels, kept_boxes = kept_scores[order], kept_labels[order], kept_boxes[order]

        for score, label, box in zip(kept_scores, kept_labels, kept_boxes):
            predictions.append({"image_id": image_id, "score": float(score), "label": int(label), "box": [float(x) for x in box.tolist()]})

        gt_boxes = box_cxcywh_to_xyxy(target["boxes"].detach().cpu()).clamp(0.0, 1.0)
        for label, box in zip(target["labels"].detach().cpu(), gt_boxes):
            ground_truths.append({"image_id": image_id, "label": int(label), "box": [float(x) for x in box.tolist()]})
    return predictions, ground_truths


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(precisions) - 1, 0, -1):
        precisions[idx - 1] = max(precisions[idx - 1], precisions[idx])
    grid = np.linspace(0, 1, 101)
    return float(np.mean([precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0 for t in grid]))


def compute_detection_metrics(predictions: list[dict], ground_truths: list[dict], num_classes: int):
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    ap_by_threshold = {threshold: [] for threshold in thresholds}
    for class_id in range(num_classes):
        gt_class = [g for g in ground_truths if int(g["label"]) == class_id]
        pred_class = sorted([p for p in predictions if int(p["label"]) == class_id], key=lambda p: float(p["score"]), reverse=True)
        for threshold in thresholds:
            if not gt_class:
                continue
            gt_by_image = defaultdict(list)
            for gt in gt_class:
                item = dict(gt)
                item["matched"] = False
                gt_by_image[int(gt["image_id"])].append(item)
            tp, fp = np.zeros(len(pred_class), dtype=np.float32), np.zeros(len(pred_class), dtype=np.float32)
            for idx, pred in enumerate(pred_class):
                candidates = gt_by_image.get(int(pred["image_id"]), [])
                if not candidates:
                    fp[idx] = 1.0
                    continue
                pred_box = torch.tensor(pred["box"], dtype=torch.float32).view(1, 4)
                gt_boxes = torch.tensor([gt["box"] for gt in candidates], dtype=torch.float32)
                ious = pairwise_iou(pred_box, gt_boxes)[0]
                matched = False
                for gt_idx in torch.argsort(ious, descending=True).tolist():
                    best_gt = candidates[gt_idx]
                    if float(ious[gt_idx]) >= threshold and not best_gt["matched"]:
                        tp[idx] = 1.0
                        best_gt["matched"] = True
                        matched = True
                        break
                if not matched:
                    fp[idx] = 1.0
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            recalls = cum_tp / max(len(gt_class), 1)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)
            ap_by_threshold[threshold].append(_ap_from_pr(recalls, precisions))
    mean_by_threshold = {t: float(np.mean(v)) if v else 0.0 for t, v in ap_by_threshold.items()}
    return {"mAP": float(np.mean(list(mean_by_threshold.values()))), "mAP50": mean_by_threshold.get(0.5, 0.0), "metric_backend": "simple"}


def compute_coco_metrics(predictions: list[dict], annotation_file: str | Path, class_names: list[str]):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Для COCO mAP необходимо установить pycocotools.") from exc

    try:
        coco_gt = COCO(str(annotation_file))
        name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in coco_gt.loadCats(coco_gt.getCatIds())}
        label_to_cat_id = {idx: name_to_cat_id[name] for idx, name in enumerate(class_names) if name in name_to_cat_id}
        image_info = {int(img["id"]): img for img in coco_gt.dataset["images"]}

        coco_predictions = []
        for pred in predictions:
            image_id = int(pred["image_id"])
            if int(pred["label"]) not in label_to_cat_id or image_id not in image_info:
                continue
            width = image_info[image_id]["width"]
            height = image_info[image_id]["height"]
            x0, y0, x1, y1 = pred["box"]
            coco_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": label_to_cat_id[int(pred["label"])],
                    "bbox": [x0 * width, y0 * height, max(0.0, (x1 - x0) * width), max(0.0, (y1 - y0) * height)],
                    "score": float(pred["score"]),
                }
            )

        if not coco_predictions:
            return {"mAP": 0.0, "mAP50": 0.0, "metric_backend": "coco"}
        coco_dt = coco_gt.loadRes(coco_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = list(label_to_cat_id.values())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {"mAP": float(coco_eval.stats[0]), "mAP50": float(coco_eval.stats[1]), "metric_backend": "coco"}
    except Exception as exc:
        raise RuntimeError("COCOeval завершился с ошибкой.") from exc


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    num_classes: int,
    metric_score_threshold: float = 0.0,
    top_k: int = 100,
    metric_backend: str = "coco",
    annotation_file: str | Path | None = None,
    class_names: list[str] | None = None,
):
    model.eval()
    predictions, ground_truths, loss_totals, steps = [], [], {}, 0
    for samples, targets in loader:
        samples = {key: value.to(device) for key, value in samples.items()}
        targets = move_targets_to_device(targets, device)
        outputs = model(samples, targets)
        losses = loss_from_outputs(outputs)
        for key, value in losses.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())
        batch_preds, batch_gts = batch_to_predictions(outputs, targets, metric_score_threshold, top_k)
        predictions.extend(batch_preds)
        ground_truths.extend(batch_gts)
        steps += 1

    if metric_backend == "coco":
        if annotation_file is None or class_names is None:
            raise RuntimeError("COCOeval требует annotation_file и class_names.")
        metrics = compute_coco_metrics(predictions, annotation_file, class_names)
    elif metric_backend == "simple":
        metrics = compute_detection_metrics(predictions, ground_truths, num_classes)
    else:
        raise ValueError(f"Unknown metric backend: {metric_backend}")

    metrics["metric_score_threshold"] = metric_score_threshold
    metrics.update({f"val_{key}": value / max(steps, 1) for key, value in loss_totals.items()})
    return metrics, predictions, ground_truths


def plot_losses(metrics_csv: str | Path, output: str | Path):
    with Path(metrics_csv).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(float(row["epoch"])) for row in rows]
    series = {"classification": "train_loss_ce", "bbox L1": "train_loss_bbox", "GIoU": "train_loss_giou", "total": "train_loss_total"}
    plt.figure(figsize=(9, 5))
    for label, key in series.items():
        values = [float(row[key]) for row in rows if row.get(key)]
        if len(values) == len(epochs):
            plt.plot(epochs, values, marker="o", label=label)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    plt.legend()
    ensure_dir(Path(output).parent)
    plt.tight_layout()
    plt.savefig(output, dpi=160)


def analyze_errors(predictions: list[dict], ground_truths: list[dict], iou_match: float = 0.5):
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for gt in ground_truths:
        gt_by_image[int(gt["image_id"])].append(gt)
    for pred in predictions:
        pred_by_image[int(pred["image_id"])].append(pred)

    summary = {
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "classification_errors": [],
        "localization_errors": [],
        "duplicate_false_positives": [],
        "background_false_positives": [],
    }
    matched_gt = set()
    for image_id, preds in pred_by_image.items():
        gts = gt_by_image.get(image_id, [])
        if not gts:
            summary["background_false_positives"].extend(preds)
            summary["false_positives"].extend(preds)
            continue

        gt_boxes = torch.tensor([gt["box"] for gt in gts], dtype=torch.float32)
        for pred in sorted(preds, key=lambda item: float(item["score"]), reverse=True):
            ious = pairwise_iou(torch.tensor(pred["box"], dtype=torch.float32).view(1, 4), gt_boxes)[0]
            order = torch.argsort(ious, descending=True).tolist()
            best_all_idx = order[0]
            best_all_iou = float(ious[best_all_idx])
            best_all_key = (image_id, best_all_idx)

            if best_all_iou >= iou_match and best_all_key in matched_gt:
                item = {**pred, "best_iou": best_all_iou, "gt_label": int(gts[best_all_idx]["label"])}
                summary["duplicate_false_positives"].append(item)
                summary["false_positives"].append(item)
                continue

            unmatched = [idx for idx in order if (image_id, idx) not in matched_gt]
            if not unmatched:
                item = {**pred, "best_iou": best_all_iou}
                summary["duplicate_false_positives"].append(item)
                summary["false_positives"].append(item)
                continue

            best_idx = unmatched[0]
            gt = gts[best_idx]
            gt_key = (image_id, best_idx)
            best_iou = float(ious[best_idx])
            item = {**pred, "best_iou": best_iou, "gt_label": int(gt["label"])}

            if best_iou >= iou_match and int(pred["label"]) == int(gt["label"]):
                matched_gt.add(gt_key)
                summary["true_positives"].append(item)
            else:
                summary["false_positives"].append(item)
                if best_iou >= iou_match and int(pred["label"]) != int(gt["label"]):
                    summary["classification_errors"].append(item)
                elif int(pred["label"]) == int(gt["label"]) and best_iou >= 0.1:
                    summary["localization_errors"].append(item)
                else:
                    summary["background_false_positives"].append(item)

    for image_id, gts in gt_by_image.items():
        for idx, gt in enumerate(gts):
            if (image_id, idx) not in matched_gt:
                summary["false_negatives"].append(gt)
    summary["counts"] = {key: len(value) for key, value in summary.items() if isinstance(value, list)}
    return summary


def draw_detections(image: torch.Tensor, size: torch.Tensor, gt_items: list[dict], pred_items: list[dict], class_names: list[str], out_path: str | Path):
    h, w = [int(x) for x in size.detach().cpu().tolist()]
    image = image.detach().cpu()[:, :h, :w]
    image = (image * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)
    pil = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    draw = ImageDraw.Draw(pil)
    width, height = pil.size
    font = ImageFont.load_default()

    def draw_box(item, color, prefix):
        x0, y0, x1, y1 = item["box"]
        box = [x0 * width, y0 * height, x1 * width, y1 * height]
        label = class_names[int(item["label"])]
        text = f"{prefix}:{label}" if "score" not in item else f"{label}:{float(item['score']):.2f}"
        draw.rectangle(box, outline=color, width=3)
        text_box = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), text, fill="white", font=font)

    for gt in gt_items:
        draw_box(gt, "lime", "gt")
    for pred in sorted(pred_items, key=lambda item: float(item["score"]), reverse=True)[:20]:
        draw_box(pred, "red", "pred")
    ensure_dir(Path(out_path).parent)
    pil.save(out_path)


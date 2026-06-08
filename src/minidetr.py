import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
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
    """Fallback scalar logger when tensorboard is not installed."""

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
    """Small COCO-style detection dataset reader without pycocotools."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        class_names: list[str],
        max_size: int | None = 640,
        normalize: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.class_names = class_names
        self.max_size = max_size
        self.normalize = normalize

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

        if self.max_size is not None and max(orig_h, orig_w) > self.max_size:
            scale = self.max_size / max(orig_h, orig_w)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
        else:
            new_w, new_h = orig_w, orig_h

        boxes, labels = [], []
        for ann in self.anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([(x + 0.5 * w) / orig_w, (y + 0.5 * h) / orig_h, w / orig_w, h / orig_h])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).clamp(0.0, 1.0),
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
        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor


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


class SinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 256, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4.")
        self.num_pos_feats = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = (~mask.bool()).float()
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNNBackbone(nn.Module):
    def __init__(self, d_model: int = 256, channels: tuple[int, ...] = (64, 128, 256, 256)):
        super().__init__()
        blocks, in_channels = [], 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(in_channels, d_model, 1)

    def forward(self, images: torch.Tensor, mask: torch.Tensor | None = None):
        feats = self.proj(self.body(images))
        if mask is None:
            mask = torch.zeros(images.shape[0], images.shape[-2], images.shape[-1], dtype=torch.bool, device=images.device)
        feat_mask = F.interpolate(mask[:, None].float(), size=feats.shape[-2:], mode="nearest")
        return feats, feat_mask[:, 0].to(torch.bool)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: torch.Tensor, pad_mask: torch.Tensor | None):
        x = src
        norm = self.norm1(x)
        x2, _ = self.self_attn(norm + pos, norm + pos, norm, key_padding_mask=pad_mask)
        x = x + self.drop1(x2)
        return x + self.drop2(self.ffn(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor, memory_pad_mask: torch.Tensor | None):
        x = tgt
        norm = self.norm1(x)
        x2, _ = self.self_attn(norm, norm, norm)
        x = x + self.drop1(x2)
        mem = self.norm2(memory) + memory_pos
        x2, _ = self.cross_attn(self.norm2(x), mem, mem, key_padding_mask=memory_pad_mask)
        x = x + self.drop2(x2)
        return x + self.drop3(self.ffn(self.norm3(x)))


class MiniTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, enc_layers: int, dec_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(dec_layers)])

    def forward(self, src: torch.Tensor, mask: torch.Tensor, queries: torch.Tensor, pos: torch.Tensor):
        memory = src
        for layer in self.encoder:
            memory = layer(memory, pos, mask)
        tgt = queries
        for layer in self.decoder:
            tgt = layer(tgt, memory, pos, mask)
        return memory, tgt


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class MiniDETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        d_model: int = 128,
        nhead: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SimpleCNNBackbone(d_model=d_model)
        self.posenc = SinePositionalEncoding2D(d_model=d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.transformer = MiniTransformer(d_model, nhead, enc_layers, dec_layers, dim_ff, dropout)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, samples: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(samples, dict):
            images, pad_mask = samples["images"], samples.get("masks")
        else:
            images, pad_mask = samples, None
        feats, feat_mask = self.backbone(images, pad_mask)
        pos = self.posenc(feat_mask).permute(0, 3, 1, 2)
        src = feats.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)
        mask = feat_mask.flatten(1)
        queries = self.query_embed.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        memory, tgt = self.transformer(src, mask, queries, pos)
        return {"logits": self.class_embed(tgt), "boxes": self.bbox_embed(tgt).sigmoid(), "mask": mask, "memory": memory}


class HFDetr(nn.Module):
    """Thin wrapper around pretrained facebook/detr-resnet-50 for real fine-tuning."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        pretrained_model: str = "facebook/detr-resnet-50",
        hf_config: dict | None = None,
    ):
        super().__init__()
        from transformers import DetrConfig, DetrForObjectDetection

        id2label = {idx: name for idx, name in enumerate(class_names)}
        label2id = {name: idx for idx, name in id2label.items()}
        if hf_config is None:
            self.model = DetrForObjectDetection.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            config = DetrConfig.from_dict(hf_config)
            config.num_labels = num_classes
            config.id2label = id2label
            config.label2id = label2id
            config.use_pretrained_backbone = False
            self.model = DetrForObjectDetection(config)

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


@dataclass
class HungarianMatcher:
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0

    @torch.no_grad()
    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        matches = []
        for logits, boxes, target in zip(outputs["logits"], outputs["boxes"], targets):
            if target["boxes"].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue

            prob = logits.softmax(-1)[:, :-1]
            cls_cost = -prob[:, target["labels"]]
            l1_cost = torch.cdist(boxes, target["boxes"], p=1)
            giou_cost = 1.0 - generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(target["boxes"]))
            cost = self.cost_class * cls_cost + self.cost_bbox * l1_cost + self.cost_giou * giou_cost
            pred_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((
                torch.as_tensor(pred_idx, dtype=torch.long, device=logits.device),
                torch.as_tensor(tgt_idx, dtype=torch.long, device=logits.device),
            ))
        return matches


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        indices = self.matcher(outputs, targets)
        logits, pred_boxes = outputs["logits"], outputs["boxes"]
        batch_size, num_queries, _ = logits.shape
        target_classes = torch.full((batch_size, num_queries), self.num_classes, dtype=torch.long, device=logits.device)
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]
        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, weight=self.empty_weight)

        src_boxes, tgt_boxes = [], []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                src_boxes.append(pred_boxes[batch_idx, src_idx])
                tgt_boxes.append(targets[batch_idx]["boxes"][tgt_idx])
        num_boxes = max(sum(len(target["labels"]) for target in targets), 1)
        if src_boxes:
            src_boxes = torch.cat(src_boxes)
            tgt_boxes = torch.cat(tgt_boxes)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
            giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            loss_giou = (1.0 - torch.diag(giou)).sum() / num_boxes
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        total = loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou
        return {
            "loss_total": total,
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }


def loss_from_outputs(outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], criterion: nn.Module | None):
    if criterion is not None:
        return criterion(outputs, targets)

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
def batch_to_predictions(outputs: dict[str, torch.Tensor], targets: list[dict], score_threshold: float = 0.05, top_k: int = 100):
    logits = outputs["logits"].detach().cpu()
    boxes = box_cxcywh_to_xyxy(outputs["boxes"].detach().cpu()).clamp(0.0, 1.0)
    probs = logits.softmax(-1)[..., :-1]
    scores, labels = probs.max(dim=-1)
    predictions, ground_truths = [], []

    for batch_idx, target in enumerate(targets):
        image_id = int(target["image_id"].detach().cpu())
        keep = scores[batch_idx] >= score_threshold
        kept_scores, kept_labels, kept_boxes = scores[batch_idx][keep], labels[batch_idx][keep], boxes[batch_idx][keep]
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
                order = torch.argsort(ious, descending=True)
                matched = False
                for gt_idx in order.tolist():
                    best_gt = candidates[gt_idx]
                    if float(ious[gt_idx]) >= threshold and not best_gt["matched"]:
                        tp[idx] = 1.0
                        best_gt["matched"] = True
                        matched = True
                        break
                if not matched:
                    fp[idx] = 1.0
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            ap_by_threshold[threshold].append(_ap_from_pr(cum_tp / max(len(gt_class), 1), cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)))
    mean_by_threshold = {t: float(np.mean(v)) if v else 0.0 for t, v in ap_by_threshold.items()}
    return {"mAP": float(np.mean(list(mean_by_threshold.values()))), "mAP50": mean_by_threshold.get(0.5, 0.0)}


def compute_coco_metrics(predictions: list[dict], annotation_file: str | Path, class_names: list[str]):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception:
        return None

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
        return {"mAP": 0.0, "mAP50": 0.0}
    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = list(label_to_cat_id.values())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP": float(coco_eval.stats[0]), "mAP50": float(coco_eval.stats[1])}


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    score_threshold: float = 0.05,
    top_k: int = 100,
    metric_backend: str = "simple",
    annotation_file: str | Path | None = None,
    class_names: list[str] | None = None,
):
    model.eval()
    predictions, ground_truths, loss_totals, steps = [], [], {}, 0
    for samples, targets in loader:
        samples = {key: value.to(device) for key, value in samples.items()}
        targets = move_targets_to_device(targets, device)
        outputs = model(samples, targets) if criterion is None else model(samples)
        losses = loss_from_outputs(outputs, targets, criterion)
        for key, value in losses.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value.detach().cpu())
        batch_preds, batch_gts = batch_to_predictions(outputs, targets, score_threshold, top_k)
        predictions.extend(batch_preds)
        ground_truths.extend(batch_gts)
        steps += 1
    metrics = None
    if metric_backend == "coco" and annotation_file is not None and class_names is not None:
        metrics = compute_coco_metrics(predictions, annotation_file, class_names)
    if metrics is None:
        metrics = compute_detection_metrics(predictions, ground_truths, num_classes)
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
        "classification_errors": [],
        "localization_errors": [],
        "duplicate_false_positives": [],
        "background_false_positives": [],
        "false_positives": [],
        "false_negatives": [],
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
            best_any_idx = order[0]
            best_any_iou = float(ious[best_any_idx])
            best_unmatched_idx = next((idx for idx in order if (image_id, idx) not in matched_gt), None)

            if best_unmatched_idx is None:
                item = {**pred, "best_iou": best_any_iou}
                summary["duplicate_false_positives"].append(item)
                summary["false_positives"].append(item)
                continue

            gt = gts[best_unmatched_idx]
            gt_key = (image_id, best_unmatched_idx)
            best_iou = float(ious[best_unmatched_idx])
            item = {**pred, "best_iou": best_iou, "gt_label": int(gt["label"])}
            if best_iou >= iou_match and int(pred["label"]) == int(gt["label"]):
                matched_gt.add(gt_key)
            elif best_iou >= iou_match and int(pred["label"]) != int(gt["label"]):
                matched_gt.add(gt_key)
                summary["classification_errors"].append(item)
            elif int(pred["label"]) == int(gt["label"]) and best_iou >= 0.1:
                matched_gt.add(gt_key)
                summary["localization_errors"].append(item)
            else:
                summary["background_false_positives"].append(item)
                summary["false_positives"].append(item)
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

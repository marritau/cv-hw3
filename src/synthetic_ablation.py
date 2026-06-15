import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def class_counts(annotation_file: str | Path):
    coco = read_json(annotation_file)
    cat_id_to_name = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    counts = {name: 0 for name in cat_id_to_name.values()}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        name = cat_id_to_name.get(int(ann["category_id"]))
        if name is not None:
            counts[name] += 1
    return counts


def cmd_select_rare(args):
    counts = class_counts(args.train_annotations)
    rare = sorted(counts.items(), key=lambda item: (item[1], item[0]))[: args.num_classes]
    data = {
        "rare_classes": [name for name, _ in rare],
        "counts": dict(rare),
        "all_counts": counts,
    }
    save_json(args.output, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def crop_xyxy_from_coco_bbox(bbox, width: int, height: int):
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_crop_split(image_dir: str | Path, annotation_file: str | Path, out_dir: str | Path, min_crop_size: int):
    image_dir = Path(image_dir)
    out_dir = ensure_dir(out_dir)
    coco = read_json(annotation_file)
    images = {int(img["id"]): img for img in coco["images"]}
    cat_id_to_name = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    manifest = []
    counters = {name: 0 for name in cat_id_to_name.values()}

    for ann in tqdm(coco["annotations"], desc=f"crops {Path(annotation_file).name}"):
        if ann.get("iscrowd", 0):
            continue
        img_info = images.get(int(ann["image_id"]))
        class_name = cat_id_to_name.get(int(ann["category_id"]))
        if img_info is None or class_name is None:
            continue
        image_path = image_dir / img_info["file_name"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        crop_box = crop_xyxy_from_coco_bbox(ann["bbox"], image.width, image.height)
        if crop_box is None:
            continue
        x1, y1, x2, y2 = crop_box
        if min(x2 - x1, y2 - y1) < min_crop_size:
            continue
        crop = image.crop(crop_box)
        counters[class_name] += 1
        class_dir = ensure_dir(out_dir / class_name)
        out_path = class_dir / f"{int(ann['image_id']):012d}_{int(ann['id'])}.jpg"
        crop.save(out_path, quality=95)
        manifest.append({"path": str(out_path), "class_name": class_name, "source_image_id": int(ann["image_id"])})

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_name", "source_image_id"])
        writer.writeheader()
        writer.writerows(manifest)
    save_json(out_dir / "class_counts.json", counters)


def cmd_make_crops(args):
    build_crop_split(args.train_images, args.train_annotations, Path(args.output_dir) / "train", args.min_crop_size)
    build_crop_split(args.val_images, args.val_annotations, Path(args.output_dir) / "val", args.min_crop_size)


def load_rare_classes(path: str | Path):
    data = read_json(path)
    if isinstance(data, list):
        return data
    return data["rare_classes"]


def make_control_image(image: Image.Image, size: int):
    image = ImageOps.fit(image.convert("RGB"), (size, size))
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    return Image.merge("RGB", (edges, edges, edges))


def collect_conditioning_images(crops_dir: str | Path, class_name: str):
    class_dir = Path(crops_dir) / class_name
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(class_dir.glob(pattern))
    return sorted(images)


def cmd_generate(args):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    rare_classes = load_rare_classes(args.rare_classes)
    out_root = ensure_dir(args.output_dir)
    control_root = ensure_dir(out_root / "_control")

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    if args.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    manifest = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for class_name in rare_classes:
        candidates = collect_conditioning_images(args.crops_dir, class_name)
        if not candidates:
            raise RuntimeError(f"No crop images found for class '{class_name}' in {args.crops_dir}")
        class_out = ensure_dir(out_root / class_name)
        class_control = ensure_dir(control_root / class_name)
        for idx in tqdm(range(args.images_per_class), desc=f"generate {class_name}"):
            source_path = candidates[idx % len(candidates)]
            source = Image.open(source_path).convert("RGB")
            control = make_control_image(source, args.resolution)
            prompt = args.prompt_template.format(class_name=class_name)
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=control,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            image_path = class_out / f"{class_name.replace(' ', '_')}_{idx:04d}.png"
            control_path = class_control / f"{class_name.replace(' ', '_')}_{idx:04d}.png"
            image.save(image_path)
            control.save(control_path)
            manifest.append(
                {
                    "path": str(image_path),
                    "control_path": str(control_path),
                    "class_name": class_name,
                    "prompt": prompt,
                    "source_crop": str(source_path),
                }
            )

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "control_path", "class_name", "prompt", "source_crop"])
        writer.writeheader()
        writer.writerows(manifest)
    save_synthetic_grid(out_root, manifest, args.grid_output)


def save_synthetic_grid(out_root: Path, manifest: list[dict], output: str | Path, limit: int = 16):
    if not manifest:
        return
    paths = [Path(item["path"]) for item in manifest[:limit]]
    images = [ImageOps.fit(Image.open(path).convert("RGB"), (160, 160)) for path in paths]
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    grid = Image.new("RGB", (cols * 160, rows * 160), "white")
    for idx, image in enumerate(images):
        grid.paste(image, ((idx % cols) * 160, (idx // cols) * 160))
    output = Path(output)
    if not output.is_absolute():
        output = out_root / output
    ensure_dir(output.parent)
    grid.save(output)


class FolderClassificationDataset(Dataset):
    def __init__(self, root: str | Path, class_to_idx: dict[str, int], transform=None):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []
        for class_name, idx in class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                for path in class_dir.glob(pattern):
                    self.samples.append((path, idx))
        self.samples.sort(key=lambda item: str(item[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_transforms(train: bool, image_size: int):
    from torchvision import transforms

    aug = [transforms.Resize((image_size, image_size))]
    if train:
        aug.append(transforms.RandomHorizontalFlip())
    aug.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(aug)


def compute_classification_metrics(y_true, y_pred, class_names: list[str]):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1_scores = []
    per_class = {}
    for idx, class_name in enumerate(class_names):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_scores.append(f1)
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1, "support": int((y_true == idx).sum())}
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0, "per_class": per_class}


def build_classifier(num_classes: int, pretrained: bool):
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_classifier_once(args, use_synthetic: bool):
    set_seed(args.seed)
    device = torch.device(args.device)
    class_names = sorted([p.name for p in Path(args.real_train_dir).iterdir() if p.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_transform = make_transforms(True, args.image_size)
    val_transform = make_transforms(False, args.image_size)
    real_train = FolderClassificationDataset(args.real_train_dir, class_to_idx, train_transform)
    train_datasets = [real_train]
    if use_synthetic:
        synthetic = FolderClassificationDataset(args.synthetic_dir, class_to_idx, train_transform)
        train_datasets.append(synthetic)
    train_data = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_data = FolderClassificationDataset(args.val_dir, class_to_idx, val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_classifier(len(class_names), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"classifier epoch {epoch} synthetic={use_synthetic}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * images.size(0)
        print({"epoch": epoch, "use_synthetic": use_synthetic, "train_loss": total_loss / max(len(train_data), 1)})

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["use_synthetic"] = use_synthetic
    metrics["num_train_real"] = len(real_train)
    metrics["num_train_total"] = len(train_data)
    metrics["num_val"] = len(val_data)
    return metrics


def cmd_ablation(args):
    out_dir = ensure_dir(args.output_dir)
    rows = []
    results = []
    for use_synthetic in (False, True):
        metrics = train_classifier_once(args, use_synthetic=use_synthetic)
        results.append(metrics)
        rows.append(
            {
                "experiment": "with_synthetic" if use_synthetic else "real_only",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "num_train_real": metrics["num_train_real"],
                "num_train_total": metrics["num_train_total"],
                "num_val": metrics["num_val"],
            }
        )
    with (out_dir / "synthetic_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out_dir / "synthetic_ablation.json", results)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion + ControlNet synthetic-data ablation.")
    sub = parser.add_subparsers(dest="command", required=True)

    rare_p = sub.add_parser("select-rare")
    rare_p.add_argument("--train-annotations", required=True)
    rare_p.add_argument("--num-classes", type=int, default=3)
    rare_p.add_argument("--output", default="reports/rare_classes.json")
    rare_p.set_defaults(func=cmd_select_rare)

    crops_p = sub.add_parser("make-crops")
    crops_p.add_argument("--train-images", required=True)
    crops_p.add_argument("--train-annotations", required=True)
    crops_p.add_argument("--val-images", required=True)
    crops_p.add_argument("--val-annotations", required=True)
    crops_p.add_argument("--output-dir", default="data/classification_crops")
    crops_p.add_argument("--min-crop-size", type=int, default=24)
    crops_p.set_defaults(func=cmd_make_crops)

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--rare-classes", default="reports/rare_classes.json")
    gen_p.add_argument("--crops-dir", default="data/classification_crops/train")
    gen_p.add_argument("--output-dir", default="data/synthetic_controlnet")
    gen_p.add_argument("--images-per-class", type=int, default=50)
    gen_p.add_argument("--resolution", type=int, default=512)
    gen_p.add_argument("--steps", type=int, default=25)
    gen_p.add_argument("--guidance-scale", type=float, default=7.5)
    gen_p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    gen_p.add_argument("--controlnet-model", default="lllyasviel/sd-controlnet-canny")
    gen_p.add_argument("--prompt-template", default="a realistic photo of a {class_name}, natural background, COCO dataset style")
    gen_p.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark")
    gen_p.add_argument("--grid-output", default="synthetic_examples.png")
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    gen_p.add_argument("--enable-xformers", action="store_true")
    gen_p.set_defaults(func=cmd_generate)

    abl_p = sub.add_parser("ablation")
    abl_p.add_argument("--real-train-dir", default="data/classification_crops/train")
    abl_p.add_argument("--val-dir", default="data/classification_crops/val")
    abl_p.add_argument("--synthetic-dir", default="data/synthetic_controlnet")
    abl_p.add_argument("--output-dir", default="reports")
    abl_p.add_argument("--epochs", type=int, default=5)
    abl_p.add_argument("--batch-size", type=int, default=32)
    abl_p.add_argument("--num-workers", type=int, default=2)
    abl_p.add_argument("--image-size", type=int, default=224)
    abl_p.add_argument("--lr", type=float, default=1e-4)
    abl_p.add_argument("--weight-decay", type=float, default=1e-4)
    abl_p.add_argument("--seed", type=int, default=42)
    abl_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    abl_p.add_argument("--pretrained", action="store_true")
    abl_p.set_defaults(func=cmd_ablation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

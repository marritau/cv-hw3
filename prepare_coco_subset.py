import argparse
import json
import os
import random
import shutil
import csv
from pathlib import Path


DEFAULT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "stop sign",
    "dog",
]


def parse_classes(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_CLASSES)
    class_names = [item.strip() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(class_names))


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or mode == "none":
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def save_distribution(path: Path, class_counts: dict[str, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "instances"])
        writer.writeheader()
        for class_name, count in class_counts.items():
            writer.writerow({"class_name": class_name, "instances": count})


def filter_split(
    coco_root: Path,
    out_root: Path,
    split: str,
    class_names: list[str],
    max_images: int | None,
    min_instances_per_class: int,
    seed: int,
    link_mode: str,
):
    ann_path = coco_root / "annotations" / f"instances_{split}.json"
    image_src_dir = coco_root / split
    image_out_dir = out_root / split
    suffix = f"{len(class_names)}cls"
    out_ann_path = out_root / "annotations" / f"instances_{split}_{suffix}.json"

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    name_to_cat = {cat["name"]: cat for cat in coco["categories"]}
    missing = [name for name in class_names if name not in name_to_cat]
    if missing:
        raise ValueError(f"Classes not present in COCO annotations: {missing}")
    selected_cat_ids = {int(name_to_cat[name]["id"]) for name in class_names}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if int(ann["category_id"]) in selected_cat_ids and not ann.get("iscrowd", 0):
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    images = [img for img in coco["images"] if int(img["id"]) in anns_by_image]
    images.sort(key=lambda img: int(img["id"]))
    if max_images is not None and len(images) > max_images:
        images = random.Random(seed).sample(images, k=max_images)
        images.sort(key=lambda img: int(img["id"]))

    image_ids = {int(img["id"]) for img in images}
    annotations = [ann for img_id in sorted(image_ids) for ann in anns_by_image[img_id]]
    categories = [name_to_cat[name] for name in class_names]
    cat_id_to_name = {int(cat["id"]): cat["name"] for cat in categories}
    class_counts = {name: 0 for name in class_names}
    for ann in annotations:
        name = cat_id_to_name[int(ann["category_id"])]
        class_counts[name] += 1

    missing_after_sampling = [name for name, count in class_counts.items() if count < min_instances_per_class]
    if missing_after_sampling:
        raise RuntimeError(
            f"{split}: after sampling these classes have fewer than {min_instances_per_class} objects: {missing_after_sampling}. "
            "Increase --max-train-images/--max-val-images or choose another class list."
        )

    for img in images:
        copy_or_link(image_src_dir / img["file_name"], image_out_dir / img["file_name"], link_mode)

    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "licenses": coco.get("licenses", []),
                "info": coco.get("info", {}),
            },
            f,
            ensure_ascii=False,
        )
    print(f"{split}: {len(images)} images, {len(annotations)} boxes -> {out_ann_path}")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    save_distribution(out_root / "annotations" / f"class_distribution_{split}.csv", class_counts)


def main():
    parser = argparse.ArgumentParser(description="Prepare a 10-class COCO subset.")
    parser.add_argument("--coco-root", required=True, help="Path with train2017, val2017, annotations.")
    parser.add_argument("--out-root", default="data/coco_10cls")
    parser.add_argument("--classes", default=None, help="Comma-separated COCO class names.")
    parser.add_argument("--max-train-images", type=int, default=3000)
    parser.add_argument("--max-val-images", type=int, default=500)
    parser.add_argument("--min-train-instances-per-class", type=int, default=50)
    parser.add_argument("--min-val-instances-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "none"], default="copy")
    args = parser.parse_args()

    class_names = parse_classes(args.classes)
    if len(class_names) < 10:
        raise ValueError("The homework requires at least 10 unique classes.")
    coco_root, out_root = Path(args.coco_root), Path(args.out_root)
    filter_split(
        coco_root,
        out_root,
        "train2017",
        class_names,
        args.max_train_images,
        args.min_train_instances_per_class,
        args.seed,
        args.link_mode,
    )
    filter_split(
        coco_root,
        out_root,
        "val2017",
        class_names,
        args.max_val_images,
        args.min_val_instances_per_class,
        args.seed,
        args.link_mode,
    )


if __name__ == "__main__":
    main()

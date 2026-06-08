import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.detr import (
    CocoDetectionSubset,
    analyze_errors,
    box_cxcywh_to_xyxy,
    collate_fn,
    compute_detection_metrics,
    generalized_box_iou,
    loss_from_outputs,
)
from src.train import data_config_from_args, model_config_from_args, save_checkpoint, train_config_from_args


class Args:
    pretrained_model = "facebook/detr-resnet-50"
    max_size = 640
    train_annotations = "train.json"
    val_annotations = "val.json"
    lr = 1e-4
    lr_backbone = 1e-5
    lr_drop = 10
    weight_decay = 1e-4
    batch_size = 2
    seed = 42
    metric_backend = "coco"
    metric_score_threshold = 0.0
    hflip_prob = 0.5


class CoreTests(unittest.TestCase):
    def test_box_conversion_and_giou(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        xyxy = box_cxcywh_to_xyxy(boxes)
        expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]])
        self.assertTrue(torch.allclose(xyxy, expected, atol=1e-6))
        giou = generalized_box_iou(xyxy, xyxy)
        self.assertTrue(torch.allclose(giou, torch.ones(1, 1), atol=1e-6))

    def test_dataset_clips_boxes_and_collates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(image_dir / "img.jpg")
            ann_path = root / "ann.json"
            ann_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [-10, -5, 30, 20], "area": 600, "iscrowd": 0}
                        ],
                        "categories": [{"id": 1, "name": "person"}],
                    }
                ),
                encoding="utf-8",
            )
            dataset = CocoDetectionSubset(image_dir, ann_path, ["person"], max_size=100)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 100, 100))
            self.assertTrue(torch.allclose(target["boxes"][0], torch.tensor([0.10, 0.075, 0.20, 0.15]), atol=1e-6))
            samples, targets = collate_fn([(image, target), (image, target)])
            self.assertEqual(tuple(samples["images"].shape), (2, 3, 100, 100))
            self.assertEqual(tuple(samples["masks"].shape), (2, 100, 100))
            self.assertEqual(len(targets), 2)

    def test_loss_from_hf_style_outputs(self):
        logits = torch.randn(2, 100, 11)
        outputs = {
            "logits": logits,
            "loss": torch.tensor(3.0, requires_grad=True),
            "loss_dict": {
                "loss_ce": torch.tensor(1.0),
                "loss_bbox": torch.tensor(0.5),
                "loss_giou": torch.tensor(0.25),
            },
        }
        losses = loss_from_outputs(outputs)
        self.assertEqual(float(losses["loss_total"]), 3.0)
        self.assertEqual(float(losses["loss_ce"]), 1.0)

    def test_map50_perfect_prediction(self):
        predictions = [{"image_id": 1, "label": 0, "score": 0.9, "box": [0.1, 0.1, 0.4, 0.4]}]
        ground_truths = [{"image_id": 1, "label": 0, "box": [0.1, 0.1, 0.4, 0.4]}]
        metrics = compute_detection_metrics(predictions, ground_truths, num_classes=1)
        self.assertAlmostEqual(metrics["mAP50"], 1.0, places=5)

    def test_map_uses_second_unmatched_gt(self):
        predictions = [
            {"image_id": 1, "label": 0, "score": 0.99, "box": [0.10, 0.10, 0.40, 0.40]},
            {"image_id": 1, "label": 0, "score": 0.98, "box": [0.12, 0.10, 0.42, 0.40]},
        ]
        ground_truths = [
            {"image_id": 1, "label": 0, "box": [0.10, 0.10, 0.40, 0.40]},
            {"image_id": 1, "label": 0, "box": [0.14, 0.10, 0.44, 0.40]},
        ]
        metrics = compute_detection_metrics(predictions, ground_truths, num_classes=1)
        self.assertAlmostEqual(metrics["mAP50"], 1.0, places=5)

    def test_error_analysis_marks_duplicate_before_far_unmatched_gt(self):
        predictions = [
            {"image_id": 1, "label": 0, "score": 0.99, "box": [0.10, 0.10, 0.40, 0.40]},
            {"image_id": 1, "label": 0, "score": 0.90, "box": [0.10, 0.10, 0.40, 0.40]},
        ]
        ground_truths = [
            {"image_id": 1, "label": 0, "box": [0.10, 0.10, 0.40, 0.40]},
            {"image_id": 1, "label": 0, "box": [0.70, 0.70, 0.90, 0.90]},
        ]
        errors = analyze_errors(predictions, ground_truths)
        self.assertEqual(errors["counts"]["true_positives"], 1)
        self.assertEqual(errors["counts"]["duplicate_false_positives"], 1)
        self.assertEqual(errors["counts"]["false_negatives"], 1)
        self.assertEqual(errors["counts"]["localization_errors"], 0)

    def test_checkpoint_contains_metadata(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        class_names = [f"class_{idx}" for idx in range(10)]
        args = Args()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                0,
                {"mAP50": 0.5},
                model_config_from_args(args, class_names),
                data_config_from_args(args, class_names),
                train_config_from_args(args),
                0.5,
            )
            ckpt = torch.load(path, map_location="cpu")
        self.assertEqual(ckpt["data_config"]["class_names"], class_names)
        self.assertEqual(ckpt["data_config"]["max_size"], 640)
        self.assertEqual(ckpt["train_config"]["lr_backbone"], 1e-5)
        self.assertIn("scheduler", ckpt)
        self.assertEqual(ckpt["best_map50"], 0.5)


if __name__ == "__main__":
    unittest.main()


import unittest

import torch

from src.minidetr import (
    HungarianMatcher,
    MiniDETR,
    SetCriterion,
    analyze_errors,
    box_cxcywh_to_xyxy,
    compute_detection_metrics,
    generalized_box_iou,
)
from src.train import model_config_from_args, save_checkpoint


class Args:
    model_type = "mini"
    pretrained_model = "facebook/detr-resnet-50"
    num_queries = 5
    d_model = 32
    nhead = 4
    enc_layers = 1
    dec_layers = 1
    dim_ff = 64
    dropout = 0.1
    cost_class = 1.0
    cost_bbox = 5.0
    cost_giou = 2.0
    eos_coef = 0.1


class CoreTests(unittest.TestCase):
    def test_box_conversion_and_giou(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        xyxy = box_cxcywh_to_xyxy(boxes)
        expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]])
        self.assertTrue(torch.allclose(xyxy, expected, atol=1e-6))
        giou = generalized_box_iou(xyxy, xyxy)
        self.assertTrue(torch.allclose(giou, torch.ones(1, 1), atol=1e-6))

    def test_minidetr_forward_and_loss(self):
        torch.manual_seed(0)
        model = MiniDETR(
            num_classes=10,
            num_queries=5,
            d_model=32,
            nhead=4,
            enc_layers=1,
            dec_layers=1,
            dim_ff=64,
        )
        samples = {
            "images": torch.rand(2, 3, 64, 80),
            "masks": torch.zeros(2, 64, 80, dtype=torch.bool),
        }
        outputs = model(samples)
        self.assertEqual(tuple(outputs["logits"].shape), (2, 5, 11))
        self.assertEqual(tuple(outputs["boxes"].shape), (2, 5, 4))

        targets = [
            {
                "labels": torch.tensor([1, 3], dtype=torch.long),
                "boxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.7, 0.5, 0.2, 0.3]]),
            },
            {
                "labels": torch.tensor([2], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
            },
        ]
        criterion = SetCriterion(
            num_classes=10,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
        )
        losses = criterion(outputs, targets)
        self.assertIn("loss_total", losses)
        self.assertTrue(torch.isfinite(losses["loss_total"]))

    def test_map50_perfect_prediction(self):
        predictions = [
            {"image_id": 1, "label": 0, "score": 0.9, "box": [0.1, 0.1, 0.4, 0.4]}
        ]
        ground_truths = [
            {"image_id": 1, "label": 0, "box": [0.1, 0.1, 0.4, 0.4]}
        ]
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

    def test_error_analysis_marks_duplicates(self):
        predictions = [
            {"image_id": 1, "label": 0, "score": 0.99, "box": [0.10, 0.10, 0.40, 0.40]},
            {"image_id": 1, "label": 0, "score": 0.90, "box": [0.10, 0.10, 0.40, 0.40]},
        ]
        ground_truths = [
            {"image_id": 1, "label": 0, "box": [0.10, 0.10, 0.40, 0.40]},
        ]
        errors = analyze_errors(predictions, ground_truths)
        self.assertEqual(errors["counts"]["duplicate_false_positives"], 1)
        self.assertEqual(errors["counts"]["localization_errors"], 0)
        self.assertEqual(errors["counts"]["false_negatives"], 0)

    def test_checkpoint_contains_metadata(self):
        import tempfile
        from pathlib import Path

        model = MiniDETR(num_classes=10, num_queries=5, d_model=32, nhead=4, enc_layers=1, dec_layers=1, dim_ff=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        class_names = [f"class_{idx}" for idx in range(10)]
        config = model_config_from_args(Args(), class_names)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, 0, {"mAP50": 0.5}, config, class_names, 0.5)
            ckpt = torch.load(path, map_location="cpu")
        self.assertEqual(ckpt["class_names"], class_names)
        self.assertEqual(ckpt["model_config"]["model_type"], "mini")
        self.assertEqual(ckpt["best_map50"], 0.5)


if __name__ == "__main__":
    unittest.main()

import argparse
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.detr import (
    CocoDetectionSubset,
    HFDetr,
    analyze_errors,
    append_metrics_row,
    batch_to_predictions,
    collate_fn,
    draw_detections,
    ensure_dir,
    evaluate_model,
    loss_from_outputs,
    make_summary_writer,
    move_targets_to_device,
    parse_class_names,
    plot_losses,
    save_json,
    set_seed,
)


def build_loaders(args, class_names: list[str]):
    pin_memory = args.device.startswith("cuda")
    persistent_workers = args.num_workers > 0
    train_loader = None
    if getattr(args, "train_images", None):
        train_set = CocoDetectionSubset(
            args.train_images,
            args.train_annotations,
            class_names=class_names,
            max_size=args.max_size,
            train=True,
            hflip_prob=args.hflip_prob,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    val_set = CocoDetectionSubset(
        args.val_images,
        args.val_annotations,
        class_names=class_names,
        max_size=args.max_size,
        train=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def build_model(args, class_names: list[str], device: torch.device):
    model = HFDetr(
        num_classes=len(class_names),
        class_names=class_names,
        pretrained_model=args.pretrained_model,
        local_pretrained_dir=getattr(args, "local_pretrained_dir", None),
    )
    return model.to(device)


def build_optimizer(model, args):
    backbone_params, other_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )


def model_config_from_args(args, class_names: list[str]):
    return {
        "pretrained_model": args.pretrained_model,
        "num_classes": len(class_names),
    }


def data_config_from_args(args, class_names: list[str]):
    return {
        "class_names": class_names,
        "max_size": args.max_size,
        "train_annotations": getattr(args, "train_annotations", None),
        "val_annotations": args.val_annotations,
    }


def train_config_from_args(args):
    return {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "lr_drop": args.lr_drop,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "metric_backend": args.metric_backend,
        "metric_score_threshold": args.metric_score_threshold,
        "hflip_prob": args.hflip_prob,
    }


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    model_config: dict,
    data_config: dict,
    train_config: dict,
    best_map50: float,
):
    path = Path(path)
    ensure_dir(path.parent)
    checkpoint_model_config = dict(model_config)
    hf_model = getattr(model, "model", None)
    if hf_model is not None and hasattr(hf_model, "save_pretrained"):
        hf_dir = path.with_suffix("")
        hf_dir = hf_dir.parent / f"{hf_dir.name}_hf"
        hf_model.save_pretrained(hf_dir)
        checkpoint_model_config["local_pretrained_dir"] = str(hf_dir)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "model_config": checkpoint_model_config,
            "data_config": data_config,
            "train_config": train_config,
            "class_names": data_config["class_names"],
            "best_map50": best_map50,
        },
        path,
    )


def apply_checkpoint_config(args, checkpoint: dict, for_train: bool = False):
    model_config = checkpoint.get("model_config", {})
    data_config = checkpoint.get("data_config", {})
    train_config = checkpoint.get("train_config", {})

    for key, value in model_config.items():
        setattr(args, key, value)
    if "max_size" in data_config:
        args.max_size = data_config["max_size"]
    if for_train:
        for key in ("lr", "lr_backbone", "lr_drop", "weight_decay", "batch_size", "seed", "hflip_prob"):
            if key in train_config:
                setattr(args, key, train_config[key])
    return data_config.get("class_names", checkpoint.get("class_names"))


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def train_one_epoch(model, loader, optimizer, device, epoch: int, writer, profile_dir: Path | None, step_base: int):
    model.train()
    running, steps = {}, 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)

    def run_step(samples, targets):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(samples, targets)
        losses = loss_from_outputs(outputs)
        loss = losses["loss_total"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        return losses

    for step, (samples, targets) in enumerate(progress):
        samples = {key: value.to(device) for key, value in samples.items()}
        targets = move_targets_to_device(targets, device)

        if profile_dir is not None and epoch == 0 and step == 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            ) as prof:
                losses = run_step(samples, targets)
                prof.step()
        else:
            losses = run_step(samples, targets)

        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())
        steps += 1
        global_step = step_base + step
        for key, value in losses.items():
            writer.add_scalar(f"train/{key}", float(value.detach().cpu()), global_step)
        progress.set_postfix({key: f"{value / steps:.4f}" for key, value in running.items()})
    return {key: value / max(steps, 1) for key, value in running.items()}, steps


def cmd_train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    class_names = parse_class_names(args.class_names)
    start_epoch = 0
    best_map50 = -1.0
    checkpoint = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        restored_classes = apply_checkpoint_config(args, checkpoint, for_train=True)
        if restored_classes:
            class_names = restored_classes
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_map50 = float(checkpoint.get("best_map50", checkpoint.get("metrics", {}).get("mAP50", -1.0)))

    train_loader, val_loader = build_loaders(args, class_names)
    model = build_model(args, class_names, device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    if args.resume:
        load_checkpoint(args.resume, model, optimizer, scheduler, device)

    out = Path(args.output_dir)
    writer = make_summary_writer(out / "runs" / "detr_resnet50_coco10")
    checkpoint_dir = ensure_dir(out / "checkpoints")
    profile_dir = ensure_dir(out / "profiler_traces") if args.profile else None
    metrics_csv = out / "reports" / "metrics.csv"
    if metrics_csv.exists() and start_epoch == 0 and not args.append_metrics:
        metrics_csv.unlink()

    model_config = model_config_from_args(args, class_names)
    data_config = data_config_from_args(args, class_names)
    train_config = train_config_from_args(args)
    step_base = 0

    for epoch in range(start_epoch, args.epochs):
        train_iter = islice(train_loader, args.limit_train_batches) if args.limit_train_batches else train_loader
        train_metrics, epoch_steps = train_one_epoch(model, train_iter, optimizer, device, epoch, writer, profile_dir, step_base)
        step_base += epoch_steps
        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
            device,
            len(class_names),
            metric_score_threshold=args.metric_score_threshold,
            top_k=args.top_k,
            metric_backend=args.metric_backend,
            annotation_file=args.val_annotations,
            class_names=class_names,
            limit_batches=args.limit_val_batches,
        )
        scheduler.step()

        for key, value in train_metrics.items():
            writer.add_scalar(f"epoch_train/{key}", value, epoch)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"epoch_val/{key}", value, epoch)

        row = {
            "epoch": epoch,
            "train_loss_total": train_metrics.get("loss_total", 0.0),
            "train_loss_ce": train_metrics.get("loss_ce", 0.0),
            "train_loss_bbox": train_metrics.get("loss_bbox", 0.0),
            "train_loss_giou": train_metrics.get("loss_giou", 0.0),
            "mAP": val_metrics.get("mAP", 0.0),
            "mAP50": val_metrics.get("mAP50", 0.0),
            "metric_backend": val_metrics.get("metric_backend", args.metric_backend),
            "metric_score_threshold": val_metrics.get("metric_score_threshold", args.metric_score_threshold),
            "val_loss_total": val_metrics.get("val_loss_total", 0.0),
            "val_loss_ce": val_metrics.get("val_loss_ce", 0.0),
            "val_loss_bbox": val_metrics.get("val_loss_bbox", 0.0),
            "val_loss_giou": val_metrics.get("val_loss_giou", 0.0),
        }
        append_metrics_row(metrics_csv, row)
        current_best = max(best_map50, row["mAP50"])
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, val_metrics, model_config, data_config, train_config, current_best)
        if row["mAP50"] > best_map50:
            best_map50 = row["mAP50"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, val_metrics, model_config, data_config, train_config, best_map50)
        print(row)
    writer.close()


@torch.no_grad()
def cmd_eval(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = apply_checkpoint_config(args, checkpoint) or parse_class_names(args.class_names)
    _, val_loader = build_loaders(args, class_names)
    model = build_model(args, class_names, device)
    load_checkpoint(args.checkpoint, model, device=device)
    metrics, predictions, ground_truths = evaluate_model(
        model,
        val_loader,
        device,
        len(class_names),
        metric_score_threshold=args.metric_score_threshold,
        top_k=args.top_k,
        metric_backend=args.metric_backend,
        annotation_file=args.val_annotations,
        class_names=class_names,
        limit_batches=args.limit_val_batches,
    )
    save_json(args.output, metrics)
    save_json(args.predictions, {"predictions": predictions, "ground_truths": ground_truths})
    print(metrics)


@torch.no_grad()
def cmd_errors(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = apply_checkpoint_config(args, checkpoint) or parse_class_names(args.class_names)
    _, val_loader = build_loaders(args, class_names)
    model = build_model(args, class_names, device)
    load_checkpoint(args.checkpoint, model, device=device)
    metrics, predictions, ground_truths = evaluate_model(
        model,
        val_loader,
        device,
        len(class_names),
        metric_score_threshold=args.metric_score_threshold,
        top_k=args.top_k,
        metric_backend=args.metric_backend,
        annotation_file=args.val_annotations,
        class_names=class_names,
        limit_batches=args.limit_val_batches,
    )
    error_predictions = [pred for pred in predictions if pred["score"] >= args.error_score_threshold]
    errors = analyze_errors(error_predictions, ground_truths)
    save_json(
        args.output,
        {
            "metrics": metrics,
            "thresholds": {
                "metric_score_threshold": args.metric_score_threshold,
                "error_score_threshold": args.error_score_threshold,
                "visual_score_threshold": args.visual_score_threshold,
            },
            "errors": errors,
        },
    )

    visual_dir = ensure_dir(args.visual_dir)
    saved = 0
    model.eval()
    for samples, targets in val_loader:
        samples_device = {key: value.to(device) for key, value in samples.items()}
        targets_device = move_targets_to_device(targets, device)
        outputs = model(samples_device)
        preds, gts = batch_to_predictions(outputs, targets_device, args.visual_score_threshold, top_k=20)
        for batch_idx, target in enumerate(targets):
            image_id = int(target["image_id"])
            draw_detections(
                samples["images"][batch_idx],
                target["size"],
                [gt for gt in gts if int(gt["image_id"]) == image_id],
                [pred for pred in preds if int(pred["image_id"]) == image_id],
                class_names,
                visual_dir / f"{image_id}.png",
            )
            saved += 1
            if saved >= args.max_visuals:
                print(errors["counts"])
                return
    print(errors["counts"])


def add_common_data_args(parser):
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-annotations", "--val-ann", required=True, dest="val_annotations")
    parser.add_argument("--class-names", default="person,bicycle,car,motorcycle,bus,train,truck,traffic light,stop sign,dog")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def add_model_args(parser):
    parser.add_argument("--pretrained-model", default="facebook/detr-resnet-50")


def main():
    parser = argparse.ArgumentParser(description="DETR fine-tuning homework runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--train-images", required=True)
    train_p.add_argument("--train-annotations", "--train-ann", required=True, dest="train_annotations")
    add_common_data_args(train_p)
    add_model_args(train_p)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--lr-backbone", type=float, default=1e-5)
    train_p.add_argument("--lr-drop", type=int, default=10)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--hflip-prob", type=float, default=0.5)
    train_p.add_argument("--output-dir", default=".")
    train_p.add_argument("--resume", default=None)
    train_p.add_argument("--profile", action="store_true")
    train_p.add_argument("--limit-train-batches", type=int, default=None)
    train_p.add_argument("--limit-val-batches", type=int, default=None)
    train_p.add_argument("--append-metrics", action="store_true")
    train_p.add_argument("--metric-backend", choices=["simple", "coco"], default="coco")
    train_p.add_argument("--metric-score-threshold", type=float, default=0.0)
    train_p.add_argument("--top-k", type=int, default=100)
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval")
    add_common_data_args(eval_p)
    add_model_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output", default="reports/eval_metrics.json")
    eval_p.add_argument("--predictions", default="outputs/predictions.json")
    eval_p.add_argument("--limit-val-batches", type=int, default=None)
    eval_p.add_argument("--metric-score-threshold", type=float, default=0.0)
    eval_p.add_argument("--metric-backend", choices=["simple", "coco"], default="coco")
    eval_p.add_argument("--top-k", type=int, default=100)
    eval_p.set_defaults(func=cmd_eval)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--metrics", default="reports/metrics.csv")
    plot_p.add_argument("--output", default="outputs/plots/losses.png")
    plot_p.set_defaults(func=lambda args: plot_losses(args.metrics, args.output))

    errors_p = sub.add_parser("errors")
    add_common_data_args(errors_p)
    add_model_args(errors_p)
    errors_p.add_argument("--checkpoint", required=True)
    errors_p.add_argument("--output", default="outputs/error_analysis/errors.json")
    errors_p.add_argument("--visual-dir", default="outputs/visualizations")
    errors_p.add_argument("--max-visuals", type=int, default=16)
    errors_p.add_argument("--limit-val-batches", type=int, default=None)
    errors_p.add_argument("--metric-score-threshold", type=float, default=0.0)
    errors_p.add_argument("--metric-backend", choices=["simple", "coco"], default="coco")
    errors_p.add_argument("--error-score-threshold", type=float, default=0.3)
    errors_p.add_argument("--visual-score-threshold", type=float, default=0.5)
    errors_p.add_argument("--top-k", type=int, default=100)
    errors_p.set_defaults(func=cmd_errors)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

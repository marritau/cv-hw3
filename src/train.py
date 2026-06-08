import argparse
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.minidetr import (
    CocoDetectionSubset,
    HFDetr,
    HungarianMatcher,
    MiniDETR,
    SetCriterion,
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
    train_loader = None
    if getattr(args, "train_images", None):
        train_set = CocoDetectionSubset(
            args.train_images,
            args.train_annotations,
            class_names=class_names,
            max_size=args.max_size,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    val_set = CocoDetectionSubset(
        args.val_images,
        args.val_annotations,
        class_names=class_names,
        max_size=args.max_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model_and_loss(args, class_names: list[str], device: torch.device):
    if args.model_type == "hf-detr":
        model = HFDetr(
            num_classes=len(class_names),
            class_names=class_names,
            pretrained_model=args.pretrained_model,
            hf_config=getattr(args, "hf_config", None),
        ).to(device)
        criterion = None
    else:
        model = MiniDETR(
            num_classes=len(class_names),
            num_queries=args.num_queries,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
        ).to(device)
        criterion = SetCriterion(
            num_classes=len(class_names),
            matcher=HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou),
            eos_coef=args.eos_coef,
        ).to(device)
    return model, criterion


def model_config_from_args(args, class_names: list[str]):
    return {
        "model_type": args.model_type,
        "pretrained_model": args.pretrained_model,
        "num_classes": len(class_names),
        "num_queries": args.num_queries,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
        "cost_class": args.cost_class,
        "cost_bbox": args.cost_bbox,
        "cost_giou": args.cost_giou,
        "eos_coef": args.eos_coef,
    }


def apply_model_config(args, config: dict):
    for key, value in config.items():
        setattr(args, key, value)


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict, model_config: dict, class_names: list[str], best_map50: float):
    ensure_dir(Path(path).parent)
    checkpoint_model_config = dict(model_config)
    hf_model = getattr(model, "model", None)
    if hf_model is not None and hasattr(hf_model, "config"):
        checkpoint_model_config["hf_config"] = hf_model.config.to_dict()
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "model_config": checkpoint_model_config,
            "class_names": class_names,
            "best_map50": best_map50,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_one_epoch(model, criterion, loader, optimizer, device, epoch: int, writer, profile_dir: Path | None):
    model.train()
    running, steps = {}, 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    loader_len = len(loader) if hasattr(loader, "__len__") else 1

    def run_step(samples, targets):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(samples, targets) if criterion is None else model(samples)
        losses = loss_from_outputs(outputs, targets, criterion)
        losses["loss_total"].backward()
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
        global_step = epoch * loader_len + step
        for key, value in losses.items():
            writer.add_scalar(f"train/{key}", float(value.detach().cpu()), global_step)
        progress.set_postfix({key: f"{value / steps:.4f}" for key, value in running.items()})
    return {key: value / max(steps, 1) for key, value in running.items()}


def cmd_train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    class_names = parse_class_names(args.class_names)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        class_names = resume_checkpoint.get("class_names", class_names)
        if "model_config" in resume_checkpoint:
            apply_model_config(args, resume_checkpoint["model_config"])

    train_loader, val_loader = build_loaders(args, class_names)
    model, criterion = build_model_and_loss(args, class_names, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_map50 = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_map50 = float(checkpoint.get("best_map50", checkpoint.get("metrics", {}).get("mAP50", -1.0)))

    out = Path(args.output_dir)
    run_name = "detr_resnet50_coco10" if args.model_type == "hf-detr" else "minidetr_coco10"
    writer = make_summary_writer(out / "runs" / run_name)
    checkpoint_dir = ensure_dir(out / "checkpoints")
    profile_dir = ensure_dir(out / "profiler_traces") if args.profile else None
    metrics_csv = out / "reports" / "metrics.csv"
    if metrics_csv.exists() and start_epoch == 0 and not args.append_metrics:
        metrics_csv.unlink()
    model_config = model_config_from_args(args, class_names)

    for epoch in range(start_epoch, args.epochs):
        train_iter = islice(train_loader, args.limit_train_batches) if args.limit_train_batches else train_loader
        train_metrics = train_one_epoch(model, criterion, train_iter, optimizer, device, epoch, writer, profile_dir)
        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            len(class_names),
            args.score_threshold,
            args.top_k,
            metric_backend=args.metric_backend,
            annotation_file=args.val_annotations,
            class_names=class_names,
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f"epoch_train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"epoch_val/{key}", value, epoch)

        row = {
            "epoch": epoch,
            "train_loss_total": train_metrics.get("loss_total", 0.0),
            "train_loss_ce": train_metrics.get("loss_ce", 0.0),
            "train_loss_bbox": train_metrics.get("loss_bbox", 0.0),
            "train_loss_giou": train_metrics.get("loss_giou", 0.0),
            "mAP": val_metrics.get("mAP", 0.0),
            "mAP50": val_metrics.get("mAP50", 0.0),
            "val_loss_total": val_metrics.get("val_loss_total", 0.0),
            "val_loss_ce": val_metrics.get("val_loss_ce", 0.0),
            "val_loss_bbox": val_metrics.get("val_loss_bbox", 0.0),
            "val_loss_giou": val_metrics.get("val_loss_giou", 0.0),
        }
        append_metrics_row(metrics_csv, row)
        current_best = max(best_map50, row["mAP50"])
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, val_metrics, model_config, class_names, current_best)
        if row["mAP50"] > best_map50:
            best_map50 = row["mAP50"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, val_metrics, model_config, class_names, best_map50)
        print(row)
    writer.close()


@torch.no_grad()
def cmd_eval(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint.get("class_names", parse_class_names(args.class_names))
    if "model_config" in checkpoint:
        apply_model_config(args, checkpoint["model_config"])
    _, val_loader = build_loaders(args, class_names)
    model, criterion = build_model_and_loss(args, class_names, device)
    load_checkpoint(args.checkpoint, model, device=device)
    metrics, predictions, ground_truths = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        len(class_names),
        args.score_threshold,
        args.top_k,
        metric_backend=args.metric_backend,
        annotation_file=args.val_annotations,
        class_names=class_names,
    )
    save_json(args.output, metrics)
    save_json(args.predictions, {"predictions": predictions, "ground_truths": ground_truths})
    print(metrics)


@torch.no_grad()
def cmd_errors(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint.get("class_names", parse_class_names(args.class_names))
    if "model_config" in checkpoint:
        apply_model_config(args, checkpoint["model_config"])
    _, val_loader = build_loaders(args, class_names)
    model, criterion = build_model_and_loss(args, class_names, device)
    load_checkpoint(args.checkpoint, model, device=device)
    metrics, predictions, ground_truths = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        len(class_names),
        args.score_threshold,
        args.top_k,
        metric_backend=args.metric_backend,
        annotation_file=args.val_annotations,
        class_names=class_names,
    )
    errors = analyze_errors(predictions, ground_truths)
    save_json(args.output, {"metrics": metrics, "errors": errors})

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
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--class-names", default="person,bicycle,car,motorcycle,bus,train,truck,traffic light,stop sign,dog")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)


def add_model_args(parser):
    parser.add_argument("--model-type", choices=["hf-detr", "mini"], default="hf-detr")
    parser.add_argument("--pretrained-model", default="facebook/detr-resnet-50")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cost-class", type=float, default=1.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-giou", type=float, default=2.0)
    parser.add_argument("--eos-coef", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Minimal DETR homework runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--train-images", required=True)
    train_p.add_argument("--train-annotations", required=True)
    add_common_data_args(train_p)
    add_model_args(train_p)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--output-dir", default=".")
    train_p.add_argument("--resume", default=None)
    train_p.add_argument("--profile", action="store_true")
    train_p.add_argument("--limit-train-batches", type=int, default=None)
    train_p.add_argument("--append-metrics", action="store_true")
    train_p.add_argument("--metric-backend", choices=["simple", "coco"], default="coco")
    train_p.add_argument("--score-threshold", type=float, default=0.05)
    train_p.add_argument("--top-k", type=int, default=100)
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval")
    add_common_data_args(eval_p)
    add_model_args(eval_p)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--output", default="reports/eval_metrics.json")
    eval_p.add_argument("--predictions", default="outputs/predictions.json")
    eval_p.add_argument("--score-threshold", type=float, default=0.05)
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
    errors_p.add_argument("--score-threshold", type=float, default=0.05)
    errors_p.add_argument("--metric-backend", choices=["simple", "coco"], default="coco")
    errors_p.add_argument("--visual-score-threshold", type=float, default=0.3)
    errors_p.add_argument("--top-k", type=int, default=100)
    errors_p.set_defaults(func=cmd_errors)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

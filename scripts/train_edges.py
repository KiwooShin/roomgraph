"""Train structural edges with AMP, TensorBoard, checkpointing, and held-out validation."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from roomgraph.learning.data import EdgeDataset
from roomgraph.learning.model import EdgeNet, edge_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/headcam_v1.json"))
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Diagnostic: fit and validate the same eight training images",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.overfit:
        config.update(
            run_dir=config["run_dir"] + "_overfit",
            epochs=100,
            batch_size=8,
            patience=100,
            eval_every=5,
        )
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = config["amp"] and device.type == "cuda"
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train = EdgeDataset(config["dataset"], "train", limit=8 if args.overfit else None)
    val = train if args.overfit else EdgeDataset(config["dataset"], "val")
    loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
    )
    validation = DataLoader(
        val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = EdgeNet(pretrained=config["pretrained"]).to(device, memory_format=torch.channels_last)
    encoder = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": config["learning_rate"] * 0.2}, {"params": decoder}],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["epochs"], eta_min=1e-6
    )
    best = float("inf")
    start = 0
    best_epoch = 0
    history = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, start, best_epoch = state["best"], state["epoch"] + 1, state["best_epoch"]
        history = state["history"]
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    writer = SummaryWriter(str(output / "tensorboard"))
    provenance = {
        "config": config,
        "diagnostic_overfit": args.overfit,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset_sha256": hashlib.sha256(
            (Path(config["dataset"]) / "index.json").read_bytes()
        ).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    writer.add_text("configuration", json.dumps(provenance, indent=2))
    beginning = time.perf_counter()
    for epoch in range(start, config["epochs"]):
        model.train()
        total = 0.0
        seen = 0
        started = time.perf_counter()
        for image, target in loader:
            image = (
                image.to(device, non_blocking=True)
                .float()
                .div_(255)
                .contiguous(memory_format=torch.channels_last)
            )
            target = target.to(device, non_blocking=True).float()
            if not args.overfit:
                if random.random() < 0.5:
                    image = image.flip(-1)
                    target = target.flip(-1)
                gain = torch.empty((len(image), 3, 1, 1), device=device).uniform_(0.85, 1.15)
                exposure = torch.empty((len(image), 1, 1, 1), device=device).uniform_(0.8, 1.2)
                image = (image * gain * exposure + torch.randn_like(image) * 0.006).clamp_(0, 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(image)
                loss = edge_loss(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(image)
            seen += len(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": total / seen,
            "seconds": duration,
            "images_per_second": seen / duration,
            "lr": optimizer.param_groups[1]["lr"],
        }
        if epoch % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            model.eval()
            value = 0.0
            count = 0
            with torch.inference_mode():
                for image, target in validation:
                    image = (
                        image.to(device)
                        .float()
                        .div_(255)
                        .contiguous(memory_format=torch.channels_last)
                    )
                    target = target.to(device).float()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        logits = model(image)
                        loss = edge_loss(logits, target)
                    value += loss.item() * len(image)
                    count += len(image)
            row["val_loss"] = value / count
            if row["val_loss"] < best:
                best = row["val_loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": config,
                        "provenance": provenance,
                    },
                    output / "best.pt",
                )
            probability = logits.float().sigmoid()
            preview = image[0].clone()
            mask = probability[0, 0] > 0.5
            preview[:, mask] = torch.tensor([0.2, 1, 0.75], device=device).view(3, 1)
            writer.add_image("validation/visible_prediction", preview, epoch)
            writer.add_image("validation/visible_target", target[0, :1], epoch)
        scheduler.step()
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        if device.type == "cuda":
            writer.add_scalar(
                "memory/peak_allocated_mb", torch.cuda.max_memory_allocated() / 2**20, epoch
            )
        writer.flush()
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best": best,
                "best_epoch": best_epoch,
                "history": history,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            },
            output / "last.pt",
        )
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= config["patience"]:
            print(f"Early stop; best epoch {best_epoch}", flush=True)
            break
    provenance.update(
        best_epoch=best_epoch,
        best_val_loss=best,
        total_seconds=time.perf_counter() - beginning,
        epochs_completed=len(history),
    )
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    writer.close()
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()

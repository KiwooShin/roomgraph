"""Train structural edges with AMP, TensorBoard, and resumable atomic checkpoints."""

import argparse
import hashlib
import json
import os
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
from roomgraph.learning.training import (
    EpochRandomSampler,
    atomic_torch_save,
    atomic_write_json,
    recover_best_checkpoint,
)


def check_resume(state, config, dataset_sha256, diagnostic):
    """Reject changes that silently turn a continuation into a different experiment."""
    if "provenance" not in state or "config" not in state:
        raise ValueError("Legacy checkpoint has no resume provenance; start a new run directory")
    runtime_keys = {"run_dir", "workers", "checkpoint_every"}
    previous = {k: v for k, v in state["config"].items() if k not in runtime_keys}
    current = {k: v for k, v in config.items() if k not in runtime_keys}
    if previous != current:
        raise ValueError("Resume configuration differs from the saved training experiment")
    if state["provenance"]["dataset_sha256"] != dataset_sha256:
        raise ValueError("Resume dataset identity differs from the saved experiment")
    if state["provenance"]["diagnostic_overfit"] != diagnostic:
        raise ValueError("Cannot resume between diagnostic and held-out training")


def prepare_batch(image, target, device, augment):
    image = (
        image.to(device, non_blocking=True)
        .float()
        .div_(255)
        .contiguous(memory_format=torch.channels_last)
    )
    target = target.to(device, non_blocking=True).float()
    if augment:
        if random.random() < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
        gain = torch.empty((len(image), 3, 1, 1), device=device).uniform_(0.85, 1.15)
        exposure = torch.empty((len(image), 1, 1, 1), device=device).uniform_(0.8, 1.2)
        image = (image * gain * exposure + torch.randn_like(image) * 0.006).clamp_(0, 1)
    return image, target


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/headcam_v1.json"))
    parser.add_argument("--run-dir", type=Path, help="Override output without changing the recipe")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Diagnostic: fit and validate the same eight training images",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.setdefault("compile_loss", False)
    config.setdefault("compile_model", False)
    config.setdefault("checkpoint_every", 1)
    if args.run_dir:
        config["run_dir"] = str(args.run_dir)
    if args.overfit:
        config.update(
            run_dir=config["run_dir"] + "_overfit",
            epochs=100,
            batch_size=8,
            patience=100,
            eval_every=5,
        )
    for key in ["epochs", "batch_size", "eval_every", "checkpoint_every", "patience"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["workers"] < 0:
        raise ValueError("workers must be nonnegative")
    output = Path(config["run_dir"])
    dataset_sha256 = hashlib.sha256(
        (Path(config["dataset"]) / "index.json").read_bytes()
    ).hexdigest()
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume(state, config, dataset_sha256, args.overfit)
        # Keep the best checkpoint and event history in the same experiment directory.
        if args.resume.resolve().parent != output.resolve():
            raise ValueError("Resume in the original run directory so best.pt remains available")
        if recover_best_checkpoint(state, output):
            print("Recovered best.pt from committed last.pt:", output, flush=True)
        if state["epoch"] + 1 >= config["epochs"] or state.get("early_stopped", False):
            print("Training already complete:", output, flush=True)
            return
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory is not empty: {output}; use --resume or --run-dir")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = config["amp"] and device.type == "cuda"
    output.mkdir(parents=True, exist_ok=True)
    train = EdgeDataset(config["dataset"], "train", limit=8 if args.overfit else None)
    val = train if args.overfit else EdgeDataset(config["dataset"], "val")
    sampler = EpochRandomSampler(train, seed=seed)
    loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    validation = DataLoader(
        val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed + 2),
    )
    model = EdgeNet(pretrained=config["pretrained"] and state is None).to(
        device, memory_format=torch.channels_last
    )
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
    loss_function = torch.compile(edge_loss) if config["compile_loss"] else edge_loss
    forward = torch.compile(model) if config["compile_model"] else model
    best, start, best_epoch = float("inf"), 0, 0
    history = []
    previous_seconds = 0.0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, start, best_epoch = state["best"], state["epoch"] + 1, state["best_epoch"]
        history = state["history"]
        previous_seconds = state.get("session_seconds", 0.0)
        torch.set_rng_state(state["rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=start if state else None)
    provenance = {
        "config": config,
        "diagnostic_overfit": args.overfit,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset_sha256": dataset_sha256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "sampling": "epoch-seeded private CPU generator; worker RNG isolated",
        "timing_scope": "process_seconds starts at main() (excludes Python imports)",
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    if state:
        provenance["resumed_from"] = {"epoch": state["epoch"], "provenance": state["provenance"]}
    atomic_write_json(provenance, output / "run.json")
    writer.add_text("configuration", json.dumps(provenance, indent=2))
    beginning = time.perf_counter()
    for epoch in range(start, config["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        total = torch.zeros((), device=device)
        seen = 0
        started = time.perf_counter()
        for image, target in loader:
            image, target = prepare_batch(image, target, device, augment=not args.overfit)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = forward(image)
                loss = loss_function(logits, target)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # One synchronization checks both loss and gradients before changing weights.
            if not (torch.isfinite(loss.detach()) & torch.isfinite(norm)):
                raise RuntimeError(f"Non-finite loss or gradient at epoch {epoch}, sample {seen}")
            optimizer.step()
            total += loss.detach() * len(image)
            seen += len(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": total.item() / seen,
            "seconds": duration,
            "images_per_second": seen / duration,
            "train_images": seen,
            "optimizer_steps": (epoch + 1) * len(loader),
            "lr": optimizer.param_groups[1]["lr"],
        }
        improved = False
        validation_started = time.perf_counter()
        if epoch % config["eval_every"] == 0 or epoch == config["epochs"] - 1:
            model.eval()
            value = torch.zeros((), device=device)
            count = 0
            with torch.inference_mode():
                for image, target in validation:
                    image, target = prepare_batch(image, target, device, augment=False)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        logits = model(image)
                        # Eager validation avoids a second compilation for inference-only graphs.
                        loss = edge_loss(logits, target)
                    value += loss * len(image)
                    count += len(image)
            row["val_loss"] = value.item() / count
            if not np.isfinite(row["val_loss"]):
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
            if row["val_loss"] < best:
                best, best_epoch = row["val_loss"], epoch
                improved = True
            probability = logits.float().sigmoid()
            preview = image[0].clone()
            mask = probability[0, 0] > 0.5
            preview[:, mask] = torch.tensor([0.2, 1, 0.75], device=device).view(3, 1)
            writer.add_image("validation/visible_prediction", preview, epoch)
            writer.add_image("validation/visible_target", target[0, :1], epoch)
        row["validation_seconds"] = time.perf_counter() - validation_started
        scheduler.step()
        early_stopped = epoch - best_epoch >= config["patience"]
        final_epoch = epoch == config["epochs"] - 1 or early_stopped
        # The checkpoint records timing up to serialization; history.json includes its write time.
        row["elapsed_seconds"] = previous_seconds + time.perf_counter() - beginning
        history.append(row)
        checkpoint_started = time.perf_counter()
        if improved or (epoch + 1) % config["checkpoint_every"] == 0 or final_epoch:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": config,
                    "provenance": provenance,
                    "early_stopped": early_stopped,
                    "session_seconds": row["elapsed_seconds"],
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                },
                output / "last.pt",
            )
        if improved:
            # Commit the resumable state first; recovery can recreate best.pt after a crash.
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "provenance": provenance,
                },
                output / "best.pt",
            )
        row["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        row["elapsed_seconds"] = previous_seconds + time.perf_counter() - beginning
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        if device.type == "cuda":
            writer.add_scalar(
                "memory/peak_allocated_mb", torch.cuda.max_memory_allocated() / 2**20, epoch
            )
        atomic_write_json(history, output / "history.json")
        print(json.dumps(row), flush=True)
        if early_stopped:
            print(f"Early stop; best epoch {best_epoch}", flush=True)
            break
    writer.close()
    provenance.update(
        best_epoch=best_epoch,
        best_val_loss=best,
        total_seconds=previous_seconds + time.perf_counter() - beginning,
        process_seconds=time.perf_counter() - process_started,
        epochs_completed=len(history),
        optimizer_steps=history[-1]["optimizer_steps"],
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
    )
    atomic_write_json(provenance, output / "run.json")
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()

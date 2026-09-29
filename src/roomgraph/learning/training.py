"""Deterministic epoch sampling and interruption-safe training output writes."""

import json
import os
import pickle
import tempfile
from collections.abc import Callable, Iterator, Sized
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler[int]):
    """Shuffle reproducibly from epoch identity without consuming global RNG state.

    Call ``set_epoch`` before iterating the training loader. Give the loader its
    own ``torch.Generator`` as well so worker seeding cannot affect model RNG.
    This supports exact sample-order restoration at epoch boundaries; it does
    not attempt to restore a partially consumed epoch.
    """

    def __init__(self, data_source: Sized, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def _atomic_write(path: str | Path, write: Callable[[BinaryIO], object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Replace a checkpoint only after its complete serialization succeeds."""

    _atomic_write(path, lambda stream: torch.save(payload, stream))


def atomic_write_json(payload: Any, path: str | Path) -> None:
    """Write strict JSON atomically, retaining an existing file on failure."""

    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def recover_best_checkpoint(state: dict[str, Any], output: str | Path) -> bool:
    """Reconcile best.pt after a committed last.pt update interrupted publication.

    The trainer commits the resume checkpoint first on each validation
    improvement, then publishes best.pt. Recovery is possible only when that
    resume checkpoint itself contains the best model. Return whether recovery
    was necessary; refuse to silently substitute a different epoch's weights.
    """

    path = Path(output) / "best.pt"
    best_state = None
    if path.is_file():
        try:
            best_state = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
            # A valid committed last.pt can repair a damaged publication too.
            pass
    if (
        isinstance(best_state, dict)
        and "model" in best_state
        and best_state.get("epoch") == state["best_epoch"]
    ):
        actual_sha = best_state.get("provenance", {}).get("dataset_sha256")
        expected_sha = state.get("provenance", {}).get("dataset_sha256")
        if actual_sha is None or expected_sha is None or actual_sha == expected_sha:
            return False
    if state["best_epoch"] != state["epoch"]:
        raise ValueError(
            "Cannot recover best.pt: it is missing or mismatched, and the resume "
            f"checkpoint epoch {state['epoch']} is not the best epoch {state['best_epoch']}"
        )
    atomic_torch_save(
        {
            "model": state["model"],
            "epoch": state["best_epoch"],
            "config": state["config"],
            "provenance": state["provenance"],
        },
        path,
    )
    return True

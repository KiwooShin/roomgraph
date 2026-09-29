"""Append frozen-checkpoint evaluation metrics to the local TensorBoard run."""

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def log_evaluation(run_directory, metrics, reconstructions):
    step = metrics["checkpoint_epoch"]
    writer = SummaryWriter(str(run_directory / "tensorboard"))
    for split in ["train_subset", "validation", "test"]:
        for kind in ["visible", "amodal"]:
            for name, value in metrics[split][kind].items():
                writer.add_scalar(f"evaluation/{split}/{kind}_{name}", value, step)
    writer.add_scalar(
        "evaluation/canny/test_visible_f1", metrics["canny"]["test_visible"]["f1"], step
    )
    for key in [
        "dimension_mae_m",
        "edge_chamfer_m",
        "edge_accuracy_10cm",
        "edge_completeness_10cm",
    ]:
        writer.add_scalar(
            f"reconstruction/test/{key}",
            np.mean([r["metrics"][key] for r in reconstructions]),
            step,
        )
    writer.add_text("evaluation/protocol", metrics["metric"], step)
    writer.flush()
    writer.close()

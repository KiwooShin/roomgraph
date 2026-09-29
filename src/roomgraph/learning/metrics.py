"""Thin-boundary distance metrics; no background-dominated pixel accuracy."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def thin_prediction(probability, threshold):
    return skeletonize(np.asarray(probability) >= threshold)


def boundary_counts(prediction, target, tolerance=2):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    matched_prediction = (
        int(np.count_nonzero(prediction & (distance_transform_edt(~target) <= tolerance)))
        if target.any()
        else 0
    )
    matched_target = (
        int(np.count_nonzero(target & (distance_transform_edt(~prediction) <= tolerance)))
        if prediction.any()
        else 0
    )
    return np.array(
        [matched_prediction, prediction.sum(), matched_target, target.sum()], dtype=np.int64
    )


def summarize_counts(counts):
    a, b, c, d = map(float, counts)
    precision = a / b if b else 0.0
    recall = c / d if d else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_boundaries(probabilities, targets, threshold, tolerance=2):
    def one(pair):
        probability, target = pair
        return np.stack(
            [
                boundary_counts(thin_prediction(p, threshold), t, tolerance)
                for p, t in zip(probability, target, strict=True)
            ]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = np.sum(list(pool.map(one, zip(probabilities, targets, strict=True))), axis=0)
    return {
        "visible": summarize_counts(counts[0]),
        "amodal": summarize_counts(counts[1:].sum(axis=0)),
        "per_channel": [summarize_counts(c) for c in counts],
        "counts": counts.tolist(),
    }

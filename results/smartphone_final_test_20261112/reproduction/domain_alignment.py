"""Differentiable RBF MMD and Deep-CORAL in fixed Source-Train coordinates."""

from __future__ import annotations

import numpy as np


def losses(source, target, reference):
    import tensorflow as tf

    source, target = tf.cast(source, tf.float32), tf.cast(target, tf.float32)
    tf.debugging.assert_rank(source, 2)
    tf.debugging.assert_rank(target, 2)
    tf.debugging.assert_equal(tf.shape(source)[1], tf.shape(target)[1])
    tf.debugging.assert_greater_equal(tf.shape(source)[0], 2)
    tf.debugging.assert_greater_equal(tf.shape(target)[0], 2)
    mean = tf.convert_to_tensor(reference["mean"], dtype=tf.float32)
    std = tf.convert_to_tensor(reference["std"], dtype=tf.float32)
    tf.debugging.assert_positive(std)
    a, b = (source - mean) / std, (target - mean) / std
    bandwidth = tf.convert_to_tensor(reference["bandwidth_sq"], dtype=tf.float32)
    tf.debugging.assert_positive(bandwidth)

    def kernel(x, y):
        distance = tf.maximum(
            0.0,
            tf.reduce_sum(x * x, 1)[:, None]
            + tf.reduce_sum(y * y, 1)[None, :]
            - 2 * tf.matmul(x, y, transpose_b=True),
        )
        kernels = [
            tf.exp(-distance / (2 * bandwidth * float(multiplier)))
            for multiplier in reference.get("bandwidth_multipliers", [1.0])
        ]
        return tf.add_n(kernels) / len(kernels)

    mmd = tf.maximum(
        0.0,
        tf.reduce_mean(kernel(a, a))
        + tf.reduce_mean(kernel(b, b))
        - 2 * tf.reduce_mean(kernel(a, b)),
    )
    ac, bc = a - tf.reduce_mean(a, 0), b - tf.reduce_mean(b, 0)
    na, nb = tf.cast(tf.shape(a)[0] - 1, tf.float32), tf.cast(tf.shape(b)[0] - 1, tf.float32)
    # Gram form is equivalent to covariance Frobenius loss without D x D allocation.
    norm = (
        tf.reduce_sum(tf.square(tf.matmul(ac, ac, transpose_b=True))) / na**2
        + tf.reduce_sum(tf.square(tf.matmul(bc, bc, transpose_b=True))) / nb**2
        - 2 * tf.reduce_sum(tf.square(tf.matmul(ac, bc, transpose_b=True))) / (na * nb)
    )
    dimension = tf.cast(tf.shape(a)[1], tf.float32)
    coral = tf.maximum(0.0, norm) / (4 * dimension**2)
    tf.debugging.assert_all_finite(mmd, "Non-finite MMD loss")
    tf.debugging.assert_all_finite(coral, "Non-finite CORAL loss")
    return mmd, coral


def balanced_schedule(rows, source_batches, seed, epoch):
    """At most one Target image pass per epoch, balanced over the four recorded people."""
    groups = {}
    for index, row in enumerate(rows):
        if row["split"] != "calibration" or row.get("label") is not None:
            raise ValueError("Only unlabeled calibration rows may be Target training inputs")
        groups.setdefault(row["person"], []).append(index)
    if len(groups) != 4:
        raise ValueError("Target sampling must retain the four participant IDs")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x544152]))
    shuffled = [rng.permutation(groups[p]) for p in sorted(groups)]
    count = min(min(map(len, shuffled)), len(source_batches))
    if count == 0:
        return {}
    slots = np.linspace(0, len(source_batches) - 1, count, dtype=int)
    return {
        source_batches[int(slot)]: [int(participant[i]) for participant in shuffled]
        for i, slot in enumerate(slots)
    }

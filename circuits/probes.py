"""Per-(concept, layer) linear probe training.

Paper §7.2: train a logistic-regression probe to discriminate concept
prompts from checker prompts using MLP activations at the last-token
position. Per-fold accuracy is the §7.3 "concept-vs-token distinction
is linearly encoded at every layer" claim; the unit-normalised
weight vector is the §7.3 probe direction used in the Jaccard-cosine
cross-validation.

The probe configuration is fixed (`LogisticRegression(C=1.0, max_iter=1000,
solver='lbfgs')`) — these were the hyperparameters under which the
0.97–1.00 accuracy band and the L20 r = 0.645 result were obtained.
Changing them invalidates the locked numbers.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class ProbeResult(TypedDict):
    accuracy: float
    accuracy_std: float
    weight_vector: np.ndarray
    top10_neurons: list[int]


def train_concept_probe(
    X_object: np.ndarray,
    X_checker: np.ndarray,
    *,
    seed: int = 42,
    cv_folds: int = 5,
    min_samples_per_class: int = 5,
) -> ProbeResult | None:
    """Train a logistic-regression probe on `object vs checker` activations.

    Lazy imports of sklearn so the heavy dependency only loads when this
    function is actually called.

    Parameters
    ----------
    X_object, X_checker
        Per-prompt MLP activation matrices, shape `(n_prompts, mlp_dim)`,
        last-token position. One class each.
    seed
        Forwarded to `LogisticRegression(random_state=...)` and the
        cross-validation splitter.
    cv_folds
        Number of stratified folds for cross-validation; capped at the
        smaller class count.
    min_samples_per_class
        Skip training if either class has fewer rows than this. Returns
        `None` in that case.

    Returns
    -------
    Per-(concept, layer) record `{accuracy, accuracy_std, weight_vector,
    top10_neurons}`, or `None` if one class lacks enough samples to
    train. The weight vector is L2-normalised; `top10_neurons` are the
    indices of the ten largest |weights|.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    assert X_object.ndim == 2 and X_checker.ndim == 2, (
        f"both inputs must be 2-D, got shapes {X_object.shape}, {X_checker.shape}"
    )
    assert X_object.shape[1] == X_checker.shape[1], (
        f"feature dim mismatch: {X_object.shape[1]} vs {X_checker.shape[1]}"
    )

    n_obj, n_chk = len(X_object), len(X_checker)
    if n_obj < min_samples_per_class or n_chk < min_samples_per_class:
        return None

    X = np.vstack([X_object, X_checker]).astype(np.float32)
    y = np.array([1] * n_obj + [0] * n_chk, dtype=np.int64)

    X_scaled = StandardScaler().fit_transform(X)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed,
    )

    n_folds = min(cv_folds, n_obj, n_chk)
    if n_folds >= 2:
        scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")
        accuracy = float(scores.mean())
        accuracy_std = float(scores.std())
    else:
        accuracy = float("nan")
        accuracy_std = float("nan")

    # Refit on all data for the released weight vector.
    clf.fit(X_scaled, y)
    w = clf.coef_[0]
    norm = float(np.linalg.norm(w))
    w_unit = w / (norm + 1e-10)

    top10 = np.argsort(np.abs(w))[::-1][:10].tolist()

    return {
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "weight_vector": w_unit,
        "top10_neurons": top10,
    }

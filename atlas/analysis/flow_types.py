"""Flow-type classification of per-concept circuit-size profiles.

Paper §6.1: each concept's circuit-size-by-layer curve is categorised by
its temporal shape into one of:

  - `two_phase`       — sharp early spike, dip, late re-explosion
                         (Qwen's atomicity-super-cluster signature)
  - `build_and_hold`  — monotone gradual rise to a broad plateau
                         (DeepSeek's atomicity signature)
  - `late_emergence`  — flat first half, peak in the second half
                         (most non-atomicity concepts in both models)
  - `flash`           — single narrow spike that dominates the curve
                         (defined in code but doesn't trigger on the
                         published data — no concept matches)
  - `empty`           — all-zero curve
  - `unclassified`    — doesn't fit any rule above

The classifier is a rule cascade — first matching rule wins. Thresholds
are calibrated to reproduce the §6.1 counts (Python × Qwen: 7 two_phase,
95 late_emergence, 4 unclassified; etc.). These counts are locked by
`tests/test_paper_numbers.py::test_f9_f12_flow_type_counts`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# Thresholds named for the rule they belong to. Calibrated against
# the §6.1 counts on the frozen R/P × QW/DS L14 data.
_HIGH_PLATEAU_RATIO = 0.5          # "wide" plateau means sizes > this × max
_FLASH_HEIGHT_VS_MEAN = 5.0        # peak height must exceed this × mean
_FLASH_MAX_WIDTH = 3               # plateau width at-or-below this is narrow
_LATE_EMERGENCE_FIRST_HALF_RATIO = 0.1   # first half must be quiet to this depth
_TWO_PHASE_PEAK_MIN_HEIGHT = 0.3   # candidate peaks must exceed this × max
_TWO_PHASE_TROUGH_MAX_RATIO = 0.5  # trough between peaks must dip this far
_TWO_PHASE_MIN_PEAK_SEPARATION = 3 # peaks must be ≥ this many layers apart
_BUILD_AND_HOLD_MIN_MONOTONE_RUN = 4   # longest non-decreasing run ≥ this
_BUILD_AND_HOLD_MIN_PLATEAU_DIVISOR = 3  # plateau width ≥ n / this


def classify_flow_type(sizes: Sequence[float]) -> str:
    """Classify a per-layer circuit-size curve by its temporal shape.

    Returns one of `{"two_phase", "build_and_hold", "late_emergence",
    "flash", "empty", "unclassified"}` — see module docstring.

    Pure. Input is a 1-D sequence (length = number of layers).
    """
    arr = np.asarray(sizes, dtype=float)
    assert arr.ndim == 1 and len(arr) >= 4, (
        f"sizes must be a 1-D sequence of length ≥ 4, got shape {arr.shape}"
    )

    max_size = arr.max()
    if max_size == 0:
        return "empty"

    mean_size = arr.mean()
    peak_idx = int(arr.argmax())
    n = len(arr)

    # Width of the high plateau (>50% of max).
    width = int((arr > _HIGH_PLATEAU_RATIO * max_size).sum())

    # Rule 1 — Flash: single narrow spike.
    if max_size > _FLASH_HEIGHT_VS_MEAN * mean_size and width <= _FLASH_MAX_WIDTH:
        return "flash"

    # Rule 2 — Late emergence: quiet first half, peak past midpoint.
    if (arr[:n // 2].max() < _LATE_EMERGENCE_FIRST_HALF_RATIO * max_size
            and peak_idx >= n // 2):
        return "late_emergence"

    # Rule 3 — Two-phase: two peaks separated by a deep trough.
    peaks = [
        i for i in range(1, n - 1)
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
        and arr[i] > _TWO_PHASE_PEAK_MIN_HEIGHT * max_size
    ]
    if len(peaks) >= 2:
        trough = arr[peaks[0]:peaks[-1] + 1].min()
        smaller_peak = min(arr[peaks[0]], arr[peaks[-1]])
        if (trough < _TWO_PHASE_TROUGH_MAX_RATIO * smaller_peak
                and peaks[-1] - peaks[0] >= _TWO_PHASE_MIN_PEAK_SEPARATION):
            return "two_phase"

    # Rule 4 — Build-and-hold: long monotone run + wide plateau.
    longest = current = 0
    for i in range(1, n):
        if arr[i] >= arr[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if (longest >= _BUILD_AND_HOLD_MIN_MONOTONE_RUN
            and width >= n // _BUILD_AND_HOLD_MIN_PLATEAU_DIVISOR):
        return "build_and_hold"

    return "unclassified"


def classify_all_flow_types(
    sizes_by_concept: dict[str, dict[int, int]],
) -> dict[str, str]:
    """Batch-classify every concept in `sizes_by_concept`.

    `sizes_by_concept` is the output of `load_concept_sizes_by_layer` —
    concept → layer → size. Returns concept → flow-type label.
    """
    out: dict[str, str] = {}
    for concept, layer_sizes in sizes_by_concept.items():
        layers = sorted(layer_sizes.keys())
        sizes = [layer_sizes[L] for L in layers]
        out[concept] = classify_flow_type(sizes)
    return out

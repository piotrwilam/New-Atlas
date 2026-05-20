"""Layer 1: artifact-generation pipeline (frozen, GPU-required to rerun).

Module-level re-exports are intentionally omitted because individual
modules pull in heavy optional dependencies:
    - extraction, binarization, marginalization, pipeline, ablation → torch
    - probes → scikit-learn
    - metrics, io_utils → numpy / h5py only

Importing the package shouldn't force the heavy deps. Always import the
specific submodule you need:

    from circuits.extraction import ActivationExtractor
    from circuits.probes import train_concept_probe
    from circuits.ablation import AblationHook
    from circuits.metrics import jaccard_similarity
    from circuits.io_utils import load_atlas_hdf5
"""

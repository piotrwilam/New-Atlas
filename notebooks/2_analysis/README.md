# Layer 2 — Analysis notebooks

These are the **exploratory analysis** notebooks used while writing the
paper. They read the frozen artifacts produced by Layer 1 and compute
statistics, run validations, generate intermediate plots.

For the **canonical, locked-in versions of the paper figures**, use the
scripts in [../../experiments/](../../experiments/) instead — those are what's covered by
the golden-numbers tests in `tests/test_paper_numbers.py`.

These notebooks are kept here for context (they're what the agent
exploration looked like) and for any future re-analysis that doesn't
fit the paper's published figures.

## Notebook map

| Notebook | Paper section | What it explores |
|---|---|---|
| `6_ablation.ipynb` | §7.1 | Single-concept zero-ablation of concept-only neurons; concept-vs-checker dissociation. |
| `6b_ablation_double.ipynb` | §7.1 | Refined double-dissociation test (revised reporting; the 4 cleanly-passing concepts). |
| `7_E3_meta_circuits.ipynb` | §6.2 | Hierarchical clustering of concept-only sets per (lang, model, layer) — produces F5/F13 inputs. |
| `7_E6_layer_dynamics.ipynb` | §6.1 | Flow-type classification (two_phase / build_and_hold / late_emergence). |
| `7_E7_cross_language.ipynb` | §5.3 | Per-(model, equivalence-class) cross-language sharing — produces F3 inputs. |
| `8_E4_wellformedness.ipynb` | (extension) | Well-formedness probes — exploratory, not in the paper. |
| `8_E5_composition.ipynb` | (extension) | Composition tests — exploratory, not in the paper. |
| `9_results_stage1.ipynb` | §5.1 | Per-concept aggregation → `9_results_*.xlsx` (the input to F1, F2). |
| `10_E8_cross_model.ipynb` | §4.2 | Cross-model Spearman ρ on concept fractions (the 0.638 / 0.673 number). |
| `V2_probes.ipynb` | §7.2–§7.3 | Linear-probe training + Jaccard-cosine cross-validation — produces F7 / F8 inputs. |

## A note on outputs

Most notebooks have outputs stripped (nbstripout configured in `.gitattributes`).
A few may still carry embedded outputs from runs prior to the strip hook — these
are kept as historical records.

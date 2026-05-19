# Coding Guidelines and Experiment Protocol

Principle: **a reader who didn't write this should be able to audit it.**
Keep this document short. Enforce it. Add a rule each time a bug would have been caught by one.

---

## Part 1 — Code Guidelines

### Structure

- One package (e.g. `atlas/`), organised by pipeline stage: `extraction/`, `binarisation/`, `decomposition/`, `probes/`, `analysis/`, `plotting/`. No `utils/`, no `helpers/`.
- One script per paper-result in `experiments/`, named after what it produces (`fig3_concept_only_fraction.py`).
- Notebooks for narrative and analysis only. No function defined in a notebook is imported elsewhere.

### Build phase vs final phase

Code goes through two phases:

**Build phase.** Logic may live in notebooks. Composition across stages, inline functions, exploratory work. Acceptable while the abstraction is still settling.

**Refactor phase, after each stage is done.** A stage is "done" when its artifact looks right, you've stopped tweaking, and you can write a one-line contract for each function. At that point:

- Move substantive functions from notebook to the package.
- Add docstring stating the contract.
- Add assertions at the top for the assumptions.
- Write a small synthetic test for anything correctness-critical.
- Update the notebook to import from the package.
- Rerun the notebook end-to-end to confirm nothing broke.

Per-stage refactor is a gate, not an aspiration. Before starting the next paper, all reusable logic from the previous paper is in the package.

**Exception during build phase:** a function goes to the package immediately if (a) it's reused across notebooks, or (b) it implements something where wrong output looks similar to right output. "Substantive" = ≥10 lines of logic OR anything numerical (math, slicing, masking, set ops), regardless of length. Pure plumbing (read → loop → write) can stay in the script.

**Refactor-on-branch.** Any refactor touching > 50 lines or > 2 files goes on a feature branch and merges as a single squash commit. The repo is public; reviewers should see the story of a stage being formalised, not 30 incremental commits.

### Functions

- Docstring with: one-line purpose, inputs (shapes and semantics for tensors), outputs (same), assumptions, side effects.
- Assertions at the top check the assumptions. Shape, dtype, value range, set membership.
- Pure where possible. I/O isolated to thin wrappers at the script level.
- Soft cap at ~60 lines of body (excluding docstring + assertions).
- No mutation of inputs.

### Naming

- Variables name what the thing *is*: `activations` not `arr`.
- Tensor shape in comment or jaxtyping annotation: `Float[Tensor, "prompt position neuron"]`.
- Functions are verbs (`extract_activations`), data are nouns.

### Configs and constants

- No hardcoded numbers in code. Thresholds, layer ranges, model names in config files or named constants.
- No hardcoded paths. Structural paths in `atlas/paths.py` (package root, results dir, paper dir, drive mount). Per-experiment paths in the config.
- Configs use Hydra (structured, with CLI overrides and composition), not flag soup.

### Imports and errors

- Imports at top, grouped: stdlib, third-party, local. No top-level side effects on import.
- Fail loudly. No silent `except: pass`. Crash with a useful message.

### Tests

- Each module has a test file. Pytest. Mirror the package layout: `atlas/extraction/foo.py` → `tests/extraction/test_foo.py`.
- One test per public function with a tiny synthetic input. Shared fixtures in `tests/conftest.py`.
- Tests are runnable documentation.

### Agent-specific rules

- A `CLAUDE.md` at the repo root summarises project quirks and points to this file. Claude Code reads it automatically; no need to paste the guidelines each session.
- Every generated function includes the docstring contract and assertions.
- Ask the agent to explain non-obvious choices in comments.
- After a function is written, ask the agent to write the test that would have caught its most likely failure.

---

## Part 2 — Notebooks

Two types.

**Experiment notebooks.** Compose package functions across stages. Prose explains the why. Sanity checks inline after each stage. End with an artifact-recording cell (output path, config hash, git commit).

**Analysis notebooks.** Read artifacts from disk. Produce figures and tables. Never recompute the expensive thing. One notebook per paper section or figure cluster.

**Notebook cell contract.** Every cell either (a) imports and sets up, (b) calls a package function and assigns the result, or (c) inspects or plots. No cell defines a function that gets reused two cells later — that's the package's job.

Both should pass kernel-restart-and-run-all. Logic stays in the package; narrative stays in the notebook.

---

## Part 3 — Figures

### Plotting functions

- Live in `atlas/plotting/`. Take data, return a matplotlib Figure. No file reads, no file writes.
- Style settings (fonts, colors, dimensions) centralised in `atlas/plotting/style.py`, exposed as an `apply_style(name)` function (`name` ∈ {"paper", "poster", "slides"}). Every figure script calls it once at the top; plotting functions never apply style themselves. Changing the paper's font size means editing one file.

### One script per figure

- `experiments/fig3_concept_only_fraction.py`, `experiments/table2_flow_types.py`, etc.
- Each script: load config, `apply_style()`, load artifact, call a plotting function, write the figure. Nothing else.
- The reviewer-or-future-self test: pick any figure in the paper, find the script by filename, read it top to bottom in under five minutes, rerun it, get the same figure.

### Where figures live

- **Generated:** in the results directory of the run that produced them, e.g. `results/240516_a3f2c1d_eps0.5/figures/fig3.png`. Inherits provenance from the directory.
- **Paper-final:** copied deliberately to `paper/figures/` when promoted. This is the version cited in LaTeX.
- Track `paper/figures/` in git. Gitignore `results/.../figures/` — regenerate from script + config if needed.

---

## Part 4 — Experiment Protocol

### Configuration

Every experiment has a Hydra config. No parameters hardcoded. Same config + same commit = same result.

### Naming

Single convention applied everywhere: `{paper}_{stage}_{variant}_{date}` (e.g. `atlas2x2_extraction_qw_eps05_240516`). Used for run IDs, output directories, W&B run names, artifact filenames.

### Output directories

Every run writes to a directory named with: date, git commit hash (short), config hash (short). Example: `results/240516_a3f2c1d_eps0.5/`. Contains resolved config, outputs, `run.log`. Never overwrite — new run, new directory.

### Provenance

At the start of every run, log: git commit, git status, resolved config, library versions, hardware, seeds, start time. At the end: end time, output files, warnings. A `run_info.json` in the output directory.

### Seeds

Set explicitly, log explicitly. For stochastic results, multiple seeds with variance — not a single run.

### Stage handoff

Each stage's output is the next stage's input, with a stable schema. Document the schema once per stage. Version it when it changes.

### Failed runs

Keep the data in `results/failed/` (gitignored). Log a one-line entry per failure in `RUNS.md` at the repo root: date + reason + config delta. The directory is for inspection; `RUNS.md` is what's reviewed at post-mortem.

### Reproducibility check

Before a result goes in a paper, rerun the experiment from the config alone. If it doesn't reproduce, that's the bug to find before anything else.

---

## Part 5 — W&B

- One project per paper. Run names match output directory names.
- Log: config, metrics, key plots, run metadata (git commit, seed, duration), tags by stage and variant.
- Log the Drive path of artifacts as a summary field. W&B is the index, not the storage.
- Don't use W&B Artifacts for large files. Use Drive.
- Optionally log the rclone command needed to pull the artifact, so future-you has it immediately.

---

## Part 6 — Storage (Drive + rclone)

- **Working storage:** Drive. Active experiments write here from Colab.
- **Local access:** rclone, with explicit sync commands. Pull before a Claude Code session, push after local changes. The state is always known.
- **Archive:** Hugging Face for stable artifacts at the end of a stage or paper. Not for iterating.
- Sync only the active results directory, not all of Drive.
- Drive has no version control. The new-directory-per-run convention prevents overwrite.

---

## Part 7 — Post-mortem (after each paper)

Half a page on: what went wrong technically, what the agent got wrong, what you missed in review, what one rule would have prevented it. Feed rules back into this document.

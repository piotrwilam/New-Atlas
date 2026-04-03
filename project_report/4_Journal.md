# Journal of Experiments

This document records the experimental history of the project, from the original pitch through to the current CSP-Atlas work. Each experiment is framed as a hypothesis or discovery, with architecture, results, and conclusions where applicable. Experiments that became part of the main findings or active research are referenced to the relevant documents rather than duplicated here.

This is a living document — it will be refined over subsequent iterations.

---

## The Original Pitch: Neural Masquerading

The project began with a proposal titled *"Neural Masquerading: Investigating Metacognitive Deception and Representational Drift in Large Language Models."* The core question was: **can a language model generate harmful output while maintaining an internally "calm" state — effectively lying to safety probes?**

The proposal outlined a methodology using Orthogonal Procrustes Analysis to measure "representational drift" between a model's safe baseline activations and its activations when instructed to produce harmful content while appearing safe internally. The hypothesis was that larger, more capable models would be better at this decoupling — a form of metacognitive control.

This pitch defined the project's DNA: using mechanistic interpretability tools (probing, activation extraction, circuit analysis) to understand what models *actually do internally* versus what they *appear to do externally*. All subsequent work flows from this framing.

**Reference:** [Original proposal 1](https://docs.google.com/document/d/1LZLIVPNrYeU8Z6PiaRddSVjnBn0x4k0Qhn2ogXm7eR4/edit?usp=sharing)

---

## Phase 1: Probing Internal States

Phase 1 translated the Neural Masquerading pitch into concrete experiments. Rather than jumping directly to deception detection, the team first needed to establish that internal ethical/safety signals could be reliably extracted from model activations at all. Work was distributed across the team as documented in the progress reports (06-02, 20-02, 06-03).

---

### Experiment 1: Neurofeedback on Qwen 0.5B (Piotr)

**Hypothesis:** A language model's internal ethical judgment about a scenario can be detected via linear probing of hidden states *before* the model finishes generating text. The model already "knows" its verdict internally — we just need to train it to surface that signal through a specific token channel.

**Architecture:**
- Base model: Qwen/Qwen2.5-0.5B (494M parameters)
- Fine-tuning: LoRA (rank 16, alpha 32) on all attention + MLP projections
- Special tokens: `<|verdict_start|>` and `<|verdict_end|>` added to vocabulary
- Training data: ETHICS commonsense dataset, filtered to examples where the base model already agrees with the label (~54% retention), then balanced to ~500 ethical + ~500 unethical examples
- Template: `"User: {scenario}\nModel: <|verdict_start|> {ETHICAL|UNETHICAL} <|verdict_end|> Because {explanation}"`
- Probing: LogisticRegression (linear) and MLP (128, 64) per layer on hidden states at the verdict token position

**Results (Experiment 1, baseline):**
- The model learned the reporting format (100% format compliance)
- Ethical/unethical clusters were separable in PCA space
- Linear probe accuracy was modest but above chance
- The experiment established the pipeline: train → extract → probe → visualize

**Conclusions:**
- Internal ethical signals exist and are extractable
- The neurofeedback approach (training the model to *report* what it already knows) is viable
- Pipeline validated for iteration

**Reference:** [NLP Project 01, Experiment_01](/Users/piotrwilam/Code/NLP%20Project%2001/Experiment_01/)

---

### Experiment 2: Scaling to Qwen 1.5B (Piotr)

**Hypothesis:** Scaling from 0.5B to 1.5B parameters improves the separability of ethical judgment signals and shifts the critical layer deeper into the network.

**Architecture:** Same as Experiment 1, but with Qwen/Qwen2.5-1.5B. Added confidence-based quality filtering (threshold 0.7) and improved label quality with ETHICS subset selection. Training: 636 examples, 6 epochs, final loss 0.538.

**Results:**
- Critical layer (activation patching): Layer 24
- Critical layer (linear probe): Layer 20
- Critical layer (MLP probe): Layer 20
- PCA explained variance: 44.7% + 8.25% = ~53% in first two components
- LDA classification score: 0.965 (excellent separation)
- Linear probe accuracy: 69.25%; MLP probe: 86.25% (both at layer 20)
- 123 out of 400 eval examples misclassified

**Conclusions:**
- Scaling improves signal quality significantly (86.25% MLP vs. lower on 0.5B)
- Ethical judgment crystallizes in a specific "decision zone" (layers 20–24 in a 28-layer model, roughly 70–85% depth)
- Patching and probing identify slightly different critical layers, suggesting the signal builds gradually
- The internal signal is real, localizable, and layer-specific

**Reference:** [NLP Project 01, Experiment_03](/Users/piotrwilam/Code/NLP%20Project%2001/Experiment_03/)

---

### Experiment 2b: Vignesh's MLP Classifiers Across Layers (Vignesh)

**Hypothesis:** MLP classifiers trained iteratively through all layers of model output can classify activations into ethical/unethical categories, revealing at which layer the ethical signal is strongest.

**Architecture:** MLP classifiers trained per layer on activation outputs from the Qwen models, classifying ethical vs. unethical based on ETHICS dataset labels.

**Results:** Achieved 70% accuracy. The per-layer results showed that classification performance varies across depth.

**Conclusions:** Confirmed that ethical signals are present in internal activations and localizable to specific layers, independently validating the probing approach.

**Reference:** 20-02 Progress Report

---

### Experiment 2c: Matt's PCA Discovery (Matt)

**Hypothesis:** PCA analysis of activations across all layers reveals how ethical decisions evolve through the network.

**Architecture:** Plotted activations for all layers against top 3 PCAs of the penultimate layer.

**Results:** Discovered that decisions are effectively made in the first layer and that activations simply disperse through subsequent layers, solidifying initial decisions.

**Conclusions:** This was a surprising finding — it suggested that the model's "decision" about ethical content is largely set early, with later layers refining rather than fundamentally changing it. This contrasted with the probing results showing peak accuracy at middle layers, suggesting that *separability* and *decision-making* may occur at different depths.

**Reference:** 20-02 Progress Report

---

### Experiment 2d: Efe's Probing on Sparse Model (Efe)

**Hypothesis:** Internal layers of the weight-sparse CSP model capture ethical values, detectable via probes after MLP and Attention layers.

**Architecture:** Trained probes after MLP and Attention layers of the OpenAI circuit-sparsity model on ETHICS-derived data.

**Results:** No internal layers captured ethical values — the sparse model does not encode ethical judgment in a probe-accessible way.

**Conclusions:** Important negative result. The weight-sparse architecture, while excellent for code-level circuit analysis, does not internalize ethical concepts the way dense models do. This steered the project toward using the sparse model for *code structure* analysis rather than *ethical judgment* detection.

**Reference:** 20-02 Progress Report; [Efe's literature review](https://docs.google.com/document/d/1sn2eGEoBcA6ukZEBCFPklYmdZ2P_bAHsiw_Zt5fl2nQ/edit?tab=t.0)

---

### Experiment 2e: Asha's Sparse Model Neurofeedback (Asha)

**Hypothesis:** The neurofeedback approach (fine-tuning to surface internal signals) works on weight-sparse models as well as dense ones.

**Results:** Sparsity led to a lack of ability to produce neurofeedback — the sparse model could not be trained to report its internal state through the neurofeedback channel.

**Conclusions:** Another important negative result confirming that the sparse model's architecture is fundamentally different from dense models for this kind of probing. Reinforced the pivot toward using the sparse model for structural circuit analysis.

**Reference:** 20-02 Progress Report

---

### Experiment 2f: Sparse Model Capability Test (Efe)

**Discovery:** The CSP sparse model cannot even correctly complete simple code like `def multiply(x,y):` — its generation capabilities are severely limited.

**Conclusions:** The model is useful for *understanding* code (activations encode meaningful structure) but not for *generating* it. This set expectations for the CSP-Atlas work: we analyze what the model encodes, not what it produces.

**Reference:** 06-03 Progress Report

---

### Experiment 3: Security Circuit Probing on CSP Transformer (Piotr)

**Hypothesis:** The CSP (circuit-sparsity) transformer, a weight-sparse GPT trained on Python code, contains localizable circuits that distinguish secure from insecure code patterns.

**Architecture:**
- Model: OpenAI circuit-sparsity (419M parameters, 8 layers, weight-sparse)
- Dataset: 192 minimal pairs (384 probes) — secure vs. insecure variants of SQL injection, command injection, unsafe deserialization, weak cryptography, path traversal, insecure YAML parsing
- Probes: LogisticRegression (linear) + MLP (128, 64) per layer
- Split: 80/20 stratified

**Results:**
- Best linear probe: Layer 4 — **87.0% accuracy**
- Best MLP probe: Layer 6 — 84.4% accuracy
- The security signal is largely linearly separable in mid-network residual stream

**Conclusions:**
- Security-relevant information is concentrated in the middle layers of the CSP transformer
- Linear separability at 87% suggests the model encodes security patterns as a relatively simple geometric distinction, not a deeply entangled representation
- This experiment bridged from the ethical judgment work (Phase 1) to the code-specific analysis (Phase 2) — the CSP transformer became the focus model going forward

**Reference:** [CSP-Ablation-Project](https://github.com/piotrwilam/CSP-Ablation-Project), Phase 1

---

### Experiment 4: Ablation Sweep — Are the Circuits Compact? (Piotr)

**Hypothesis:** A small set of neurons (~20) carries most of the security signal. Ablating these neurons should drop probe accuracy to chance level (~50%), while ablating fewer should have minimal effect.

**Architecture:**
- Ablation strategies: zero-ablation (set activations to 0) and mean-ablation (set to dataset mean)
- Sweep: k = {1, 3, 5, 10, 20, 50, 100, 200} neurons ablated
- Measurement: probe accuracy after ablation
- Generation sanity check: 40 prompts (20 secure, 20 insecure) — verify ablation doesn't cause nonsense

**Results:**
- Baseline accuracy: 86.7%
- At k=20 ablation: accuracy drops to ~50% (chance level)
- The generation sanity check confirmed the model still produces coherent code after ablation — only the security discrimination is destroyed

**Conclusions:**
- The security signal is highly concentrated: ~20 neurons out of the full network carry the discriminative information
- This confirms compact, surgically identifiable circuits exist in the CSP transformer
- The result motivated the next question: if compact circuits exist for *security*, do they exist for *every* Python concept? This question became CSP-Atlas.

**Reference:** [CSP-Ablation-Project](https://github.com/piotrwilam/CSP-Ablation-Project), Phase 2; artifacts on [HuggingFace](https://huggingface.co/piotrwilam/CSP-Ablation-Project)

---

## Transition: Proposals That Shaped Phase 2

Between Phase 1 and Phase 2, several theoretical proposals were developed that influenced the direction of CSP-Atlas.

---

### Team convergence (late Feb, after meeting with Karen)

After the 06-03 meeting with Karen, the team discussed and synthesized the various experimental directions. Efe's idea of combining Piotr's ablation approach with circuit topology analysis became the central direction. The goal crystallized into identifying three types of circuit structure:

1. **Feature-specific neurons** — small sets responsible for a single feature
2. **Hub neurons** — shared across multiple features
3. **Ablation and steering** — identifying directions in activation space that influence generation (building on Vignesh's finding of prior steering work)

Piotr and Efe refined two complementary plans: circuit topology mapping and ablation-based behavior steering. These became the foundation for CSP-Atlas.

**Reference:** 06-03 Progress Report; [Multi-thread document](https://docs.google.com/document/d/1NobR1gUdB3SSshw0aJRPCw9ybtRaOK6D3PgcFAJZ0hc/edit?usp=sharing)

---

### Proposal: STATNLP-FFS — Finding Feature Strands (Piotr + Efe)

**Discovery topic:** Feature strands — syntax-level behaviors that remain active across multiple layers, forming wire-like circuits in sparse transformers.

Three methods were proposed:

1. **Minimal Cut Sets** — Use graph theory to identify the minimal set of neurons whose removal breaks a feature. For each Python concept (def, return, if-else, try-except), collect feature prompts, record activations per layer, select top-k neurons, perform causal removal testing. The novelty over OpenAI's published work was the use of graph theory metrics rather than simple pruning.

2. **High Betweenness** — Identify neurons that act as bottleneck hubs where multiple feature strands converge. Example hypothesis: Layer 4, neuron 812 receives activations from Layer 3 neurons handling API calls, type tokens, and variable names — a convergence point.

3. **Low Rank Residual Direction** — Find a single direction **v** in residual space that represents a feature: **v** = mean(**h_feature**) - mean(**h_not_feature**). This could enable model steering: **h_new** = **h** + α × **v**.

These proposals were not executed as standalone experiments but directly influenced CSP-Atlas's design — particularly the idea that circuits for specific Python concepts can be isolated and characterized. Vignesh concurrently proposed using automated circuit discovery (Conmy et al., 2023).

**Reference:** [archive.md](../docs/archive.md); [Ablation literature review](https://docs.google.com/document/d/1OA5YawEelKwvi_eblhHAJq2iRgsdsqvS9a_XlyJuP9M/edit?usp=sharing)

---

### Exploratory Work: Matt's Analysis (Matt, week of Feb 6–8)

**Discovery topic:** Category-specific activation geometries in the CSP transformer.

**Methods applied** to 28 categories of Python code stubs:
- Top-5 next-token predictions, entropy, KL divergence baselines
- Layer-by-layer logit lens analysis (predictions crystallize mid-to-late layers)
- Direct logit attribution (per-head and per-MLP signed contributions)
- Residual stream PCA trajectories (semantic divergence from layer 3)
- UMAP projection of attribution vectors into 3D interactive atlas

**Key observation:** Category-specific activation geometries exist — class/def are adjacent but distinct in UMAP space; data=/buf= are far apart. This was early evidence that the CSP transformer organizes Python concepts geometrically, which CSP-Atlas later confirmed systematically.

Matt also explored PCA of ETHICS vs. Reddit data, showing clear cluster separation, and proposed the question: "Is it Code? or Is it Cake?" — exploring whether the model's internal representation of code vs. natural language is geometrically separable.

**Dataset research (Matt + Efe):** Surveyed BugsInPy, QuixBugs, DiverseVul, HumanEval, and LeetCode datasets for suitability as code analysis benchmarks.

**Reference:** [archive.md](../docs/archive.md); 06-03 Progress Report

---

## Phase 2: CSP-Atlas — Universal Circuit Discovery

Phase 2 shifted focus from probing specific behaviors (ethics, security) to mapping the *entire* circuit structure of the CSP transformer. The motivating question: if compact circuits exist for security, do they exist for every Python concept?

The CSP-Atlas project developed a systematic pipeline: generate controlled prompts (Module 1), extract and binarize activations (Module 2A/2B), compute universal circuits via marginalization, and evaluate their properties.

---

### Experiment 5: Prompt Generation Pipeline (Module 1)

**Hypothesis:** By generating prompts that share only one structural invariant (a specific AST node applied to a specific builtin) while varying everything else (variable names, context, padding), we can create a dataset where intersection of activation patterns isolates concept-specific neurons.

**Architecture and results:** 43 AST nodes × 63 builtins → 1,276 pairs × 50 prompts = 63,800 total prompts. All prompts AST-verified, quality-filtered by sequence loss.

**Conclusion:** The controlled variance approach successfully produced a dataset suitable for circuit extraction. See [2_Findings.md](2_Findings.md), Section 2.

---

### Experiments 6–9: Universal Circuit Extraction and Analysis

The core CSP-Atlas experiments — extraction, binarization, marginalization, evaluation, and modularity scoring — are documented in detail in the findings and current research documents:

- **Universal circuit extraction and genuineness** → [2_Findings.md](2_Findings.md), Finding 1
- **Modularity and the AST/builtin asymmetry** → [2_Findings.md](2_Findings.md), Finding 2
- **Layer architecture and the inverted-U pattern** → [2_Findings.md](2_Findings.md), Finding 3
- **Compositionality and the Entanglement Index** → [2_Findings.md](2_Findings.md), Finding 4
- **Active research (ablation, SAE, analytical tools, visualizations, thresholds)** → [3_Current.md](3_Current.md)

---

## Summary Timeline

| Date | Experiment | Who | Key Result |
|------|-----------|-----|------------|
| ~Feb 6 | Progress report: papers narrowed | Team | Converged on metacognition + sparse models |
| ~Feb 6–8 | Activation geometry analysis | Matt | Category-specific geometries in CSP |
| ~Feb 13–15 | Neurofeedback on Qwen 0.5B | Piotr | Pipeline validated, signal exists |
| ~Feb 15–16 | Neurofeedback on Qwen 1.5B | Piotr | 86.25% MLP probe at layer 20 |
| ~Feb 15–16 | MLP classifiers across layers | Vignesh | 70% accuracy on ethical classification |
| ~Feb 15–16 | PCA of penultimate layer | Matt | Decisions made in first layer |
| ~Feb 17 | CSP probing (MLP + Attn layers) | Efe | No ethical signal in sparse model (negative) |
| ~Feb 17 | Sparse model neurofeedback | Asha | Sparsity blocks neurofeedback (negative) |
| ~Feb 17 | CSP-Ablation Phase 1: Probing | Piotr | 87% linear probe at layer 4 |
| ~Feb 20 | Progress report: experiments | Team | Distributed workload, narrowing direction |
| ~Feb 21 | CSP-Ablation Phase 2: Ablation | Piotr | ~20 neurons carry security signal |
| Late Feb | Team convergence after Karen meeting | Team | Topology + ablation direction agreed |
| Late Feb | STATNLP-FFS proposals | Piotr + Efe | Feature strands framework |
| ~Mar 6 | Progress report: sparse model focus | Team | Confirmed CSP-Atlas direction |
| Mar 2026 | CSP-Atlas Module 1 | Piotr | 63,800 prompts generated |
| Mar 2026 | CSP-Atlas Module 2 | Piotr | 43 AST + 63 builtin universals extracted |
| Mar 2026 | CSP-Atlas evaluation | Piotr | Findings 1–4 established |

---

## References

- [Original proposal — Neural Masquerading](https://docs.google.com/document/d/1LZLIVPNrYeU8Z6PiaRddSVjnBn0x4k0Qhn2ogXm7eR4/edit?usp=sharing)
- [Multi-thread document](https://docs.google.com/document/d/1NobR1gUdB3SSshw0aJRPCw9ybtRaOK6D3PgcFAJZ0hc/edit?usp=sharing)
- [Original proposal 2](https://docs.google.com/spreadsheets/d/1lCTl9By1A9PuvQ3B_l8LBJ9nGK20z338eRGuGGcJ7i8/edit?usp=sharing)
- [Ablation literature review](https://docs.google.com/document/d/1OA5YawEelKwvi_eblhHAJq2iRgsdsqvS9a_XlyJuP9M/edit?usp=sharing)
- [CSP-Atlas on GitHub](https://github.com/piotrwilam/CSP-Atlas)
- [CSP-Ablation-Project on GitHub](https://github.com/piotrwilam/CSP-Ablation-Project)
- [CSP-Atlas on HuggingFace](https://huggingface.co/CSP-Atlas)

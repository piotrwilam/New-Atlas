# Universal Circuit Discovery in a Weight-Sparse Code Transformer

**Course:** COMP0087 — Statistical NLP, UCL

**Team:** Piotr, Efe, Vignesh, Matt, Asha

**TA:** Karen

## Organization

This report is organized in two parts:

- **Part 1 — Findings:** The principal results of the CSP-Atlas project, presenting the discovery and analysis of universal circuits in a weight-sparse code transformer. This is the core scientific contribution.

- **Part 2 — Journal:** A chronological record of all experiments conducted during the project, including negative results, proposals that shaped the direction, and the evolution of ideas across the team. This provides context for how the findings emerged.

---

# Part 1 — Findings: Analysis of Universal Circuits in a CSP Transformer

**Repository:** [CSP-Atlas](https://github.com/piotrwilam/CSP-Atlas)
**Model:** [openai/circuit-sparsity](https://huggingface.co/openai/circuit-sparsity)
**Data & artifacts:** [CSP-Atlas on HuggingFace](https://huggingface.co/CSP-Atlas)

## 1. Summary

This document presents findings from the CSP-Atlas project, which investigates the internal representations of Python language constructs inside a code-understanding transformer. The central question is: **does the model develop dedicated neural circuits for specific Python concepts, and if so, how are they organized?**

We find that the CSP transformer contains stable, identifiable neural circuits — which we call *universal objects* — that correspond to fundamental Python language elements such as AST (Abstract Syntax Tree) nodes and builtin objects. These circuits can be isolated through a systematic process of controlled prompt generation, activation extraction, binarization, and intersection-based marginalization. The resulting circuits are measurably distinct from random neural activity and exhibit structured properties including compositionality, layer-specific specialization, and varying degrees of modularity.

### What is a CSP transformer?

CSP (Circuit Sparsity) is a transformer model released by OpenAI, designed for code understanding tasks. It is a GPT-architecture causal language model with 8 layers and 2048 neurons per MLP layer, trained on Python code. The model is publicly available on HuggingFace (`openai/circuit-sparsity`) and was specifically designed for mechanistic interpretability research — its relatively small size makes internal circuit analysis tractable.

### What is a universal circuit?

A universal circuit (or *universal object*) is a set of neurons in the CSP transformer that consistently activate whenever a specific Python concept appears in the input, **regardless of the surrounding code context**. For example, the universal circuit for the `For` loop is the set of neurons that fire every time the model processes a `for` statement — whether the loop iterates over a list of financial transactions, DNA sequences, or game objects. By systematically varying the surface form of the code while keeping the structural concept fixed, we can isolate these concept-specific neural representations from noise.

There are two types of universal objects:
- **Universal AST objects** — circuits for syntactic constructs (e.g., `For`, `If`, `ListComp`, `Try`)
- **Universal Builtin objects** — circuits for Python builtin types and functions (e.g., `list`, `dict`, `int`, `ValueError`)

### Flow of this document

Section 2 explains the building blocks of the analysis: AST nodes, builtins, and the prompt generation system. Section 3 describes the pipeline for extracting and constructing universal circuits from model activations. Section 4 presents the analytical methods and key results. Section 5 discusses conclusions.

---

## 2. Concepts and Prompt Generation

### AST nodes and builtins

Every valid Python program can be parsed into an Abstract Syntax Tree (AST) — a hierarchical representation of its syntactic structure. Each node in the tree corresponds to a language construct: `For` (for-loops), `If` (conditionals), `FunctionDef` (function definitions), `ListComp` (list comprehensions), and so on. Python defines approximately 35 concrete AST node types that appear in everyday code.

Separately, Python provides a set of *builtin objects* — types (`int`, `list`, `dict`), functions (`print`, `len`, `range`), and exceptions (`ValueError`, `TypeError`) — that are available without any imports. There are approximately 153 such objects.

These two dimensions define the space of investigation: every Python snippet can be characterized by which AST nodes it uses and which builtins it involves. The CSP-Atlas project explores the full cross-product of these two dimensions.

### Prompt generation

To isolate a neural circuit for a specific concept, we need many code examples that share that concept but differ in everything else. The prompt generator (Module 1) creates these examples through a controlled process:

1. **Concept Matrix** — The Cartesian product of AST nodes × builtin objects defines all (AST, builtin) pairs to investigate. In the "small" configuration, this is approximately 40 AST nodes × 50 builtins = 2,000 pairs.

2. **Essence construction** — For each pair, a structurally valid Python snippet is built programmatically using `ast.parse()` and `ast.unparse()`. The snippet is guaranteed to contain the target AST node applied to the target builtin. For example, the pair (For, list) produces a for-loop that iterates over a list.

3. **Variance injection** — Each essence is then varied along three orthogonal dimensions:
   - *Lexical/semantic variance* — Variable names and values are drawn from different domains (finance, biology, gaming, physics, e-commerce). A `For` loop over a list looks different when the list is `ledger_entries` versus `dna_samples`.
   - *Structural variance* — The snippet is wrapped at global scope (~40%), inside a function (~30%), or inside a class method (~30%).
   - *Padding variance* — Unrelated code (assignments, print statements) is optionally added before or after the snippet.

4. **Quality filtering** — Each generated prompt is run through the CSP model. Prompts with excessively high sequence loss (indicating the model cannot process them well) are discarded. The top prompts per pair are retained.

The result is a dataset where, for each (AST, builtin) pair, there are multiple prompt variations that share *only* the structural essence. Any neuron that fires consistently across all variations must be responding to the concept itself, not to surface-level features like variable names or code layout.

**Dataset used in this report:** 43 AST nodes × 63 builtins = 1,276 valid pairs, with 50 prompt variations per pair, totaling **63,800 prompts**.

---

## 3. Building Universal Circuits

The process of constructing universal circuits from raw model activations has three stages.

### Stage 1: Activation extraction

Each prompt is fed through the CSP transformer, and the activation values at each of the 8 MLP layers are recorded. Each layer has 2048 neurons, so a single prompt produces 8 vectors of 2048 values. The raw activation values are continuous floating-point numbers representing how strongly each neuron responded to the input.

For efficiency, prompts are processed in batches of 64 using GPU parallelism, with left-padded tokenization for the causal (decoder) architecture. This achieves approximately 14 pairs per second on a T4 GPU.

### Stage 2: Binarization and pair representation

Raw activations are converted to binary masks through a two-step process:

1. **Epsilon thresholding** — A neuron is considered "active" for a given prompt if its absolute activation exceeds a threshold ε (set to 0.001 in this report).

2. **Consistency filtering** — Across all 100 prompt variations for a pair, the *consistency score* of each neuron is the fraction of prompts where it was active. Only neurons exceeding a consistency threshold (typically 0.8, meaning the neuron fired in at least 80 of 100 prompts) are retained. This removes neurons that fire sporadically.

The output is a *pair representation*: a binary mask per layer indicating which neurons reliably encode the (AST, builtin) pair.

### Stage 3: Marginalization (intersection)

Universal circuits are extracted by intersecting pair representations:

- **Universal AST circuit** for a given AST node (e.g., `For`): take the intersection of all pair representations that involve `For` across every builtin (`For`×`list`, `For`×`dict`, `For`×`int`, ...). The surviving neurons fire for `For` regardless of the builtin context.

- **Universal Builtin circuit** for a given builtin (e.g., `list`): take the intersection across all AST nodes (`For`×`list`, `If`×`list`, `ListComp`×`list`, ...). The surviving neurons fire for `list` regardless of the syntactic context.

This intersection operation is the key insight: by marginalizing over one dimension, we isolate the circuit that is invariant to the other dimension.

---

### Methodological contribution

The approach used in this work — controlled variance injection followed by intersection-based marginalization — is distinct from the dominant methods in the mechanistic interpretability literature:

- **Activation patching / causal tracing** (Conmy et al., 2023; Goldowsky-Dill et al., 2023) identifies circuits by intervening on activations and measuring downstream effects. This is powerful but operates one circuit at a time, making population-level mapping expensive.
- **Sparse Autoencoders (SAEs)** (Cunningham et al., 2023; Bricken et al., 2023) decompose activations into monosemantic features. This scales well but produces features without guaranteed semantic labels — interpretation requires post-hoc analysis.
- **Direct logit attribution** (Nostalgebraist, 2020) traces token predictions back to individual components. This is efficient but limited to output-facing circuits.

Our approach is complementary: by controlling the *input* rather than intervening on *activations*, we extract circuits with known, pre-defined semantic identity (the AST node or builtin that defines the prompt invariant). The method scales naturally — 106 universal circuits were extracted in a single pass — and requires no causal intervention, no learned decomposition, and no post-hoc labeling. The tradeoff is that it requires the ability to generate controlled input variations, which is straightforward for code but may be harder in other domains.

---

## 4. Analysis and Findings

The following findings are structured as theses, each supported by the logic behind the claim and the empirical results that support it.

---

### Finding 1: Universal circuits are genuine representations

**Thesis:** The CSP transformer develops dedicated, stable neural circuits for every Python concept tested. These are not artifacts of the extraction method — they are genuine internal representations of language constructs.

**Logic and meaning:** The marginalization procedure is extremely aggressive. To survive, a neuron must fire consistently (≥80% of prompts) across 100 prompt variations that share nothing but the structural essence, and then survive intersection across all builtins (for AST universals) or all AST nodes (for builtin universals). If the model did not truly encode these concepts, the intersection would collapse to an empty set — just as intersecting random binary vectors of this density would yield nothing.

**Supporting results:**

- **100% survival rate.** From 1,276 (AST, builtin) pairs, all 43 AST nodes and all 63 builtin objects produced non-empty universal circuits (ε = 0.001, consistency ≥ 0.8). Not a single concept was lost through the marginalization process.
- **Zero empty pair masks.** Out of 10,208 individual layer-level pair masks (1,276 pairs × 8 layers), every single one contained active neurons. The model dedicates neurons to every (AST, builtin) combination at every layer of the network.

These results constitute strong evidence that the CSP transformer has learned structured, concept-level internal representations of Python syntax and semantics — not merely statistical correlations.

**Significance:** This is the most clearly supported finding — the evidence is binary and exhaustive, with no edge cases or exceptions. It establishes the foundation for all subsequent analysis: the objects we study are real, not artifacts.

---

### Finding 2: The model separates syntax from semantics and allocates them different circuit architectures

**Thesis:** The CSP transformer draws a sharp internal boundary between syntactic structure (AST nodes) and semantic content (builtin types and functions). Not only does it represent these as distinct categories, it processes them through fundamentally different circuit architectures: syntax receives dedicated, modular circuits while semantics is processed through shared, overlapping neural populations.

**Logic and meaning:** If every concept used the same neurons, the model would have no way to distinguish between them. Conversely, if every concept used entirely separate neurons, the model would be inefficient. The modularity score quantifies where each concept falls on this spectrum by testing, via permutation tests at each of 8 layers, whether an object's mean Jaccard similarity to all other objects is significantly lower than chance (p < 0.05). A score of 8/8 means the circuit is distinctive at every layer; 0/8 means indistinguishable from the population.

**Supporting results (106 universal objects, 500 permutations per condition):**

- Only **12 out of 106** circuits scored above 0. All 12 are AST nodes. Zero builtins scored above 1/8. If modularity were unrelated to AST vs. builtin status, we would expect the 12 scoring objects to be roughly proportional — about 5 AST (43/106) and 7 builtin (63/106). The observed 12/0 split is the core finding.
- **The highest-scoring objects are structurally simple AST nodes:** `Import` (7/8), `Break` (6/8), `Pass` (6/8), `ImportFrom` (5/8), `Continue` (4/8). These scores are extremely unlikely by chance (e.g., P(≥7/8) < 0.00001%).
- **Most AST nodes (31 out of 43) also score 0** — the finding is not that "all AST circuits are modular" but that **the only modular circuits are AST circuits**. No builtin is distinctive at any layer.
- Mean Jaccard similarity to others is consistently lower for AST nodes (~0.36–0.46) than for builtins (~0.49–0.56). This difference does not depend on any significance threshold — AST circuits occupy more distinctive neural territory regardless of how the test is configured.

The asymmetry is absolute: the model allocates distinctive circuits only to AST constructs, while all builtins share overlapping neural populations. The model treats "what kind of statement is this?" as a more separable question than "what type of object is involved?"

*A note on terminology:* We use "syntax" and "semantics" as shorthand, but the deeper distinction may be **structural uniqueness vs. slot interchangeability**. Each AST node defines a unique syntactic pattern — you cannot substitute `For` for `If` without changing the code's structure. Builtins, by contrast, are largely interchangeable within the same syntactic slot — `list`, `dict`, and `tuple` can all appear as arguments to the same function or in the same type annotation. Builtins share circuits not because they are "semantic" in a linguistic sense, but because they occupy the same structural roles. The model allocates dedicated circuits to constructs that define unique structural patterns, and shared circuits to constructs that are substitutable.

**Significance:** This is the most surprising finding. There is no a priori reason to predict that AST nodes would receive dedicated circuitry while builtins would not — in programming, types are arguably as fundamental as syntax. Yet the asymmetry is absolute: every top-scoring circuit is an AST node, no builtin scores above 1/8. This has not been reported in the circuit discovery literature and suggests that sparse code transformers organize knowledge along an AST/builtin axis that mirrors the structure of programming languages themselves. It also has applied potential: if AST nodes and builtins use different circuit architectures (dedicated vs. shared), targeted interventions could modify one without disrupting the other.

---

### Finding 3: Circuit architecture varies across layers, suggesting a processing pipeline

**Thesis:** Universal circuits do not have a uniform shape. The number of active neurons changes across the 8 layers of the transformer, suggesting that different layers play different roles in processing a concept.

**Logic and meaning:** If circuits had the same size at every layer, it would suggest a flat, homogeneous representation — the model simply "remembers" the concept uniformly throughout its depth. Instead, the variation in circuit size across layers implies a processing pipeline: early layers may perform broad pattern detection (larger circuits), while deeper layers may compress information into more specific representations (smaller circuits), or vice versa. The shape of a circuit — its profile of neuron counts across layers — is a signature of how the model processes that concept.

**Supporting results (mean circuit size in neurons, per layer):**

| Layer | Pair circuits | Universal AST | Universal Builtin |
|-------|--------------|---------------|-------------------|
| 0     | 385.6        | 263.1         | 238.6             |
| 1     | 498.7        | 460.4         | 452.4             |
| 2     | 496.8        | 422.1         | 401.9             |
| 3     | 280.1        | 138.8         | 96.5              |
| 4     | 247.8        | 131.7         | 93.9              |
| 5     | 206.3        | 91.1          | 53.8              |
| 6     | 265.8        | 128.7         | 88.5              |
| 7     | 290.9        | 174.7         | 138.0             |

- **Inverted-U pattern.** Circuits are moderate at layer 0, expand to peak size at layers 1–2 (~450–500 neurons), then compress sharply to a minimum at layer 5 (~54–91 neurons for universals), before re-expanding at layers 6–7. This pattern holds for pair circuits, AST universals, and builtin universals alike.
- **Builtin circuits compress more aggressively.** At layer 5, builtin universals average only 53.8 neurons versus 91.1 for AST universals — builtins are squeezed into a tighter representation at the network's bottleneck.
- **The shape is preserved through marginalization.** Pair representations are larger than universal ones (intersection reduces size), but the layer-to-layer profile is consistent — the processing signature is robust, not an artifact.

The inverted-U shape suggests a staged computation: early layers (1–2) perform broad feature detection with large circuits, middle layers (3–5) compress into abstract representations, and late layers (6–7) expand again for output preparation. This is consistent with the "compression then generation" pattern observed in other transformer architectures.

**Significance:** The layer evolution data is unambiguous — the inverted-U pattern holds consistently across pair circuits, AST universals, and builtin universals, and it survives the aggressive marginalization procedure. This means the staged processing signature is not an artifact of any particular concept or threshold — it is a structural property of the network itself. The fact that the pattern is preserved through marginalization further reinforces Finding 1: these circuits reflect genuine, stable representations whose architecture is dictated by the transformer's computational pipeline.

---

### Finding 4: Pair circuits are only partially compositional

**Thesis:** The circuit for a specific (AST, builtin) pair is not simply the union of its AST universal and builtin universal components. A substantial fraction of each pair's neurons belong to neither universal — they encode the *interaction* between the syntactic construct and the type it operates on.

**Logic and meaning:** If the model were perfectly compositional, knowing the circuit for `For` and the circuit for `list` would fully predict the circuit for (`For`, `list`). The Entanglement Index (E_I) measures exactly this: E_I = 0 means the pair circuit is entirely explained by the union of its universals; E_I = 1 means it shares nothing with them. A high E_I indicates that the model encodes context-specific information — neurons that fire specifically when a `For` loop operates on a `list`, but not for `For` in general or `list` in general.

**Supporting results (1,276 pairs at layer 4):**

- **Mean E_I = 0.57, median E_I = 0.59.** On average, 57% of a pair circuit's neurons are not explained by the union of its AST and builtin universals. The model devotes more than half of each pair's neural resources to interaction-specific processing.
- **Only 0.47% of pairs have E_I < 0.2** (highly compositional). Near-perfect compositionality is rare and limited to isolated nodes (`Pass`, `Break`, `Import` paired with `_isolated_`) where there is no builtin interaction by construction.
- **Most compositional real pairs:** `UnaryOp` × `complex` (E_I = 0.30), `Set` × `float` (E_I = 0.30), `Raise` × `FileNotFoundError` (E_I = 0.33) — cases where the AST node and builtin have a tight, natural relationship.
- **Least compositional pairs:** `AnnAssign` × `float` (E_I = 0.72), `AnnAssign` × `bool` (E_I = 0.71), `Compare` × `bool` (E_I = 0.69) — cases where the interaction produces a representation qualitatively different from either component.

This finding means that universal circuits capture only part of the model's knowledge. The model does build on compositional building blocks, but it also develops rich, pair-specific representations that encode how a particular syntactic construct interacts with a particular type. Understanding the full picture requires analyzing both the universal components and their interaction terms.

**Significance:** This result has deep implications for the limits of mechanistic interpretability. If 57% of each pair's neurons encode *interactions* rather than *components*, then understanding individual universal circuits gives you less than half the picture. The Gao et al. (2025) paper demonstrates that circuits in sparse transformers are interpretable; our E_I result quantifies a concrete boundary on that interpretability — the real computation lives substantially in the interaction terms, not the parts. This challenges the implicit assumption in circuit-level interpretability that understanding parts gives you understanding of wholes.

---

### Analytical protocols

The findings above are supported by the following analytical methods, implemented in the evaluation notebooks ([3A_evaluation](../notebooks/3A_evaluation.ipynb), [4_modularity_scores](../notebooks/4_modularity_scores.ipynb)):

**Topology map (UMAP).** Universal circuits are projected into 2D using UMAP with Jaccard distance. This reveals which concepts have similar neural representations. Clusters indicate groups of concepts that share neural substrates.

**Circuit overlap (Jaccard heatmaps).** Pairwise Jaccard similarity between all universal circuits quantifies neural overlap. High similarity between two concepts suggests shared computational mechanisms; low similarity suggests dedicated circuitry.

**Compositionality (Entanglement Index).** The Entanglement Index (E_I) measures whether a pair circuit can be explained as the union of its AST and builtin universal components. E_I = 0 means fully compositional (pair = union of parts); E_I = 1 means entirely unique.

**Marginalization robustness.** Starting with one pair representation and progressively intersecting more, the circuit size should stabilize — confirming that the universal circuit is a genuine invariant, not an artifact of insufficient data.

**Modularity scoring.** Per-object permutation tests across all layers quantify how distinctive each circuit is relative to the population. Implemented in [4_modularity_scores](../notebooks/4_modularity_scores.ipynb).

---

## 5. Conclusions

*[To be written after analysis results are finalized.]*

---

## References

- **Repository:** [CSP-Atlas on GitHub](https://github.com/piotrwilam/CSP-Atlas)
- **Ablation studies:** [CSP-Ablation-Project on GitHub](https://github.com/piotrwilam/CSP-Ablation-Project)
- **Data artifacts:** [CSP-Atlas on HuggingFace](https://huggingface.co/CSP-Atlas)
- **Model:** [openai/circuit-sparsity on HuggingFace](https://huggingface.co/openai/circuit-sparsity)

---
---

# Part 2 — Journal of Experiments

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

**Conclusion:** The controlled variance approach successfully produced a dataset suitable for circuit extraction. See Part 1, Section 2.

---

### Experiments 6–9: Universal Circuit Extraction and Analysis

The core CSP-Atlas experiments — extraction, binarization, marginalization, evaluation, and modularity scoring — are documented in detail in Part 1:

- **Universal circuit extraction and genuineness** → Part 1, Finding 1
- **Modularity and the syntax/semantics separation** → Part 1, Finding 2
- **Layer architecture and the inverted-U pattern** → Part 1, Finding 3
- **Compositionality and the Entanglement Index** → Part 1, Finding 4
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

# Findings: Analysis of Universal Circuits in a CSP Transformer

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

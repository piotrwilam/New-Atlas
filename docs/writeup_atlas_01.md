# CSP-Atlas

---

## 3. Prompt Generation and Universal Circuits

### 3.1 Concept space: AST nodes and builtins

The investigation targets two families of Python concepts. **AST (Abstract Syntax Tree) nodes** are the syntactic constructs that define a program's structure: `For`, `If`, `FunctionDef`, `Import`, `Break`, `Pass`, `ListComp`, and so on — 43 node types in total. **Builtin objects** are the types (`int`, `list`, `dict`, `str`, `float`, `bool`, `tuple`, `set`, `complex`, `frozenset`, `bytes`, `bytearray`, `object`, `type`, `memoryview`), functions (`print`, `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `reversed`, `min`, `max`, `sum`, `abs`, `round`, `any`, `all`, `isinstance`, `issubclass`, `hasattr`, `getattr`, `setattr`, `delattr`, `callable`, `iter`, `next`, `hash`, `id`, `repr`, `input`, `open`, `super`, `property`, `staticmethod`, `classmethod`), and exceptions (`Exception`, `ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `RuntimeError`, `StopIteration`, `FileNotFoundError`, `OSError`, `ImportError`, `NotImplementedError`, `ZeroDivisionError`) that Python provides without imports — 63 in total.

The full Cartesian product — 43 × 63 = **1,276 (AST, builtin) pairs** — defines the concept space.

### 3.2 Object prompt generation

To isolate a neural circuit for a specific concept, many code examples must share that concept but differ in everything else. The prompt generator (`11A_object_prompts.ipynb`) creates 50 prompt variations per pair:

1. **Essence construction.** For each pair, a structurally valid Python snippet is built programmatically using `ast.parse()` and `ast.unparse()`, guaranteed to contain the target AST node applied to the target builtin.

2. **Variance injection** along three orthogonal dimensions: *lexical/semantic* (variable names and values drawn from finance, biology, gaming, physics, e-commerce domains), *structural* (global scope ~40%, inside a function ~30%, inside a class method ~30%), and *padding* (unrelated code optionally added before or after).

3. **Quality filtering.** Prompts with excessively high sequence loss are discarded; the top 50 per pair are retained.

The total dataset is 1,276 × 50 = **63,800 prompts**. Samples are shown in Appendix A.

### 3.3 Activation extraction and binarisation

Each prompt is processed in a single forward pass through the CSP transformer (`12_extraction.ipynb`). No text is generated — the model is used purely as a measuring instrument. Forward hooks registered on the `.mlp` module at each of the 8 layers intercept the MLP output. The activation vector is extracted at the **last token position** only, because the last token's residual stream integrates information from the entire input sequence through causal attention.

Each prompt produces **8 vectors of 2048 values** — one per layer. Each value represents how strongly the MLP at that layer contributed to a particular residual stream dimension. (The term "neuron" throughout this report refers to one dimension of this 2048-d MLP output vector — the composite signal that enters the residual stream — not an internal MLP neuron in the expanded hidden layer.)

Raw activations are then converted to binary masks:

1. **Epsilon thresholding.** A neuron is considered "active" if its absolute activation exceeds ε.
2. **Consistency filtering.** Across all 50 prompt variations for a pair, only neurons active in at least a given fraction of prompts are retained.

### 3.4 Marginalisation into universal circuits

A universal circuit for a concept is obtained by intersecting across the complementary dimension (`13_universals.ipynb`). For an AST node: the intersection of all pair masks across all 63 builtins. For a builtin: the intersection across all 43 AST nodes. A neuron survives only if it fires consistently regardless of which complementary object is involved.

This marginalisation is aggressive. If the model did not encode these concepts, the intersection would collapse to an empty set — just as intersecting random binary vectors of this density would yield nothing. The result is **106 universal circuits** (43 AST + 63 builtin), each a binary mask of 2048 dimensions at each of 8 layers.

### 3.5 Parameters: epsilon and consistency threshold

Two parameters control the extraction sensitivity:

- **Epsilon (ε)** sets the minimum activation magnitude for a neuron to count as "active." ε = 0.001 includes faint signals; ε = 0.5 retains only the loudest neurons.
- **Consistency threshold** sets the fraction of prompts a neuron must fire across. 80% means ≥ 40 of 50 prompts; 50% means ≥ 25; 20% means ≥ 10.

Both the universal circuit targets and the checker masks (Section 4) are rebuilt at every combination in the 3 × 3 grid: ε ∈ {0.001, 0.1, 0.5} × consistency ∈ {20%, 50%, 80%} = 9 parameter settings.

---

## 4. Checker Prompts and Checker Circuits

### 4.1 The token-vs-concept question

Every keyword-bearing Python concept comes with an inherent confound: the `import` statement always contains the token `import`, the `for` loop always contains the token `for`. A universal circuit that fires whenever `import` is present might be the model's reaction to seeing that token, not a representation of the import *concept*.

### 4.2 Testable objects

The check applies to objects whose keyword can plausibly appear in code without the associated concept. Of the 43 AST nodes, 24 have testable keywords (`import`, `for`, `if`, `break`, `pass`, `continue`, `while`, `return`, `class`, `def`, `yield`, `try`, `raise`, `assert`, `with`, `del`, `lambda`, `global`, `nonlocal`, and several async variants). The remaining 19 AST nodes — `Assign`, `AugAssign`, `AnnAssign`, `BoolOp`, `BinOp`, `UnaryOp`, `IfExp`, `Dict`, `Set`, `ListComp`, `SetComp`, `DictComp`, `GeneratorExp`, `Compare`, `Call`, `Attribute`, `Subscript`, `Starred`, `Slice` — have no distinctive keyword token; the token confound does not apply to them.

Of the 63 builtins, 34 have testable keywords (`int`, `float`, `str`, `list`, `dict`, `set`, `print`, `len`, `range`, `map`, `filter`, `min`, `max`, `sum`, `open`, `type`, etc.). The remaining 29 have tokens too technical or compound to appear naturally without the concept (e.g., `ValueError`, `isinstance`, `frozenset`).

This gives **58 testable objects** (24 AST + 34 builtins).

### 4.3 Checker prompt generation

For each testable object, 50 checker prompts are generated (`11B_checker_prompts.ipynb`) across five categories where the keyword token appears but the concept does not:

- **Category A — String literal:** `msg = "waiting for results"`
- **Category B — Comment:** `x = 42  # break here`
- **Category C — Variable/function name:** `breakdown_count = 5`
- **Category D — Dictionary key:** `config = {"break": True}`
- **Category E — Print call:** `print("break time")`

Each prompt is validated: (a) parses as valid Python, (b) target concept absent from the AST, (c) keyword token present in the CSP tokenizer output. Samples are shown in Appendix B.

### 4.4 Checker circuit extraction

The checker prompts are fed through the **identical** extraction pipeline: same model, same hooks, same ε, same consistency threshold, same last-token position. This methodological identity is critical. The result is a token checker mask at all 8 layers for each of the 58 testable objects.

---

## 5. Cross-Section Experiment: Concept vs. Token Decomposition

### 5.1 Method

For each testable object at each layer, the universal circuit mask (set A) and the token checker mask (set B) are compared (`15_neuron_list.ipynb`). The 2048 dimensions are partitioned into three disjoint groups:

- **A \ B = concept-only neurons:** In the universal circuit but *not* the token checker. These fire when the concept is present but not for the bare keyword.
- **A ∩ B = shared neurons:** In both masks. These fire for the concept and also for the token alone.
- **B \ A = token-only neurons:** In the token checker but not the universal circuit.

Two fractions summarise the decomposition: concept_fraction = |A \ B| / |A| and token_fraction = |A ∩ B| / |A|.

### 5.2 Result files

The cross-section is computed for all 9 (ε, consistency) parameter combinations. Each result file is named:

`15_neuron_list_eps{E}_cons{C}_L0_1_2_3_4_5_6_7_both.csv`

where `{E}` ∈ {0.001, 0.1, 0.5} and `{C}` ∈ {0.2, 0.5, 0.8}. There are **9 files** in total.

Each file contains 464 rows (58 objects × 8 layers) with columns: `object`, `layer`, `n_concept_only`, `n_both`, `n_token_only`, and the comma-separated neuron indices for each partition (`concept_only`, `both`, `token_only`). This enables both count-level statistics and neuron identity analysis from the same files.

### 5.3 How to read the result files

A row such as:

```
ast__Assert, 3, 91, 233, 38, "[10, 41, 49, ...]", "[1, 11, 22, ...]", "[32, 38, ...]"
```

means that at layer 3, the Assert universal circuit has 91 concept-only neurons (IDs listed), 233 shared neurons, and 38 token-only neurons. The circuit size is 91 + 233 = 324 neurons, of which 28% are concept-only.

---

## 6. Finding 1: Universal Circuits Are Genuine Representations

### 6.1 Thesis

The CSP transformer develops dedicated, stable neural circuits for every Python concept tested. These circuits are not artefacts of the extraction method — they are genuine internal representations of language constructs.

### 6.2 Logic

The marginalisation procedure is extremely aggressive. To survive, a neuron must fire consistently (≥ 80% of prompts) across 50 prompt variations that share nothing but the structural essence, and then survive intersection across all 63 builtins (for AST universals) or all 43 AST nodes (for builtin universals). If the model did not truly encode these concepts, the intersection would collapse to an empty set.

### 6.3 Evidence

**100% survival rate.** From 1,276 (AST, builtin) pairs, all 43 AST nodes and all 63 builtin objects produced non-empty universal circuits (ε = 0.001, consistency ≥ 80%). Not a single concept was lost through the marginalisation process. Out of 10,208 individual layer-level pair masks (1,276 pairs × 8 layers), every single one contained active neurons.

**Non-empty at every parameter setting.** The cross-section results (`15_neuron_list` files) confirm that all 58 testable objects have non-empty universal circuits (A = concept-only ∪ shared) at every one of the 9 (ε, consistency) combinations. Zero of the 464 object × layer combinations has |A| = 0 at ε = 0.001, consistency 80%.

### 6.4 Scope

This finding is binary and exhaustive, with no edge cases. It establishes the foundation for all subsequent analysis: the objects we study are real, not artefacts. The detailed composition of these circuits — what fraction is concept-driven vs. token-driven — is the subject of Finding 2.

---

## 7. Finding 2: AST Circuits Contain a Concept Component Distinct From Token Activation

### 7.1 Thesis

Universal circuits for AST nodes are not merely token detectors. They contain a substantial set of neurons that fire when the syntactic concept is present but do *not* fire when the keyword token appears without the concept. This concept component is strongest at mid-to-late layers and at high activation magnitudes. Builtin circuits, by contrast, are almost entirely token-driven.

### 7.2 Logic

The cross-section experiment (Section 5) compares each universal circuit mask (set A) with a token checker mask (set B) built from prompts where the keyword appears in strings, comments, variable names, dictionary keys, and print calls — but the actual Python construct is absent. Neurons in A \ B respond to the concept, not the token. If A \ B is empty at all parameter settings, the circuit is purely token-driven. If A \ B is consistently non-empty, the circuit encodes something beyond the token.

Note that 19 of the 43 AST nodes — those representing structural patterns without a distinctive keyword (e.g., `Assign`, `BinOp`, `ListComp`, `Subscript`) — are not subject to the token confound at all. Their universal circuits cannot be explained by any single keyword token, since no such token exists. These are inherently concept-driven representations, and the token independence question does not arise for them.

### 7.3 Evidence: concept-only neurons exist across all parameter settings

At every (ε, consistency) combination in the 3 × 3 sweep, AST circuits contain concept-only neurons:

| ε | Consistency | AST concept fraction | Builtin concept fraction |
|---|---|---|---|
| 0.001 | 20% | 1.7% | 0.4% |
| 0.001 | 50% | 3.4% | 1.1% |
| 0.001 | 80% | 7.7% | 1.5% |
| 0.1 | 20% | 9.5% | 1.5% |
| 0.1 | 50% | 9.6% | 1.7% |
| 0.1 | 80% | 12.1% | 2.1% |
| 0.5 | 20% | 12.3% | 2.1% |
| 0.5 | 50% | 12.5% | 2.2% |
| 0.5 | 80% | 11.6% | 1.3% |

The AST concept fraction exceeds the builtin fraction by 4–9× at every setting. The existence of concept-only neurons is not an artefact of any single threshold.

### 7.4 Evidence: the loudest neurons are predominantly concept-specific

At ε = 0.5, consistency 80%, layer 5 — retaining only the loudest-firing neurons:

| Group | Concept-only | Shared | Circuit size | Concept fraction |
|---|---|---|---|---|
| Modular ASTs (6) | 10 | 6 | 16 | **62.5%** |
| Non-modular ASTs (18) | 32 | 23 | 55 | **58.2%** |
| Builtins (34) | 0 | 36 | 36 | **0.0%** |

The majority of the loudest neurons in AST circuits are concept-specific. For builtins, every concept-only neuron that existed at low ε vanishes — they fire too weakly to survive ε = 0.5.

### 7.5 Evidence: layer profile of the concept signal

At ε = 0.001, consistency 80%, the concept fraction follows an inverted-U across layers:

| Layer | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| Modular AST | 8.1% | 0.1% | 4.9% | 23.1% | 26.9% | **30.4%** | 23.4% | 17.6% |
| Non-modular AST | 4.7% | 0.0% | 3.2% | 6.3% | 8.4% | 9.0% | 9.7% | 6.1% |
| Builtin | 0.9% | 0.0% | 2.0% | 1.8% | 4.9% | 3.0% | 4.4% | 2.6% |

Layer 1 is essentially pure token (concept fraction ≈ 0%). The concept signal peaks at layers 3–5 for modular ASTs, consistent with the mid-layer compression pattern observed in the logit lens analysis.

### 7.6 Four-tier structure

The results reveal a consistent hierarchy across the full set of 106 universal objects:

1. **Tokenless ASTs** (19 objects: `Assign`, `BinOp`, `ListComp`, `Compare`, `Call`, `Attribute`, `Subscript`, etc.): These AST nodes have no distinctive keyword token. Their universal circuits are inherently concept-driven — no single token can explain the activation pattern, since the prompts that produce them share only syntactic structure.

2. **Modular keyword ASTs** (6 objects: `Import`, `Break`, `Pass`, `ImportFrom`, `Continue`, `Assert`): Strongest concept signal among testable objects, peaking at 30% by neuron count and 62.5% by activation magnitude at layer 5.

3. **Non-modular keyword ASTs** (18 objects: `For`, `While`, `If`, `Return`, `ClassDef`, `FunctionDef`, etc.): Weaker but genuine concept residual — 6–10% by count, 58% by magnitude at ε = 0.5.

4. **Builtins** (63 objects: `int`, `list`, `print`, etc.): Near-zero concept signal at all settings. Universal circuits for builtins are effectively subsets of their token activation patterns. The few concept-only neurons that appear at low ε fire weakly and vanish at ε = 0.5.

---

## Key Files Reference

### Notebooks

| Notebook | Role |
|----------|------|
| `11A_object_prompts.ipynb` | Generates 63,800 object prompts (50 per pair, 1,276 pairs) |
| `11B_checker_prompts.ipynb` | Generates checker prompts (50 per testable object, 58 objects) |
| `12_extraction.ipynb` | Activation extraction: forward passes, hooks, batching |
| `13_universals.ipynb` | Marginalisation of pair masks into 106 universal circuits |
| `15_neuron_list.ipynb` | Cross-section decomposition across all 9 parameter settings |

### Output files by notebook

| Notebook | Output files |
|----------|-------------|
| `11A_object_prompts.ipynb` | `11A_object_prompts.parquet` |
| `11B_checker_prompts.ipynb` | `11B_checker_prompts.parquet` |
| `12_extraction.ipynb` | `12_object_activations.h5`, `12_checker_activations.h5` |
| `13_universals.ipynb` | 9× `13_object_masks_eps{E}_cons{C}.h5` + 9× `13_checker_masks_eps{E}_cons{C}.h5` = 18 files |
| `15_neuron_list.ipynb` | 9× `15_neuron_list_eps{E}_cons{C}_L0_1_2_3_4_5_6_7_both.csv` |


---

## Appendix A: Sample Object Prompts

Prompts selected from the 63,800-prompt dataset illustrating the controlled variance injection. Each set of 10 variations shares only the structural essence — the target AST node applied to the target builtin — while variable names, domains, wrappers, and padding all vary.

### (For, list) — 10 of 50 variations

```python
result = None
measurements = [9.81, 300000000.0, 6.674e-11, 1.602e-19]
for reading in measurements:
    compute_trajectory(reading)
```

```python
result = None
shopping_cart = [29.99, 15.5, 89.0, 4.99]
for order_item in shopping_cart:
    process_checkout(order_item)
counter += 1
```

```python
class PortfolioManagerMain:

    def calculate_returns_run(self):
        result = None
        ledger_entries = [1050.5, -20.0, 400.25, 88.1]
        for transaction in ledger_entries:
            audit_record(transaction)
```

```python
ledger_entries = [1050.5, -20.0, 400.25, 88.1]
for transaction in ledger_entries:
    audit_record(transaction)
print('Done')
```

```python
def update_leaderboard_main():
    print('Starting process')
    player_scores = [9500, 8700, 12400, 6300]
    for player_entry in player_scores:
        update_leaderboard(player_entry)
    print('Done')
```

```python
def compute_trajectory_main():
    status = True
    measurements = [9.81, 300000000.0, 6.674e-11, 1.602e-19]
    for reading in measurements:
        compute_trajectory(reading)
    result = None
```

```python
def analyze_genome_main():
    print('Starting process')
    dna_samples = ['ACTG', 'GCTA', 'CGAT', 'TTAC']
    for genome_sequence in dna_samples:
        analyze_genome(genome_sequence)
    counter += 1
```

```python
def audit_record_main():
    result = None
    ledger_entries = [1050.5, -20.0, 400.25, 88.1]
    for transaction in ledger_entries:
        audit_record(transaction)
    status = False
```

```python
result = None
player_scores = [9500, 8700, 12400, 6300]
for player_entry in player_scores:
    update_leaderboard(player_entry)
print('Done')
```

```python
def process_checkout_main():
    counter = 0
    shopping_cart = [29.99, 15.5, 89.0, 4.99]
    for order_item in shopping_cart:
        process_checkout(order_item)
    status = False
```

### (Try, ValueError) — 10 of 50 variations

```python
def compute_trajectory_main():
    print('Starting process')
    try:
        velocity = 299792458
        result = int(velocity)
    except ValueError as e:
        compute_trajectory(str(e))
    result = None
```

```python
counter = 0
try:
    total_price = 142
    result = int(total_price)
except ValueError as e:
    process_checkout(str(e))
counter += 1
```

```python
print('Starting process')
try:
    balance = 42500
    result = int(balance)
except ValueError as e:
    audit_record(str(e))
print('Done')
```

```python
def audit_record_main():
    print('Starting process')
    try:
        balance = 42500
        result = int(balance)
    except ValueError as e:
        audit_record(str(e))
    result = None
```

```python
class GameEngineMain:

    def spawn_entity_run(self):
        try:
            hit_points = 9500
            result = int(hit_points)
        except ValueError as e:
            update_leaderboard(str(e))
        counter += 1
```

```python
status = True
try:
    velocity = 299792458
    result = int(velocity)
except ValueError as e:
    compute_trajectory(str(e))
```

```python
print('Starting process')
try:
    mutation_rate = 43044295
    result = int(mutation_rate)
except ValueError as e:
    analyze_genome(str(e))
print('Done')
```

```python
def audit_record_main():
    status = True
    try:
        balance = 42500
        result = int(balance)
    except ValueError as e:
        audit_record(str(e))
    status = False
```

```python
def update_leaderboard_main():
    try:
        hit_points = 9500
        result = int(hit_points)
    except ValueError as e:
        update_leaderboard(str(e))
```

```python
try:
    total_price = 142
    result = int(total_price)
except ValueError as e:
    process_checkout(str(e))
status = False
```

---

## Appendix B: Sample Checker Prompts

Prompts from the token checker dataset. Each contains the keyword token but **not** the associated Python concept. The keyword appears in strings, comments, variable names, dictionary keys, and print calls — but never as the actual Python construct.

### `import` token without `Import` concept — 10 of 50

```python
def process():
    record = 42  # import fragment see docs
```

```python
class Obj:
    def method(self):
        limit = {"action": "pattern", "import": "waiting"}
```

```python
def func():
    cycle = {"import": True, "building": 42}
    return None
```

```python
data = []
print("starting import reference")
result = None
```

```python
class Obj:
    def method(self):
        phase = []  # TODO import this later
```

```python
class Obj:
    def method(self):
        print("threshold import receiving")
```

```python
data = []
important_data = True
result = None
```

```python
data = []
reimport_flag = True
result = None
```

```python
data = []
batch = {"import": True, "analysis": 42}
result = None
```

```python
source = []  # TODO import this later
```

### `for` token without `For` concept — 10 of 50

```python
def process():
    data = []  # TODO for this later
```

```python
class Obj:
    def method(self):
        print("for completed for protocol")
```

```python
def func():
    print("for completed for tracking")
    return None
```

```python
def process():
    index = {"action": "processing", "for": "analysis"}
```

```python
class Obj:
    def method(self):
        phase = 42  # for analysis see docs
```

```python
# for running algorithm reference
target = 10
```

```python
queue = {"action": "documentation", "for": "validation"}
```

```python
item = {"action": "pattern", "for": "workflow"}
```

```python
print("completed for protocol")
```

```python
x = 1
cycle = 42  # for deployment see docs
y = 2
```

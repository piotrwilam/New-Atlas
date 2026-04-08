# Plan: `R1A_object_prompts.ipynb` + `R1B_checker_prompts.ipynb` — Rust Prompt Generation

## Context

Extending New-Atlas to Rust. The Python pipeline (1A/1B) generates prompts containing specific (AST node, builtin) pairs for Qwen activation extraction. We need identical-structure Rust notebooks that generate prompts with Rust syntactic constructs (~35) and standard library objects (~45), using tree-sitter-rust for validation instead of Python's `ast` module.

---

## Files to Create

| File | Based On | Purpose |
|---|---|---|
| `notebooks/R1A_object_prompts.ipynb` | `1A_object_prompts.ipynb` | Rust construct × object prompt generation with perplexity filter |
| `notebooks/R1B_checker_prompts.ipynb` | `1B_checker_prompts.ipynb` | Rust token-without-concept prompts for checker circuit extraction |

No changes to existing files. No new `src/` modules — all logic lives in the notebooks (matching Python pattern).

---

## R1A_object_prompts.ipynb — Cell Structure

Mirror the Python 1A structure (23 cells). Same stages A→D, substituted internals.

### Cell 1 — Dependencies
Same as Python + add `tree-sitter`, `tree-sitter-rust`:
```
pkgs = ["transformers", "torch", "pyarrow", "accelerate", "tree-sitter", "tree-sitter-rust"]
```

### Cell 2 — Configuration
Same MODE system (test/small/full), same thresholds, same DATA_DIR/SRC_PATH pattern.
```python
MODE = "test"  # "test" | "small" | "full"
run_name = "R1A_object"
OUTPUT_FILE = "R1A_object_prompts.parquet"
STATS_FILE = "R1A_object_stats.json"
```
Mode thresholds identical to Python 1A.

### Cell 3 — Imports
Same as Python 1A but add tree-sitter imports:
```python
import tree_sitter_rust as ts_rust
from tree_sitter import Language, Parser
```

### Cell 4 — Google Drive mount
Identical to Python 1A.

### Cell 5 — tree-sitter setup & validation helpers
**NEW — replaces Python's ast module.** Core functions:
```python
RUST_LANGUAGE = Language(ts_rust.language())
parser = Parser(RUST_LANGUAGE)

def parse_rust(code: str):
    return parser.parse(bytes(code, "utf-8"))

def get_rust_node_types(code: str) -> set[str]:
    # Walk tree, collect all node.type values
    
def has_errors(code: str) -> bool:
    return parse_rust(code).root_node.has_error

def verify_concept(code: str, concept: str) -> bool:
    # Dispatcher: simple node lookup for most, custom for compound concepts
    # Compound: let_mut, mutable_reference, macro_invocation_question_mark
```

Compound concept verification:
- `let_mut`: look for `let_declaration` node with `mutable_specifier` child
- `mutable_reference`: look for `reference_expression` with `mutable_specifier` child  
- `macro_invocation_question_mark` (the `?` operator): look for `try_expression` node

Smoke test: parse a few Rust snippets, verify node types.

### Cell 6 — DOMAINS definition
5 domains (finance, biology, gaming, physics, ecommerce), same structure as Python but with **Rust-typed mock data**:
```python
"finance": {
    "var_names": {
        "item": "transaction", "func": "audit_record",
        "struct_name": "Portfolio", "method": "calculate_returns",
        "value": "balance", "key": "ticker", "module": "accounting",
    },
    "mock_data": {
        "i32": "42500_i32", "i64": "42500_i64", "f64": "1050.50_f64",
        "u32": "150_u32", "u64": "150_u64", "usize": "150_usize",
        "bool": "true", "char": "'$'",
        "str_literal": '"USD-2024-Q3-REPORT"',
        "string_expr": 'String::from("USD-2024-Q3")',
        "vec_i32": "vec![1050, -20, 400, 88]",
        "vec_str": 'vec!["AAPL", "GOOG", "MSFT"]',
        "option_i32": "Some(42500_i32)",
        "result_i32": "Ok(42500_i32)",
        "hashmap": 'HashMap::from([("AAPL", 178.50), ("GOOG", 140.20)])',
    },
}
```
All 5 domains must be fully defined with all mock_data keys.

### Cell 7 — Concept space: families, compatibility, sparse matrix
**Direct translation from user spec Section 3.**
- `RUST_CONSTRUCT_FAMILIES` — 15 families, ~35 constructs
- `RUST_OBJECT_FAMILIES` — 5 families, ~45 objects
- `RUST_COMPATIBILITY` — which object families pair with which construct families
- `CONSTRUCT_DISPLAY_NAMES` — tree-sitter node name → short display name
- `ITEM_LEVEL` vs `STMT_LEVEL` sets — determines wrapping behavior
- `build_sparse_matrix(mode)` — same logic as Python, produces `(construct, object)` pairs

### Cell 8 — RustPromptGenerator class
Mirror `ASTPromptGenerator` exactly. Key differences:

**Template dict** — Rust lambdas taking `(d, m)` for all ~30 constructs. Starting with the templates from user spec Section 6, expanded to cover all constructs. Each lambda returns a Rust code string.

**Object-conditional logic** — Some templates need to branch based on the object type:
- `For` with `Vec` vs `HashMap` uses different iteration syntax
- `Fn` with primitive vs trait uses different signatures
- `Match` with `Option` vs `Result` uses different arms

**Wrappers** — `RUST_WRAPPERS` with `bare_fn_main`, `inside_fn`, `inside_impl` and weights `[0.5, 0.3, 0.2]`. Must respect `ITEM_LEVEL` vs `STMT_LEVEL`:
- `STMT_LEVEL` constructs → always inside a function body
- `ITEM_LEVEL` constructs → at module level, wrapper's `fn main()` calls/uses them

**Padding** — Rust equivalents:
```python
PADDING_BEFORE = ["", "let _result: Option<i32> = None;", 'println!("Starting process");', "let _status = true;", "let mut _counter = 0;"]
PADDING_AFTER = ["", 'println!("Done");', "let _status = false;", "_counter += 1;", "let _result: Option<i32> = None;"]
```

**Verification** — uses `has_errors()` + `verify_concept()` from cell 5 instead of `ast.parse()` + `_verify()`.

**`generate_batch(construct, object, n=150) -> list[dict]`** — Same loop: cycle domains, build essence, pad, wrap, parse-verify, collect.

Smoke test at end of cell.

### Cell 9 — PerplexityFilter class
**Identical to Python 1A.** Same model (Qwen2.5-Coder-7B), same `F.cross_entropy` loss computation, same `filter_batch()` with catastrophic threshold. No changes needed — the model handles Rust code fine.

### Cell 10 — Pipeline orchestration function
**Identical structure to Python 1A.** `run_pipeline()` iterates pairs, generates, filters, checkpoints.

Output parquet schema:
```
construct: string      (was ast_node in Python)
object: string         (was builtin_obj in Python)
variation_id: int64
prompt_text: string    (Rust code)
sequence_loss: float64
token_length: int64
tree_sitter_verified: bool  (was ast_verified in Python)
```

### Cell 11 — Execute pipeline
Same as Python. Call `run_pipeline()` with mode-appropriate params.

### Cells 12-15 — Quality inspection, stats, upload, cleanup
Same structure as Python 1A. Adapt display formatting for Rust constructs.

---

## R1B_checker_prompts.ipynb — Cell Structure

Mirror Python 1B (9 cells). tree-sitter replaces ast for validation.

### Cell 1 — Configuration
```python
N_TARGET = 50
OVERSHOOT_FACTOR = 3
run_name = "R1B_checker"
OUTPUT_FILE = "R1B_checker_prompts.parquet"
```

### Cell 2 — Imports + Drive mount
Same as Python 1B + tree-sitter imports.

### Cell 3 — Load tokenizer (no model needed)
Identical to Python 1B.

### Cell 4 — tree-sitter setup
Same helpers as R1A cell 5 (parse_rust, get_rust_node_types, has_errors). Plus:
```python
def get_rust_identifiers(code: str) -> set[str]:
    """Walk tree, collect text of all identifier and type_identifier nodes."""
```

### Cell 5 — RUST_KEYWORD_MAP
From user spec Section 8.1. Two sections:
- **Construct keywords** (~22): `rust__For`, `rust__While`, etc. with `forbidden_nodes`
- **Object keywords** (~10-15): `rust__Vec`, `rust__String`, etc. with `forbidden_idents`

### Cell 6 — Word banks and identifier variants
**RUST_IDENTIFIER_VARIANTS** — keyword → list of Rust-appropriate identifiers containing it:
```python
"for": ["format_string", "information", "formula_result", "before_update"]
"fn": ["fn_name_str", "config_fn_ptr"]  # tricky — "fn" is short
"struct": ["restructure", "constructor", "infrastructure"]
"let": ["letter_count", "outlet_mode", "newsletter_flag"]
...
```

**RUST_VAR_NAMES** — keyword-safe Rust variable names (snake_case):
```python
["data", "result", "value", "item", "record", "entry", "output", "signal", ...]
```

**RUST_STRING_WORDS** — Same 40+ context words as Python.

**RUST_CONTEXT_WRAPPERS** — Rust equivalents:
```python
[
    "{snippet}",
    "let x = 1;\n{snippet}\nlet y = 2;",
    "let data: Vec<i32> = vec![];\n{snippet}\nlet result: Option<i32> = None;",
    "fn func() {{\n    {snippet}\n}}",
    "fn process() {{\n    {snippet}\n}}",
    "struct Obj;\nimpl Obj {{\n    fn method(&self) {{\n        {snippet}\n    }}\n}}",
]
```

### Cell 7 — Validation functions
```python
def check_concept_absent_rust(code: str, obj_key: str) -> bool:
    """tree-sitter analog of Python's check_concept_absent."""
    # Check forbidden_nodes via get_rust_node_types()
    # Check forbidden_idents via get_rust_identifiers()

def check_token_present(code: str, keyword: str, tokenizer) -> bool:
    # Identical to Python — tokenizer-based check

def validate_prompt_rust(code: str, obj_key: str) -> bool:
    # check_concept_absent_rust AND check_token_present
    # tree-sitter parse must succeed (no errors)
```

### Cell 8 — Generation categories + main loop
**5 categories adapted for Rust syntax:**

| Cat | Python | Rust |
|-----|--------|------|
| A: String | `x = "the import of data"` | `let x = "the import of data";` |
| B: Comment | `x = 42  # import see docs` | `let x = 42;  // import see docs` |
| C: Identifier | `important_data = 42` | `let important_data = 42;` |
| D: Dict key → array tuple | `x = {"import": True}` | `let x = [("import", true), ("status", false)];` |
| E: Print | `print("starting import")` | `println!("starting import");` |

Main loop: iterate `RUST_KEYWORD_MAP`, generate with 3x overshoot, validate, balance categories.

### Cell 9 — Save parquet
Output schema (identical structure to Python):
```
object: string     (e.g., "rust__For", "rust__Vec")
keyword: string
variation_id: int64
prompt_text: string (Rust code)
```
Save to `{DATA_DIR}/R1B_checker_prompts.parquet`.

---

## Key Risks & Mitigations

1. **Template syntax errors** — Rust is much stricter than Python. Many templates will produce tree-sitter errors on first attempt. Mitigation: build incrementally (5 constructs first in test mode), inspect failures, iterate.

2. **`fn` keyword in identifiers** — "fn" is only 2 chars, hard to embed in identifiers without it being a Rust keyword. `fn_name_str` would tokenize as `fn` + `_name_str`. Mitigation: use identifiers where "fn" is a substring not at word boundary: `config`, `infn` won't work... May need to skip `fn` from checker or use longer embeddings like `function_name`.

3. **Closure keyword** — Closures use `||` not a single keyword. `move` only applies to move closures. The checker for `Closure` may need special handling or be skipped.

4. **`?` operator** — single character, not a keyword. Cannot be checked as a token substring. Skip from checker prompts.

5. **Lifetime `'a`** — the `'` token overlaps with char literals. tree-sitter distinguishes them correctly, but the tokenizer check needs care.

6. **Template count** — ~30 construct templates need writing, each with object-conditional branches. This is the bulk of the work. Estimate: ~300 lines of template lambdas.

---

## Verification Plan

1. **Cell-by-cell in test mode first** — Run R1A with 5 constructs × 5 objects × 10 kept. Inspect generated prompts visually. Check tree-sitter verification rate.
2. **Tree-sitter smoke test** — Cell 5 should parse known-good Rust and verify all construct types are detectable.
3. **Checker validation** — Run R1B on 5 objects, inspect that keyword is present and concept is absent in each prompt.
4. **Perplexity sanity** — Verify Qwen assigns reasonable loss to Rust code (it should — Qwen2.5-Coder supports Rust).

---

## Implementation Order

1. **R1A cells 1-5** — Setup, config, tree-sitter helpers. Can be tested standalone.
2. **R1A cell 6** — All 5 DOMAINS with Rust mock data.
3. **R1A cell 7** — Concept space dicts, sparse matrix builder.
4. **R1A cell 8** — RustPromptGenerator. Start with 5 easy constructs (For, If, Let, Fn, Struct), get pipeline working, then fill remaining ~25.
5. **R1A cells 9-15** — PerplexityFilter (copy from Python), pipeline, quality inspection.
6. **R1B cells 1-4** — Setup, tree-sitter, tokenizer.
7. **R1B cells 5-6** — RUST_KEYWORD_MAP, word banks, identifier variants.
8. **R1B cells 7-9** — Validation, generation, save.

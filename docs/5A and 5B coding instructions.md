Experiment 2 — Token Independence Check
Coding Instructions for Claude Code
Purpose: For every universal circuit that has a keyword token, build a "token checker" mask from prompts where the token appears without the concept. Compare with the universal circuit to measure what fraction of the circuit is token-driven vs concept-driven.
Two notebooks:

5A — Prompt generation, validation, activation extraction, token checker mask creation
5B — Comparison with universal circuits, reporting

GitHub repo: https://github.com/piotrwilam/CSP-Atlas
Data sources:

Universal circuit masks: universal_106x50.h5 in Drive
CSP model: openai/circuit-sparsity on HuggingFace
Output: token_checker_masks.h5 + comparison CSV


1. Keyword Mapping
1.1 Which objects have testable keywords
Not all 106 objects have a keyword token. The mapping below defines which objects can be tested and what token to look for. Objects not in this map are reported as "no keyword — token confound not applicable."
python# Primary keyword for each object.
# Key = universal circuit name (as in HDF5)
# Value = dict with:
#   "keyword": the token string to embed in prompts
#   "forbidden_nodes": AST node types that must NOT appear in the parse tree
#   "notes": any special handling

KEYWORD_MAP = {
    # --- AST nodes with dedicated keywords ---
    "ast__Import":       {"keyword": "import",   "forbidden_nodes": ["Import", "ImportFrom"]},
    "ast__ImportFrom":   {"keyword": "from",     "forbidden_nodes": ["Import", "ImportFrom"]},
    "ast__Break":        {"keyword": "break",    "forbidden_nodes": ["Break"]},
    "ast__Pass":         {"keyword": "pass",     "forbidden_nodes": ["Pass"]},
    "ast__Continue":     {"keyword": "continue", "forbidden_nodes": ["Continue"]},
    "ast__Assert":       {"keyword": "assert",   "forbidden_nodes": ["Assert"]},
    "ast__If":           {"keyword": "if",       "forbidden_nodes": ["If", "IfExp"]},
    "ast__While":        {"keyword": "while",    "forbidden_nodes": ["While"]},
    "ast__For":          {"keyword": "for",      "forbidden_nodes": ["For", "AsyncFor", "ListComp", "SetComp", "DictComp", "GeneratorExp"]},
    "ast__Return":       {"keyword": "return",   "forbidden_nodes": ["Return"]},
    "ast__With":         {"keyword": "with",     "forbidden_nodes": ["With", "AsyncWith"]},
    "ast__Raise":        {"keyword": "raise",    "forbidden_nodes": ["Raise"]},
    "ast__Delete":       {"keyword": "del",      "forbidden_nodes": ["Delete"]},
    "ast__Global":       {"keyword": "global",   "forbidden_nodes": ["Global"]},
    "ast__Nonlocal":     {"keyword": "nonlocal", "forbidden_nodes": ["Nonlocal"]},
    "ast__Lambda":       {"keyword": "lambda",   "forbidden_nodes": ["Lambda"]},
    "ast__Yield":        {"keyword": "yield",    "forbidden_nodes": ["Yield", "YieldFrom"]},
    "ast__YieldFrom":    {"keyword": "yield",    "forbidden_nodes": ["Yield", "YieldFrom"]},
    "ast__Try":          {"keyword": "try",      "forbidden_nodes": ["Try"]},
    "ast__ClassDef":     {"keyword": "class",    "forbidden_nodes": ["ClassDef"]},
    "ast__FunctionDef":  {"keyword": "def",      "forbidden_nodes": ["FunctionDef", "AsyncFunctionDef"]},
    "ast__AsyncFor":     {"keyword": "async",    "forbidden_nodes": ["AsyncFor", "AsyncFunctionDef", "AsyncWith"]},
    "ast__AsyncFunctionDef": {"keyword": "async", "forbidden_nodes": ["AsyncFor", "AsyncFunctionDef", "AsyncWith"]},
    "ast__AsyncWith":    {"keyword": "async",    "forbidden_nodes": ["AsyncFor", "AsyncFunctionDef", "AsyncWith"]},

    # --- Builtins with short common-English tokens ---
    # These words appear naturally in strings/comments in real Python code
    "builtin__list":     {"keyword": "list",     "forbidden_nodes": [],  "forbidden_names": ["list"]},
    "builtin__set":      {"keyword": "set",      "forbidden_nodes": [],  "forbidden_names": ["set"]},
    "builtin__map":      {"keyword": "map",      "forbidden_nodes": [],  "forbidden_names": ["map"]},
    "builtin__open":     {"keyword": "open",     "forbidden_nodes": [],  "forbidden_names": ["open"]},
    "builtin__type":     {"keyword": "type",     "forbidden_nodes": [],  "forbidden_names": ["type"]},
    "builtin__hash":     {"keyword": "hash",     "forbidden_nodes": [],  "forbidden_names": ["hash"]},
    "builtin__id":       {"keyword": "id",       "forbidden_nodes": [],  "forbidden_names": ["id"]},
    "builtin__all":      {"keyword": "all",      "forbidden_nodes": [],  "forbidden_names": ["all"]},
    "builtin__any":      {"keyword": "any",      "forbidden_nodes": [],  "forbidden_names": ["any"]},
    "builtin__sum":      {"keyword": "sum",      "forbidden_nodes": [],  "forbidden_names": ["sum"]},
    "builtin__min":      {"keyword": "min",      "forbidden_nodes": [],  "forbidden_names": ["min"]},
    "builtin__max":      {"keyword": "max",      "forbidden_nodes": [],  "forbidden_names": ["max"]},
    "builtin__next":     {"keyword": "next",     "forbidden_nodes": [],  "forbidden_names": ["next"]},
    "builtin__input":    {"keyword": "input",    "forbidden_nodes": [],  "forbidden_names": ["input"]},
    "builtin__len":      {"keyword": "len",      "forbidden_nodes": [],  "forbidden_names": ["len"]},
    "builtin__range":    {"keyword": "range",    "forbidden_nodes": [],  "forbidden_names": ["range"]},
    "builtin__filter":   {"keyword": "filter",   "forbidden_nodes": [],  "forbidden_names": ["filter"]},
    "builtin__print":    {"keyword": "print",    "forbidden_nodes": [],  "forbidden_names": ["print"]},
    "builtin__int":      {"keyword": "int",      "forbidden_nodes": [],  "forbidden_names": ["int"]},
    "builtin__float":    {"keyword": "float",    "forbidden_nodes": [],  "forbidden_names": ["float"]},
    "builtin__str":      {"keyword": "str",      "forbidden_nodes": [],  "forbidden_names": ["str"]},
    "builtin__bool":     {"keyword": "bool",     "forbidden_nodes": [],  "forbidden_names": ["bool"]},
    "builtin__round":    {"keyword": "round",    "forbidden_nodes": [],  "forbidden_names": ["round"]},
    "builtin__zip":      {"keyword": "zip",      "forbidden_nodes": [],  "forbidden_names": ["zip"]},
    "builtin__sorted":   {"keyword": "sorted",   "forbidden_nodes": [],  "forbidden_names": ["sorted"]},
    "builtin__super":    {"keyword": "super",    "forbidden_nodes": [],  "forbidden_names": ["super"]},
    "builtin__iter":     {"keyword": "iter",     "forbidden_nodes": [],  "forbidden_names": ["iter"]},
    "builtin__object":   {"keyword": "object",   "forbidden_nodes": [],  "forbidden_names": ["object"]},
    "builtin__bytes":    {"keyword": "bytes",    "forbidden_nodes": [],  "forbidden_names": ["bytes"]},
    "builtin__dict":     {"keyword": "dict",     "forbidden_nodes": [],  "forbidden_names": ["dict"]},
    "builtin__tuple":    {"keyword": "tuple",    "forbidden_nodes": [],  "forbidden_names": ["tuple"]},
    "builtin__property": {"keyword": "property", "forbidden_nodes": [],  "forbidden_names": ["property"]},
    "builtin__complex":  {"keyword": "complex",  "forbidden_nodes": [],  "forbidden_names": ["complex"]},
    "builtin__reversed": {"keyword": "reversed", "forbidden_nodes": [],  "forbidden_names": ["reversed"]},
}
IMPORTANT — Builtin validation: For builtins, the "concept absent" condition is different from AST nodes. You cannot check the AST for absence of a builtin — list is just a Name node. Instead, check that the keyword does NOT appear as a function call or type reference in the code. The forbidden_names field lists identifiers that must not appear as ast.Name nodes used in ast.Call context or bare references. The keyword must appear ONLY inside strings, comments, or as part of a longer identifier.
1.2 Objects without testable keywords
The following objects have no testable keyword and are excluded from 5A. Report them in 5B as "N/A — no keyword confound."
AST nodes without a unique keyword: Assign, AnnAssign, AugAssign, Attribute, BinOp, BoolOp, Call, Compare, Dict, DictComp, GeneratorExp, IfExp, ListComp, Set, SetComp, Slice, Starred, Subscript, UnaryOp.
Builtins with tokens too technical to appear without the concept: ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, StopIteration, FileNotFoundError, OSError, ImportError, NotImplementedError, ZeroDivisionError, Exception, enumerate, isinstance, issubclass, frozenset, bytearray, memoryview, callable, classmethod, staticmethod, delattr, getattr, hasattr, setattr, repr, abs, _isolated_.
NOTE: The boundary is a judgment call. If Claude Code is unsure whether a builtin is testable, include it. The validation filter will reject prompts that don't work. Better to try and fail than to exclude a testable object.

2. Prompt Generation (5 Categories × 10 Prompts)
2.1 The 5 categories
For each keyword, generate 10 prompts per category. Each prompt is a complete, valid Python snippet where the keyword token appears but the associated concept does not.
Category A — String literal assignment. The keyword appears inside a quoted string assigned to a variable.
Template patterns (pick randomly, vary words):
pythonf'{var1} = "waiting {keyword} {var2}"'
f'{var1} = "{var2} {keyword} {var3} reference"'
f'{var1} = "ready {keyword} deployment"'
f'{var1} = "the {keyword} of {var2}"'
f'{var1} = "no {keyword} needed here"'
Category B — Comment. The keyword appears inside a # comment.
Template patterns:
pythonf'{var1} = 42  # {keyword} {var2} see docs'
f'# {keyword} {var2} details here\n{var1} = 10'
f'{var1} = []  # TODO {keyword} this later'
f'# note: {keyword} not applicable for {var2}\n{var1} = True'
Category C — Variable/function name containing the keyword. The keyword string is embedded in an identifier.
This is keyword-specific. Define a lookup:
pythonIDENTIFIER_VARIANTS = {
    "import":   ["important_data", "reimport_flag", "import_path_str"],
    "break":    ["breakdown_count", "breakpoint_flag", "unbreakable"],
    "pass":     ["passport_number", "password_hash", "bypass_mode"],
    "continue": ["continuous_mode", "discontinue_flag", "continuum"],
    "for":      ["format_string", "information", "formula_result", "before_update"],
    "if":       ["notification", "verification", "modification", "amplifier"],
    "while":    ["meanwhile_flag", "worthwhile", "awhile_counter"],
    "return":   ["return_value_str", "unreturnable", "return_label"],
    "assert":   ["assertive_mode", "reassert_count"],
    "with":     ["width_value", "withdraw_amount", "within_range"],
    "class":    ["classification", "classic_mode", "subclass_name"],
    "def":      ["default_value", "undefined_flag", "defiant_mode", "defense_level"],
    "yield":    ["yielded_count", "unyielding"],
    "try":      ["retry_count", "country_code", "entry_point"],
    "raise":    ["raised_flag", "fundraise_total", "praise_count"],
    "del":      ["delivery_date", "delta_value", "model_name", "delegate_to"],
    "global":   ["global_str_ref", "globalization"],
    "nonlocal": ["nonlocal_ref_str"],
    "lambda":   ["lambda_str_ref"],
    "async":    ["async_str_label"],
    # Builtins
    "list":     ["checklist_items", "listing_data", "blacklist_mode"],
    "set":      ["settings_config", "reset_flag", "offset_value", "sunset_time"],
    "map":      ["mapped_result", "roadmap_items", "bitmap_size"],
    "dict":     ["dictionary_words", "verdict_text", "dictation_mode"],
    "open":     ["opening_hours", "reopening", "openly_shared"],
    "type":     ["typewriter_mode", "archetype", "prototype_id"],
    "print":    ["printable_chars", "fingerprint_id", "blueprint_ref"],
    "int":      ["interval_ms", "painting_id", "intelligence", "pointer_val"],
    "float":    ["floating_label", "afloat_status"],
    "str":      ["string_data", "stream_id", "structure_type", "destroy_flag"],
    "range":    ["arrangement", "strange_value", "orange_count"],
    "sum":      ["summary_text", "consumer_id", "assumption"],
    "max":      ["maximum_str", "climax_point"],
    "min":      ["minimum_str", "admin_level", "minute_count", "dominion"],
    "len":      ["length_str", "lender_id", "silent_mode", "calendar_ref"],
    "hash":     ["hashtag_count", "rehash_data"],
    "bool":     ["boolean_str", "booleanize"],
    "round":    ["roundup_notes", "background_val", "surround_count"],
    "zip":      ["zipcode_val", "unzip_path"],
    "all":      ["alliance_name", "install_path", "wallpaper"],
    "any":      ["company_name", "many_items", "tyranny"],
    "next":     ["next_str_ref", "annex_id"],
    "input":    ["input_str_ref"],
    "filter":   ["filter_str_ref", "unfiltered_data"],
    "object":   ["objection_text", "objective_id"],
    "id":       ["idea_count", "identity_str", "video_ref", "oxide_level"],
    "super":    ["supervisor_name", "superfluous"],
    "iter":     ["literal_count", "iterator_str_ref"],
    "bytes":    ["bytes_str_ref", "gigabytes_val"],
    "sorted":   ["sorted_str_ref", "unsorted_data"],
    "tuple":    ["tuple_str_ref", "quintuple_val"],
    "property": ["property_str_ref"],
    "complex":  ["complexity_score", "complex_str_ref"],
    "reversed": ["reversed_str_ref"],
}
CRITICAL: Not all of these will work. The CSP tokenizer may or may not split the identifier to include the keyword as a separate token. The validation step (Section 3) will check this. Generate all variants, let validation filter.
Category D — Dictionary key. The keyword appears as a string key in a dict literal.
Template patterns:
pythonf'{var1} = {{"{keyword}": True, "{var2}": 42}}'
f'{var1} = {{"{var2}": 10, "mode": "{keyword}"}}'
f'config = {{"action": "{var1}", "{keyword}": "{var2}"}}'
Category E — Print/logging call with string argument. The keyword appears inside a string passed to print().
Template patterns:
pythonf'print("starting {keyword} {var1}")'
f'print("{var1} {keyword} completed")'
f'print("ready {keyword} {var2} now")'
2.2 Randomization
Each prompt should vary:

Variable names: Draw from a word bank of 30+ neutral Python variable names (not overlapping with any keyword).
Surrounding words in strings: Draw from a bank of 50+ common English words.
Context wrapper: Wrap the snippet randomly in one of:

Global scope with padding: x = 1\n{snippet}\ny = 2
Inside a function: def func():\n    {snippet}\n    return None
Inside a class method: class Obj:\n    def method(self):\n        {snippet}



2.3 Word banks
pythonVAR_NAMES = [
    "data", "result", "value", "item", "record", "entry", "output",
    "signal", "status", "config", "mode", "level", "target", "source",
    "handler", "token", "state", "buffer", "counter", "index",
    "offset", "block", "phase", "cycle", "frame", "chunk", "batch",
    "queue", "score", "label", "flag", "limit", "depth", "width_val",
]

STRING_WORDS = [
    "waiting", "ready", "starting", "completed", "processing",
    "running", "loading", "checking", "updating", "building",
    "reading", "writing", "sending", "receiving", "tracking",
    "reference", "documentation", "deployment", "production",
    "analysis", "operation", "execution", "validation", "review",
    "schedule", "planning", "progress", "workflow", "pipeline",
    "resource", "component", "structure", "algorithm", "protocol",
    "standard", "baseline", "threshold", "interval", "duration",
    "capacity", "frequency", "priority", "sequence", "pattern",
    "segment", "fragment", "section", "portion", "division",
]

CONTEXT_WRAPPERS = [
    "{snippet}",
    "x = 1\n{snippet}\ny = 2",
    "data = []\n{snippet}\nresult = None",
    "def func():\n    {snippet}\n    return None",
    "def process():\n    {snippet}",
    "class Obj:\n    def method(self):\n        {snippet}",
]
CRITICAL — Variable names must not contain any keyword. Scan VAR_NAMES and remove any that contain a testable keyword as a substring. For example, "counter" contains no keywords. But "format" contains "for". Validate this programmatically at startup.
2.4 The generation function
pythonimport random

def generate_raw_prompts(keyword, n_per_category=10):
    """
    Generate candidate prompts for one keyword across all 5 categories.
    Returns list of (prompt_text, category_label) tuples.
    No validation yet — that happens in Section 3.
    """
    candidates = []

    for _ in range(n_per_category):
        w1, w2, w3 = random.sample(STRING_WORDS, 3)
        v1, v2 = random.sample(VAR_NAMES, 2)
        wrapper = random.choice(CONTEXT_WRAPPERS)

        # Category A — String literal
        snippet_a = random.choice([
            f'{v1} = "waiting {keyword} {w1}"',
            f'{v1} = "{w1} {keyword} {w2} reference"',
            f'{v1} = "the {keyword} of {w1}"',
            f'{v1} = "no {keyword} needed for {w1}"',
            f'{v1} = "{w1} {keyword} {w2}"',
        ])
        candidates.append((wrapper.format(snippet=snippet_a), "A_string"))

        # Category B — Comment
        snippet_b = random.choice([
            f'{v1} = 42  # {keyword} {w1} see docs',
            f'# {keyword} {w1} {w2} reference\n{v1} = 10',
            f'{v1} = []  # TODO {keyword} this later',
            f'# note: {keyword} not needed for {w1}\n{v1} = True',
        ])
        candidates.append((wrapper.format(snippet=snippet_b), "B_comment"))

        # Category C — Identifier
        ident_variants = IDENTIFIER_VARIANTS.get(keyword, [])
        if ident_variants:
            ident = random.choice(ident_variants)
            snippet_c = random.choice([
                f'{ident} = 42',
                f'{ident} = "{w1}"',
                f'{ident} = [{v1} for {v1} in []]' if keyword != "for" else f'{ident} = 42',
                f'{ident} = True',
            ])
            candidates.append((wrapper.format(snippet=snippet_c), "C_identifier"))

        # Category D — Dict key
        snippet_d = random.choice([
            f'{v1} = {{"{keyword}": True, "{w1}": 42}}',
            f'{v1} = {{"{w1}": 10, "mode": "{keyword}"}}',
            f'{v1} = {{"action": "{w1}", "{keyword}": "{w2}"}}',
        ])
        candidates.append((wrapper.format(snippet=snippet_d), "D_dictkey"))

        # Category E — Print call
        snippet_e = random.choice([
            f'print("{w1} {keyword} {w2}")',
            f'print("starting {keyword} {w1}")',
            f'print("{keyword} completed for {w1}")',
        ])
        candidates.append((wrapper.format(snippet=snippet_e), "E_print"))

    return candidates
2.5 Over-generate to guarantee 50 valid prompts
The validation filter (Section 3) will reject some prompts. Generate 3x the needed amount (150 raw → keep first 50 valid). If fewer than 50 survive, increase generation or relax the identifier category.
pythonN_TARGET = 50
OVERSHOOT_FACTOR = 3

def generate_for_keyword(keyword, target_n=N_TARGET):
    """Generate and validate prompts. Returns exactly target_n or fewer."""
    raw = generate_raw_prompts(keyword, n_per_category=target_n * OVERSHOOT_FACTOR // 5)
    valid = [p for p in raw if validate_prompt(p[0], keyword)]
    
    # Balance categories: take up to target_n // 5 per category, fill remainder from any
    by_cat = {}
    for text, cat in valid:
        by_cat.setdefault(cat, []).append(text)
    
    per_cat = target_n // 5  # = 10
    selected = []
    for cat in ["A_string", "B_comment", "C_identifier", "D_dictkey", "E_print"]:
        selected.extend(by_cat.get(cat, [])[:per_cat])
    
    # If short (e.g. C_identifier had few valid), fill from others
    if len(selected) < target_n:
        remaining = [t for t, c in valid if t not in selected]
        selected.extend(remaining[:target_n - len(selected)])
    
    return selected[:target_n]

3. Validation
Every prompt must pass ALL THREE checks.
3.1 Parse check
pythonimport ast as ast_module

def get_all_ast_info(code_string):
    """Return set of all AST node type names and all Name identifiers."""
    try:
        tree = ast_module.parse(code_string)
    except SyntaxError:
        return None, None
    
    node_types = set()
    name_ids = set()  # identifiers used as Name nodes
    call_names = set()  # identifiers used as function calls
    
    for node in ast_module.walk(tree):
        node_types.add(type(node).__name__)
        if isinstance(node, ast_module.Name):
            name_ids.add(node.id)
        if isinstance(node, ast_module.Call):
            if isinstance(node.func, ast_module.Name):
                call_names.add(node.func.id)
    
    return node_types, call_names
3.2 Concept absence check
pythondef check_concept_absent(code_string, obj_key):
    """
    Verify the target concept is NOT present in the parse tree.
    """
    info = KEYWORD_MAP[obj_key]
    node_types, call_names = get_all_ast_info(code_string)
    
    if node_types is None:
        return False  # parse failed
    
    # Check forbidden AST nodes
    for forbidden in info["forbidden_nodes"]:
        if forbidden in node_types:
            return False
    
    # Check forbidden builtin names (for builtins only)
    if "forbidden_names" in info:
        for forbidden_name in info["forbidden_names"]:
            if forbidden_name in call_names:
                return False
    
    return True
3.3 Token presence check
pythondef check_token_present(code_string, keyword, tokenizer):
    """
    Verify the keyword appears as a token in the tokenizer output.
    
    We check if any individual token decodes to exactly the keyword,
    OR if the keyword appears as a substring in the decoded token
    (for cases where the tokenizer keeps it as part of a larger token
    inside a string).
    """
    token_ids = tokenizer.encode(code_string, add_special_tokens=False)
    
    # Method 1: exact match
    keyword_token_ids = tokenizer.encode(keyword, add_special_tokens=False)
    # Check if keyword_token_ids appears as a contiguous subsequence
    ids_list = token_ids
    kw_len = len(keyword_token_ids)
    for i in range(len(ids_list) - kw_len + 1):
        if ids_list[i:i+kw_len] == keyword_token_ids:
            return True
    
    # Method 2: decoded substring match (fallback)
    for tid in token_ids:
        decoded = tokenizer.decode([tid]).strip()
        if decoded == keyword:
            return True
    
    return False
3.4 Combined validation
pythondef validate_prompt(code_string, obj_key, tokenizer):
    """Returns True if prompt is valid for token-without-concept testing."""
    keyword = KEYWORD_MAP[obj_key]["keyword"]
    
    # 1. Must parse
    node_types, call_names = get_all_ast_info(code_string)
    if node_types is None:
        return False
    
    # 2. Concept must be absent
    if not check_concept_absent(code_string, obj_key):
        return False
    
    # 3. Token must be present
    if not check_token_present(code_string, keyword, tokenizer):
        return False
    
    return True

4. Activation Extraction and Token Checker Mask
Use the IDENTICAL pipeline as Module 2. Same model, same hooks, same epsilon, same consistency threshold. This is critical — any methodological difference could create spurious differences between the token checker and the universal circuit.
4.1 Parameters — must match Module 2 exactly
pythonEPSILON = 0.001
CONSISTENCY_THRESHOLD = 0.8
N_LAYERS = 8
TOKEN_POSITION = "last"  # final token position
4.2 Extraction
Copy the activation extraction code from Module 2 (ActivationExtractor class). Do not reimplement. If Module 2 code is available in the repo, import it directly. If not, copy the class.
python# For each object in KEYWORD_MAP:
#   1. Generate 50 validated prompts
#   2. Feed each through CSP model
#   3. Record MLP output at last token position, all 8 layers
#   4. Apply epsilon threshold → binary mask per prompt per layer
#   5. Compute consistency score across 50 prompts
#   6. Apply consistency threshold → token checker mask per layer

# Output: token_checker_masks[obj_key][layer_id] = np.array(bool, 2048)
4.3 Save to HDF5
python# Save with same structure as the universal masks
OUTPUT_FILE = DATA_DIR / "token_checker_masks.h5"

with h5py.File(OUTPUT_FILE, "w") as f:
    tcm = f.create_group("token_checker_masks")
    
    for obj_key, layers in token_checker_masks.items():
        for layer_id, mask in layers.items():
            tcm.create_dataset(
                f"layer_{layer_id}/{obj_key}",
                data=mask.astype(np.bool_),
                compression="gzip"
            )
    
    # Metadata
    md = f.create_group("metadata")
    md.attrs["epsilon"] = EPSILON
    md.attrs["consistency_threshold"] = CONSISTENCY_THRESHOLD
    md.attrs["n_prompts_per_object"] = N_TARGET
    md.attrs["categories"] = "A_string,B_comment,C_identifier,D_dictkey,E_print"
    
    # Per-object stats
    stats = f.create_group("generation_stats")
    for obj_key in token_checker_masks:
        stats.attrs[f"{obj_key}_n_valid"] = prompt_counts[obj_key]  # actual count used

print(f"Saved: {OUTPUT_FILE}")
4.4 Also save the prompts themselves
For reproducibility and debugging, save all validated prompts.
pythonPROMPTS_FILE = DATA_DIR / "token_checker_prompts.parquet"

records = []
for obj_key, prompts in all_validated_prompts.items():
    for i, text in enumerate(prompts):
        records.append({
            "object": obj_key,
            "keyword": KEYWORD_MAP[obj_key]["keyword"],
            "variation_id": i,
            "prompt_text": text,
        })

pd.DataFrame(records).to_parquet(PROMPTS_FILE)
print(f"Saved: {PROMPTS_FILE}")

5. Notebook 5A Structure
CellTypeContent0MarkdownTitle, purpose, method summary1CodeConfiguration: paths, parameters (must match Module 2)2CodeImports, pip installs3CodeMount Drive / detect environment4CodeLoad CSP model and tokenizer5CodeKEYWORD_MAP definition (Section 1.1)6CodeIDENTIFIER_VARIANTS definition (Section 2.3)7CodeWord banks: VAR_NAMES, STRING_WORDS, CONTEXT_WRAPPERS (Section 2.3)8CodeValidation functions: get_all_ast_info, check_concept_absent, check_token_present, validate_prompt (Section 3)9CodeStartup validation: scan VAR_NAMES for keyword conflicts, remove any10CodeGeneration functions: generate_raw_prompts, generate_for_keyword (Section 2.4–2.5)11CodeMain generation loop: for each object in KEYWORD_MAP, generate 50 validated prompts. Print per-object stats (how many raw, how many valid, category breakdown). Flag objects with < 50 valid.12CodeActivationExtractor class (copied from Module 2)13CodeToken checker mask extraction loop: for each object, run 50 prompts through the model, apply epsilon + consistency filter. Store masks.14CodeSave token_checker_masks.h5 (Section 4.3)15CodeSave token_checker_prompts.parquet (Section 4.4)16CodeSummary: how many objects tested, how many had enough prompts, mask sizes per layer

6. Notebook 5B — Comparison and Reporting
6.1 Load both mask sets
python# Load universal circuit masks (from Module 2 atlas)
# Load token checker masks (from 5A output)

universal = {}   # {obj_key: {layer: bool_array}}
token_chk = {}   # {obj_key: {layer: bool_array}}

# ... load from respective HDF5 files ...
6.2 Core comparison
For each object that has both a universal circuit and a token checker mask, at each layer, compute:
pythondef compare_masks(universal_mask, token_mask):
    """
    Compare universal circuit (A) with token checker (B).
    
    Returns dict with:
        A_only: neurons in universal but not in token checker (concept-driven)
        A_and_B: neurons in both (possibly token-driven)
        B_only: neurons in token checker but not in universal (token-reactive, not concept)
        token_fraction: |A∩B| / |A| (fraction of universal that is token-explained)
        concept_fraction: |A\B| / |A| (fraction of universal that is concept-only)
    """
    A = universal_mask
    B = token_mask
    
    a_only = np.logical_and(A, ~B).sum()
    a_and_b = np.logical_and(A, B).sum()
    b_only = np.logical_and(~A, B).sum()
    a_size = A.sum()
    b_size = B.sum()
    
    token_fraction = float(a_and_b / a_size) if a_size > 0 else 0.0
    concept_fraction = float(a_only / a_size) if a_size > 0 else 0.0
    
    return {
        "A_size": int(a_size),
        "B_size": int(b_size),
        "A_only": int(a_only),
        "A_and_B": int(a_and_b),
        "B_only": int(b_only),
        "token_fraction": round(token_fraction, 4),
        "concept_fraction": round(concept_fraction, 4),
    }
6.3 Main comparison loop
pythoncomparison_records = []

for obj_key in sorted(KEYWORD_MAP.keys()):
    if obj_key not in token_chk:
        continue  # no token checker for this object (insufficient prompts)
    if obj_key not in universal:
        continue  # no universal circuit for this object
    
    for layer_id in range(N_LAYERS):
        result = compare_masks(
            universal[obj_key][layer_id],
            token_chk[obj_key][layer_id]
        )
        result["object"] = obj_key
        result["type"] = "ast" if obj_key.startswith("ast__") else "builtin"
        result["keyword"] = KEYWORD_MAP[obj_key]["keyword"]
        result["layer"] = layer_id
        comparison_records.append(result)

comparison_df = pd.DataFrame(comparison_records)
6.4 Output tables
Table 1 — Per-object summary (one row per object):
For each object, average concept_fraction across all 8 layers. Sort descending. This is the "how much of this circuit is NOT explained by the token" table.
pythonsummary = comparison_df.groupby(["object", "type", "keyword"]).agg({
    "concept_fraction": "mean",
    "token_fraction": "mean",
    "A_size": "mean",
    "A_and_B": "mean",
    "A_only": "mean",
    "B_only": "mean",
}).round(4).sort_values("concept_fraction", ascending=False)
Table 2 — Per-object per-layer detail (full table):
Save as CSV for inspection.
Table 3 — Objects without keywords:
List all objects not in KEYWORD_MAP with a note: "No keyword token — token confound does not apply to this circuit."
6.5 Key summary statistics
pythonprint("=== TOKEN INDEPENDENCE RESULTS ===\n")

# Overall
mean_concept = comparison_df["concept_fraction"].mean()
mean_token = comparison_df["token_fraction"].mean()
print(f"Across all tested circuits and layers:")
print(f"  Mean concept-only fraction: {mean_concept:.3f}")
print(f"  Mean token-overlap fraction: {mean_token:.3f}")
print()

# By type
for obj_type in ["ast", "builtin"]:
    subset = comparison_df[comparison_df["type"] == obj_type]
    if len(subset) == 0:
        continue
    print(f"{obj_type.upper()} circuits:")
    print(f"  Mean concept fraction: {subset['concept_fraction'].mean():.3f}")
    print(f"  Mean token fraction:   {subset['token_fraction'].mean():.3f}")
    print()

# Highlight: modular circuits (the top scorers from Experiment 1)
MODULAR = ["ast__Import", "ast__Break", "ast__Pass", "ast__ImportFrom",
           "ast__Continue", "ast__Assert"]
mod_subset = comparison_df[comparison_df["object"].isin(MODULAR)]
if len(mod_subset) > 0:
    print("MODULAR circuits (score > 0 in Experiment 1):")
    print(f"  Mean concept fraction: {mod_subset['concept_fraction'].mean():.3f}")
    print(f"  Mean token fraction:   {mod_subset['token_fraction'].mean():.3f}")
    print()

# Any circuit where token_fraction > 0.5 at any layer?
high_token = comparison_df[comparison_df["token_fraction"] > 0.5]
if len(high_token) > 0:
    print(f"WARNING: {high_token['object'].nunique()} circuits have token_fraction > 0.5 at some layer:")
    for obj in high_token["object"].unique():
        layers = high_token[high_token["object"] == obj]["layer"].tolist()
        print(f"  {obj}: layers {layers}")
6.6 Save outputs
python# Full detail
comparison_df.to_csv(DATA_DIR / "token_independence_detail.csv", index=False)

# Summary per object
summary.to_csv(DATA_DIR / "token_independence_summary.csv")

# Objects without keywords
no_keyword = [obj for obj in all_objects if obj not in KEYWORD_MAP]
pd.DataFrame({"object": no_keyword, "status": "no_keyword"}).to_csv(
    DATA_DIR / "token_independence_no_keyword.csv", index=False
)

7. Notebook 5B Structure
CellTypeContent0MarkdownTitle, purpose1CodeConfiguration, imports2CodeMount Drive / detect environment3CodeLoad universal circuit masks from Module 2 HDF54CodeLoad token checker masks from 5A HDF55CodeKEYWORD_MAP (same as 5A, needed for metadata)6Codecompare_masks function (Section 6.2)7CodeMain comparison loop (Section 6.3)8CodeTable 1: per-object summary (Section 6.4)9CodeTable 2: per-object per-layer detail (Section 6.4)10CodeTable 3: objects without keywords (Section 6.4)11CodeSummary statistics (Section 6.5)12CodeSave all outputs (Section 6.6)

8. Critical Implementation Notes

Parameter identity. EPSILON, CONSISTENCY_THRESHOLD, and TOKEN_POSITION must be identical to Module 2. Any difference invalidates the comparison. Print them at the start of 5A and verify against the Module 2 HDF5 metadata.
Tokenizer. Use the CSP tokenizer (AutoTokenizer.from_pretrained("openai/circuit-sparsity")), not a generic GPT-2 tokenizer. Token boundaries differ.
Comment tokenization. The # comment character and everything after it may or may not be tokenized. Verify with a test prompt that the keyword token actually appears in the token list when inside a comment. If the tokenizer strips comments, Category B is invalid.
The for keyword is the hardest. It appears in For loops, comprehensions (ListComp, SetComp, DictComp, GeneratorExp), and AsyncFor. The forbidden_nodes list must include ALL of these. A prompt like x = [i for i in range(10)] contains for but is a ListComp, not a For — and we want to exclude it from the For token checker.
The from keyword appears in ImportFrom (from X import Y) but can also appear in yield from (YieldFrom). The forbidden_nodes for ImportFrom should include both ImportFrom and Import (since we're testing the from token). Do NOT include YieldFrom in forbidden — that uses a different AST relationship.
Builtins with the forbidden_names check. For builtins like list, the validation must ensure the word list does not appear as a function call (list(...)) or bare reference (x = list). But it CAN appear inside strings, comments, or as part of identifiers (checklist). The get_all_ast_info function captures Call names separately for this purpose.
Empty token checker masks. If a keyword is very rare in non-concept contexts (e.g., nonlocal, lambda), the token checker mask may have very few or zero active neurons after consistency filtering. This is a valid result — report it as "token produces no consistent activation pattern when concept is absent."
Do not filter prompts by model perplexity. Module 2 filters prompts by sequence loss to ensure the model can process them. For the token checker, skip this filter. The prompts are simple Python (string assignments, comments, print calls) — the model will handle them fine. Adding a perplexity filter would bias toward prompts the model "recognizes," which is the opposite of what we want.
Batch processing. Process prompts in batches of 64 with left-padded tokenization, same as Module 2. Use the same attention mask handling.
Reproducibility. Set random.seed(42) and np.random.seed(42) before generation. Save all prompts to parquet so the experiment can be reproduced.
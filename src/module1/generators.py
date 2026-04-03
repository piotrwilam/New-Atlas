"""
Stage B — Variance Engine / AST Prompt Generator.

For each (AST node, builtin) pair, generates N semantically orthogonal prompts
that share the same structural essence but differ in lexical domain, wrapper
type, and padding. All output is routed through ast.parse() → ast.unparse()
to guarantee canonical PEP 8 spacing (Atlas 2 §2).
"""

import ast
import builtins as builtins_module
import logging
import random
import textwrap

from .variance_schema import (
    DOMAINS,
    DOMAIN_KEYS,
    PADDING_BEFORE,
    PADDING_AFTER,
    WRAPPER_TYPES,
    WRAPPER_WEIGHTS,
)

logger = logging.getLogger(__name__)


class ASTPromptGenerator:
    """Generate structurally verified, semantically diverse Python prompts."""

    def __init__(self, domains: dict = None):
        self.domains = domains or DOMAINS
        self.domain_keys = list(self.domains.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self, ast_node: str, builtin_obj: str, n: int = 150
    ) -> list[dict]:
        """Generate up to *n* prompt dicts for the (ast_node, builtin_obj) pair.

        Returns a (possibly shorter) list; entries that fail to parse are
        silently dropped.
        """
        results = []
        for i in range(n):
            try:
                dk = self.domain_keys[i % len(self.domain_keys)]
                domain = self.domains[dk]
                d, m = domain["var_names"], domain["mock_data"]

                essence = self._build_essence(ast_node, builtin_obj, d, m)
                if essence is None:
                    continue

                pad_b = random.choice(PADDING_BEFORE)
                pad_a = random.choice(PADDING_AFTER)
                body = "\n".join(p for p in [pad_b, essence, pad_a] if p)

                wrapper = random.choices(WRAPPER_TYPES, weights=WRAPPER_WEIGHTS, k=1)[0]
                wrapped = self._apply_wrapper(body, wrapper, d)

                tree = ast.parse(wrapped)
                prompt_text = ast.unparse(tree)
                verified = self._verify(tree, ast_node)

                results.append({
                    "prompt_text": prompt_text,
                    "ast_verified": verified,
                    "domain": dk,
                    "wrapper": wrapper,
                })
            except SyntaxError:
                continue
            except Exception as exc:
                logger.debug("Error (%s, %s) #%d: %s", ast_node, builtin_obj, i, exc)
                continue

        if not results:
            logger.warning("ZERO prompts for (%s, %s)", ast_node, builtin_obj)
        return results

    # ------------------------------------------------------------------
    # Essence builder — dispatch table over ~35 AST node types
    # ------------------------------------------------------------------

    def _build_essence(self, node: str, bobj: str, d: dict, m: dict) -> str | None:
        def mock(t=None):
            return m.get(t or bobj, m.get("list", "[1, 2, 3]"))

        def iterable_expr():
            if bobj in ("list", "tuple", "set", "frozenset"):
                return f"{d['list']} = {mock('list')}"
            elif bobj == "dict":
                return f"{d['dict']} = {mock('dict')}"
            elif bobj == "range":
                return f"{d['list']} = list(range(10))"
            elif bobj == "str":
                return f"{d['list']} = list({mock('str')})"
            elif bobj == "int":
                return f"{d['list']} = list(range({mock('int')} % 20))"
            elif bobj in ("bytes", "bytearray"):
                return f"{d['list']} = list(b'hello')"
            else:
                return f"{d['list']} = [str({bobj}), str(type({bobj}))]"

        T = {
            "For": lambda: f"""{iterable_expr()}
for {d['item']} in {d['list']}:
    {d['func']}({d['item']})""",

            "While": lambda: f"""{d['value']} = {mock('int')}
while {d['value']} > 0:
    {d['func']}({d['value']})
    {d['value']} -= 1""",

            "AsyncFor": lambda: f"""async def _run():
    {iterable_expr()}
    async for {d['item']} in {d['list']}:
        {d['func']}({d['item']})""",

            "If": lambda: f"""{d['value']} = {mock()}
if isinstance({d['value']}, {bobj}):
    {d['func']}({d['value']})
else:
    pass""",

            "IfExp": lambda: f"""{d['value']} = {mock()}
result = {d['func']}({d['value']}) if isinstance({d['value']}, {bobj}) else None""",

            "ListComp": lambda: f"""{iterable_expr()}
result = [{d['item']} for {d['item']} in {d['list']}]""",

            "DictComp": lambda: f"""{iterable_expr()}
result = {{str(k): k for k in {d['list']}}}""",

            "SetComp": lambda: f"""{iterable_expr()}
result = {{{d['item']} for {d['item']} in {d['list']}}}""",

            "GeneratorExp": lambda: f"""{iterable_expr()}
result = list({d['item']} for {d['item']} in {d['list']})""",

            "Try": lambda: self._try_tpl(bobj, d, m),
            "ExceptHandler": lambda: self._try_tpl(bobj, d, m),

            "Raise": lambda: (
                f"if not isinstance({mock()}, {bobj}):\n    raise {bobj}('Invalid: ' + str({mock()}))"
                if self._is_exc(bobj) else
                f"{d['value']} = {mock()}\nif {d['value']} is None:\n    raise ValueError(str({bobj}))"
            ),

            "FunctionDef": lambda: (
                f"def {d['func']}({d['item']}: {bobj}) -> {bobj}:\n    result = {bobj}({d['item']})\n    return result"
                if self._is_call(bobj) else
                f"def {d['func']}({d['item']}):\n    \"\"\"Process using {bobj}.\"\"\"\n    return str({d['item']})"
            ),

            "AsyncFunctionDef": lambda: f"""async def {d['func']}({d['item']}):
    result = {bobj}({d['item']}) if callable({bobj}) else str({d['item']})
    return result""",

            "ClassDef": lambda: f"""class {d['class']}:
    data_type = {bobj}
    def __init__(self):
        self.{d['value']} = {mock()}
    def {d['method']}(self):
        return self.{d['value']}""",

            "Return": lambda: f"""def {d['func']}():
    {d['value']} = {mock()}
    return {bobj}({d['value']}) if callable({bobj}) else {d['value']}""",

            "Assign": lambda: f"{d['value']} = {mock()}",
            "AugAssign": lambda: f"{d['value']} = {mock('int')}\n{d['value']} += 1",
            "AnnAssign": lambda: (
                f"{d['value']}: {bobj} = {mock()}" if self._is_call(bobj)
                else f"{d['value']}: str = str({mock()})"
            ),
            "Import": lambda: "import ast",
            "ImportFrom": lambda: "from collections import OrderedDict",

            "Break": lambda: f"""{iterable_expr()}
for {d['item']} in {d['list']}:
    break""",

            "Continue": lambda: f"""{iterable_expr()}
for {d['item']} in {d['list']}:
    if {d['item']}:
        continue
    {d['func']}({d['item']})""",

            "Pass": lambda: f"class {d['class']}:\n    pass",
            "Delete": lambda: f"{d['value']} = {mock()}\ndel {d['value']}",

            "With": lambda: f"""class {d['class']}:
    def __enter__(self): return self
    def __exit__(self, *a): pass
with {d['class']}() as ctx:
    {d['value']} = {mock()}""",

            "AsyncWith": lambda: f"""async def _run():
    class {d['class']}:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
    async with {d['class']}() as ctx:
        {d['value']} = {mock()}""",

            "Call": lambda: (
                f"result = {bobj}({mock()})" if self._is_call(bobj)
                else f"result = str({mock()})"
            ),
            "Attribute": lambda: f"{d['value']} = {mock()}\nresult = {d['value']}.__class__.__name__",
            "Subscript": lambda: f"{iterable_expr()}\nresult = {d['list']}[0]",
            "Lambda": lambda: f"transform = lambda x: {bobj}(x) if callable({bobj}) else x\nresult = transform({mock()})",
            "BinOp": lambda: f"{d['value']} = {mock('int')}\nresult = {d['value']} + 1",
            "BoolOp": lambda: f"{d['value']} = {mock()}\nresult = {d['value']} and True",
            "UnaryOp": lambda: f"{d['value']} = {mock('int')}\nresult = -{d['value']}",
            "Compare": lambda: f"{d['value']} = {mock()}\nresult = {d['value']} == {mock()}",
            "Assert": lambda: f"{d['value']} = {mock()}\nassert {d['value']} is not None",

            "Global": lambda: f"""{d['value']} = {mock()}
def {d['func']}():
    global {d['value']}
    {d['value']} = {mock()}""",

            "Nonlocal": lambda: f"""def {d['func']}_outer():
    {d['value']} = {mock()}
    def inner():
        nonlocal {d['value']}
        {d['value']} = {mock()}
    inner()""",

            "Yield": lambda: f"""def {d['func']}():
    {iterable_expr()}
    for {d['item']} in {d['list']}:
        yield {d['item']}""",

            "YieldFrom": lambda: f"""def {d['func']}():
    {iterable_expr()}
    yield from {d['list']}""",

            "Starred": lambda: f"{iterable_expr()}\nfirst, *rest = {d['list']}",
            "Slice": lambda: f"{iterable_expr()}\nresult = {d['list']}[1:3]",
            "Dict": lambda: f"result = {mock('dict')}",
            "Set": lambda: f"result = {mock('set')}",
        }

        builder = T.get(node)
        if builder is None:
            return None
        try:
            return builder()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_tpl(self, bobj: str, d: dict, m: dict) -> str:
        if self._is_exc(bobj):
            return (
                f"try:\n    {d['value']} = {m.get('int', '42')}\n"
                f"    result = int({d['value']})\n"
                f"except {bobj} as e:\n    {d['func']}(str(e))"
            )
        return (
            f"try:\n    result = {bobj}({m.get('list', '[1,2,3]')})\n"
            f"except Exception as e:\n    {d['func']}(str(e))"
        )

    def _apply_wrapper(self, body: str, wrapper: str, d: dict) -> str:
        if wrapper == "global":
            return body
        elif wrapper == "function":
            return f"def {d['func']}_main():\n{textwrap.indent(body, '    ')}"
        elif wrapper == "method":
            return (
                f"class {d['class']}Main:\n"
                f"    def {d['method']}_run(self):\n"
                f"{textwrap.indent(body, '        ')}"
            )
        return body

    def _verify(self, tree: ast.AST, target: str) -> bool:
        tc = getattr(ast, target, None)
        return tc is not None and any(isinstance(n, tc) for n in ast.walk(tree))

    @staticmethod
    def _is_exc(name: str) -> bool:
        obj = getattr(builtins_module, name, None)
        return obj is not None and isinstance(obj, type) and issubclass(obj, BaseException)

    @staticmethod
    def _is_call(name: str) -> bool:
        obj = getattr(builtins_module, name, None)
        return obj is not None and callable(obj)

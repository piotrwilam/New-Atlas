"""
Stage A — Concept Matrix Generator.

Introspects Python's `ast` and `builtins` modules to produce the (AST node,
builtin object) pairs that drive the rest of the pipeline.
"""

import ast
import builtins as builtins_module


# Abstract / meta nodes that are never concrete prompt targets
_ABSTRACT_AST = {
    'AST', 'mod', 'stmt', 'expr', 'expr_context', 'boolop',
    'operator', 'unaryop', 'cmpop', 'comprehension', 'excepthandler',
    'arguments', 'arg', 'keyword', 'alias', 'withitem',
    'match_case', 'pattern', 'type_ignore', 'type_param',
    'ParamSpec', 'TypeVar', 'TypeVarTuple',
}


def get_ast_nodes() -> list[str]:
    """Return sorted list of concrete AST node class names."""
    nodes = []
    for name in dir(ast):
        obj = getattr(ast, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ast.AST)
            and name not in _ABSTRACT_AST
            and not name.startswith('_')
        ):
            nodes.append(name)
    return sorted(nodes)


def get_builtin_objects() -> list[str]:
    """Return sorted list of all public builtin names."""
    return sorted([n for n in dir(builtins_module) if not n.startswith('_')])


def generate_concept_matrix(
    ast_nodes: list[str],
    builtin_objs: list[str],
) -> list[tuple[str, str]]:
    """Return the Cartesian product of ast_nodes × builtin_objs."""
    return [(a, b) for a in ast_nodes for b in builtin_objs]


# ---------------------------------------------------------------------------
# Pre-defined subsets for 'small' and 'test' run modes
# ---------------------------------------------------------------------------

SMALL_AST: list[str] = [
    # Statements
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Delete",
    "Assign", "AugAssign", "AnnAssign", "For", "AsyncFor", "While", "If",
    "With", "AsyncWith", "Raise", "Try", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Pass", "Break", "Continue",
    # Expressions
    "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp", "Dict", "Set",
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "Yield", "YieldFrom",
    "Compare", "Call", "Attribute", "Subscript", "Starred", "Slice",
]

SMALL_BUILTINS: list[str] = [
    # Types
    "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "frozenset", "bytes", "bytearray", "complex", "object", "type",
    "memoryview",
    # Functions
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "min", "max", "sum", "abs", "round", "any", "all",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "callable", "iter", "next", "hash", "id", "repr", "input", "open",
    "super", "property", "staticmethod", "classmethod",
    # Exceptions
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "FileNotFoundError",
    "OSError", "ImportError", "NotImplementedError", "ZeroDivisionError",
]

TEST_AST: list[str] = ["For", "If", "ListComp", "Try", "FunctionDef"]
TEST_BUILTINS: list[str] = ["list", "dict", "int", "ValueError", "range"]


def get_pairs_for_mode(mode: str) -> list[tuple[str, str]]:
    """Return concept pairs for the given run mode."""
    all_ast = get_ast_nodes()
    all_builtins = get_builtin_objects()

    if mode == "test":
        return generate_concept_matrix(TEST_AST, TEST_BUILTINS)
    elif mode == "small":
        selected_ast = [n for n in SMALL_AST if n in all_ast]
        selected_builtins = [n for n in SMALL_BUILTINS if n in all_builtins]
        return generate_concept_matrix(selected_ast, selected_builtins)
    else:  # "full"
        return generate_concept_matrix(all_ast, all_builtins)

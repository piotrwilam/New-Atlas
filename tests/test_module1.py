"""
Unit tests for Module 1 — run with `pytest tests/test_module1.py`.

These tests exercise Stages A and B (concept matrix + generator) without
requiring the CSP model or GPU. Stage C (PerplexityFilter) is tested with a
mock model to avoid loading the real weights.
"""

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from module1.concept_matrix import (
    TEST_AST,
    TEST_BUILTINS,
    generate_concept_matrix,
    get_ast_nodes,
    get_builtin_objects,
    get_pairs_for_mode,
)
from module1.generators import ASTPromptGenerator
from module1.variance_schema import DOMAINS


# ---------------------------------------------------------------------------
# Stage A — Concept Matrix
# ---------------------------------------------------------------------------


class TestConceptMatrix:
    def test_get_ast_nodes_returns_list(self):
        nodes = get_ast_nodes()
        assert isinstance(nodes, list)
        assert len(nodes) >= 50

    def test_get_ast_nodes_are_sorted(self):
        nodes = get_ast_nodes()
        assert nodes == sorted(nodes)

    def test_no_abstract_nodes(self):
        nodes = get_ast_nodes()
        abstract = {"mod", "stmt", "expr", "boolop", "operator", "unaryop",
                    "cmpop", "comprehension", "excepthandler"}
        assert not abstract.intersection(nodes)

    def test_get_builtin_objects_returns_list(self):
        builtins = get_builtin_objects()
        assert isinstance(builtins, list)
        assert len(builtins) >= 100

    def test_common_builtins_present(self):
        builtins = get_builtin_objects()
        for name in ("int", "str", "list", "dict", "ValueError", "range"):
            assert name in builtins

    def test_generate_concept_matrix_size(self):
        pairs = generate_concept_matrix(["For", "If"], ["list", "dict"])
        assert len(pairs) == 4

    def test_generate_concept_matrix_contents(self):
        pairs = generate_concept_matrix(["For"], ["list", "dict"])
        assert ("For", "list") in pairs
        assert ("For", "dict") in pairs

    def test_get_pairs_for_mode_test(self):
        pairs = get_pairs_for_mode("test")
        assert len(pairs) == len(TEST_AST) * len(TEST_BUILTINS)

    def test_get_pairs_for_mode_small(self):
        pairs = get_pairs_for_mode("small")
        assert len(pairs) > 0


# ---------------------------------------------------------------------------
# Stage B — AST Prompt Generator
# ---------------------------------------------------------------------------


class TestASTPromptGenerator:
    @pytest.fixture
    def gen(self):
        return ASTPromptGenerator(DOMAINS)

    def test_generate_batch_returns_list(self, gen):
        results = gen.generate_batch("For", "list", n=5)
        assert isinstance(results, list)

    def test_generate_batch_not_empty(self, gen):
        results = gen.generate_batch("For", "list", n=5)
        assert len(results) > 0

    def test_prompt_text_is_valid_python(self, gen):
        results = gen.generate_batch("For", "list", n=5)
        for r in results:
            try:
                ast.parse(r["prompt_text"])
            except SyntaxError as exc:
                pytest.fail(f"SyntaxError in prompt: {exc}\n{r['prompt_text']}")

    @pytest.mark.parametrize("node,builtin", [
        ("For", "list"),
        ("If", "ValueError"),
        ("Try", "ValueError"),
        ("ListComp", "int"),
        ("FunctionDef", "str"),
        ("ClassDef", "dict"),
        ("Lambda", "range"),
        ("While", "int"),
    ])
    def test_core_pairs_produce_output(self, gen, node, builtin):
        results = gen.generate_batch(node, builtin, n=3)
        assert len(results) >= 1, f"No output for ({node}, {builtin})"

    def test_ast_verified_flag(self, gen):
        results = gen.generate_batch("For", "list", n=10)
        verified = [r for r in results if r["ast_verified"]]
        assert len(verified) / max(len(results), 1) >= 0.8

    def test_domain_cycling(self, gen):
        results = gen.generate_batch("Assign", "int", n=10)
        domains_seen = {r["domain"] for r in results}
        assert len(domains_seen) > 1

    def test_unsupported_node_returns_empty(self, gen):
        # 'Module' is not in the dispatch table
        results = gen.generate_batch("Module", "list", n=5)
        assert results == []

    def test_exception_builtin_try_template(self, gen):
        results = gen.generate_batch("Try", "ValueError", n=3)
        for r in results:
            assert "ValueError" in r["prompt_text"]

    def test_prompt_text_is_string(self, gen):
        results = gen.generate_batch("BinOp", "int", n=3)
        for r in results:
            assert isinstance(r["prompt_text"], str)
            assert len(r["prompt_text"]) > 0


# ---------------------------------------------------------------------------
# Stage C — PerplexityFilter (mocked)
# ---------------------------------------------------------------------------


class TestPerplexityFilter:
    """Test PerplexityFilter behaviour without loading the real CSP model."""

    @pytest.fixture
    def mock_filter(self):
        with patch("module1.filters.AutoTokenizer") as mock_tok_cls, \
             patch("module1.filters.AutoModelForCausalLM") as mock_model_cls:

            # Tokenizer returns a dict-like object with input_ids of shape (1, 10)
            import torch
            mock_tok = MagicMock()
            mock_tok.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
            mock_tok_cls.from_pretrained.return_value = mock_tok

            # Model returns loss tensor
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(2.5)
            mock_model.return_value = mock_output
            mock_model_cls.from_pretrained.return_value = mock_model

            from module1.filters import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.device = "cpu"
            f.tokenizer = mock_tok
            f.model = mock_model
            yield f

    def test_filter_batch_keeps_top_k(self, mock_filter):
        import torch
        # Give each prompt a distinct loss
        mock_model = mock_filter.model
        losses = [5.0, 1.0, 3.0, 2.0, 4.0]
        side_effects = []
        for l in losses:
            m = MagicMock()
            m.loss = torch.tensor(l)
            side_effects.append(m)
        mock_model.side_effect = side_effects

        prompts = [{"prompt_text": f"x = {i}"} for i in range(5)]
        result = mock_filter.filter_batch(prompts, top_k=3, catastrophic_threshold=10.0)
        assert len(result) == 3
        losses_out = [r["sequence_loss"] for r in result]
        assert losses_out == sorted(losses_out)

    def test_filter_batch_catastrophic(self, mock_filter):
        import torch
        mock_model = mock_filter.model
        side_effects = []
        for _ in range(3):
            m = MagicMock()
            m.loss = torch.tensor(15.0)
            side_effects.append(m)
        mock_model.side_effect = side_effects

        prompts = [{"prompt_text": "x = 1"} for _ in range(3)]
        result = mock_filter.filter_batch(prompts, top_k=3, catastrophic_threshold=10.0)
        assert result == []

    def test_filter_batch_empty_input(self, mock_filter):
        result = mock_filter.filter_batch([], top_k=10)
        assert result == []


# ---------------------------------------------------------------------------
# Pipeline smoke test (no model required)
# ---------------------------------------------------------------------------


class TestPipelineSmoke:
    def test_run_pipeline_with_mock_components(self, tmp_path):
        from module1.pipeline import run_pipeline

        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = [
            {"prompt_text": "x = 1", "ast_verified": True}
        ]

        mock_filter = MagicMock()
        mock_filter.filter_batch.return_value = [
            {
                "prompt_text": "x = 1",
                "ast_verified": True,
                "sequence_loss": 1.5,
                "token_length": 5,
            }
        ]

        pairs = [("For", "list"), ("If", "dict")]
        df = run_pipeline(
            concept_pairs=pairs,
            generator=mock_gen,
            pfilter=mock_filter,
            n_generate=5,
            n_keep=2,
            checkpoint_dir=str(tmp_path),
            checkpoint_every=100,
            run_name="smoke_test",
            catastrophic_threshold=10.0,
            mode="test",
        )

        assert len(df) == 2  # 1 prompt × 2 pairs
        assert set(df.columns) >= {"ast_node", "builtin_obj", "variation_id",
                                    "prompt_text", "sequence_loss",
                                    "token_length", "ast_verified"}

        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

        stats_files = list(tmp_path.glob("*.json"))
        assert len(stats_files) == 1

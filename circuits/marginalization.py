import logging

import numpy as np

logger = logging.getLogger(__name__)


class UniversalModuleComputer:
    """
    Computes Universal AST and Universal Builtin modules by
    intersecting Pair Representations.

    Universal AST (e.g. "For"): intersect across all builtins
        → neurons that fire for For regardless of what is iterated
    Universal Builtin (e.g. "list"): intersect across all AST nodes
        → neurons that fire for list regardless of the AST context
    """

    def compute_universal_ast(self, ast_node: str, pair_masks: dict,
                               all_builtins: list) -> dict:
        """
        For a given AST node, intersect its Pair Representations across all builtins.

        Returns:
            {layer_id: bool_mask} — the Universal AST mask
        """
        relevant = {b: pair_masks[(ast_node, b)]
                    for (a, b) in pair_masks if a == ast_node}
        if not relevant:
            logger.warning(f"No pair masks found for AST node: {ast_node}")
            return {}

        n_used = len(relevant)
        n_total = len(all_builtins)
        if n_used < n_total:
            logger.info(f"Universal AST {ast_node}: intersection across {n_used}/{n_total} builtins")

        result = {}
        # Find all layer ids present
        all_lids = set()
        for masks in relevant.values():
            all_lids.update(masks.keys())

        for lid in all_lids:
            layer_masks = [v[lid] for v in relevant.values() if lid in v]
            if not layer_masks:
                continue
            intersection = layer_masks[0].copy()
            for m in layer_masks[1:]:
                intersection = np.logical_and(intersection, m)
            result[lid] = intersection

        return result

    def compute_universal_builtin(self, builtin_obj: str, pair_masks: dict,
                                   all_ast_nodes: list) -> dict:
        """
        For a given builtin, intersect its Pair Representations across all AST nodes.

        Returns:
            {layer_id: bool_mask} — the Universal Builtin mask
        """
        relevant = {a: pair_masks[(a, builtin_obj)]
                    for (a, b) in pair_masks if b == builtin_obj}
        if not relevant:
            logger.warning(f"No pair masks found for builtin: {builtin_obj}")
            return {}

        n_used = len(relevant)
        n_total = len(all_ast_nodes)
        if n_used < n_total:
            logger.info(f"Universal Builtin {builtin_obj}: intersection across {n_used}/{n_total} AST nodes")

        result = {}
        all_lids = set()
        for masks in relevant.values():
            all_lids.update(masks.keys())

        for lid in all_lids:
            layer_masks = [v[lid] for v in relevant.values() if lid in v]
            if not layer_masks:
                continue
            intersection = layer_masks[0].copy()
            for m in layer_masks[1:]:
                intersection = np.logical_and(intersection, m)
            result[lid] = intersection

        return result

    def compute_all(self, pair_masks: dict, ast_nodes: list,
                    builtin_objs: list) -> dict:
        """
        Compute all Universal Modules (125 AST + 153 Builtin).

        Returns:
            {
                "ast": {ast_node: {layer: mask}},
                "builtin": {builtin_obj: {layer: mask}},
            }
        """
        universal_ast = {}
        for ast_n in ast_nodes:
            result = self.compute_universal_ast(ast_n, pair_masks, builtin_objs)
            if result:
                universal_ast[ast_n] = result

        universal_builtin = {}
        for blt_o in builtin_objs:
            result = self.compute_universal_builtin(blt_o, pair_masks, ast_nodes)
            if result:
                universal_builtin[blt_o] = result

        logger.info(f"Universal AST modules: {len(universal_ast)}")
        logger.info(f"Universal Builtin modules: {len(universal_builtin)}")
        return {"ast": universal_ast, "builtin": universal_builtin}

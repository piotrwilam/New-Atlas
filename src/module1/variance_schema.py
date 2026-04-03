"""
Variance Schema — semantic domains and structural constants.

Five domains provide the Lexical/Semantic variance axis (Atlas 2 §9).
Each domain supplies domain-specific variable names and mock data so that
every prompt variation reads as plausible code from a different field.
"""

DOMAINS: dict = {
    "finance": {
        "var_names": {
            "list": "ledger_entries", "dict": "account_records",
            "item": "transaction", "func": "audit_record",
            "class": "PortfolioManager", "method": "calculate_returns",
            "value": "balance", "key": "ticker",
        },
        "mock_data": {
            "list": "[1050.50, -20.00, 400.25, 88.10]",
            "dict": "{'ticker': 'AAPL', 'shares': 150, 'price': 178.50}",
            "int": "42500", "str": "'USD-2024-Q3-REPORT'",
            "float": "3.75", "set": "{'NYSE', 'NASDAQ', 'LSE'}",
            "tuple": "('BUY', 150, 178.50)", "bool": "True",
        },
    },
    "biology": {
        "var_names": {
            "list": "dna_samples", "dict": "genome_annotations",
            "item": "genome_sequence", "func": "analyze_genome",
            "class": "SequenceAnalyzer", "method": "run_alignment",
            "value": "mutation_rate", "key": "gene_id",
        },
        "mock_data": {
            "list": "['ACTG', 'GCTA', 'CGAT', 'TTAC']",
            "dict": "{'gene': 'BRCA1', 'chromosome': 17, 'position': 43044295}",
            "int": "43044295", "str": "'BRCA1-VARIANT-P53'",
            "float": "0.95", "set": "{'adenine', 'thymine', 'guanine'}",
            "tuple": "('chr17', 43044295, 43125483)", "bool": "False",
        },
    },
    "gaming": {
        "var_names": {
            "list": "player_scores", "dict": "character_stats",
            "item": "player_entry", "func": "update_leaderboard",
            "class": "GameEngine", "method": "spawn_entity",
            "value": "hit_points", "key": "player_id",
        },
        "mock_data": {
            "list": "[9500, 8700, 12400, 6300]",
            "dict": "{'health': 100, 'mana': 50, 'attack': 25}",
            "int": "9500", "str": "'LEVEL_42_BOSS'",
            "float": "1.5", "set": "{'warrior', 'mage', 'rogue'}",
            "tuple": "('dragon', 5000, 'fire')", "bool": "True",
        },
    },
    "physics": {
        "var_names": {
            "list": "measurements", "dict": "particle_data",
            "item": "reading", "func": "compute_trajectory",
            "class": "SimulationRunner", "method": "step_forward",
            "value": "velocity", "key": "particle_id",
        },
        "mock_data": {
            "list": "[9.81, 3.0e8, 6.674e-11, 1.602e-19]",
            "dict": "{'mass': 1.67e-27, 'charge': 1.6e-19, 'spin': 0.5}",
            "int": "299792458", "str": "'MUON_DECAY_EVENT'",
            "float": "9.81", "set": "{'proton', 'neutron', 'electron'}",
            "tuple": "(0.0, 9.81, -3.2)", "bool": "False",
        },
    },
    "ecommerce": {
        "var_names": {
            "list": "shopping_cart", "dict": "product_catalog",
            "item": "order_item", "func": "process_checkout",
            "class": "OrderProcessor", "method": "apply_discount",
            "value": "total_price", "key": "sku",
        },
        "mock_data": {
            "list": "[29.99, 15.50, 89.00, 4.99]",
            "dict": "{'sku': 'WDG-4420', 'price': 29.99, 'stock': 142}",
            "int": "142", "str": "'ORDER-2024-XK9001'",
            "float": "29.99", "set": "{'shipped', 'pending', 'delivered'}",
            "tuple": "('WDG-4420', 29.99, 3)", "bool": "True",
        },
    },
}

DOMAIN_KEYS: list[str] = list(DOMAINS.keys())

# Structural/Contextual Variance — wrapper proportions (Atlas 2 §9)
WRAPPER_TYPES: list[str] = ["global", "function", "method"]
WRAPPER_WEIGHTS: list[float] = [0.40, 0.30, 0.30]

# Padding Variance — unrelated statements injected before/after the essence
PADDING_BEFORE: list[str] = [
    "", "result = None", "print('Starting process')",
    "status = True", "counter = 0",
]
PADDING_AFTER: list[str] = [
    "", "print('Done')", "status = False",
    "counter += 1", "result = None",
]

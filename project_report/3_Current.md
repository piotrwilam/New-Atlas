# Current Research

Active areas of investigation. The purpose of this work is understanding universal objects — how circuits function within a CSP transformer.

## Research experiments

1. **Ablation studies** (Piotr) — Prove that identified neurons are essential for universal circuits by selectively disabling them and measuring the impact on model behavior. Extends the CSP-Ablation-Project work from security circuits to universal circuits.

2. **Causal path analysis and SAE** (Vignesh) — Sparse Autoencoder-based investigation of causal pathways through the transformer. Building on Vignesh's earlier observations about model behavior steering and automated circuit discovery (Conmy et al., 2023).

3. **Analytical tools X01–X06** (Efe) — Efe's toolset for structured analysis of universal circuit properties. Extends the probing and literature review work into systematic analytical tooling.

4. **Circuit graph visualizations** — Representing circuits as graphs within the transformer architecture to reveal connectivity and information flow patterns. Related to the STATNLP-FFS high betweenness proposal.

5. **Neuron activity threshold exploration** — Testing different thresholds for neuron activity during activation extraction to understand sensitivity and robustness of circuit definitions.

6. **Literature review** (Efe + Asha) — Compile a bibliography of up to 10 key papers covering mechanistic interpretability, circuit discovery in transformers, sparse autoencoders, and code understanding models. The review should contextualize the CSP-Atlas findings within the broader field. Building on Efe's and Asha's earlier literature review work.

7. **Report visualizations** — Generate and save plots to visually support the four findings. Most already exist in notebooks 3A and 4; they need to be run and exported. Key plots: marginalization convergence curves (Finding 1), modularity histogram split by AST/builtin (Finding 2), layer evolution line chart (Finding 3), E_I histogram and Jaccard heatmaps (Finding 4), combined heatmap of all circuits × layers (cross-finding).

## Links

- [CSP-Atlas on GitHub](https://github.com/piotrwilam/CSP-Atlas)
- [CSP-Ablation-Project on GitHub](https://github.com/piotrwilam/CSP-Ablation-Project)
- [CSP-Atlas on HuggingFace](https://huggingface.co/CSP-Atlas)
- [CSP-Ablation-Project artifacts on HuggingFace](https://huggingface.co/piotrwilam/CSP-Ablation-Project)
- [openai/circuit-sparsity model](https://huggingface.co/openai/circuit-sparsity)

# Extended Data figure audit

## Extended Data Figure 1c

- Frozen input: `results/aggregated/perpair_all.csv`.
- Plotting entry point: `scripts/viz/ed_fig1_gut_hippo.py`, function
  `panel_c_hippo_scatter`.
- The plotted estimand first intersects DLPFC and hippocampus BMI pairs,
  selects the 15 highest-ranked DLPFC pairs, and calculates Spearman
  correlation between their DLPFC and hippocampus ranks.
- The frozen input contains 218 DLPFC rows, 536 hippocampus rows, and 194
  common pairs. The plotted top-15 rank statistic is rho = 0.6607142857
  (P = 0.00733057), and the top-five intersection is 3 of 5.
- Across all 194 common pairs, the raw z-score Spearman correlation is
  rho = 0.9858160108. This is a different estimand and is not the statistic
  displayed by panel c. The caption therefore reports rho = 0.66 and 3/5.

## Extended Data Figures 3d and 4a

- Both panels used the same six unique Kuppe heart-section communication-score
  correlations from `results/cross_replication/replication_correlation.csv`.
- The duplicate panel was removed from Extended Data Figure 3 and retained as
  Extended Data Figure 4a, where it is grouped with pair-level robustness.

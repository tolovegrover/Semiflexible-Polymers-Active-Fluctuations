#!/bin/bash
set -e

echo "Step 1: Fitting mode spectrum..."
python3 02_fit_spectrum.py

echo "Step 2: Checking bond statistics and fluctuations..."
python3 03_calc_bond_stats.py
python3 04_calc_bond_fluctuations.py

echo "Step 3: Plotting bond metrics..."
python3 05_plot_bond_metrics.py

echo "Step 4: Performing Rg reconstruction..."
cd rg_reconstruction_analysis
python3 06_reconstruct_rg.py
python3 07_plot_comparison_all_taus.py

echo "Step 5: Verifying numerical consistency..."
python3 08_verify_numerics.py > ../../data/verification_log.txt
cd ..

echo "Pipeline complete. Plots are available in ../plots/ directory."

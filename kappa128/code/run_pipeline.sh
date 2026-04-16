#!/bin/bash
# Master Pipeline for Active Polymer Mode Analysis
# This script runs the analysis from processed data to final plots.

echo "Step 1: Fitting Mode Spectrum..."
python3 02_fit_spectrum.py

echo "Step 2: Calculating Bond Statistics & Fluctuations..."
python3 03_calc_bond_stats.py
python3 04_calc_bond_fluctuations.py

echo "Step 3: Plotting Bond Metrics..."
python3 05_plot_bond_metrics.py

echo "Step 4: Performing Rg Reconstruction..."
cd rg_reconstruction_analysis
python3 06_reconstruct_rg.py
python3 07_plot_comparison_all_taus.py

echo "Step 5: Verifying Numerical Consistency..."
python3 08_verify_numerics.py > ../../data/verification_log.txt

echo "Pipeline Complete. Plots are available in the ../plots/ directory."

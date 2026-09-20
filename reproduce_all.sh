#!/bin/bash
# Global Reproduction Script
# Runs the full analysis pipeline for all stiffness parameters.

set -e

echo "Starting Global Reproduction Pipeline..."

echo "------------------------------------------"
echo "Processing kappa = 32"
echo "------------------------------------------"
cd kappa32/code
./run_pipeline.sh
cd ../..

echo "------------------------------------------"
echo "Processing kappa = 128"
echo "------------------------------------------"
cd kappa128/code
./run_pipeline.sh
cd ../..

echo "------------------------------------------"
echo "Processing kappa = 256"
echo "------------------------------------------"
cd kappa256/code
./run_pipeline.sh
cd ../..

echo "------------------------------------------"
echo "Analysis complete for all systems."
echo "Results can be found in the respective /plots/ folders."
echo "------------------------------------------"

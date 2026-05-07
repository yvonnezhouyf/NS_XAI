#!/bin/bash
# Run ADA-MCTS on Paratransit domain with REAL Chattanooga Data

echo "============================================================"
echo "Starting ADA-MCTS for Paratransit with REAL Data"
echo "============================================================"
echo ""

# BUG FIX: Validate real data availability - FAIL if missing
echo "Validating real data availability..."
DATA_DIR="../../use_cases/paratransit/data"
REQUIRED_FILES=("travel_time_matrix.csv" "travel_time_matrix_cong.csv" "train_chains.csv")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "ERROR: Missing required REAL data files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $DATA_DIR/$file"
    done
    echo ""
    echo "This script requires REAL Chattanooga data."
    echo "Please ensure all data files are present before running."
    exit 1
fi

echo "✓ All required real data files found."
echo ""

# Collect experiences
echo "Step 1: Collecting experiences with real data..."
python -m data_collection.data_generation paratransit

# Initialize MDP0
echo ""
echo "Step 2: Training initial BNN (MDP0)..."
python -m data_collection.train_model paratransit

# Perform Act As You Learn
echo ""
echo "Step 3: Running Act-As-You-Learn with real data..."
python act_learn.py paratransit

echo ""
echo "============================================================"
echo "ADA-MCTS for Paratransit completed!"
echo "Results saved in results/ directory"
echo "============================================================"

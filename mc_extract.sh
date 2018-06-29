#!/bin/bash

set -euo pipefail

# MC_RAW_DIR=/scratch/q/qmorris/gurnit/new_pcawg3
# MC_DATA=$HOME/data/mc_100.npz
# B_DATA=$HOME/results/b_100.npz
# CLON=True
# OUTPUT_FILE=$SCRATCH/results/vectors.npz
PYTHON_MOD=python/3.6.4-anaconda5.1.0

sbatch --nodes=1 --ntasks-per-node=40 --job-name=mc_extract --output=$SCRATCH/outputs/mc_extract.out --error=$SCRATCH/outputs/mc_extract.err --time=24:00:00 --export=PYTHON_MOD=$PYTHON_MOD,ALL $SCRATCH/mc_extract_sbatch.sh
#!/bin/bash

set -euo pipefail

MC_RAW_DIR=/scratch/q/qmorris/gurnit/new_pcawg3
MC_DATA=$HOME/data/mc_100.npz
B_DATA=$HOME/results/b_100.npz
CLON=True
OUTPUT_FILE=$SCRATCH/results/vectors.npz
PYTHON_MOD=python/3.6.4-anaconda5.1.0

sbatch --nodes=1 --ntasks-per-node=40 --time=24:00:00 --export=MC_RAW_DIR=$MC_RAW_DIR,MC_DATA=$MC_DATA,B_DATA=$B_DATA,CLON=$CLON,OUTPUT_FILE=$OUTPUT_FILE,PYTHON_MOD=$PYTHON_MOD,ALL $SCRATCH/mc_gurnit_sbatch.sh
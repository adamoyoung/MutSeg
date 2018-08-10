#!/bin/bash

set -euo pipefail

LENS=(NaN 32218 37057 30941 31304 29139 24624 25917 26479 14153 18095 19702 19197 16842 12926 10242 10836 8240 12706 7126 9032 5585 3484)
L_KS=(NaN 250 244 199 192 181 172 160 147 142 136 136 134 116 108 103 91 82 79 60 64 49 52)
N_KS=(NaN 230 265 221 223 208 176 185 189 101 129 141 137 120 92 73 77 59 91 51 64 40 25)
K_MODES=(n l)
T=35
CHRM=$1
K_MODE=$2
FOLD=$3
MP=40
LEN=${LENS[$CHRM]}

if [ $K_MODE -eq 0 ]
then K=${N_KS[$CHRM]}
fi

if [ $K_MODE -eq 1 ]
then K=${L_KS[$CHRM]}
fi

RESULTS_DIR="${SCRATCH}/results/${K_MODES[$K_MODE]}/fold_${FOLD}"
DATA_DIR="${HOME}/data/mc_chrms_f"
DATA_FILE="${DATA_DIR}/fold_${FOLD}_chrm_${CHRM}.dat"
# $FOLD == 5 implies that we are training on all folds!
if [ $FOLD -eq 5 ]; then
	RESULTS_DIR="${SCRATCH}/results/${K_MODES[$K_MODE]}/all"
	DATA_FILE="${DATA_DIR}/all_chrm_${CHRM}.dat"
fi

JOB_NAME="${CHRM}_${K_MODES[$K_MODE]}_$FOLD"
OUT_FILE="${SCRATCH}/outputs/${CHRM}_${K_MODES[$K_MODE]}_$FOLD.out"
ERR_FILE="${SCRATCH}/outputs/${CHRM}_${K_MODES[$K_MODE]}_$FOLD.err"

sbatch --nodes=1 --ntasks-per-node=$MP --time=24:00:00 --job-name=$JOB_NAME --output=$OUT_FILE --error=$ERR_FILE --export=DATA_FILE=$DATA_FILE,RESULTS_DIR=$RESULTS_DIR,LEN=$LEN,K=$K,T=$T,MP=$MP,CHRM=$CHRM,FOLD=$FOLD,ALL seg_sbatch.sh
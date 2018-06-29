#!/bin/bash

set -euo pipefail

LENS=(NaN 32218 37057 30941 31304 29139 24624 25917 26479 14153 18095 19702 19197 16842 12926 10242 10836 8240 12706 7126 9032 5585 3484)
L_KS=(NaN 250 244 199 192 181 172 160 147 142 136 136 134 116 108 103 91 82 79 60 64 49 52)
#N_KS=(NaN 247 284 237 240 224 189 199 203 109 139 151 147 129 99 79 83 63 97 55 69 43 27)
N_KS=(NaN 230 265 221 223 208 176 185 189 101 129 141 137 120 92 73 77 59 91 51 64 40 25)
#KS=(NaN 794 913 762 771 718 607 639 652 349 446 485 473 415 318 252 267 203 313 176 223 138 86)
#SEXES=(male female both)
#SEX_ABBREVS=(m f b)
K_MODES=(n l)
T=35
CHRM=$1
K_MODE=$2
FOLD=$3 # fold_5 is the one that contains all of them!
#SEX=$4
RESULTS_DIR="${SCRATCH}/results/${K_MODES[$K_MODE]}/fold_${FOLD}"
DATA_FILE="${HOME}/data/mc_chrms/mc_100_fold_${FOLD}_chrm_${CHRM}.dat"
MP=40

if [ $K_MODE -eq 0 ]
then K=${N_KS[$CHRM]}
fi

if [ $K_MODE -eq 1 ]
then K=${L_KS[$CHRM]}
fi

# $FOLD == 5 implies that we are training on all folds!
if [ $FOLD -eq 5 ]; then
	RESULTS_DIR="${SCRATCH}/results/${K_MODES[$K_MODE]}/all"
	DATA_FILE="${HOME}/data/mc_chrms/mc_100_all_chrm_${CHRM}.dat"
fi


LEN=${LENS[$CHRM]}

sbatch --nodes=1 --ntasks-per-node=${MP} --time=24:00:00 --job-name=c_${CHRM}_k_${K_MODES[$K_MODE]}_f_$FOLD --output=${SCRATCH}/outputs/c_${CHRM}_${K_MODES[$K_MODE]}_$FOLD.out --error=${SCRATCH}/outputs/c_${CHRM}_${K_MODES[$K_MODE]}_$FOLD.err --export=DATA_FILE=$DATA_FILE,RESULTS_DIR=$RESULTS_DIR,LEN=$LEN,K=$K,T=$T,MP=$MP,CHRM=$CHRM,FOLD=$FOLD,ALL seg_sbatch.sh
#!/bin/bash

set -euo pipefail

LENS=(NaN 32218 37057 30941 31304 29139 24624 25917 26479 14153 18095 19702 19197 16842 12926 10242 10836 8240 12706 7126 9032 5585 3484)
#N_KS=(NaN 2300 2650 2210 2230 2080 1760 1850 1890 1010 1290 1410 1370 1200 920 730 770 590 910 510 640 400 250)
#L_KS=(NaN 2500 2440 1990 1920 1810 1720 1600 1470 1420 1360 1360 1340 1160 1080 1030 910 820 790 600 640 490 520)
N_KS=(NaN 2288 2632 2197 2223 2069 1749 1841 1881 1005 1285 1399 1363 1196 918 727 770 585 902 506 641 397 247)
L_KS=(NaN 2493 2432 1981 1912 1810 1712 1592 1464 1413 1356 1351 1339 1152 1074 1026 904 812 781 592 631 482 514)
K_MODES=(n l)
T=35
CHRM=$1
K_MODE=$2
FOLD=$3
RUN=$4
MP=40
LEN=${LENS[$CHRM]}
K_THRESH=1500

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

if [ $RUN -eq 0 ]; then
	PREV_K=0
	if [ $K -gt $K_THRESH ]; then
		K=$K_THRESH
	fi
elif [ $RUN -eq 1 ]; then
	if [ $K -gt $K_THRESH ]; then
		K=`expr $K - $K_THRESH`
		PREV_K=$K_THRESH
	else
		echo "no more work to be done!"
		exit 0
	fi 
else
	exit 0
fi

sbatch --nodes=1 --ntasks-per-node=$MP --time=24:00:00 --job-name=$JOB_NAME --output=$OUT_FILE --error=$ERR_FILE --export=DATA_FILE=$DATA_FILE,RESULTS_DIR=$RESULTS_DIR,LEN=$LEN,K=$K,T=$T,MP=$MP,CHRM=$CHRM,FOLD=$FOLD,PREV_K=$PREV_K,ALL seg_sbatch.sh
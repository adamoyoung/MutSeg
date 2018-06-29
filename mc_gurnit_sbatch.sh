#!/bin/bash

# echo $PYTHON_MOD > $SCRATCH/testfile.txt
module load $PYTHON_MOD
python3 $HOME/extract_features.py $MC_RAW_DIR $MC_DATA $B_DATA $CLON $OUTPUT_FILE
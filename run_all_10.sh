#!/bin/bash

r=$1 # run -- 0 or 1
for m in `seq 0 1`
do
	# set to `seq 0 5` if you want to include an "all" run
	for f in `seq 5 5`
	do
		for c in `seq 1 22`
		do
			./seg_10.sh $c $m $f $r
		done
	done
done
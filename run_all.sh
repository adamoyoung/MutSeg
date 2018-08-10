#!/bin/bash

for m in `seq 0 1`
do
	# set to `seq 0 5` if you want to include an "all" run
	for f in `seq 0 5`
	do
		for c in `seq 1 22`
		do
			./seg.sh $c $m $f
		done
	done
done

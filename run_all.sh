#!/bin/bash

for i in `seq 0 2`
do
	for j in `seq 1 22`
	do
		./seg.sh $j $i
	done
done
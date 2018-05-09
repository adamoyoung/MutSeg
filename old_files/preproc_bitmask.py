import sys
import numpy as np
import time

def preproc(file):

	beg_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print("starting preproc")

	# remove header
	file.readline()

	mut_pos = 22*[set()]
	mut_tups = 22*[ [] ]
	typs_set = set()

	line = file.readline()
	while line != '':
		line = line.split()
		chrm, pos, typ = line[1], line[2], line[9]
		if chrm == "X" or chrm == "Y" or chrm == "chr":
			line = file.readline()
			continue
		chrm = int(chrm) - 1
		pos = int(pos) - 1
		assert( chrm >= 0 and chrm <= 22 )
		mut_pos[chrm].add( pos )
		mut_tups[chrm].append( (pos,typ) )
		typs_set.add( typ )
		line = file.readline()

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{}] read in all of the lines".format(cur_time-beg_time) )

	for i in range(22):
		print("starting mut_pos {}".format(i)) 
		mut_pos[i] = np.array( sorted(list(mut_pos[i])) , dtype=np.int)
	comb_mut_pos = np.concatenate(mut_pos)	

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{}] created the combined mutation array".format(cur_time-beg_time) )

	# one dict per chromosome, converts absolute postition (on the chromosome) 
	# to the relative position in the mutation array
	abs_to_relative = {}

	for i in range(22):
		abs_to_relative[i] = {}
		prev_num = sum([mut_pos[k].size for k in range(i)])
		for j in range(len(mut_pos[i])):
			abs_to_relative[i][mut_pos[i][j]] = prev_num + j
	# get rid of mut_pos to save memory
	del mut_pos

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{}] created the abs_to_relative dictionary".format(cur_time-beg_time) )

	typs_list = sorted(list(typs_set))
	typ_to_bitmask = {}
	one = np.ones([1], dtype=np.uint64)
	for i in range(len(typs_list)):
		typ_to_bitmask[typs_list[i]] = ( one << i )

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{}] created the typ_to_bitmask dictionary".format(cur_time-beg_time) )

	#print( "final array dimensions: [ {} x {} ]".format( comb_mut_pos.size, len(typs_list) ) )
	#final_array = np.zeros([comb_mut_pos.size, len(typs_list)], dtype=np.bool)
	print( "final array dimensions: [ {} x 1 ]".format(comb_mut_pos.size) )
	final_array = np.zeros([comb_mut_pos.size], dtype=np.uint64)
	print( "final array size: {}".format(sys.getsizeof(final_array) / 1000000) )

	for i in range(22):
		print( "starting mut_tups {}".format(i) )
		for j in range(len(mut_tups[i])):
			pos, typ = mut_tups[i][j]
			assert( type(pos) == int and type(typ) == str )
			typ_bitmask = typ_to_bitmask[typ]
			pos_ind = abs_to_relative[i][pos]
			final_array[pos_ind] |= typ_bitmask

	print( "final array is complete" )

	assert( (final_array != 0).all() ) 

	return final_array

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: preproc.py [src file (.txt)] [dest file (.npy)]")
		exit(1)
	file = open(sys.argv[1], 'r')
	array = preproc(file)
	np.save(sys.argv[2], array)

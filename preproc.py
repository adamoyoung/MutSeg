import sys
import numpy as np
import time

float_t = np.float64

def preproc(file):

	beg_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print("[0] starting preproc")

	# remove header
	file.readline()

	mut_pos = [
				set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),
				set(),set(),set(),set(),set(),set(),set(),set(),set(),set(),set()
			]
	mut_tups = [
				[],[],[],[],[],[],[],[],[],[],[],
				[],[],[],[],[],[],[],[],[],[],[]
			]
	assert( len(mut_pos) == 22 and len(mut_tups) == 22 )
	typs_set = set()

	line = file.readline()
	while line != '':
		line = line.split()
		chrm, pos, dens, typ = line[1], line[2], line[8], line[9]
		if chrm == "X" or chrm == "Y" or chrm == "chr":
			line = file.readline()
			continue
		chrm, pos, dens = int(chrm)-1, int(pos)-1, float(line[8])
		assert( chrm >= 0 and chrm <= 22 )
		mut_pos[chrm].add( pos )
		mut_tups[chrm].append( (pos,typ,dens) )
		typs_set.add( typ )
		line = file.readline()

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] read in all of the lines".format(cur_time-beg_time) )

	comb_mut_pos_size = 0
	for i in range(22):
		print("starting mut_pos {}".format(i)) 
		mut_pos[i] = np.array( sorted(list(mut_pos[i])) , dtype=np.int)
		comb_mut_pos_size += mut_pos[i].size
	# comb_mut_pos = np.concatenate(mut_pos)
	# assert( comb_mut_pos.shape[0] == comb_mut_pos.size )

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created the combined mutation array".format(cur_time-beg_time) )
	print( "combined array dimensions: [ {} ]".format(comb_mut_pos_size) )

	# one dict per chromosome, converts absolute postition (on the chromosome) 
	# to the relative position in the mutation array
	abs_to_relative = {}

	for i in range(22):
		abs_to_relative[i] = {}
		prev_num = sum([mut_pos[k].size for k in range(i)])
		for j in range(len(mut_pos[i])):
			abs_to_relative[i][mut_pos[i][j]] = prev_num + j

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created the abs_to_relative dictionary".format(cur_time-beg_time) )

	typs_list = sorted(list(typs_set))
	typ_to_ind = {}
	for i in range(len(typs_list)):
		typ_to_ind[typs_list[i]] = i

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created the typ_to_ind dictionary".format(cur_time-beg_time) )

	print( "final array dimensions: [ {} x {} ]".format( comb_mut_pos_size, len(typs_list) ) )
	final_array = np.zeros([comb_mut_pos_size, len(typs_list)], dtype=float_t)
	print( "final array size: {}".format(sys.getsizeof(final_array) / 1000000) )

	for i in range(22):
		print( "starting mut_tups {}".format(i) )
		for j in range(len(mut_tups[i])):
			pos, typ, dens = mut_tups[i][j]
			assert( type(pos) == int and type(typ) == str and type(dens) == float)
			typ_ind = typ_to_ind[typ]
			pos_ind = abs_to_relative[i][pos]
			final_array[pos_ind][typ_ind] = dens

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array is complete".format(cur_time-beg_time) )

	assert( np.sum(final_array, axis=1).all() )

	return final_array, mut_pos

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: preproc.py [ src file (.txt) ] [ dest file (.npy) ]")
		exit(1)
	file = open(sys.argv[1], 'r')
	array,mut_pos = preproc(file)
	np.savez(sys.argv[2], array=array, mut_pos=mut_pos)
import sys
import numpy as np
import time

# argv[1]: input file
# argv[2]: output file
# argv[3]: mode

float_t = np.float64
NUM_CHRMS = 23 # does not include the X chromosome, too difficult to make CN calls

def preproc(file, group_by=1):

	beg_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print("[0] starting preproc")

	mut_pos = [set() for i in range(NUM_CHRMS)]
	mut_tups = [[] for i in range(NUM_CHRMS)]
	typs_set = set()

	# remove header
	file.readline()
	
	line = file.readline()
	while line != '':
		line = line.split(",")
		if len(line) < 2:
			line = file.readline()
			continue
		chrm, pos, dens, typ = line[1], line[2], line[8], line[9]
		if chrm == "chr":
			line = file.readline()
			continue
		elif chrm == "X":
			#chrm = 22
			line = file.readline()
		elif chrm == "Y":
			chrm = 22
		else:
			chrm = int(chrm)-1
		pos, dens = int(pos)-1, float(line[8])
		assert( chrm >= 0 and chrm <= NUM_CHRMS )
		mut_pos[chrm].add( pos )
		mut_tups[chrm].append( (pos,typ,dens) )
		typs_set.add( typ )
		line = file.readline()
	
	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] read in all of the lines".format(cur_time-beg_time) )

	comb_mut_pos_size = 0
	for i in range(NUM_CHRMS):
		print("starting mut_pos {}".format(i)) 
		mut_pos[i] = np.array( sorted(mut_pos[i]) , dtype=np.int)
		comb_mut_pos_size += mut_pos[i].size

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created mut_pos arrays".format(cur_time-beg_time) )
	print( "combined number of mutation positions: {}".format(comb_mut_pos_size) )

	# one dict per chromosome, converts absolute postition (on the chromosome) 
	# to the relative position in the mutation array
	abs_to_relative = {}
	for i in range(NUM_CHRMS):
		abs_to_relative[i] = {}
		prev_num = sum([mut_pos[k].size for k in range(i)])
		for j in range(len(mut_pos[i])):
			abs_to_relative[i][mut_pos[i][j]] = prev_num + j

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created the abs_to_relative dictionary".format(cur_time-beg_time) )

	typs_list = sorted(typs_set)
	typ_to_ind = {}
	for i in range(len(typs_list)):
		typ_to_ind[typs_list[i]] = i

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created the typ_to_ind dictionary".format(cur_time-beg_time) )

	print( "final array dimensions: [ {} x {} ]".format( comb_mut_pos_size, len(typs_list) ) )
	final_array = np.zeros([comb_mut_pos_size, len(typs_list)], dtype=float_t)
	print( "final array size: {}".format(sys.getsizeof(final_array) / 1000000) )

	for i in range(NUM_CHRMS):
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

	if group_by > 1:
		# group into groups the size of group_by
		# note that any mutations at the end that are not in a group of size group_by are discarded
		mut_pos_g_sizes = [mut_pos[i].size // group_by for i in range(NUM_CHRMS)]
		final_array_g = np.zeros([np.sum(mut_pos_g_sizes),len(typs_list)], dtype=float_t)
		mut_pos_g = [np.zeros([mut_pos_g_sizes[i],2], dtype=np.int) for i in range(NUM_CHRMS)]
		for i in range(NUM_CHRMS):
			print( "g arrays: starting chrm {}".format(i) )
			prev = sum(mut_pos_g_sizes[0:i])
			for j in range(mut_pos_g_sizes[i]-1):
				final_array_g[prev + j] = np.sum(final_array[ prev + j*group_by : prev + (j+1)*group_by ], axis=0)
				if np.sum(final_array_g[prev + j]) == 0.:
					print("final_g_array[{}]".format(j))
				mut_pos_g[i][j][0] = mut_pos[i][ j*group_by ]
				mut_pos_g[i][j][1] = mut_pos[i][ (j+1)*group_by ]
			last = mut_pos_g_sizes[i]-1
			final_array_g[prev+ last] = np.sum(final_array[ prev + last*group_by : prev + (last+1)*group_by ], axis=0)
			mut_pos_g[i][last][0] = mut_pos[i][ last*group_by ]
			mut_pos_g[i][last][1] = -1 # last one
	else:
		final_array_g = final_array
		mut_pos_g = mut_pos

	print( np.sum(final_array_g, axis=1).all() )
	#assert( np.sum(final_array_g, axis=1).all() )

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array_g and mut_g_array are complete".format(cur_time-beg_time) )
	

	chrm_begs = []
	prev = 0
	for i in range(NUM_CHRMS):
		chrm_begs.append(prev)
		prev += mut_pos_g[i].shape[0]
	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] chrm_begs is complete".format(cur_time-beg_time) )

	return final_array_g, mut_pos_g, chrm_begs

if __name__ == "__main__":

	if len(sys.argv) != 4:
		print("Usage: preproc.py [ src file (.txt) ] [ dest file (.npy) ] [ mode ]")
		exit(1)
	file = open(sys.argv[1], 'r')
	array,mut_pos,chrm_begs = preproc(file, int(sys.argv[3]))
	np.savez(sys.argv[2], array=array, mut_pos=mut_pos, chrm_begs=chrm_begs)
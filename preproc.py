import sys
import numpy as np
import time
from convert_to_C import convert

# argv[1]: input file
# argv[2]: output file
# argv[3]: mode

float_t = np.float64
NUM_CHRMS = 22 # does not include the X chromosome or Y chromosome

def preproc(file_name, group_by=1, mut_sex='both'):

	num_chrms = NUM_CHRMS

	beg_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print("[0] starting preproc, file = {}, group_by = {}, mut_sex = {}".format(file_name,group_by,mut_sex) )

	mut_pos = [set() for i in range(num_chrms)]
	mut_tups = [[] for i in range(num_chrms)]
	typs_set = set()

	file = open(file_name, 'r')

	# remove header
	file.readline()
	
	line = file.readline()
	line_count = 1
	while line != '':
		line = line.split(",")
		if len(line) < 2:
			line = file.readline()
			continue
		chrm, pos, dens, typ, sex = line[1], line[2], line[8], line[9], line[10].strip()
		assert( "chr" != "X" )
		if chrm == "chr" or chrm == "Y" or (mut_sex == "male" and sex != "M") or (mut_sex == "female" and sex != "F"):
			line = file.readline()
			line_count += 1
			continue
		# if (sex == "F" and int(chrm)-1 == 21):
		# 	print(line_count)
		# 	quit()
		# elif chrm == "Y":
		# 	assert(mut_sex != "female")
		# 	chrm = 22
		else:
			chrm = int(chrm)-1
		pos, dens = int(pos)-1, float(line[8])
		assert( chrm >= 0 and chrm < num_chrms )
		assert( sex == "M" or sex == "F")
		mut_pos[chrm].add( pos )
		mut_tups[chrm].append( (pos,typ,dens) )
		typs_set.add( typ )
		line = file.readline()
		line_count += 1

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] read in all of the lines".format(cur_time-beg_time) )

	comb_mut_pos_size = 0
	for i in range(num_chrms):
		print( "starting mut_pos {}: {} unique positions".format(i,len(mut_pos[i])) ) 
		mut_pos[i] = np.array( sorted(mut_pos[i]) , dtype=np.int)
		comb_mut_pos_size += mut_pos[i].size

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] created mut_pos arrays".format(cur_time-beg_time) )
	print( "combined number of mutation positions: {}".format(comb_mut_pos_size) )

	# one dict per chromosome, converts absolute postition (on the chromosome) 
	# to the relative position in the mutation array
	abs_to_relative = {}
	for i in range(num_chrms):
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

	for i in range(num_chrms):
		print( "starting mut_tups {}".format(i) )
		for j in range(len(mut_tups[i])):
			pos, typ, dens = mut_tups[i][j]
			assert( type(pos) == int and type(typ) == str and type(dens) == float )
			typ_ind = typ_to_ind[typ]
			pos_ind = abs_to_relative[i][pos]
			final_array[pos_ind][typ_ind] = dens

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array is complete".format(cur_time-beg_time) )

	assert( np.sum(final_array, axis=1).all() )

	# group into groups the size of group_by
	# note that any mutations at the end that are not in a group of size group_by are discarded
	# also note that the mut_pos_g array is [beg_pt, end_pt] inclusive
	mut_pos_g_sizes = [mut_pos[i].size // group_by for i in range(num_chrms)]
	final_array_g = np.zeros([np.sum(mut_pos_g_sizes),len(typs_list)], dtype=float_t)
	mut_pos_g = [np.zeros([mut_pos_g_sizes[i],2], dtype=np.int) for i in range(num_chrms)]
	for i in range(num_chrms):
		print( "g arrays: starting chrm {}".format(i) )
		prev = sum(mut_pos_g_sizes[0:i])
		for j in range(mut_pos_g_sizes[i]-1):
			final_array_g[prev + j] = np.sum(final_array[ prev + j*group_by : prev + (j+1)*group_by ], axis=0)
			if np.sum(final_array_g[prev + j]) == 0.:
				print("final_g_array[{}]".format(j))
			mut_pos_g[i][j][0] = mut_pos[i][ j*group_by ]
			mut_pos_g[i][j][1] = mut_pos[i][ (j+1)*group_by-1 ]
		last = mut_pos_g_sizes[i]-1
		final_array_g[prev + last] = np.sum(final_array[ prev + last*group_by : prev + (last+1)*group_by ], axis=0)
		mut_pos_g[i][last][0] = mut_pos[i][ last*group_by ]
		mut_pos_g[i][last][1] = mut_pos[i][ (last+1)*group_by-1 ]

	#print( "Every position has at least one mutation? {}".format(np.sum(final_array_g, axis=1).all()) )
	assert( np.sum(final_array_g, axis=1).all() )

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array_g and mut_g_array are complete".format(cur_time-beg_time) )
	
	chrm_begs = []
	prev = 0
	for i in range(num_chrms):
		chrm_begs.append(prev)
		prev += mut_pos_g[i].shape[0]
	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] chrm_begs is complete".format(cur_time-beg_time) )

	return final_array_g, mut_pos_g, chrm_begs

if __name__ == "__main__":

	if len(sys.argv) != 4:
		print("Usage: python3 preproc.py [ src file (.txt) ] [ dest file (.npy) ] [ group by ]")
		exit(1)
	csv_file_name = sys.argv[1]
	mc_data_file_name = sys.argv[2]
	group_by = int(sys.argv[3])
	m_array,m_mut_pos,m_chrm_begs = preproc(csv_file_name, group_by, "male")
	f_array,f_mut_pos,f_chrm_begs = preproc(csv_file_name, group_by, "female")
	b_array,b_mut_pos,b_chrm_begs = preproc(csv_file_name, group_by, "both")
	np.savez(mc_data_file_name + "_m", array=m_array, mut_pos=m_mut_pos, chrm_begs=m_chrm_begs)
	np.savez(mc_data_file_name + "_f", array=f_array, mut_pos=f_mut_pos, chrm_begs=f_chrm_begs)
	np.savez(mc_data_file_name + "_b", array=b_array, mut_pos=b_mut_pos, chrm_begs=b_chrm_begs)
	m_mc_data = {"array": m_array, "mut_pos": m_mut_pos, "chrm_begs":  m_chrm_begs}
	f_mc_data = {"array": f_array, "mut_pos": f_mut_pos, "chrm_begs":  f_chrm_begs}
	b_mc_data = {"array": b_array, "mut_pos": b_mut_pos, "chrm_begs":  b_chrm_begs}
	cfile_core_name = "data/mc_chrms/mc_100"
	for i in range(NUM_CHRMS):
		convert(m_mc_data, i, cfile_core_name + "_m")
		convert(f_mc_data, i, cfile_core_name + "_f")
		convert(b_mc_data, i, cfile_core_name + "_b")
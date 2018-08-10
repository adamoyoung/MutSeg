import sys
import numpy as np
import time
from convert_to_C import convert
import os

# argv[1]: input file
# argv[2]: output file
# argv[3]: group_by

np.random.seed(373)
np.set_printoptions(threshold=1000)
float_t = np.float64
NUM_CHRMS = 22 # does not include the X chromosome or Y chromosome
#CHRM_LENS = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468] #,57227415 ]
CHRM_LENS = [ 249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566 ]
sex_to_ind = {"m": 0, "M": 0, "male": 0, "f": 1, "F": 1, "female": 1, "b": 2, "both": 2}

def preproc(dir_name, group_by=1, num_folds=5):

	num_chrms = NUM_CHRMS

	beg_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print("[0] starting preproc, file = {}, group_by = {}, num_folds = {}".format(dir_name,group_by,num_folds) )

	mut_pos = [set() for i in range(num_chrms)]
	mut_tups = []
	typs_set = set()

	file_paths = []
	file_count = 0
	entries = sorted(os.listdir(dir_name))
	for entry in entries:
		entry_path = os.path.join(dir_name,entry)
		if os.path.isfile(entry_path):
			file_paths.append(entry_path)
			file_count += 1
	assert( file_count == len(entries) )

	for file_path in file_paths:

		file_mut_tups = []

		file = open(file_path, 'r')

		# remove header
		file.readline()
		
		line = file.readline()
		line_count = 1
		total_ints = 0.
		while line != '':
			line = line.split(",")
			#assert(len(line) == 12)
			if len(line) != 12:
				print(line_count)
				print(line)
				quit()
			# if len(line) < 2:
			# 	line = file.readline()
			# 	continue
			chrm, pos, ints, typ, sex = line[2], line[3], line[7], line[10], line[11].strip()
			assert( chrm != "X" )
			if chrm == "chr" or chrm == "Y":
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
			pos, ints = int(pos)-1, float(ints)
			assert( chrm >= 0 and chrm < num_chrms )
			assert( sex == "M" or sex == "F")
			mut_pos[chrm].add( pos )
			file_mut_tups.append( (chrm,pos,typ,ints,sex) )
			typs_set.add( typ )
			line = file.readline()
			line_count += 1
			total_ints += ints

		for mut_tup in file_mut_tups:
			# convert intensities to frequencies
			freq_mut_tup = (mut_tup[0], mut_tup[1], mut_tup[2], mut_tup[3]/total_ints, mut_tup[4])
			mut_tups.append(freq_mut_tup)

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


	print( "final array dimensions: [ {} x {} x {} ]".format( num_folds, comb_mut_pos_size, len(typs_list) ) )
	final_array = np.zeros([num_folds, comb_mut_pos_size, len(typs_list)], dtype=float_t)
	print( "final array size: {} MB".format(sys.getsizeof(final_array) // 1000000) )

	num_data_pts = len(mut_tups)
	indices = np.arange(num_data_pts)
	np.random.shuffle(indices)
	part_size = num_data_pts // num_folds
	parts = [ (i*part_size, (i+1)*part_size) for i in range(num_folds-1) ] + [ ((num_folds-1)*part_size, num_data_pts) ]

	for p in range(len(parts)):
		for i in range(parts[p][0],parts[p][1]):
			chrm, pos, typ, freq, sex = mut_tups[indices[i]]
			fold = p
			typ_ind = typ_to_ind[typ]
			pos_ind = abs_to_relative[chrm][pos]
			final_array[fold][pos_ind][typ_ind] = freq

	# for i in range(num_chrms):
	# 	print( "starting chrm {} mut_tups".format(i+1) )
	# 	for j in range(len(mut_tups[i])):
	# 		fold = get_fold(i,j)
	# 		assert( fold >= 0 )
	# 		chrm, pos, typ, dens, sex = mut_tups[i][j]
	# 		assert( type(pos) == int and type(typ) == str and type(dens) == float and type(sex) == str)
	# 		typ_ind = typ_to_ind[typ]
	# 		pos_ind = abs_to_relative[i][pos]
	# 		final_array[fold][pos_ind][typ_ind] = dens

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array is complete".format(cur_time-beg_time) )

	assert( np.sum(final_array, axis=(0,2)).all() )

	# group into groups the size of group_by
	# note that any mutations at the end that are not in a group of size group_by are discarded
	# also note that the mut_pos_g array is [beg_pt, end_pt] inclusive
	mut_pos_g_sizes = [mut_pos[i].size // group_by for i in range(num_chrms)]
	final_array_g = np.zeros([num_folds,np.sum(mut_pos_g_sizes),len(typs_list)], dtype=float_t)
	mut_pos_g = [np.zeros([mut_pos_g_sizes[i],2], dtype=np.int) for i in range(num_chrms)]
	for i in range(num_chrms):
		print( "g arrays: starting chrm {}".format(i+1) )
		prev = sum(mut_pos_g_sizes[0:i])
		for j in range(mut_pos_g_sizes[i]-1):
			final_array_g[:,prev+j] = np.sum(final_array[:,prev + j*group_by : prev + (j+1)*group_by], axis=1)
			# final_array_g[0][prev + j] = np.sum(final_array[0][ prev + j*group_by : prev + (j+1)*group_by ], axis=0)
			# final_array_g[1][prev + j] = np.sum(final_array[1][ prev + j*group_by : prev + (j+1)*group_by ], axis=0)
			# final_array_g[2][prev + j] = np.sum(final_array[2][ prev + j*group_by : prev + (j+1)*group_by ], axis=0)
			mut_pos_g[i][j][0] = mut_pos[i][ j*group_by ]
			mut_pos_g[i][j][1] = mut_pos[i][ (j+1)*group_by-1 ]
		last = mut_pos_g_sizes[i]-1
		final_array_g[:,prev+last] = np.sum(final_array[:,prev + last*group_by : prev + (last+1)*group_by], axis=1)
		# final_array_g[0][prev + last] = np.sum(final_array[0][ prev + last*group_by : prev + (last+1)*group_by ], axis=0)
		# final_array_g[1][prev + last] = np.sum(final_array[1][ prev + last*group_by : prev + (last+1)*group_by ], axis=0)
		# final_array_g[2][prev + last] = np.sum(final_array[2][ prev + last*group_by : prev + (last+1)*group_by ], axis=0)
		mut_pos_g[i][last][0] = mut_pos[i][ last*group_by ]
		mut_pos_g[i][last][1] = mut_pos[i][ (last+1)*group_by-1 ]

	assert( np.sum(final_array_g, axis=(0,2)).all() )

	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] final_array_g and mut_g_array are complete".format(cur_time-beg_time) )
	
	chrm_begs = []
	prev = 0
	for i in range(num_chrms):
		chrm_begs.append(prev)
		prev += mut_pos_g[i].shape[0]
	cur_time = time.clock_gettime(time.CLOCK_MONOTONIC)
	print( "[{0:.0f}] chrm_begs is complete".format(cur_time-beg_time) )

	return final_array_g, mut_pos_g, chrm_begs, typs_list

if __name__ == "__main__":

	if len(sys.argv) != 6:
		eprint("Usage: python3 preproc.py [ src directory (full of .txt files) ] [ dest file (.npz) ] [ group by ] [ num folds ] [ cfile dir path ]")
		exit(1)
	dir_name = sys.argv[1]
	mc_data_file_name = sys.argv[2]
	group_by = int(sys.argv[3])
	num_folds = int(sys.argv[4])
	cfile_dir_path = sys.argv[5] # often "data/mc_chrms/mc_100"
	array, mut_pos, chrm_begs, typs_list = preproc(dir_name, group_by, num_folds)
	np.savez(mc_data_file_name, array=array, mut_pos=mut_pos, chrm_begs=chrm_begs, typs_list=typs_list, chrm_lens=CHRM_LENS, float_t=[float_t], sex_to_ind=[sex_to_ind])
	# mc_data = np.load(mc_data_file_name)
	# array = mc_data["array"]
	# mut_pos = mc_data["mut_pos"]
	# chrm_begs = mc_data["chrm_begs"]
	# typs_list = mc_data["typs_list"]
	folds = [n for n in range(num_folds)]
	for n in range(num_folds):
		if num_folds == 1:
			train_folds = [0]
			valid_fold = None
		else:
			train_folds = folds[:n] + folds[n+1:]
			valid_fold = n
		print("train folds = {}, valid fold = [{}]".format(train_folds,valid_fold))
		train_array = np.sum([array[f] for f in train_folds], axis=0)
		# m_mc_data = {"array": train_arrays[0], "mut_pos": mut_pos, "chrm_begs":  chrm_begs, "typs_list": typs_list}
		# f_mc_data = {"array": train_arrays[1], "mut_pos": mut_pos, "chrm_begs":  chrm_begs, "typs_list": typs_list}
		data = {"array": train_array, "mut_pos": mut_pos, "chrm_begs":  chrm_begs, "typs_list": typs_list}
		for c in range(NUM_CHRMS):
			# convert(m_mc_data, c, cfile_core_name + "_m")
			# convert(f_mc_data, c, cfile_core_name + "_f")
			convert(data, c, "{}/fold_{}_chrm_{}.dat".format(cfile_dir_path,n,c+1))

	# also do one with no folds
	data = {"array": np.sum(array, axis=0), "mut_pos": mut_pos, "chrm_begs":  chrm_begs, "typs_list": typs_list}
	for c in range(NUM_CHRMS):
		convert(data, c, "{}/all_chrm_{}.dat".format(cfile_dir_path,c+1))
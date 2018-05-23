import sys
import numpy as np

# argv[1]:
# argv[2]:

data = np.load(sys.argv[1])
cfile_name_core = "mutation_counts"

array = data["array"]
mut_pos = data["mut_pos"]
chrm_begs = data["chrm_begs"]
if len(sys.argv) == 3:
	chrm_num = int(sys.argv[2]) - 1
	chrm_start = chrm_begs[chrm_num]
	if chrm_num == len(chrm_begs)-1:
		chrm_end = array.shape[0]
	else:
		chrm_end = chrm_begs[chrm_num+1]
	print( "Converting chromosome {}".format(chrm_num+1) )
	print( "chrm_start = {}, chrm_end = {}, length = {}".format(chrm_start,chrm_end,chrm_end-chrm_start) )
	barray = bytes(array[chrm_start:chrm_end])
	cfile_name = "{}_chrm_{}.dat".format(cfile_name_core,chrm_num+1)
else:
	print( "Converting entire genome" )
	barray = bytes(array)
	cfile_name = "{}.dat".format(cfile_name_core)

with open(cfile_name, 'wb') as file:
	file.write(barray)
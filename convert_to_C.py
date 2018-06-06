import sys
import numpy as np

# argv[1]:
# argv[2]:

def convert(mc_data, chrm, cfile_name_core):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]
	chrm_start = chrm_begs[chrm]
	if chrm == len(chrm_begs)-1:
		chrm_end = array.shape[0]
	else:
		chrm_end = chrm_begs[chrm+1]
	print( "Converting chromosome {}".format(chrm+1) )
	print( "chrm_start = {}, chrm_end = {}, length = {}".format(chrm_start,chrm_end,chrm_end-chrm_start) )
	barray = bytes(array[chrm_start:chrm_end])
	assert( len(barray) == (chrm_end-chrm_start)*40*8 )
	cfile_name = "{}_chrm_{}.dat".format(cfile_name_core,chrm+1)

	with open(cfile_name, 'wb') as file:
		file.write(barray)


if __name__ == "__main__":

	mc_data_file_name = sys.argv[1]
	chrm = int(sys.argv[2]) - 1

	mc_data = np.load(mc_data_file_name)
	cfile_name_core = "data/mc_100"

	convert(mc_data,chrm,cfile_name_core)



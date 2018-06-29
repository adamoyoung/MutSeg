import sys
import numpy as np

# argv[1]:
# argv[2]:

def convert(data, chrm, cfile_name):

	array = data["array"]
	mut_pos = data["mut_pos"]
	chrm_begs = data["chrm_begs"]
	chrm_start = chrm_begs[chrm]
	if chrm == len(chrm_begs)-1:
		chrm_end = array.shape[0]
	else:
		chrm_end = chrm_begs[chrm+1]
	print( "Converting chromosome {}".format(chrm+1) )
	print( "chrm_start = {}, chrm_end = {}, length = {}".format(chrm_start,chrm_end,chrm_end-chrm_start) )

	barray = bytes(array[chrm_start:chrm_end])
	#valid_barray = bytes(valid_array[chrm_start:chrm_end])
	assert( len(barray) == (chrm_end-chrm_start)*array.shape[1]*8 )
	#assert( len(train_barray) == len(valid_barray) )

	with open(cfile_name, 'wb') as file:
		file.write(barray)


# if __name__ == "__main__":

# 	mc_data_file_name = sys.argv[1]
# 	chrm = int(sys.argv[2]) - 1

# 	mc_data = np.load(mc_data_file_name)
# 	cfile_name_core = "data/mc_100"

# 	convert(mc_data,chrm,cfile_name_core)



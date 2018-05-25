import numpy as np
import struct
import sys
import matplotlib.pyplot as plt

float_t = np.float64
np.set_printoptions(threshold=np.iinfo(np.int32).max)

# chromosome lens (for 22 autosomal chromosomes and Y, X not included)
# got these from cytoBand.txt online
#chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]
# got these from UCSC genome browser (hg38)
chrm_lens = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,57227415 ]

def median(a, b):

	return int(round((a+b)/2))

def traceback(S_s_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]
	chrm_mut_pos = mut_pos[chrm]

	# read in contents of S_s file
	S_s_file = open(S_s_file_name, 'rb')
	S_s_bytes = S_s_file.read(4*M*K)
	S_s = []
	for i in range(len(S_s_bytes) // 4):
		number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
		S_s.append(number)

	# get segmentation
	final_path = []
	final_path.insert(0,M)
	k = K-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		final_path.insert(0,col+1)
		k -= 1
	final_path.insert(0,0)

	# find the number of mutations in each segment
	num_muts = np.zeros([len(final_path)-1, T], dtype=float_t)
	for i in range(len(final_path)-1):
		cur_num_muts = np.sum(array[ chrm_beg+final_path[i] : chrm_beg+final_path[i+1] ], axis=0)
		num_muts[i] = cur_num_muts

	# get the acutal bp positions of the segment boundaries
	chrm_bounds = []
	chrm_bounds.append(0)
	for i in range(len(final_path)-2):
		beg_pt = chrm_mut_pos[final_path[i]][1]
		end_pt = chrm_mut_pos[final_path[i+1]][0]
		chrm_bounds.append(median(beg_pt,end_pt))
	chrm_bounds.append(chrm_lens[chrm])
	chrm_bounds = np.array(chrm_bounds, dtype=np.uint32)

	# save files as csv
	print(chrm_bounds)
	np.savetxt("chrm_{}_bounds.csv".format(chrm+1), chrm_bounds, "%u", delimiter=",")
	np.savetxt("chrm_{}_num_muts.csv".format(chrm+1), num_muts, "%f", delimiter=",")

def scores(E_f_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]
	chrm_mut_pos = mut_pos[chrm]

	E_f_file = open(E_f_file_name, 'rb')
	E_f_bytes = E_f_file.read(8*K)

	initial_score = struct.unpack('d', E_f_bytes[:8])[0]
	final_score = struct.unpack('d', E_f_bytes[-8:])[0]

	total_Ts = np.zeros([len(chrm_begs), T], dtype=float_t)
	for i in range(len(chrm_begs) - 1):
		total_Ts[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ], axis=0)
	total_Ts[-1] = np.sum(array[ chrm_begs[-1] : ], axis=0)
	total_Ms = np.reshape(np.sum(total_Ts, axis=1), [len(chrm_begs), 1])
	term_threes = - np.sum( (total_Ts / total_Ms) * np.log((total_Ts / total_Ms)), axis=1 )

	print( "Entropy of T = {}".format(term_threes[chrm]) )
	print( "Initial: score = {}, information = {}".format(initial_score, initial_score + term_threes[chrm]) )
	print( "Optimal: score = {}, information = {}".format(final_score, final_score + term_threes[chrm]) )

def corr(S_s_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]
	chrm_mut_pos = mut_pos[chrm]

	# read in contents of S_s file
	S_s_file = open(S_s_file_name, 'rb')
	S_s_bytes = S_s_file.read(4*M*K)
	S_s = []
	for i in range(len(S_s_bytes) // 4):
		number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
		S_s.append(number)

	# get segmentation
	final_path = []
	final_path.insert(0,M)
	k = K-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		final_path.insert(0,col+1)
		k -= 1
	final_path.insert(0,0)

	seg_counts = np.zeros([len(final_path)-1,T], dtype=float_t)
	for i in range(len(final_path)-1):
		seg_counts[i] = np.sum(array[ chrm_beg+final_path[i] : chrm_beg+final_path[i+1] ], axis=0)
	corr_matrix = np.corrcoef(seg_counts, rowvar=False)
	#print(np.corrcoef(seg_counts, rowvar=False).shape)
	plt.imshow(corr_matrix)
	plt.show()

if __name__ == "__main__":

	if len(sys.argv) != 6:
		print("Error: Usage")

	mc_data_file_name = sys.argv[1]
	file_name = sys.argv[2] # could be S_s or E_f
	total_K = int(sys.argv[3])
	chrm = int(sys.argv[4])-1
	mode = sys.argv[5]

	mc_data = np.load(mc_data_file_name)

	if mode == "s":
		traceback(file_name, mc_data, total_K, chrm)
	elif mode == "e":
		scores(file_name, mc_data, total_K, chrm)
	elif mode == "c":
		corr(file_name, mc_data, total_K, chrm)
	else:
		print("Error: Usage")
import numpy as np
import struct
import sys

# got these from cytoBand.txt
chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]

def median(a, b):

	return int(round((a+b)/2))

def traceback(S_s_file_name, M, K, chrm_mut_pos, final_pos):

	S_s_file = open(S_s_file_name, 'rb')
	S_s_bytes = S_s_file.read(4*M*K)

	S_s = []
	for i in range(len(S_s_bytes) // 4):
		number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
		S_s.append(number)

	final_path = []
	final_path.insert(0,M)
	k = K-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		final_path.insert(0,col+1)
		k -= 1
	final_path.insert(0,0)

	#print(final_path)

	chrm_bounds = []
	chrm_bounds.append(0)
	for i in range(len(final_path)-2):
		beg_pt = chrm_mut_pos[final_path[i]][1]
		end_pt = chrm_mut_pos[final_path[i+1]][0]
		chrm_bounds.append(median(beg_pt,end_pt))
	chrm_bounds.append(final_pos)

	print(chrm_bounds)
	print(len(chrm_bounds) == K+1)

	# for i in range(len(final_path)-1):
	# 	print(chrm_mut_pos[final_path[i]])
	# print(chrm_mut_pos[final_path[-1]-1])

if __name__ == "__main__":

	mc_data_file_name = sys.argv[1]
	S_s_file_name = sys.argv[2]
	K = int(sys.argv[3])
	chrm = int(sys.argv[4])-1

	mc_data = np.load(sys.argv[1])
	mut_pos = mc_data["mut_pos"]

	mut_sizes = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(mut_sizes)
	ks = [ round((mut_sizes[i]/total_num_muts)*K) for i in range(len(mut_sizes)) ]

	traceback(sys.argv[2], mut_sizes[chrm], ks[chrm], mut_pos[chrm], chrm_lens[chrm])
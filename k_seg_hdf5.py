import sys
import numpy as np
import itertools
import h5py

float_t = np.float64
eps = np.finfo(float_t).eps # 2.2e-16
ninf = -np.inf
np.random.seed(303)

def compute_counts(muts, M, T):
	# assumes that muts is a [M,T] binary array
	# returns an [M,T] array of the cumulative number of mutations, of each tumour type, from 0:m for every m = {0, ..., M}

	C = np.zeros([M+1,T], dtype=float_t)

	for i in range(1,M+1):
		C[i] = C[i-1] + muts[i-1]

	return C

def score(start, end, C, M, seed=[]):
	# start inclusive, end exclusive
	# checks for segmentations that violate seed
	# does not check for minimum segment size

	tumour_C = C[end] - C[start]
	total_C = np.sum(tumour_C)
	for seg in seed:
		if seg > start and seg < end:
			return ninf
	term_one = np.sum( (tumour_C/M) * np.log( (tumour_C/M) + eps ))
	term_two = -( total_C / M ) * np.log( ( total_C / M) + eps )

	return term_one + term_two 

def k_seg(muts, K, min_size, seed, data_file_name):
	# muts is an [M,T] binary array
	# pos is a list

	M, T = muts.shape[0], muts.shape[1]
	#N = len(pos)
	assert( M < 4294967295 )

	hfile = h5py.File(data_file_name, 'a', libver='latest')
	t_group = hfile.require_group("tables")
	# traceback array: save on disk (approx 0.16*K GB)
	S_s = t_group.require_dataset("S_s", (M,K), dtype=np.uint32)
	# final results for each k: save on disk (approx 0.32 GB)
	E_f = t_group.require_dataset("E_f", (K,), dtype=float_t)
	# working set: keep in memory (approx 0.64 GB)
	E_s = np.full([M,2], np.nan, dtype=float_t)

	C = compute_counts(muts, M, T)
	M_total = np.sum(muts)
	# we no longer need muts
	del muts

	print( "k == 0" )
	for i in range(0,M):
		if i+1 < min_size:
			E_s[i,0] = ninf
		else:
			E_s[i,0] = score(0,i+1,C,M_total,seed)
			S_s[i,0] = 0
	E_f[0] = E_s[M-1,0]

	temp_scores = np.zeros([M], dtype=float_t)
	temp_seqs = np.zeros([M], dtype=np.int32)
	# array that converts k to an index (0 or 1) in the working set array
	E_inds = np.zeros([K], dtype=np.uint8)
	E_inds[0] = 0

	for k in range(1,K): #  1 <= k <= 9
		print( "k == {}".format(k) )
		E_inds[k] = 1 - E_inds[k-1]
		for i in range(k,M): # 1 <= i <= 9
			for j in range(k,i+1): # 1 <= j <= 9
				if i-(j-1) < min_size:
					temp_scores[j-1:i] = np.full([i-(j-1)], ninf, dtype=float_t)
					break
				temp_scores[j-1] = E_s[j-1,E_inds[k-1]] + score(j,i+1,C,M_total,seed)
				temp_seqs[j-1] = j-1
			E_s[i,E_inds[k]] = np.max(temp_scores[k-1:i])
			S_s[i,k] = temp_seqs[k-1+np.argmax(temp_scores[k-1:i])]
		E_f[k] = E_s[M-1,E_inds[k]]
			
	#print(t_group.get("E_f")[:])
	#print(t_group.get("S_s")[:])

	final_score = E_f[-1]
	
	if np.isnan(final_score) or final_score == ninf:
		return None, None

	final_path = []
	final_path.insert(0,M)
	k = K-1
	row = M-1
	while k > 0:
		row = S_s[row,k]
		final_path.insert(0,row+1)
		k -= 1
	final_path.insert(0,0)

	return final_score, final_path
	
"""================================================"""

def total_score(muts, partition, min_size, M_total):

	M, T = muts.shape[0], muts.shape[1]
	total_score = 0.

	for i in range(1,len(partition)):
		if partition[i] - partition[i-1] < min_size:
			#print("this should happen")
			total_score = ninf
			return total_score

	for i in range(1,len(partition)):
		right = partition[i]
		left = partition[i-1]
		# print("{} {}".format(left,right))
		t_counts = np.sum(muts[left:right], axis=0)
		total_counts = np.sum(t_counts)
		term_one = np.sum( (t_counts/M_total) * np.log((t_counts/M_total) + eps) )
		term_two = -( (total_counts/M_total) * np.log(total_counts/M_total) )
		total_score += term_one + term_two

	return total_score

def brute_force(muts, T, K, min_size=1, seed=[]):

	M = len(muts)
	#N = len(pos)
	M_total = np.sum(muts)

	best_score = None
	best_partition = None

	partitions = itertools.combinations(range(1,M), K-1)
	skip = False
	for partition in partitions:
		partition_set = set(partition)
		for boundary in seed:
			if boundary not in partition_set:
				skip = True
				break
		if skip:
			skip = False
		else:
			partition = [0] + sorted(list(partition)) + [M]
			score = total_score(muts,partition,min_size,M_total)
			if best_score is None or score > best_score:
				best_score = score
				best_partition = partition

	return best_score, best_partition



# def main(argc, argv):

# 	if argc != 3 or int(argv[1]) > 0:
# 		print("Usage: k_seg k (k is an int > 0)")
# 	k_seg(k)

if __name__ == "__main__":

	# if len(sys.argv) != 2 or int(sys.argv[1]) > 0:
	# 	print("Usage: k_seg k (k is an int > 0)")
	# main(len(sys.argv), sys.argv)

	T = 2
	K = 4
	M = 10
	min_size = 1
	seed = [] # [2, 6] # up until 2, up until 6
	filename = "test_data_file"

	muts = np.zeros([M,T],dtype=np.int8)
	screen = np.random.choice([3,4,7], size=[M])
	#screen = np.array([3,3,3,3,7,7,4,4,4,4])
	#print(screen)
	muts[:,0] += screen % 2
	muts[:,1] += screen % 3
	#print(muts)
	#pos = [] # [2, 5, 6, 33, 40, 45, 47, 55, 57, 88]
	bf_results = brute_force(muts,T,K,min_size,seed)
	print( "BF:\nscore = {}\npositions = {}".format(bf_results[0], bf_results[1]) )
	dp_results = k_seg(muts,K,min_size,seed,filename)
	print( "DP:\nscore = {}\npositions = {}".format(dp_results[0], dp_results[1]) )
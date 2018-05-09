import sys
import numpy as np
import itertools

eps = np.finfo(np.float64).eps # 2.2e-16
ninf = -np.inf
np.random.seed(303)

def compute_counts(muts, M, T):
	# assumes that muts is a [M,T] binary array
	# returns an [M,T] array of the cumulative number of mutations, of each tumour type, from 0:m for every m = {0, ..., M}

	C = np.zeros([M+1,T], dtype=np.float64)

	#C[0][muts[0]] += 1

	for i in range(1,M+1):
		C[i] = C[i-1] + muts[i-1]

	return C

def score(start, end, C, M):
	# start inclusive, end exclusive

	tumour_C = C[end] - C[start]
	total_C = np.sum(tumour_C)
	term_one = np.sum( (tumour_C/M) * np.log( (tumour_C/M) + eps ))
	term_two = -( total_C / M ) * np.log( ( total_C / M) + eps )

	return term_one + term_two 

def k_seg(muts, pos, K, min_size=1):
	# muts is an [M,T] binary array
	# pos is a list

	M, T = muts.shape[0], muts.shape[1]
	N = len(pos)

	E_s = np.full([M+1,K+1], np.nan, dtype=np.float64)
	S_s = np.full([M+1,K+1,K], np.nan, dtype=np.int)
	C = compute_counts(muts, M, T)
	M_total = np.sum(muts)

	for i in range(1,M+1):
		if i < min_size:
			E_s[i,1] = ninf
			S_s[i,1,0] = -1
		else:
			E_s[i,1] = score(0, i, C, M_total)
			S_s[i,1,0] = i

	temp_scores = np.zeros([M+1], dtype=np.float64)
	temp_seqs = np.zeros([M+1,K], dtype=np.int)

	for k in range(2,K+1): # 2 <= k <= K
		for i in range(k+1,M+1): # k < i <= M
			for j in range(k,i): # k <= j < i
				if i-j < min_size:
					temp_scores[j:i] = np.full([i-j], ninf, dtype=np.float64)
					break
				temp_scores[j] = E_s[j,k-1] + score(j,i,C,M_total)
				temp_seqs[j, 0:k-1] = S_s[j,k-1,0:k-1]
				temp_seqs[j, k-1] = i
			E_s[i,k] = np.max(temp_scores[k:i])
			S_s[i,k,0:k] = temp_seqs[k+np.argmax(temp_scores[k:i]), 0:k]
			#print("S_s[{},{},0:{}] = {}".format(i,k,k,S_s[i,k,0:k]))

	print(E_s[1:,1:])
	#print(S_s[M,K,0:K])

	return E_s[M,K], S_s[M,K,0:K]
	
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

def brute_force(muts, pos, T, K, min_size=1):

	M = len(muts)
	N = len(pos)
	M_total = np.sum(muts)

	best_score = None
	best_partition = None

	partitions = itertools.combinations(range(1,M), K-1)

	for partition in partitions:
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
	min_size = 2

	muts = np.zeros([M,T],dtype=np.int8)
	screen = np.random.choice([3,4,7], size=[M])
	#screen = np.array([3,3,3,3,7,7,4,4,4,4])
	#print(screen)
	muts[:,0] += screen % 2
	muts[:,1] += screen % 3
	#print(muts)
	pos = [] # [2, 5, 6, 33, 40, 45, 47, 55, 57, 88]
	bf_results = brute_force(muts,pos,T,K,min_size)
	print( "BF:\nscore = {}\npositions = {}".format(bf_results[0], bf_results[1]) )
	dp_results = k_seg(muts,pos,K,min_size)
	print( "DP:\nscore = {}\npositions = {}".format(dp_results[0], dp_results[1]) )
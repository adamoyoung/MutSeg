import sys
import numpy as np
import itertools

eps = np.finfo(np.float64).eps # 2.2e-16
ninf = -np.inf
np.random.seed(303)

def compute_counts(muts, M, T):
	# returns an [M,T] array of the cumulative number of mutations, of each tumour type, from 0:m for every m = {0, ..., M}

	C = np.zeros([M+1,T], dtype=np.float64)

	#C[0][muts[0]] += 1

	for i in range(1,M+1):
		for t in range(T):
			C[i][t] = C[i-1][t]
		C[i][muts[i-1]] += 1

	return C

def score(start, end, C, M):
	# start inclusive, end exclusive

	my_C = C[end] - C[start]
	t_one = np.sum( (my_C/M) * np.log( (my_C/M) + eps ))
	t_two = -( (end-start) / M ) * np.log( ((end-start) / M) + eps )

	return t_one + t_two 

def k_seg(muts, pos, T, K, min_size=1):

	M = len(muts)
	N = len(pos)

	E_s = np.full([M+1,K+1], np.nan, dtype=np.float64)
	S_s = np.full([M+1,K+1,K], np.nan, dtype=np.int)
	C = compute_counts(muts, M, T)

	for i in range(1,M+1):
		if i < min_size:
			E_s[i,1] = ninf
		else:
			E_s[i,1] = score(0, i, C, M)
			S_s[i,1,0] = i

	temp_scores = np.zeros([M+1], dtype=np.float64)
	temp_seqs = np.zeros([M+1,K], dtype=np.int)

	for k in range(2,K+1): # 2 <= k <= K
		for i in range(k,M+1): # k <= i <= M
			for j in range(1,i): # 1 <= j < i
				if i-j < min_size:
					temp_scores[j:i] = np.full([i-j], ninf, dtype=np.float64)
					break
				if np.isnan(E_s[j,k-1]):
					print("E_s, k={},j={},i={}".format(k-1,j,i))
				if np.isnan(score(j,i,C,M)):
					print("score, k={},j={},i={}".format(k,j,i))
				temp_scores[j] = E_s[j,k-1] + score(j,i,C,M)
				temp_seqs[j, 0:k-1] = S_s[j,k-1,0:k-1]
				temp_seqs[j, k-1] = i
			E_s[i,k] = np.nanmax(temp_scores[1:i])
			S_s[i,k,0:k] = temp_seqs[1+np.nanargmax(temp_scores[1:i]), 0:k]
			#print("S_s[{},{},0:{}] = {}".format(i,k,k,S_s[i,k,0:k]))

	print("DP method:")
	print(E_s[1:,1:])
	# print(E_s[M,K])
	# print(S_s[M,K,0:K])


def brute_force(muts, pos, T, K, min_size=1):

	M = len(muts)
	N = len(pos)
	muts = np.array(muts).reshape([M,1])

	best_score = None
	best_partition = None

	partitions = itertools.combinations(range(1,M), K-1)

	for partition in partitions:
		partition = [0] + sorted(list(partition)) + [M]
		score = total_score(muts,partition,T,min_size)
		if best_score is None or score > best_score:
			best_score = score
			best_partition = partition

	print("Brute force method:")
	print(best_score)
	print(best_partition)

def total_score(muts, partition, T, min_size):

	M = muts.size
	tumours = np.arange(T) * np.ones([M,1]) # [10,2]
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
		t_counts = np.sum(muts[left:right] == tumours[left:right], axis=0)
		term_one = np.sum( (t_counts/M) * np.log((t_counts/M) + eps) )
		term_two = -( ((right-left)/M) * np.log((right-left)/M) )
		total_score += term_one + term_two

	return total_score



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

	#muts = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
	muts = np.random.randint(low=0,high=T,size=[M],dtype=np.int)
	print(muts)
	pos = [] # [2, 5, 6, 33, 40, 45, 47, 55, 57, 88]

	# print("min_size=1:")
	# brute_force(muts, pos, T, K, 1)
	# k_seg(muts, pos, T, K, 1)
	print("min_size=2:")
	brute_force(muts, pos, T, K, 2)
	k_seg(muts, pos, T, K, 2)
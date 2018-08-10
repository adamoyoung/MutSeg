import numpy as np
import struct
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import logsumexp
from math import ceil
import re
import os


SEG_SIZE = 1000000 # (1 MB)
# NUM_CHRMS = 22
# chromosome lens (for 22 autosomal chromosomes and Y, X not included)
# got these from cytoBand.txt online
#chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]
# got these from UCSC genome browser (hg38)
#chrm_lens = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,57227415 ]
#typs = ['BLCA','BOCA','BRCA','BTCA','CESC','CLLE','CMDI','COAD','DLBC','EOPC','ESAD','GACA','GBM','HNSC','KICH','KIRC','KIRP','LAML','LGG','LICA','LIHC','LINC','LIRI','LUAD','LUSC','MALY','MELA','ORCA','OV','PACA','PAEN','PBCA','PRAD','READ','RECA','SARC','SKCM','STAD','THCA','UCEC']

def median(a, b):

	return int(round((a+b)/2))

def myround(x, base):

	return int(base*round(float(x)/base))

def traceback(S_s_file_name, mc_data, Ks, chrm_ind, fold=None):

	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]
	chrm_lens = mc_data["chrm_lens"]
	if fold is None:
		f_array = np.sum(mc_data["array"], axis=0)
	else:
		f_array = np.sum(mc_data["array"],axis=0) - mc_data["array"][fold]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	# total_num_muts = sum(num_muts)
	# Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm_ind], Ks[chrm_ind], f_array.shape[1], chrm_begs[chrm_ind]
	chrm_mut_pos = mut_pos[chrm_ind]

	# read in contents of S_s file
	S_s_file = open(S_s_file_name, 'rb')
	S_s_bytes = S_s_file.read(4*M*K)
	S_s = []
	for i in range(len(S_s_bytes) // 4):
		number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
		S_s.append(number)

	# get segmentation
	mut_bounds = []
	mut_bounds.insert(0,M)
	k = K-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		mut_bounds.insert(0,col+1)
		k -= 1
	mut_bounds.insert(0,0)

	# find the number of mutations in each segment
	mut_ints = np.zeros([len(mut_bounds)-1, T], dtype=float_t)
	for i in range(len(mut_bounds)-1):
		cur_mut_ints = np.sum(f_array[ chrm_beg+mut_bounds[i] : chrm_beg+mut_bounds[i+1] ], axis=0)
		mut_ints[i] = cur_mut_ints

	# get the acutal bp positions of the segment boundaries
	bp_bounds = []
	bp_bounds.append(0)
	for i in range(len(mut_bounds)-2):
		beg_pt = chrm_mut_pos[mut_bounds[i]][1]
		end_pt = chrm_mut_pos[mut_bounds[i+1]][0]
		bp_bounds.append(median(beg_pt,end_pt))
	bp_bounds.append(chrm_lens[chrm_ind])
	bp_bounds = np.array(bp_bounds, dtype=np.uint32)

	return mut_ints, mut_bounds, bp_bounds

	# # save files as csv
	# print(bounds)
	# np.savetxt("chrm_{}_bounds.csv".format(chrm+1), bounds, "%u", delimiter=",")
	# np.savetxt("chrm_{}_num_muts.csv".format(chrm+1), num_muts, "%f", delimiter=",")

def scores(E_f_file_name, mc_data, Ks, chrm_ind):

	f_array = mc_data["array"][0] # just need it to determine T
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	# total_num_muts = sum(num_muts)
	# Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm_ind], Ks[chrm_ind], f_array.shape[1], chrm_begs[chrm_ind]

	E_f_file = open(E_f_file_name, 'rb')
	E_f_bytes = E_f_file.read(8*K)

	initial_score = struct.unpack('d', E_f_bytes[:8])[0]
	final_score = struct.unpack('d', E_f_bytes[-8:])[0]

	return final_score

	# return term_threes[chrm], term_threes[chrm] + final_score
	# print( "Entropy of T = {}".format(term_threes[chrm]) )
	# print( "Initial: score = {}, information = {}".format(initial_score, initial_score + term_threes[chrm]) )
	# print( "Optimal: score = {}, information = {}".format(final_score, final_score + term_threes[chrm]) )s

def convert_dat_to_npz(dat_dir, mc_data, total_K, bound_file_name):

	chrm_begs = mc_data["chrm_begs"]
	num_chrms = len(chrm_begs)
	float_t = mc_data["float_t"][0]
	mut_pos = mc_data["mut_pos"]
	sexes = ["male", "female", "both"]
	T = mc_data["arrays"].shape[2]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	mut_ints = np.zeros([len(sexes),num_chrms,max(Ks),T], dtype=float_t)
	bounds = np.zeros([len(sexes),num_chrms,max(Ks)+1], dtype=np.uint32)
	final_scores = np.zeros([len(sexes),num_chrms], dtype=float_t)

	for s in range(len(sexes)):
		for i in range(num_chrms):
			S_s_file_name = "{}/{}/S_s_chrm_{}.dat".format(dat_dir, sexes[s], i+1)
			chrm_mut_ints, chrm_bounds = traceback(S_s_file_name, mc_data, total_K, i, sexes[s])
			mut_ints[s,i,0:Ks[i]] = chrm_mut_ints
			bounds[s,i,0:Ks[i]+1] = chrm_bounds
			E_f_file_name = "{}/{}/E_f_chrm_{}.dat".format(dat_dir, sexes[s], i+1)
			final_scores[s,i] = scores(E_f_file_name, mc_data, total_K, i)

	np.savez(bound_file_name, Ks=Ks, mut_ints=mut_ints, bounds=bounds, final_scores=final_scores)

def mutual_info(mut_ints, M):

	eps = np.finfo(mut_ints.dtype).eps
	term_one = np.sum( (mut_ints/M) * np.log( mut_ints/M + eps ) )
	term_two = - (np.sum(mut_ints)/M) * np.log( np.sum(mut_ints)/M + eps )

	return term_one + term_two

def classify_segs(mc_data, b_data):

	""" My method -- wrong! """

	# use "both" for array and for mut_ints
	array = mc_data["arrays"][2]
	mut_pos = mc_data["mut_pos"]
	float_t = mc_data["float_t"][0]
	chrm_begs = list(mc_data["chrm_begs"]) + [array.shape[0]]
	Ks = b_data["Ks"]
	mut_ints = b_data["mut_ints"][2]
	# mut_ints is [num_chrms,max(Ks),T]
	bounds = b_data["bounds"]
	
	T = mut_ints.shape[2]
	num_chrms = mut_ints.shape[0]
	alpha = 0.5

	seg_inds = [ [] for i in range(num_chrms) ]

	for i in range(num_chrms):
		M = np.sum(array[chrm_begs[i]:chrm_begs[i+1]])
		K = Ks[i]
		best = 0
		for j in range(K):
			N = np.sum(mut_ints[i][j])
			worst = ( N/M ) * np.log( 1/T )
			seg_threshold = worst + alpha * ( best - worst )
			seg_info = mutual_info(mut_ints[i][j], M)
			assert( seg_info >= worst )
			if seg_info >= seg_threshold:
				seg_inds[i].append(j)

	for i in range(num_chrms):
		print( "chrm {}: Ks = {}, good_Ks = {}, %_good_Ks = {}".format(i+1,Ks[i],len(seg_inds[i]),len(seg_inds[i])/Ks[i]) )

def print_gurnit_2(mc_data, b_data, csv_file_name):

	chrm_begs = mc_data["chrm_begs"]
	num_chrms = chrm_begs.shape[0]

	bp_bounds = b_data["bp_bounds"]
	Ks = b_data["Ks"]
	k_modes =["n", "l"]
	assert(Ks.shape[0] <= len(k_modes))

	for k_mode in range(Ks.shape[0]):
		print(k_mode)
		k_mode_file_name = csv_file_name + "_" + str(k_modes[k_mode]) + ".csv"
		with open(k_mode_file_name, 'w') as k_mode_file:
			for c in range(num_chrms):
				for k in range(Ks[k_mode][c]):
					print( "{},{},{},{}".format(c+1,k,bp_bounds[k_mode][-1][c][k],bp_bounds[k_mode][-1][c][k+1]) , file=k_mode_file)


	print("naive")
	naive_file_name = csv_file_name + "_naive.csv"
	_, __, naive_bp_bounds, naive_Ks = naive_traceback(mc_data,SEG_SIZE)
	with open(naive_file_name, 'w') as naive_file:
		for c in range(num_chrms):
			for k in range(naive_Ks[c]):
				print( "{},{},{},{}".format(c+1,k,naive_bp_bounds[c][k],naive_bp_bounds[c][k+1]) , file=naive_file)


def calculate_entropy_of_T(mc_data):

	chrm_begs = mc_data["chrm_begs"]
	array = mc_data["arrays"][2]
	float_t = array.dtype

	T = array.shape[1]
	num_chrms = chrm_begs.shape[0]

	total_Ts = np.zeros([num_chrms, T], dtype=float_t)
	for i in range(num_chrms - 1):
		total_Ts[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ], axis=0)
	total_Ts[-1] = np.sum(array[ chrm_begs[-1] : ], axis=0)
	total_Ms = np.reshape(np.sum(total_Ts, axis=1), [num_chrms, 1])
	entropy_of_T = - np.sum( (total_Ts / total_Ms) * np.log((total_Ts / total_Ms)), axis=1 )
	assert( len(entropy_of_T.shape) == 1 and entropy_of_T.shape[0] == num_chrms )

	return entropy_of_T

def nats_to_bits(nats):

	return nats / np.log2(np.exp(1))

def compute_total_mutual_information(mc_data,mut_ints,Ks,scores):

	array = mc_data["arrays"][2]
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]
	eps = np.finfo(float_t).eps
	
	H_of_T_given_C = calculate_entropy_of_T(mc_data)
	scores += H_of_T_given_C
	# # added June 26 - possibly the source of error from before
	# log_scores = np.log(scores)

	num_chrms = chrm_begs.shape[0]
	T = array.shape[1]
	total_K = np.sum(Ks)
	chrm_begs = list(chrm_begs) + [array.shape[0]]
	# chrm_mut_dens = np.zeros([num_chrms, T], dtype=float_t)
	# for i in range(num_chrms):
	# 	chrm_mut_dens[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ], axis=0)
	chrm_mut_dens = np.sum(mut_ints,axis=1)

	# probabilities that are based off of counts
	log_P_of_C = np.log(np.sum(chrm_mut_dens, axis=1) / np.sum(chrm_mut_dens))
	assert( log_P_of_C.shape[0] == num_chrms and len(log_P_of_C.shape) == 1 and log_P_of_C.shape == scores.shape )
	log_P_of_T = np.log(np.sum(chrm_mut_dens, axis=0) / np.sum(chrm_mut_dens))
	log_P_of_B_given_C = np.full([total_K,num_chrms], eps, dtype=float_t)
	prev = 0
	for i in range(num_chrms):
		total_chrm_mut_ints = np.sum(mut_ints[i])
		for j in range(Ks[i]):
			# don't forget to sum over T
			# += is to preven 0's
			if total_chrm_mut_ints != 0.:
				log_P_of_B_given_C[prev+j,i] += np.sum(mut_ints[i,j]) / total_chrm_mut_ints 
		prev += Ks[i]
	log_P_of_B_given_C = np.log(log_P_of_B_given_C)
	log_P_of_T_given_C_and_B = np.full([T,num_chrms,total_K], eps, dtype=float_t)
	for t in range(T):
		prev = 0
		for i in range(num_chrms):
			for j in range(Ks[i]):
				if np.sum(mut_ints[i,j]) != 0.:
					log_P_of_T_given_C_and_B[t,i,prev+j] += mut_ints[i,j,t] / np.sum(mut_ints[i,j])
			prev += Ks[i]
	log_P_of_T_given_C_and_B = np.log(log_P_of_T_given_C_and_B)
	
	# new stuff
	log_P_of_C_given_B = np.full([num_chrms,total_K], eps, dtype=float_t)
	prev = 0
	for i in range(num_chrms):
		for j in range(Ks[i]):
			log_P_of_C_given_B[i,prev+j] = 1.
		prev += Ks[i]
	log_P_of_C_given_B = np.log(log_P_of_C_given_B)
	log_P_of_B = logsumexp(np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C, axis=1)
	log_P_of_T_given_B = logsumexp(np.reshape(log_P_of_C_given_B, [1,num_chrms,total_K]) + log_P_of_T_given_C_and_B, axis=1)
	log_P_of_T_given_C = logsumexp(np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_K]) + log_P_of_T_given_C_and_B, axis=2)
	log_P_of_C_and_T = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_T_given_C ).T
	# end of new stuff

	log_P_of_C_and_B = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C  ).T
	log_P_of_C_and_T_and_B = ( np.reshape(log_P_of_C, [1,num_chrms,1]) + np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_K]) + log_P_of_T_given_C_and_B ).transpose([1,0,2])
	log_P_of_T_and_B = np.reshape(log_P_of_B, [1,total_K]) + log_P_of_T_given_B

	I_of_T_and_B_given_C = np.sum(np.exp(log_P_of_C) * np.reshape(scores,[num_chrms,1]))
	H_of_C = - np.sum( np.exp(log_P_of_C) * log_P_of_C )
	H_of_C_given_T = np.sum( np.exp(log_P_of_C_and_T) * ( np.reshape(log_P_of_T, [1,T]) - log_P_of_C_and_T ) )
	H_of_C_given_B = np.sum( np.exp(log_P_of_C_and_B) * ( np.reshape(log_P_of_B, [1,total_K]) - log_P_of_C_and_B ) )
	#H_of_C_given_T_and_B = - np.sum( np.exp(log_P_of_C_and_T_and_B) * log_P_of_C_and_T_and_B ) + np.sum( np.sum( np.exp(log_P_of_C_and_T_and_B), axis=0) * log_P_of_T_and_B )
	H_of_B = - np.sum(np.exp(log_P_of_B) * log_P_of_B)
	H_of_T = - np.sum(np.exp(log_P_of_T) * log_P_of_T)
	H_of_C_and_T_and_B = - np.sum(np.exp(log_P_of_C_and_T_and_B) * log_P_of_C_and_T_and_B)
	H_of_T_and_B = - np.sum(np.exp(log_P_of_T_and_B) * log_P_of_T_and_B)
	H_of_C_given_T_and_B = H_of_C_and_T_and_B - H_of_T_and_B
	#assert( H_of_T_given_C <= H_of_T)

	I_of_T_and_B = I_of_T_and_B_given_C - H_of_C_given_T - H_of_C_given_B + H_of_C_given_T_and_B + H_of_C

	return I_of_T_and_B_given_C, I_of_T_and_B

def naive(mc_data, sex_ind):

	array = mc_data["arrays"][sex_ind]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]
	chrm_lens = mc_data["chrm_lens"]
	eps = np.finfo(float_t).eps

	for i in range(len(mut_pos)):
		assert(len(mut_pos[i].shape) == 2)
	assert(chrm_begs.shape[0] == mut_pos.shape[0])
	assert(chrm_begs.shape[0] == chrm_lens.shape[0])
	num_chrms = chrm_begs.shape[0]

	# compute total_M
	total_Ms = np.zeros([len(chrm_begs)], dtype=float_t)
	for i in range(len(chrm_begs) - 1):
		total_Ms[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ] )
	total_Ms[-1] = np.sum(array[ chrm_begs[-1] : ] )

	T = array.shape[1]
	Ks = [chrm_lens[i] // SEG_SIZE for i in range(num_chrms)]
	mut_ints = np.zeros([num_chrms,max(Ks),T], dtype=float_t)
	total_num_segs = sum(Ks)
	chrm_scores = np.zeros([num_chrms], dtype=float_t)
	# print( [ chrm_lens[i] // SEG_SIZE for i in range(len(chrm_begs)) ] )
	# print( total_num_segs )


	for i in range(len(chrm_begs)):
		#print( "chrm {}: largest mut_pos = {}, largest segment size = {}".format(i+1, mut_pos[i][-1][1], (chrm_lens[i] // SEG_SIZE)*SEG_SIZE) )
		cur_pos = 0
		for j in range( 1, Ks[i] - 1 ):
			beg_pt = (j-1)*SEG_SIZE
			end_pt = j*SEG_SIZE
			while cur_pos < mut_pos[i].shape[0] and mut_pos[i][cur_pos][0] >= beg_pt and mut_pos[i][cur_pos][1] < end_pt:
				mut_ints[i,j] += array[ chrm_begs[i]+cur_pos ]
				cur_pos += 1
			# the case where a mutation crosses over a segment boundary: put it in the segment where most of it is
			if cur_pos < mut_pos[i].shape[0] and mut_pos[i][cur_pos][0] >= beg_pt and mut_pos[i][cur_pos][1] >= end_pt:
				if mut_pos[i][cur_pos][1] - end_pt < end_pt - mut_pos[i][cur_pos][0]:
					# put it in this segment
					mut_ints[i,j] += array[ chrm_begs[i]+cur_pos ]
					cur_pos += 1
				else:
					# put it in the next segment
					mut_pos[i][cur_pos][0] = mut_pos[i][cur_pos][1]
			tumour_C = mut_ints[i,j]
			total_C = np.sum(mut_ints[i,j])
			term_one = np.sum( (tumour_C/total_Ms[i]) * np.log( (tumour_C/total_Ms[i]) + eps ))
			term_two = -( total_C / total_Ms[i] ) * np.log( ( total_C / total_Ms[i]) + eps )
			chrm_scores[i] += term_one + term_two
		# last segment gets all of mutations on the chrm that haven't been added yet
		if i == len(chrm_begs)-1:
			tumour_C = np.sum(array[ chrm_begs[i]+cur_pos: ], axis=0)
		else:
			tumour_C = np.sum(array[ chrm_begs[i]+cur_pos: chrm_begs[i+1] ], axis=0)
		total_C = np.sum(tumour_C)
		term_one = np.sum( (tumour_C / total_Ms[i]) * np.log( (tumour_C / total_Ms[i]) + eps ))
		term_two = -( total_C / total_Ms[i] ) * np.log( ( total_C / total_Ms[i]) + eps )
		chrm_scores[i] += term_one + term_two

	return mut_ints, Ks, chrm_scores

def opt_vs_naive_information(mc_data, b_data):

	# naive segmentation
	mut_ints, Ks, chrm_scores = naive(mc_data,2)
	I_of_T_and_B_given_C, I_of_T_and_B = compute_total_mutual_information(mc_data,mut_ints,Ks,chrm_scores)
	b_I_of_T_and_B = nats_to_bits(I_of_T_and_B)
	b_I_of_T_and_B_given_C = nats_to_bits(I_of_T_and_B_given_C)
	print( "NAIVE: I_of_T_and_B = {0:.6f} I_of_T_and_B_given_C = {1:.6f} Diff = {2:.6f}".format(b_I_of_T_and_B, b_I_of_T_and_B_given_C, b_I_of_T_and_B-b_I_of_T_and_B_given_C) )
	naive_values = (b_I_of_T_and_B,b_I_of_T_and_B_given_C,chrm_scores)

	# optimal segmentation	
	Ks = b_data["Ks"]
	mut_ints = b_data["mut_ints"][2]
	chrm_scores = b_data["final_scores"][2]
	I_of_T_and_B_given_C, I_of_T_and_B = compute_total_mutual_information(mc_data,mut_ints,Ks,chrm_scores)
	b_I_of_T_and_B_given_C = nats_to_bits(I_of_T_and_B_given_C)
	b_I_of_T_and_B = nats_to_bits(I_of_T_and_B)
	print( "OPT: I_of_T_and_B = {0:.6f} I_of_T_and_B_given_C = {1:.6f} Diff = {2:.6f}".format(b_I_of_T_and_B, b_I_of_T_and_B_given_C, b_I_of_T_and_B-b_I_of_T_and_B_given_C) )
	optimal_values = (b_I_of_T_and_B,b_I_of_T_and_B_given_C,chrm_scores)

	print("IMPROVEMENT: I_of_T_and_B = {0:.6f} I_of_T_and_B_given_C = {1:.6f}".format(optimal_values[0]-naive_values[0], optimal_values[1]-naive_values[1]) )

	# print segmentation scores
	print("Segmentation Scores")
	print("CHRM: NAIVE vs OPT")	
	for i in range(chrm_scores.shape[0]):
		print( "{0}: {1:.6f} vs {2:.6f}".format(i+1,nats_to_bits(naive_values[2][i]),nats_to_bits(optimal_values[2][i])) )

def classify_segs_2(mc_data, b_data):

	""" Quaid's method """

	# use "both" for array and for mut_ints
	array = mc_data["arrays"][2]
	mut_pos = mc_data["mut_pos"]
	float_t = mc_data["float_t"][0]
	chrm_begs = list(mc_data["chrm_begs"]) + [array.shape[0]]
	Ks = b_data["Ks"]
	mut_ints = b_data["mut_ints"][2]
	bounds = b_data["bounds"]
	
	T = mut_ints.shape[2]
	num_chrms = mut_ints.shape[0]
	alpha = 0.5
	eps = np.finfo(float_t).eps

	# I_of_S_and_T_given_C = np.zeros([num_chrms], dtype=float_t)

	# for i in range(num_chrms):
		
	# 	P_of_S_and_T_given_C = mut_ints[i,0:Ks[i]] / np.sum(mut_ints[i,0:Ks[i]])
	# 	P_of_S_given_C = np.sum(mut_ints[i,0:Ks[i]], axis=1) / np.sum(mut_ints[i,0:Ks[i]])
	# 	P_of_T_given_C = np.sum(mut_ints[i,0:Ks[i]], axis=0) / np.sum(mut_ints[i,0:Ks[i]])
	# 	P_of_C = np.sum(mut_ints[i,0:Ks[i]]) / np.sum(mut_ints[:,0:Ks[i]])
	# 	I_of_S_and_T_given_C[i] = np.sum( P_of_S_and_T_given_C * ( np.log(P_of_S_and_T_given_C+eps) - np.reshape(np.log(P_of_S_given_C+eps),[Ks[i],1]) - np.reshape(np.log(P_of_T_given_C+eps),[1,T]) ) )
	# 	I_of_S_and_T_given_C[i] *= P_of_C

	# I_of_S_and_T_given_C = np.sum(I_of_S_and_T_given_C)

	# print(I_of_S_and_T_given_C)

	test_results = np.zeros([num_chrms,max(Ks)], dtype=float_t)
	H_of_in_given_C = np.zeros([num_chrms,max(Ks)], dtype=float_t)

	for i in range(num_chrms):
		P_of_S_given_C = np.sum(mut_ints[i,0:Ks[i]],axis=1) / np.sum(mut_ints[i])
		P_of_T_given_C = np.sum(mut_ints[i,0:Ks[i]], axis=0) / np.sum(mut_ints[i])
		P_of_S_and_T_given_C = mut_ints[i,0:Ks[i]] / np.sum(mut_ints[i])
		for j in range(Ks[i]):
			P_of_in_given_C = P_of_S_given_C[j]
			P_of_out_given_C = 1-P_of_in_given_C
			P_of_in_and_T_given_C = np.reshape(P_of_S_and_T_given_C[j],[1,T])
			P_of_out_and_T_given_C = np.reshape(np.sum(P_of_S_and_T_given_C, axis=0),[1,T]) - P_of_in_and_T_given_C
			in_term = np.sum( P_of_in_and_T_given_C * ( np.log(P_of_in_and_T_given_C+eps) - np.reshape(np.log(P_of_in_given_C+eps),[1,1]) - np.reshape(np.log(P_of_T_given_C+eps),[1,T]) ) )
			out_term = np.sum( P_of_out_and_T_given_C * ( np.log(P_of_out_and_T_given_C+eps) - np.reshape(np.log(P_of_out_given_C+eps),[1,1]) - np.reshape(np.log(P_of_T_given_C+eps),[1,T]) ) )
			test_results[i,j] = in_term + out_term
			H_of_in_given_C[i,j] = - P_of_in_given_C * np.log(P_of_in_given_C)

	for i in range(num_chrms):
		print("chrm {}".format(i+1))
		for j in range(Ks[i]):
			print( "seg {}: test_result = {}, upper_bound = {}".format(j,test_results[i,j],H_of_in_given_C[i,j]) )


def convert_dat_to_npz_2(dat_dir, mc_data, bound_file_name, validation=False, custom_K=None):

	""" 
	For when there are (potentially) multiple folds and (potentially) multiple ways of dividing the Ks.
	Assumes that splitting by sex is not a thing
	set custom_K = None when you want to compare length mode (l) with mutation intensity mode (n)
	set validation = None when you want to get data for a segmentation that uses all of the training data
	"""

	chrm_begs = mc_data["chrm_begs"]
	num_chrms = len(chrm_begs)
	float_t = mc_data["float_t"][0]
	mut_pos = mc_data["mut_pos"]
	array = mc_data["array"]
	chrm_lens = mc_data["chrm_lens"]

	T = array.shape[2]
	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	total_chrm_lens = np.sum(chrm_lens)
	
	if custom_K is None:
		num_K_modes = 2
		l_Ks = [ ceil(chrm_lens[i] / SEG_SIZE) for i in range(len(chrm_lens))]
		total_K = sum(l_Ks)
	else:
		num_K_modes = 1
		total_K = custom_K
	k_modes = ["n","l"]
	Ks = np.zeros([num_K_modes,num_chrms], dtype=np.int)
	n_Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]
	Ks[0] = np.array(n_Ks, dtype=np.int)
	if custom_K is None:
		Ks[1] = np.array(l_Ks, dtype=np.int)

	if not validation:
		num_folds = 0
	else:
		num_folds = array.shape[0]

	mut_ints = np.zeros([num_K_modes,num_folds+1,num_chrms,np.max(Ks),T], dtype=float_t)
	mut_bounds = np.zeros([num_K_modes,num_folds+1,num_chrms,np.max(Ks)+1], dtype=np.uint32)
	bp_bounds = np.zeros([num_K_modes,num_folds+1,num_chrms,np.max(Ks)+1], dtype=np.uint32)
	final_scores = np.zeros([num_K_modes,num_folds+1,num_chrms], dtype=float_t)
	
	for m in range(num_K_modes):
		print("m = {}".format(m))
		k_mode_dir = k_modes[m]
		for n in range(num_folds): # this loop does not execute if validation is False
			print("n = {}".format(n))
			fold_dir = "fold_{}".format(n)
			for c in range(num_chrms):
				S_s_file_name = "{}/{}/{}/S_s_chrm_{}.dat".format(dat_dir, k_mode_dir, fold_dir, c+1)
				chrm_mut_ints, chrm_mut_bounds, chrm_bp_bounds = traceback(S_s_file_name, mc_data, Ks[m], c, fold=n)
				mut_ints[m,n,c,0:Ks[m][c]] = chrm_mut_ints
				mut_bounds[m,n,c,0:Ks[m][c]+1] = chrm_mut_bounds
				bp_bounds[m,n,c,0:Ks[m][c]+1] = chrm_bp_bounds
				E_f_file_name = "{}/{}/{}/E_f_chrm_{}.dat".format(dat_dir, k_mode_dir, fold_dir, c+1)
				final_scores[m,n,c] = scores(E_f_file_name, mc_data, Ks[m], c)
		fold_dir = "all"
		for c in range(num_chrms):
			S_s_file_name = "{}/{}/{}/S_s_chrm_{}.dat".format(dat_dir, k_mode_dir, fold_dir, c+1)
			chrm_mut_ints, chrm_mut_bounds, chrm_bp_bounds = traceback(S_s_file_name, mc_data, Ks[m], c, fold=None)
			mut_ints[m,-1,c,0:Ks[m][c]] = chrm_mut_ints
			mut_bounds[m,-1,c,0:Ks[m][c]+1] = chrm_mut_bounds
			bp_bounds[m,-1,c,0:Ks[m][c]+1] = chrm_bp_bounds
			E_f_file_name = "{}/{}/{}/E_f_chrm_{}.dat".format(dat_dir, k_mode_dir, fold_dir, c+1)
			#print(E_f_file_name)
			final_scores[m,-1,c] = scores(E_f_file_name, mc_data, Ks[m], c)

	np.savez(bound_file_name, Ks=Ks, mut_ints=mut_ints, mut_bounds=mut_bounds, bp_bounds=bp_bounds, final_scores=final_scores)

def compute_conditional_mutual_info(array,chrm_begs,ints,Ks):

	# array is [M,T]
	# bounds is [num_chrms,max(Ks)+1]

	num_chrms = ints.shape[0]
	T = array.shape[1]
	float_t = array.dtype
	eps = np.finfo(float_t).eps

	M_c = np.sum(ints,axis=(1,2))
	P_of_C = M_c / np.sum(ints)
	H_of_T_given_C = - np.sum( (np.sum(ints, axis=1) / np.reshape(M_c,[num_chrms,1])) * np.log(np.sum(ints, axis=1) / np.reshape(M_c,[num_chrms,1]) + eps), axis=1 )
	#print(P_of_C)
	#print(H_of_T_given_C)
	print(nats_to_bits(np.sum(H_of_T_given_C * P_of_C)))
	H_of_S_and_T_given_C = - np.sum( (ints/np.reshape(M_c,[num_chrms,1,1]) ) * np.log(ints / np.reshape(M_c,[num_chrms,1,1]) + eps), axis=(1,2) )
	H_of_S_given_C = - np.sum( ( np.sum(ints,axis=2)/np.reshape(M_c,[num_chrms,1]) ) * np.log(np.sum(ints,axis=2)/np.reshape(M_c,[num_chrms,1]) + eps), axis=1 )
	information_C = H_of_T_given_C + H_of_S_given_C - H_of_S_and_T_given_C
	I_of_T_and_B_given_C = np.sum(P_of_C * information_C)

	return I_of_T_and_B_given_C

def compute_conditional_mutual_info_2(array,chrm_begs,ints,Ks,scores):

	num_chrms = ints.shape[0]
	T = array.shape[1]
	float_t = array.dtype
	eps = np.finfo(float_t).eps

	# mut_ints = np.zeros([num_chrms,np.max(Ks),T], dtype=float_t)
	# for c in range(num_chrms):
	# 	for k in range(Ks[c]):
	# 		temp = np.sum(array[ chrm_begs[c] + bounds[c,k] : chrm_begs[c] + bounds[c,k+1] ], axis=0)
	# 		mut_ints[c,k] = temp
	M_c = np.sum(ints,axis=(1,2))
	P_of_C = M_c / np.sum(ints)
	H_of_T_given_C = - np.sum( (np.sum(ints, axis=1) / np.reshape(M_c,[num_chrms,1])) * np.log(np.sum(ints, axis=1) / np.reshape(M_c,[num_chrms,1]) + eps), axis=1 )
	I_of_T_and_B_given_C = np.sum(P_of_C * (scores+H_of_T_given_C))

	return I_of_T_and_B_given_C

def k_fold_validation_eval(mc_data,b_data):

	""" evaluate validation fold mutual info on training bounds """

	array = mc_data["array"]
	float_t = mc_data["float_t"][0]
	chrm_begs = mc_data["chrm_begs"]
	
	mut_bounds = b_data["mut_bounds"]
	mut_ints = b_data["mut_ints"]
	Ks = b_data["Ks"]
	num_K_modes = mut_bounds.shape[0]
	num_folds = mut_bounds.shape[1]-1
	assert( num_folds == array.shape[0] )
	num_chrms = mut_bounds.shape[2]
	assert( num_chrms == chrm_begs.shape[0] )

	mutual_infos = np.zeros([num_K_modes,num_folds], dtype=float_t)

	for m in range(num_K_modes):
		for n in range(num_folds):
			valid_array = array[n]
			train_bounds = mut_bounds[m,n] # num_chrms, Ks[c]+1
			I_of_T_and_B_given_C = compute_conditional_mutual_info(valid_array,chrm_begs,train_bounds,Ks[m])
			mutual_infos[m,n] = I_of_T_and_B_given_C

	return nats_to_bits(mutual_infos)

def total_train_eval(mc_data,b_data):

	array = np.sum(mc_data["array"], axis=0)
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]

	mut_bounds = b_data["mut_bounds"]
	mut_ints = b_data["mut_ints"]
	Ks = b_data["Ks"]
	final_scores = b_data["final_scores"]
	num_K_modes = mut_bounds.shape[0]
	num_chrms = mut_bounds.shape[2]
	
	mutual_infos = np.zeros([num_K_modes,2], dtype=float_t)

	for m in range(num_K_modes):
		ints = mut_ints[m,-1]
		bounds = mut_bounds[m,-1]
		scores = final_scores[m,-1]
		mutual_infos[m,0] = compute_conditional_mutual_info(array,chrm_begs,ints,Ks[m])
		mutual_infos[m,1] = compute_conditional_mutual_info_2(array,chrm_begs,ints,Ks[m],scores)
		# new_mut_ints = compute_conditional_mutual_info(array,chrm_begs,bounds,Ks[m])
		# print((mut_ints[m,-1] == new_mut_ints).all())

	return nats_to_bits(mutual_infos)

def naive_traceback(mc_data,seg_size):

	""" the naive equivalent of traceback """

	array = np.sum(mc_data["array"], axis=0)
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]
	chrm_lens = mc_data["chrm_lens"]
	eps = np.finfo(float_t).eps

	num_chrms = len(chrm_begs)
	Ks = [ceil(chrm_lens[i] / seg_size) for i in range(len(chrm_lens))]
	M = array.shape[0]
	T = array.shape[1]

	mut_ints = np.zeros([num_chrms,max(Ks),T], dtype=float_t)
	mut_bounds = np.zeros([num_chrms,max(Ks)+1], dtype=np.uint32)
	bp_bounds = np.zeros([num_chrms,max(Ks)+1], dtype=np.uint32)

	for c in range(num_chrms):
		for k in range(Ks[c]):
			bp_bounds[c,k] = k*seg_size
		bp_bounds[c,Ks[c]] = chrm_lens[c]

	saddle_mut_count = 0
	for c in range(num_chrms):
		cur_ind = 0
		mut_bounds[c,0] = 0
		chrm_mut_pos = mut_pos[c]
		for k in range(Ks[c]):
			prev_ind = cur_ind
			end_pt = bp_bounds[c,k+1]
			while cur_ind < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_ind][1] < end_pt:
				cur_ind += 1
			if cur_ind < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_ind][0] < end_pt and chrm_mut_pos[cur_ind][1] >= end_pt:
				# mutation is saddling boundaries
				saddle_mut_count += 1
				if end_pt - chrm_mut_pos[cur_ind][0] > chrm_mut_pos[cur_ind][1] - end_pt + 1:
					# mutation overlaps more with current segment, add it to the current mut_ints
					cur_ind += 1
			if prev_ind != cur_ind:
				mut_ints[c,k] = np.sum(array[chrm_begs[c]+prev_ind:chrm_begs[c]+cur_ind], axis=0)
			else: # prev_ind == cur_ind
				# there are no mutations in this segment
				mut_ints[c,k] = np.zeros([T],dtype=float_t)
			mut_bounds[c,k+1] = cur_ind
	#print(saddle_mut_count)
	# thing1 = np.sum(mut_ints, axis=(1,2))
	# thing2 = [np.sum(array[chrm_begs[i]:chrm_begs[i+1]]) for i in range(len(chrm_begs)-1)] + [np.sum(array[chrm_begs[-1]:])]
	# print(np.sum(mut_ints))
	# print(np.sum(array))
	assert( np.sum(mut_ints).astype(np.int) == np.sum(array).astype(np.int) ) 

	return mut_ints, mut_bounds, bp_bounds, Ks

def total_train_eval(mc_data,b_data):

	array = np.sum(mc_data["array"], axis=0)
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]

	mut_bounds = b_data["mut_bounds"]
	mut_ints = b_data["mut_ints"]
	Ks = b_data["Ks"]
	final_scores = b_data["final_scores"]
	num_K_modes = mut_bounds.shape[0]
	num_chrms = mut_bounds.shape[2]
	
	mutual_infos = np.zeros([num_K_modes,2], dtype=float_t)

	for m in range(num_K_modes):
		ints = mut_ints[m,-1]
		bounds = mut_bounds[m,-1]
		scores = final_scores[m,-1]
		mutual_infos[m,0] = compute_conditional_mutual_info(array,chrm_begs,ints,Ks[m])
		mutual_infos[m,1] = compute_conditional_mutual_info_2(array,chrm_begs,ints,Ks[m],scores)
		# new_mut_ints = compute_conditional_mutual_info(array,chrm_begs,bounds,Ks[m])
		# print((mut_ints[m,-1] == new_mut_ints).all())

	return nats_to_bits(mutual_infos)

def opt_vs_naive_information_2(mc_data, b_data):

	""" assumes that b_data is for a segmentation that has a consistent K-value """

	array = np.sum(mc_data["array"],axis=0)
	chrm_begs = mc_data["chrm_begs"]
	float_t = mc_data["float_t"][0]

	# naive segmentation
	mut_ints, mut_bounds, bp_bounds, Ks = naive_traceback(mc_data,SEG_SIZE)
	I_of_T_and_B_given_C = compute_conditional_mutual_info(array,chrm_begs,mut_ints,Ks)
	b_I_of_T_and_B_given_C = nats_to_bits(I_of_T_and_B_given_C)
	print( "NAIVE: I_of_T_and_B_given_C = {0:.6f}".format(b_I_of_T_and_B_given_C) )

	# optimal segmentation	
	Ks = b_data["Ks"]
	mut_ints = b_data["mut_ints"]
	I_of_T_and_B_given_C = np.zeros([2], dtype=float_t)
	I_of_T_and_B_given_C[0] = compute_conditional_mutual_info(array,chrm_begs,mut_ints[0,-1],Ks[0])
	I_of_T_and_B_given_C[1] = compute_conditional_mutual_info(array,chrm_begs,mut_ints[1,-1],Ks[1])
	b_I_of_T_and_B_given_C = nats_to_bits(I_of_T_and_B_given_C)
	print( "OPT: I_of_T_and_B_given_C, n = {0:.6f}, l = {1:.6f}".format(b_I_of_T_and_B_given_C[0],b_I_of_T_and_B_given_C[1]) )

def get_seg(pos, chrm_bounds):
	# the last item in bounds is the length of the chromosome

	seg = -1

	num_segs = chrm_bounds.shape[0]-1

	for i in range(num_segs):
		start = chrm_bounds[i]
		end = chrm_bounds[i+1]
		if pos >= start and pos <= end:
			seg = i
			return seg

	return seg

def update_mc_file(mc_data,b_data,naive_bp_bounds,naive_Ks,mc_file_path):

	"""
	DEPRECATED 
	adds two new columns to mc_file (assuming it's a csv):
	1 - optimal length-based K
	2 - optimal num of mutation-based K
	3 - naive
	"""

	n_bp_bounds = b_data["bp_bounds"][0,-1]
	l_bp_bounds = b_data["bp_bounds"][1,-1]
	n_l_Ks = b_data["Ks"]
	naive_Ks = np.reshape(np.array(naive_Ks), [1,len(naive_Ks)])
	Ks = np.concatenate([n_l_Ks,naive_Ks], axis=0)

	file = open(mc_file_path, 'r')
	lines = []
	# read in header
	header = file.readline()
	lines.append(header)
	line_count = 1
	line = file.readline()
	while line != '':
		line = line.split(",")
		# remove newline from the last entry in the line
		line[-1] = line[-1].strip()
		chrm, pos = line[2], line[3]
		if chrm == "Y" or chrm == "X":
			line = file.readline()
		chrm_ind, pos_ind = int(chrm)-1, int(pos)-1
		n_seg = sum(Ks[0,0:chrm_ind]) + get_seg(pos_ind,n_bp_bounds[chrm_ind])
		l_seg = sum(Ks[1,0:chrm_ind]) + get_seg(pos_ind,l_bp_bounds[chrm_ind])
		naive_seg = sum(Ks[2,0:chrm_ind]) + get_seg(pos_ind,naive_bp_bounds[chrm_ind])
		if n_seg < 0:
			print("Error: opt n seg is incorrect [line {}: chrm={}, pos={}]".format(line_count,chrm,pos))
			quit()
		if l_seg < 0:
			print("Error: opt l seg is incorrect [line {}: chrm={}, pos={}]".format(line_count,chrm,pos))
			quit()
		if naive_seg < 0:
			print("Error: naive seg is incorrect [line {}: chrm={}, pos={}]".format(line_count,chrm,pos))
			quit()
		line = line + [str(n_seg),str(l_seg),str(naive_seg)+"\n"]
		line = ",".join(line)
		lines.append(line)
		line = file.readline()
		line_count += 1
	lines = "".join(lines)
	file.close()

	new_mc_file_path = re.sub(r".txt", r"_p.txt", mc_file_path)
	file = open(new_mc_file_path, 'w')
	file.write(lines)
	file.close()

def update_mc_file_fast(mc_data,b_data,naive_bp_bounds,naive_Ks,mc_file_path):

	""" adds two new columns to mc_file (assuming it's a csv):
	1 - optimal length-based K
	2 - optimal num of mutation-based K
	3 - naive
	"""

	n_l_Ks = b_data["Ks"]
	n_bp_bounds = b_data["bp_bounds"][0,-1]
	l_bp_bounds = b_data["bp_bounds"][1,-1]
	bp_bounds = [n_bp_bounds, l_bp_bounds, naive_bp_bounds]
	num_chrms = n_bp_bounds.shape[0]
	chrm_lens = mc_data["chrm_lens"]
	assert( (naive_Ks == n_l_Ks[1]).all() )
	naive_Ks = np.reshape(np.array(naive_Ks), [1,len(naive_Ks)])
	Ks = np.concatenate([n_l_Ks,naive_Ks], axis=0)

	file = open(mc_file_path, 'r')
	lines = []
	pos_tups = []
	# read in header -- don't append it yeat
	header = file.readline()
	line_count = 1
	line = file.readline()
	while line != '':
		line = line.split(",")
		# remove newline from the last entry in the line
		line[-1] = line[-1].strip()
		chrm, pos = line[2], line[3]
		if chrm == "Y" or chrm == "X":
			line = file.readline()
			line_count += 1
			continue
		chrm_ind, pos_ind = int(chrm)-1, int(pos)-1
		lines.append(line)
		pos_tups.append( (chrm_ind,pos_ind) )
		line = file.readline()
		line_count += 1
	file.close()

	dtype = [("chrm",int), ("pos",int)]
	pos_array = np.array(pos_tups, dtype=dtype)
	sorted_inds = np.argsort(pos_array, order=["chrm", "pos"])
	lines = list(np.array(lines)[sorted_inds])
	
	cur_chrm_ind = 0
	cur_local_segs = np.array([0,0,0])
	cur_global_segs = np.array([0,0,0])
	proc_lines = [header]
	for i in range(len(lines)):
		# note: line is already split by commas here
		line = list(lines[i])
		chrm_ind, pos_ind = int(line[2])-1, int(line[3])-1
		while chrm_ind > cur_chrm_ind:
			cur_local_segs = np.array([0,0,0])
			cur_global_segs[0] += Ks[0,cur_chrm_ind] 
			cur_global_segs[1] += Ks[1,cur_chrm_ind]
			cur_global_segs[2] += Ks[2,cur_chrm_ind]
			cur_chrm_ind += 1
			assert(chrm_ind < num_chrms)
		for j in range(len(cur_local_segs)):
			while pos_ind >= bp_bounds[j][cur_chrm_ind][cur_local_segs[j]+1]:
				cur_local_segs[j] += 1
			if (pos_ind > bp_bounds[j][cur_chrm_ind][Ks[j,cur_chrm_ind]]):
				print("line {}, segmentation {}".format(i,j))
				print(pos_ind)
				print(bp_bounds[j][cur_chrm_ind][-1])
				quit()
			assert(cur_local_segs[j] < Ks[j,cur_chrm_ind])
			line.append(str(cur_global_segs[j] + cur_local_segs[j]))
		line[-1] = line[-1] + "\n"
		line = ",".join(line)
		proc_lines.append(line)
	# make the processed lines one giant string
	proc_lines = "".join(proc_lines)

	new_mc_file_path = re.sub(r".txt", r"_pf.txt", mc_file_path)
	file = open(new_mc_file_path, 'w')
	file.write(proc_lines)
	file.close()

def extract_features(mc_data,b_data,mc_raw_dir):

	_, __, naive_bp_bounds, naive_Ks = naive_traceback(mc_data,SEG_SIZE)

	entry_count = 0
	file_count = 0
	entries = sorted(os.listdir(mc_raw_dir))
	#print(len(entries))
	for entry in entries:
		entry_path = os.path.join(mc_raw_dir,entry)
		if os.path.isfile( entry_path ):
			#update_mc_file(mc_data,b_data,naive_bp_bounds,naive_Ks,entry_path)
			update_mc_file_fast(mc_data,b_data,naive_bp_bounds,naive_Ks,entry_path)
			file_count += 1
		entry_count += 1

	assert(entry_count == len(entries))

	print("Complete! {} entries processed, {} files".format(entry_count,file_count))

if __name__ == "__main__":

	# dat_dir = "./results/july_17"
	# mc_data_file_name = "data/mc_100.npz"
	# mc_data = np.load(mc_data_file_name)
	# total_K = 2897
	# bound_file_name = "results/b_100.npz"
	# b_data = np.load(bound_file_name)
	# csv_file_name = "results/gurnit.csv"

	#convert_dat_to_npz(dat_dir, mc_data, total_K, bound_file_name)

	#classify_segs(mc_data, b_data)

	#print_gurnit(mc_data, b_data, csv_file_name)

	#opt_vs_naive_information(mc_data, b_data)

	#classify_segs_2(mc_data, b_data)

	#convert_dat_to_npz_2(dat_dir,mc_data,bound_file_name,validation=True)

	# fold_mutual_infos = k_fold_validation_eval(mc_data,b_data)
	# total_mutual_infos = total_train_eval(mc_data,b_data)
	# print(fold_mutual_infos)
	# print(np.mean(fold_mutual_infos,axis=1))
	# print(np.std(fold_mutual_infos,axis=1))
	# print(total_mutual_infos)

	print("counts")
	mc_data_file_name = "data/mc_100.npz"
	mc_data = np.load(mc_data_file_name)
	bound_file_name = "results/b_100.npz"
	b_data = np.load(bound_file_name)
	opt_vs_naive_information_2(mc_data,b_data)

	print("freqs")
	mc_data_file_name = "data/mc_100_f.npz"
	mc_data = np.load(mc_data_file_name)
	bound_file_name = "results/b_100_f.npz"
	b_data = np.load(bound_file_name)
	opt_vs_naive_information_2(mc_data,b_data)

	#mc_data = np.load("/home/q/qmorris/youngad2/data/mc_100_f.npz")
	#b_data = np.load("/home/q/qmorris/youngad2/data/b_100_f.npz")
	#mc_raw_dir = "/scratch/q/qmorris/youngad2/new_pcawg3"
	#extract_features(mc_data,b_data,mc_raw_dir)
	#print_gurnit_2(mc_data,b_data,"/scratch/q/qmorris/youngad2/gurnit_f")


# ===========================



# def naive_traceback(mc_data, total_K, chrm):

# 	array = mc_data["array"]
# 	mut_pos = mc_data["mut_pos"]
# 	chrm_begs = mc_data["chrm_begs"]

# 	T = array.shape[1]
# 	bounds = [i*SEG_SIZE for i in range(chrm_lens[chrm] // SEG_SIZE)] + [chrm_lens[chrm]]
# 	num_muts = np.zeros([chrm_lens[chrm] // SEG_SIZE, T], dtype=float_t)

# 	chrm_beg = chrm_begs[chrm]
# 	chrm_mut_pos = mut_pos[chrm]
# 	cur_pos = 0
# 	for j in range(1, (chrm_lens[chrm] // SEG_SIZE) - 1):
# 		beg_pt = (j-1)*SEG_SIZE
# 		end_pt = j*SEG_SIZE
# 		while cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] < end_pt:
# 			num_muts[j-1] += array[ chrm_beg+cur_pos ]
# 			cur_pos += 1
# 		if cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] >= end_pt:
# 			if chrm_mut_pos[cur_pos][1] - end_pt < end_pt - chrm_mut_pos[cur_pos][0]:
# 				# put it in this segment
# 				num_muts[j-1] += array[ chrm_beg+cur_pos ]
# 				cur_pos += 1
# 			else:
# 				# put it in the next segment
# 				chrm_mut_pos[cur_pos][0] = chrm_mut_pos[cur_pos][1]
# 		# if not num_muts[j].any():
# 		# 	# can't have any segments with 0 mutations, set them to random small numbers
# 		# 	num_muts[j] = np.random.uniform(low=10*eps,high=100*eps,size=[T])
# 	if chrm == len(chrm_begs) - 1:
# 		num_muts[-1] = np.sum( array[ chrm_begs[chrm] + cur_pos : ], axis=0 )
# 	else:
# 		num_muts[-1] = np.sum( array[ chrm_begs[chrm] + cur_pos : chrm_begs[chrm+1] ], axis=0)

# 	return num_muts, bounds

# def hier_preproc(seg_c, seg_b):

# 	total_seg_c = np.reshape( np.sum(seg_c, axis=0), [1,seg_c.shape[1]] )
# 	seg_r = seg_c / total_seg_c
# 	row_means = np.reshape(np.mean(seg_r, axis=1), [seg_r.shape[0], 1])
# 	row_stds = np.reshape(np.std(seg_r, axis=1), [seg_r.shape[0], 1])
# 	norm_seg_r = ( seg_r - row_means ) / row_stds
# 	for i in range(norm_seg_r.shape[0]):
# 		if np.isnan(norm_seg_r[i]).any():
# 			assert( np.isnan(norm_seg_r[i]).all() )
# 			norm_seg_r[i, :] = 0.
# 	clipped_b = [ bound // 1000000 for bound in seg_b[:-1] ]
# 	p_norm_seg_r = pd.DataFrame(norm_seg_r, index=clipped_b, columns=typs)
# 	return p_norm_seg_r

# def plot(S_s_file_name, mc_data, total_K, chrm, sex):

# 	array = mc_data["array"]
# 	mut_pos = mc_data["mut_pos"]
# 	chrm_begs = mc_data["chrm_begs"]

# 	mut_totals = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
# 	total_mut_totals = sum(mut_totals)
# 	Ks = [ round((mut_totals[i]/total_mut_totals)*total_K) for i in range(len(mut_totals)) ]
# 	naive_seg_c, naive_seg_b = naive_traceback(mc_data, total_K, chrm)
# 	opt_seg_c, opt_seg_b = traceback(S_s_file_name, mc_data, total_K, chrm)

# 	# important constants/arrays for the current chromosome
# 	M, K, T, chrm_beg = mut_totals[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]
	
# 	sns.set(style="white")

# 	# hierarchical clustering (real data)

# 	opt_norm_seg_r = hier_preproc(opt_seg_c, opt_seg_b)

# 	g = sns.clustermap(opt_norm_seg_r, metric="sqeuclidean", row_cluster=False, col_cluster=True)
# 	g.fig.suptitle('Chromosome {} (Optimal Segmentation)'.format(chrm+1))
# 	g.ax_heatmap.set_xlabel('tumour type')
# 	g.ax_heatmap.set_ylabel('chromosome position (Mbp from the start)')
# 	plt.savefig("figures/new2/{}/chrm_{}_opt_hhmap.png".format(sex,chrm+1))
# 	#plt.show()
# 	plt.clf()


# 	# hierachical clustering (naive)

# 	naive_norm_seg_r = hier_preproc(naive_seg_c, naive_seg_b)

# 	g = sns.clustermap(naive_norm_seg_r, metric="sqeuclidean", row_cluster=False, col_cluster=True)
# 	g.fig.suptitle('Chromosome {} (Naive Segmentation)'.format(chrm+1))
# 	g.ax_heatmap.set_xlabel('tumour type')
# 	g.ax_heatmap.set_ylabel('chromosome position (Mbp from the start)')
# 	plt.savefig("figures/new2/{}/chrm_{}_naive_hhmap.png".format(sex,chrm+1))
# 	#plt.show()
# 	plt.clf()


# 	# mutation histogram

# 	g = sns.distplot(np.sum(opt_seg_c,axis=1), kde=False, bins=10) #, hist_kws={ "range": [0,myround(np.max(num_muts), 10000)] })
# 	g.set_title('Chromosome {} Segment Sizes (mutations)'.format(chrm+1))
# 	g.set_xlabel("mutation count")
# 	plt.savefig("figures/new2/{}/chrm_{}_hist_muts.png".format(sex,chrm+1))
# 	#plt.show()
# 	plt.clf()

# 	# bp histogram

# 	opt_seg_b_sizes = [np.log10(opt_seg_b[i+1] - opt_seg_b[i]) for i in range(len(opt_seg_b[0:-1]))]
# 	ax = sns.distplot(opt_seg_b_sizes, kde=False, bins=50) #, hist_kws={ "range": [0,myround(np.max(num_muts), 10000)] })
# 	ax.set_title('Chromosome {} Segment Sizes (genomic DNA)'.format(chrm+1))
# 	ax.set_xlabel("log10 bp count")
# 	plt.savefig("figures/new2/{}/chrm_{}_hist_bp_log.png".format(sex,chrm+1))
# 	#plt.show()
# 	plt.clf()

# 	# # correlation matrix
# 	# corr = np.corrcoef(seg_counts, rowvar=False)
# 	# print(np.min(corr))
# 	# print(np.max(corr))
# 	# p_corr = pd.DataFrame(corr, index=typs, columns=typs)
# 	# mask = np.zeros_like(p_corr, dtype=np.bool)
# 	# mask[np.triu_indices_from(mask)] = True
# 	# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# 	# sns.heatmap(p_corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# 	# plt.show()

# def compare_everything():

# 	sexes = ["male","female","both"]
# 	sex_abbrevs = ["m", "f", "b"]
# 	total_K = 2865

# 	seg_scores = np.zeros([len(sexes), NUM_CHRMS, 5], dtype=float_t)
# 	total_seg_scores = np.zeros([len(sexes)], dtype=float_t)
# 	total_naive_seg_scores = np.zeros([len(sexes)], dtype=float_t)

# 	for s in range(len(sexes)):
# 		mc_data = np.load("data/new2/mc_100_{}.npz".format(sex_abbrevs[s]))
# 		naive_data = np.load("results/new2/naive_100_{}.npz".format(sex_abbrevs[s]) )
# 		naive_seg_scores = naive_data["seg_scores"]
# 		#print(naive_seg_scores[0])
# 		for i in range(NUM_CHRMS):
# 			E_f_file_name = "results/new2/{}/E_f_chrm_{}.dat".format(sexes[s],i+1)
# 			seg_scores[s,i,0], seg_scores[s,i,1] = scores(E_f_file_name, mc_data, total_K, i)
# 			seg_scores[s,i,2] = np.sum(naive_seg_scores[i]) + seg_scores[s,i,0]
# 			seg_scores[s,i,3] = seg_scores[s,i,1] - seg_scores[s,i,2]
# 			seg_scores[s,i,4] = seg_scores[s,i,3] / seg_scores[s,i,1]
# 		print( "{}:".format(sexes[s]) )
# 		#print( "entropy of T | opt | naive | opt-naive | (opt-naive)/naive")
# 		#print( seg_scores[s] )

# 		array = mc_data["array"]
# 		chrm_begs = list(mc_data["chrm_begs"]) + [array.shape[0]]
# 		total_Ms = np.zeros([NUM_CHRMS], dtype=float_t)
# 		for i in range(NUM_CHRMS):
# 			total_Ms[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ])
# 		total_M = np.sum(total_Ms) # total_M == np.sum(array) (although there is some fp error)
# 		ratio_Ms = total_Ms / total_M

# 		total_seg_scores[s] = np.sum(ratio_Ms * seg_scores[s,:,1])
# 		total_naive_seg_scores[s] = np.sum(ratio_Ms * seg_scores[s,:,2] )
# 		print( "Total seg score = {}".format(total_seg_scores[s]) )
# 		print( "Total naive score = {}".format(total_naive_seg_scores[s]) )
# 		print( "% improvement = {}".format( 100*(total_seg_scores[s]-total_naive_seg_scores[s]) / total_naive_seg_scores[s] ) )

# 	# data_dict = {
# 	# 	"chromosome": 3*[i+1 for i in range(NUM_CHRMS)], 
# 	# 	"sex": NUM_CHRMS*["male"] + NUM_CHRMS*["female"] + NUM_CHRMS*["both"], 
# 	# 	"% improvement": list(100*seg_scores[0,:,-1]) + list(100*seg_scores[1,:,-1]) + list(100*seg_scores[2,:,-1]) 
# 	# }
# 	# p_data = pd.DataFrame(data_dict)
# 	# g = sns.factorplot(data=p_data, kind="bar", x="chromosome", y="% improvement", hue="sex")
# 	# g.fig.suptitle("% Improvement Over Naive Seg for Every Chromosome")
# 	# plt.show()
# 	# plt.clf()

# 		# xs = ["chrm " + str(i+1) for i in range(NUM_CHRMS)]
# 		# ys = pd.DataFrame(seg_scores[s,:,-1])
# 		# ax = sns.barplot(, )

# def plot_everything():

# 	sexes = ["female"] #["male","female","both"]
# 	sex_abbrevs = ["f"] #["m", "f", "b"]
# 	total_K = 2865

# 	for s in range(len(sexes)):
# 		print("Starting {}:".format(sexes[s]))
# 		mc_data = np.load( "data/new2/mc_100_{}.npz".format(sex_abbrevs[s]) )
# 		for i in range(NUM_CHRMS):
# 			print("Plotting Chromosome {}".format(i+1))
# 			S_s_file_name = "results/new2/{}/S_s_chrm_{}.dat".format(sexes[s], i+1)
# 			plot(S_s_file_name, mc_data, total_K, i, sexes[s])


# if __name__ == "__main__":

# 	if not ( len(sys.argv) == 6 or ( len(sys.argv) == 2 and (sys.argv[1] == "ce" or sys.argv[1] == "pe") ) ):
# 		print("Error: Usage")
# 		quit()

# 	mode = sys.argv[1]
# 	if mode == "ce":
# 		compare_everything()
# 		quit()
# 	if mode == "pe":
# 		plot_everything()
# 		quit()
# 	mc_data_file_name = sys.argv[2]
# 	file_name = sys.argv[3] # could be S_s or E_f
# 	total_K = int(sys.argv[4])
# 	chrm = int(sys.argv[5])-1

# 	mc_data = np.load(mc_data_file_name)

# 	if mode == "s":
# 		traceback(file_name, mc_data, total_K, chrm)
# 	elif mode == "e":
# 		scores(file_name, mc_data, total_K, chrm)
# 	elif mode == "p":
# 		plot(file_name, mc_data, total_K, chrm)
# 	else:
# 		print("Error: Usage")
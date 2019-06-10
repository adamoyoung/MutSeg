import chromosome as chrmlib
from chromosome import Segmentation
import numpy as np
import pickle
import struct
import os
import argparse
from scipy.special import logsumexp


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir_path", type=str, default="results", help="path to directory with S_s, E_f, and E_s files") #"/scratch/q/qmorris/youngad2/results")
parser.add_argument("--mc_file_path", type=str, default="mc_data_mp.pkl", help="path to chromosome data .pkl file") #"/home/q/qmorris/youngad2/MutSeg/mc_data_mp.pkl")
parser.add_argument("--naive_seg_size", type=int, default=1000000, help="size of segments in naive segmentation (in bp)")
parser.add_argument("--program_mode", type=str, choices=["seg", "cmi", "tmi", "ann"], default="seg")
parser.add_argument("--ann_dir_path", type=str, default="annotations", help="only useful in \'ann\' mode, path to directory where segment annotation csvs will be stored")


def median(a, b):
	return int(round((a+b)/2))


def nats_to_bits(val):
	return val / np.log(2)


def safelog(val):
	""" perform log operation while ensuring numerical stability """
	return np.log(val + chrmlib.EPS)


def load_seg_results(S_s_file_path, E_f_file_path, chrm, naive_seg_size):
	"""
	S_s_file_path: str, path to S_s file
	chrm: Chromosome, chromosome that corresponds to the S_s file
	naive_seg_size: size of the naive segments in bp, determines the number of segments per chromosome
	"""
	# get chromosome information
	M = chrm.get_unique_pos_count()
	T = chrm.get_num_cancer_types()
	num_segs = chrm.get_num_segs(naive_seg_size)
	mut_array = chrm.get_mut_array()
	mut_pos = chrm.get_mut_pos()

	# read in contents of S_s file
	fsize = os.path.getsize(S_s_file_path)
	assert fsize == 4*M*num_segs
	S_s_file = open(S_s_file_path, 'rb')
	S_s_bytes = S_s_file.read(fsize)
	S_s = np.squeeze(np.array(list(struct.iter_unpack('I',S_s_bytes)), dtype=chrmlib.INT_T), axis=1)

	# read in contents of E_f file
	fsize = os.path.getsize(E_f_file_path)
	assert fsize == 8*num_segs
	E_f_file = open(E_f_file_path, "rb")
	E_f_bytes = E_f_file.read(fsize)
	E_f = np.squeeze(np.array(list(struct.iter_unpack('d',E_f_bytes)), dtype=chrmlib.FLOAT_T), axis=1)

	# get segmentation
	mut_bounds = []
	mut_bounds.insert(0,M)
	k = num_segs-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		mut_bounds.insert(0,col+1)
		k -= 1
	mut_bounds.insert(0,0)

	# find the number of mutations in each segment
	mut_ints = np.zeros([len(mut_bounds)-1, T], dtype=chrmlib.FLOAT_T)
	for i in range(len(mut_bounds)-1):
		cur_mut_ints = np.sum(mut_array[ mut_bounds[i] : mut_bounds[i+1] ], axis=0)
		mut_ints[i] = cur_mut_ints

	# get the acutal bp positions of the segment boundaries
	bp_bounds = []
	bp_bounds.append(0)
	for i in range(len(mut_bounds)-2):
		beg_pt = mut_pos[mut_bounds[i]][1]
		end_pt = mut_pos[mut_bounds[i+1]][0]
		bp_bounds.append(median(beg_pt,end_pt))
	bp_bounds.append(chrm.get_chrm_len())
	bp_bounds = np.array(bp_bounds, dtype=chrmlib.INT_T)

	final_score = E_f[-1]

	seg = Segmentation(num_segs, mut_ints, mut_bounds, bp_bounds, final_score)
	
	return num_segs, seg


def save_seg_results(results_dir_path, mc_file_path, naive_seg_size):
	""" load all the S_s files and save them in the mc_data file """
	assert os.path.isdir(results_dir_path)
	# load chrms data
	with open(mc_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	for c in range(chrmlib.NUM_CHRMS):
		S_s_file_name = "S_s_chrm_{}.dat".format(c)
		S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
		E_f_file_name = "E_f_chrm_{}.dat".format(c)
		E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
		num_segs, segmentation = load_seg_results(S_s_file_path, E_f_file_path, chrms[c], naive_seg_size)
		chrms[c].add_seg(num_segs, segmentation)
	# save chrms data with updated segmentation results
	with open(mc_file_path, "wb") as pkl_file:
		pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


def compute_cmi_from_ints_array(ints_array):
	""" 
	compute I(B;T|C) from ints_array
	the un_ value is the value that hasn't been summed over P(C) yet (for testing purposes only)
	"""
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	total_ints_over_B = np.sum(ints_array, axis=1)
	total_ints_over_T = np.sum(ints_array, axis=2)
	total_ints_over_all = np.sum(ints_array)
	H_of_T_given_C = - np.sum((total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]) * safelog(total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]), axis=1)
	H_of_B_given_C = - np.sum((total_ints_over_T / total_ints_over_B_and_T[..., np.newaxis]) * safelog(total_ints_over_T / total_ints_over_B_and_T[..., np.newaxis] ), axis=1)
	H_of_B_and_T_given_C = - np.sum((ints_array / total_ints_over_B_and_T[..., np.newaxis, np.newaxis]) * safelog(ints_array / total_ints_over_B_and_T[..., np.newaxis, np.newaxis] ), axis=(1,2))
	un_I_of_B_and_T_given_C = H_of_T_given_C + H_of_B_given_C - H_of_B_and_T_given_C
	P_of_C = total_ints_over_B_and_T / total_ints_over_all
	I_of_B_and_T_given_C = np.sum(P_of_C * un_I_of_B_and_T_given_C) 
	return un_I_of_B_and_T_given_C, I_of_B_and_T_given_C

def compute_h_from_ints_array(ints_array):
	"""
	compute H(T|C) from ints_array
	"""
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	total_ints_over_B = np.sum(ints_array, axis=1)
	H_of_T_given_C = - np.sum((total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]) * safelog(total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]), axis=1)
	return H_of_T_given_C


def compute_cmis(chrms, naive_seg_size):
	""" 
	cmi -- conditional mutual information I(B;T|C)
	T is cancer type
	B is segmentation boundary
	C is chromosome
	I is information
	H is entropy
	"""
	T = chrms[0].get_num_cancer_types()
	num_segs = [chrm.get_num_segs(naive_seg_size) for chrm in chrms]
	max_num_segs = max(num_segs)
	
	# optimal cmis

	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	# seg_score is equivalent to H(B|C) - H(B,T|C) with log base e
	seg_scores = np.zeros([chrmlib.NUM_CHRMS], dtype=chrmlib.FLOAT_T)
	for c in range(chrmlib.NUM_CHRMS):
		seg = chrms[c].get_seg(naive_seg_size)
		seg_scores[c] = seg.final_score
		ints_array[c,:num_segs[c],:] = seg.mut_ints
	un_opt_cmi_1, opt_cmi = compute_cmi_from_ints_array(ints_array)
	# use alternate seg score approach for computing information
	un_opt_cmi_2 = seg_scores + compute_h_from_ints_array(ints_array)
	# verrify that they are similar
	assert np.isclose(un_opt_cmi_1, un_opt_cmi_2).all()

	# naive cmis

	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_segs = [chrm.get_naive_seg_arrays(naive_seg_size) for chrm in chrms]
	for c in range(chrmlib.NUM_CHRMS):
		seg = naive_segs[c]
		ints_array[c,:num_segs[c],:] = seg.mut_ints
	_, naive_cmi = compute_cmi_from_ints_array(ints_array)

	return opt_cmi, naive_cmi


def compute_tmi_from_ints_array(ints_array, num_segs):
	# get constants
	num_chrms = ints_array.shape[0]
	max_num_segs = ints_array.shape[1]
	T = ints_array.shape[2]
	total_num_segs = sum(num_segs)
	# compute totals
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	total_ints_over_all = np.sum(ints_array)
	total_ints_over_B = np.sum(ints_array, axis=1)
	total_ints_over_T = np.sum(ints_array, axis=2)
	# compute log probabilities
	log_P_of_C = safelog(total_ints_over_B_and_T / total_ints_over_all)
	log_P_of_T = safelog(np.sum(total_ints_over_B, axis=0) / total_ints_over_all)
	P_of_B_given_C = np.zeros([total_num_segs,num_chrms], dtype=chrmlib.FLOAT_T)
	prev = 0
	for c in range(num_chrms):
		for k in range(num_segs[c]):
			if total_ints_over_B_and_T[c] != 0.:
				P_of_B_given_C[prev+k,c] = total_ints_over_T[c,k] / total_ints_over_B_and_T[c] 
		prev += num_segs[c]
	log_P_of_B_given_C = safelog(P_of_B_given_C)
	del P_of_B_given_C
	P_of_T_given_C_and_B = np.zeros([T,num_chrms,total_num_segs], dtype=chrmlib.FLOAT_T)
	for t in range(T):
		prev = 0
		for c in range(num_chrms):
			for k in range(num_segs[c]):
				if total_ints_over_T[c,k] != 0.:
					P_of_T_given_C_and_B[t,c,prev+k] = ints_array[c,k,t] / total_ints_over_T[c,k]
			prev += num_segs[c]
	log_P_of_T_given_C_and_B = safelog(P_of_T_given_C_and_B)
	del P_of_T_given_C_and_B
	P_of_C_given_B = np.zeros([num_chrms,total_num_segs], dtype=chrmlib.FLOAT_T)
	prev = 0
	for c in range(num_chrms):
		for k in range(num_segs[c]):
			P_of_C_given_B[c,prev+k] = 1.
		prev += num_segs[c]
	log_P_of_C_given_B = safelog(P_of_C_given_B)
	del P_of_C_given_B
	log_P_of_B = logsumexp(np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C, axis=1)
	log_P_of_T_given_B = logsumexp(np.reshape(log_P_of_C_given_B, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B, axis=1)
	log_P_of_T_given_C = logsumexp(np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B, axis=2)
	log_P_of_C_and_T = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_T_given_C ).T
	log_P_of_C_and_B = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C  ).T
	log_P_of_C_and_B_and_T = ( np.reshape(log_P_of_C, [1,num_chrms,1]) + np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B ).transpose([1,0,2])
	log_P_of_B_and_T = np.reshape(log_P_of_B, [1,total_num_segs]) + log_P_of_T_given_B
	# compute entropies and informations with the log probabilities
	_, I_of_B_and_T_given_C = compute_cmi_from_ints_array(ints_array)
	H_of_C = - np.sum( np.exp(log_P_of_C) * log_P_of_C )
	H_of_C_given_T = np.sum( np.exp(log_P_of_C_and_T) * ( np.reshape(log_P_of_T, [1,T]) - log_P_of_C_and_T ) )
	H_of_C_given_B = np.sum( np.exp(log_P_of_C_and_B) * ( np.reshape(log_P_of_B, [1,total_num_segs]) - log_P_of_C_and_B ) )
	H_of_B = - np.sum(np.exp(log_P_of_B) * log_P_of_B)
	H_of_T = - np.sum(np.exp(log_P_of_T) * log_P_of_T)
	H_of_C_and_B_and_T = - np.sum(np.exp(log_P_of_C_and_B_and_T) * log_P_of_C_and_B_and_T)
	H_of_B_and_T = - np.sum(np.exp(log_P_of_B_and_T) * log_P_of_B_and_T)
	H_of_C_given_B_and_T = H_of_C_and_B_and_T - H_of_B_and_T

	I_of_B_and_T = I_of_B_and_T_given_C - H_of_C_given_T - H_of_C_given_B + H_of_C_given_B_and_T + H_of_C

	return I_of_B_and_T


def compute_tmis(chrms, naive_seg_size):
	"""
	tmi -- total mutual information I(B;T)
	I(B;T) = I(B;T|C) - H(C|T) - H(C|B) + H(C|B,T) + H(C)
	does into log space for numerical stability
	"""
	T = chrms[0].get_num_cancer_types()
	num_segs = [chrm.get_num_segs(naive_seg_size) for chrm in chrms]
	max_num_segs = max(num_segs)
	num_chrms = chrmlib.NUM_CHRMS
	total_num_segs = sum(num_segs)

	# optimal tmi

	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	seg_scores = np.zeros([chrmlib.NUM_CHRMS], dtype=chrmlib.FLOAT_T)
	for c in range(num_chrms):
		seg = chrms[c].get_seg(naive_seg_size)
		seg_scores[c] = seg.final_score
		ints_array[c,:num_segs[c],:] = seg.mut_ints
	opt_I_of_B_and_T = compute_tmi_from_ints_array(ints_array, num_segs)

	# naive tmi

	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_segs = [chrm.get_naive_seg_arrays(naive_seg_size) for chrm in chrms]
	for c in range(num_chrms):
		seg = naive_segs[c]
		ints_array[c,:num_segs[c],:] = seg.mut_ints
	naive_I_of_B_and_T = compute_tmi_from_ints_array(ints_array, num_segs)

	return opt_I_of_B_and_T, naive_I_of_B_and_T


if __name__ == "__main__":

	FLAGS = parser.parse_args()

	if FLAGS.program_mode == "seg":
		# interpret segmentation results and save them in an mc_data file of chromosomes
		save_seg_results(FLAGS.results_dir_path, FLAGS.mc_file_path, FLAGS.naive_seg_size)
	elif FLAGS.program_mode == "cmi":
		# compute conditional mutual information for optimal and naive
		with open(FLAGS.mc_file_path, "rb") as pkl_file:
			chrms = pickle.load(pkl_file)
		assert len(chrms) == chrmlib.NUM_CHRMS
		print("loaded chrms")
		optimal_cmi, naive_cmi = compute_cmis(chrms, FLAGS.naive_seg_size)
		print("optimal cmi", nats_to_bits(optimal_cmis))
		print("naive cmi", nats_to_bits(naive_cmis))
		print("difference", nats_to_bits(optimal_cmis-naive_cmis))
	elif FLAGS.program_mode == "tmi":
		# compute total mutual information for optimal and naive
		with open(FLAGS.mc_file_path, "rb") as pkl_file:
			chrms = pickle.load(pkl_file)
		assert len(chrms) == chrmlib.NUM_CHRMS
		print("loaded chrms")
		optimal_tmi, naive_tmi = compute_tmis(chrms, FLAGS.naive_seg_size)
		print("optimal tmi", nats_to_bits(optimal_tmi))
		print("naive tmi", nats_to_bits(naive_tmi))
		print("difference", nats_to_bits(optimal_tmi-naive_tmi))
	elif FLAGS.program_mode == "ann":
		# produce a csv where each mutation is annotated with a segment in the optimal and naive
		# with open(FLAGS.mc_file_path, "rb") as pkl_file:
		# 	chrms = pickle.load(pkl_file)
		# assert len(chrms) == chrmlib.NUM_CHRMS
		raise NotImplementedError
	else:
		raise NotImplementedError

"""
Script for intepreting and saving segmentation results.
This script should be run on boltz.
"""


import chromosome as chrmlib
from chromosome import OptimalSegmentation, NaiveSegmentation, OptimalSigSegmentation, NaiveSigSegmentation
import numpy as np
import pickle
import struct
import os
import argparse
from scipy.special import logsumexp
import multiprocessing as mp
import ctypes
import pandas as pd
from distutils.util import strtobool
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)


def median(a, b):
	return int(round((a+b)/2))


def nats_to_bits(val):
	return val / np.log(2)


def safelog(val):
	""" perform log operation while ensuring numerical stability """
	return np.log(val + chrmlib.EPS)


def load_seg_results(S_s_file_path, E_f_file_path, mut_array, mut_pos, num_segs, group_by, chrm_len, type_to_idx):
	"""
	S_s_file_path: str, path to S_s file
	E_f_file_path: str, path to E_f file
	"""
	M = mut_pos.shape[0]
	T = len(type_to_idx)
	# read in contents of S_s file
	fsize = os.path.getsize(S_s_file_path)
	# assert fsize >= 4*M*num_segs
	S_s_file = open(S_s_file_path, 'rb')
	S_s_bytes = S_s_file.read(fsize)
	S_s = np.squeeze(np.array(list(struct.iter_unpack('I',S_s_bytes)), dtype=chrmlib.INT_T), axis=1)
	# read in contents of E_f file
	fsize = os.path.getsize(E_f_file_path)
	itemsize = 8 if chrmlib.FLOAT_T == np.float64 else 4
	assert fsize >= itemsize*num_segs, (fsize, itemsize, num_segs)
	E_f_file = open(E_f_file_path, "rb")
	E_f_bytes = E_f_file.read(fsize)
	unpack_mode = 'd' if chrmlib.FLOAT_T == np.float64 else 'f'
	E_f = np.squeeze(np.array(list(struct.iter_unpack(unpack_mode,E_f_bytes)), dtype=chrmlib.FLOAT_T), axis=1)
	# get segmentation
	seg_mut_bounds = []
	seg_mut_bounds.insert(0,M)
	k = num_segs-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		seg_mut_bounds.insert(0,col+1)
		k -= 1
	seg_mut_bounds.insert(0,0)
	# find the intensity of mutations in each segment
	seg_mut_ints = np.zeros([2, len(seg_mut_bounds)-1, T], dtype=chrmlib.FLOAT_T)
	for i in range(len(seg_mut_bounds)-1):
		cur_seg_mut_ints = np.sum(mut_array[:, seg_mut_bounds[i] : seg_mut_bounds[i+1]], axis=1)
		seg_mut_ints[:,i] = cur_seg_mut_ints
	# get the acutal bp positions of the segment boundaries
	seg_bp_bounds = []
	seg_bp_bounds.append(0)
	for i in range(len(seg_mut_bounds)-2):
		beg_pt = mut_pos[seg_mut_bounds[i]][1]
		end_pt = mut_pos[seg_mut_bounds[i+1]][0]
		seg_bp_bounds.append(median(beg_pt,end_pt))
	seg_bp_bounds.append(chrm_len)
	seg_bp_bounds = np.array(seg_bp_bounds, dtype=chrmlib.INT_T)
	final_score = E_f[-1]
	seg = OptimalSegmentation(type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, final_score, group_by)
	return seg


def save_seg_results(results_dir_path, mc_dir_path, ana_file_path, naive_seg_size):
	""" load all the S_s files and save them in the mc_data file """
	assert os.path.isdir(results_dir_path), results_dir_path
	# load chrms data
	chrms = chrmlib.load_mc_data(mc_dir_path)
	for c in range(chrmlib.NUM_CHRMS):
		S_s_file_name = "S_s_chrm_{}.dat".format(c)
		S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
		E_f_file_name = "E_f_chrm_{}.dat".format(c)
		E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
		# get chromosome information
		chrm = chrms[c]
		M = chrm.get_unique_pos_count()
		T = chrm.get_num_tumour_types()
		mut_pos = chrm.get_mut_pos()
		group_by = chrm.group_by
		chrm_len = chrm.get_chrm_len()
		type_to_idx = chrm.type_to_idx
		eval_splits = ["all"]
		if chrm.valid_frac > 0.:
			eval_splits.extend(["train", "valid"])
		for eval_split in eval_splits:
			mut_array = chrm.get_mut_array(eval_split)
			for drop_zeros in [False, True]:
				# this implicitly computes the naive segmentation
				num_segs = chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split)
				# load optimal segmentation information
				opt_seg = load_seg_results(S_s_file_path, E_f_file_path, mut_array, mut_pos, num_segs, group_by, chrm_len, type_to_idx)
				chrms[c].add_opt_seg(num_segs, eval_split, opt_seg)
	# save chrms data with updated segmentation results
	for chrm in chrms:
		chrm.delete_non_ana_data()
	with open(ana_file_path, "wb") as pkl_file:
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
	# return un_I_of_B_and_T_given_C, I_of_B_and_T_given_C
	cond_vals = {
		"H_of_B_given_C": np.sum(P_of_C * H_of_B_given_C),
		"H_of_T_given_C": np.sum(P_of_C * H_of_T_given_C),
		"H_of_B_and_T_given_C": np.sum(P_of_C * H_of_B_and_T_given_C),
		"I_of_B_and_T_given_C": I_of_B_and_T_given_C
	}
	return cond_vals

def compute_h_from_ints_array(ints_array):
	"""
	compute H(T|C) from ints_array
	"""
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	total_ints_over_B = np.sum(ints_array, axis=1)
	H_of_T_given_C = - np.sum((total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]) * safelog(total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]), axis=1)
	return H_of_T_given_C


def compute_cmis(ana_file_path, naive_seg_size, drop_zeros, ana_mode, tumour_list, eval_split, print_results=True):
	""" 
	cmi -- conditional mutual information I(B;T|C)
	T is cancer type
	B is segmentation boundary
	C is chromosome
	I is mutual information
	H is entropy
	tumour_list is None only when using the sig data
	"""
	# load chrm data
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	assert len(chrms) == chrmlib.NUM_CHRMS
	# print("loaded chrms")
	# set up constants
	if not tumour_list:
		T = chrms[0].get_num_tumour_types()
	else:
		T = len(tumour_list)
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	max_num_segs = max(num_segs)
	# compute optimal cmi
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	# seg_score is equivalent to H(B|C) - H(B,T|C) with log base e
	seg_scores = np.zeros([chrmlib.NUM_CHRMS], dtype=chrmlib.FLOAT_T)
	for c in range(chrmlib.NUM_CHRMS):
		seg = chrms[c].get_opt_seg(num_segs[c], eval_split)
		seg_scores[c] = seg.final_score
		ints_array[c,:num_segs[c],:] = seg.get_mut_ints(ana_mode, tumour_list)
	# un_opt_cmi_1, optimal_cmi = compute_cmi_from_ints_array(ints_array)
	optimal_cv = compute_cmi_from_ints_array(ints_array)
	perm_cvs = []
	if chrms[0].perm_segmentations:
		# analyze permutation stuff
		perm_segs = [chrms[c].get_perm_segs(num_segs[c],drop_zeros) for c in range(chrmlib.NUM_CHRMS)]
		for p in range(len(perm_segs[0])):
			ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
			for c in range(chrmlib.NUM_CHRMS):
				perm_seg = perm_segs[c][p]
				ints_array[c,:num_segs[c],:] = perm_seg.get_mut_ints(ana_mode, tumour_list)
			perm_cv = compute_cmi_from_ints_array(ints_array)
			perm_cvs.append(perm_cv)
			# print("finished perm", p)
	num_perms = len(perm_cvs)
	# if not drop_zeros:
	# 	# use alternate seg score approach for computing information
	# 	un_opt_cmi_2 = seg_scores + compute_h_from_ints_array(ints_array)
	# 	# verify that they are similar
	# 	assert np.isclose(un_opt_cmi_1, un_opt_cmi_2).all()
	# compute naive cmi
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size, eval_split).get_mut_ints(drop_zeros, ana_mode, tumour_list) for chrm in chrms]
	for c in range(chrmlib.NUM_CHRMS):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	naive_cv = compute_cmi_from_ints_array(ints_array)
	if print_results:
		for cv in optimal_cv:
			print(">>> {}".format(cv))
			print("optimal {} = {}".format(cv,nats_to_bits(optimal_cv[cv])))
			print("naive {} = {}".format(cv,nats_to_bits(naive_cv[cv])))
			print("diff {} = {}".format(cv,nats_to_bits(optimal_cv[cv]-naive_cv[cv])))
			# if perm_cvs:
			# 	print("perm cmis: mean = {}, stddev = {}, stderr = {}".format(nats_to_bits(np.mean(perm_cmis)), nats_to_bits(np.std(perm_cmis)), nats_to_bits(np.std(perm_cmis)/np.sqrt(num_perms))))
			# 	print("opt-perm cmi difference", nats_to_bits(optimal_cmi-np.mean(perm_cmis)))
	return None


def compute_tmi_from_ints_array(ints_array, num_segs):
	# get constants
	num_chrms = ints_array.shape[0]
	max_num_segs = ints_array.shape[1]
	T = ints_array.shape[2]
	total_num_segs = sum(num_segs)
	# compute totals
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	# total_ints_over_C_and_T = np.sum(ints_array, axis=(0,2))
	total_ints_over_all = np.sum(ints_array)
	total_ints_over_B = np.sum(ints_array, axis=1)
	total_ints_over_T = np.sum(ints_array, axis=2)
	# compute log probabilities
	P_of_C = total_ints_over_B_and_T / total_ints_over_all
	P_of_T = np.sum(total_ints_over_B, axis=0) / total_ints_over_all
	log_P_of_C = safelog(P_of_C)
	log_P_of_T = safelog(P_of_T)
	P_of_B_given_C = np.zeros([total_num_segs,num_chrms], dtype=chrmlib.FLOAT_T)
	prev = 0
	for c in range(num_chrms):
		for k in range(num_segs[c]):
			if total_ints_over_B_and_T[c] != 0.:
				P_of_B_given_C[prev+k,c] = total_ints_over_T[c,k] / total_ints_over_B_and_T[c] 
		prev += num_segs[c]
	P_of_B = np.sum(P_of_B_given_C, axis=1)
	log_P_of_B = safelog(P_of_B)
	log_P_of_B_given_C = safelog(P_of_B_given_C)
	# del P_of_B_given_C
	P_of_T_given_C_and_B = np.zeros([T,num_chrms,total_num_segs], dtype=chrmlib.FLOAT_T)
	for t in range(T):
		prev = 0
		for c in range(num_chrms):
			for k in range(num_segs[c]):
				if total_ints_over_T[c,k] != 0.:
					P_of_T_given_C_and_B[t,c,prev+k] = ints_array[c,k,t] / total_ints_over_T[c,k]
			prev += num_segs[c]
	log_P_of_T_given_C_and_B = safelog(P_of_T_given_C_and_B)
	# del P_of_T_given_C_and_B
	P_of_C_given_B = np.zeros([num_chrms,total_num_segs], dtype=chrmlib.FLOAT_T)
	prev = 0
	for c in range(num_chrms):
		for k in range(num_segs[c]):
			P_of_C_given_B[c,prev+k] = 1.
		prev += num_segs[c]
	log_P_of_C_given_B = safelog(P_of_C_given_B)
	# del P_of_C_given_B
	log_P_of_B = logsumexp(np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C, axis=1)
	log_P_of_T_given_B = logsumexp(np.reshape(log_P_of_C_given_B, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B, axis=1)
	log_P_of_T_given_C = logsumexp(np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B, axis=2)
	log_P_of_C_and_T = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_T_given_C ).T
	log_P_of_C_and_B = ( np.reshape(log_P_of_C, [1,num_chrms]) + log_P_of_B_given_C  ).T
	log_P_of_C_and_B_and_T = ( np.reshape(log_P_of_C, [1,num_chrms,1]) + np.reshape(log_P_of_B_given_C.T, [1,num_chrms,total_num_segs]) + log_P_of_T_given_C_and_B ).transpose([1,0,2])
	log_P_of_B_and_T = np.reshape(log_P_of_B, [1,total_num_segs]) + log_P_of_T_given_B
	# compute entropies and informations with the log probabilities
	cond_vals = compute_cmi_from_ints_array(ints_array)
	I_of_B_and_T_given_C = cond_vals["I_of_B_and_T_given_C"]
	H_of_C = - np.sum( P_of_C * log_P_of_C )
	H_of_C_given_T = np.sum( np.exp(log_P_of_C_and_T) * ( np.reshape(log_P_of_T, [1,T]) - log_P_of_C_and_T ) )
	H_of_C_given_B = np.sum( np.exp(log_P_of_C_and_B) * ( np.reshape(log_P_of_B, [1,total_num_segs]) - log_P_of_C_and_B ) )
	H_of_B = - np.sum(P_of_B * log_P_of_B)
	H_of_T = - np.sum(P_of_T * log_P_of_T)
	H_of_C_and_B_and_T = - np.sum(np.exp(log_P_of_C_and_B_and_T) * log_P_of_C_and_B_and_T)
	H_of_B_and_T = - np.sum(np.exp(log_P_of_B_and_T) * log_P_of_B_and_T)
	H_of_C_given_B_and_T = H_of_C_and_B_and_T - H_of_B_and_T

	I_of_B_and_T = I_of_B_and_T_given_C - H_of_C_given_T - H_of_C_given_B + H_of_C_given_B_and_T + H_of_C

	total_vals = {
		"H_of_B": H_of_B,
		"H_of_T": H_of_T,
		"H_of_B_and_T": H_of_B_and_T,
		"I_of_B_and_T": I_of_B_and_T
	}
	return total_vals


def compute_tmis(ana_file_path, naive_seg_size, drop_zeros, ana_mode, tumour_list, eval_split, print_results=True):
	"""
	tmi -- total mutual information I(B;T)
	I(B;T) = I(B;T|C) - H(C|T) - H(C|B) + H(C|B,T) + H(C)
	does into log space for numerical stability
	tumour_list is None only when using the sig data
	"""
	# load chrm data
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	assert len(chrms) == chrmlib.NUM_CHRMS
	# print("loaded chrms")
	# get constants
	if not tumour_list:
		T = chrms[0].get_num_tumour_types()
	else:
		T = len(tumour_list)
	num_chrms = chrmlib.NUM_CHRMS
	# compute optimal tmi
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	max_num_segs = max(num_segs)
	total_num_segs = sum(num_segs)
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	for c in range(num_chrms):
		seg = chrms[c].get_opt_seg(num_segs[c], eval_split)
		ints_array[c,:num_segs[c],:] = seg.get_mut_ints(ana_mode, tumour_list)
	optimal_tv = compute_tmi_from_ints_array(ints_array, num_segs)
	perm_tvs = []
	if chrms[0].perm_segmentations:
		# analyze permutation stuff
		perm_segs = [chrms[c].get_perm_segs(num_segs[c],drop_zeros) for c in range(chrmlib.NUM_CHRMS)]
		for p in range(len(perm_segs[0])):
			ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
			for c in range(chrmlib.NUM_CHRMS):
				perm_seg = perm_segs[c][p]
				ints_array[c,:num_segs[c],:] = perm_seg.get_mut_ints(ana_mode, tumour_list)
			perm_tv = compute_tmi_from_ints_array(ints_array, num_segs)
			perm_tvs.append(perm_tv)
			# print("finished perm", p)
	num_perms = len(perm_tvs)
	# compute naive tmi
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	max_num_segs = max(num_segs)
	total_num_segs = sum(num_segs)
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size, eval_split).get_mut_ints(drop_zeros,ana_mode,tumour_list) for chrm in chrms]
	for c in range(num_chrms):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	naive_tv = compute_tmi_from_ints_array(ints_array, num_segs)
	if print_results:
		for tv in optimal_tv:
			print(">>> {}".format(tv))
			print("optimal {} = {}".format(tv,nats_to_bits(optimal_tv[tv])))
			print("naive {} = {}".format(tv,nats_to_bits(naive_tv[tv])))
			print("diff {} = {}".format(tv,nats_to_bits(optimal_tv[tv]-naive_tv[tv])))
		# if perm_tmis:
		# 	print("perm tmis: mean = {}, stddev = {}, stderr = {}".format(nats_to_bits(np.mean(perm_tmis)), nats_to_bits(np.std(perm_tmis)), nats_to_bits(np.std(perm_tmis)/np.sqrt(num_perms))))
		# 	print("opt-perm tmi difference", nats_to_bits(optimal_tmi-np.mean(perm_tmis)))
	return None


def plot_opt_sizes(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split):
	""" plots the sizes of the segments in the optimal segmentation (in bp) """

	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	opt_segs = [chrms[c].get_opt_seg(num_segs[c], eval_split) for c in range(len(chrms))]
	seg_sizes = []
	for c in range(len(chrms)):
		# seg_sizes = np.zeros(num_segs[c], dtype=chrmlib.INT_T)
		seg_bp_bounds = opt_segs[c].get_bp_bounds()
		for s in range(num_segs[c]):
			# seg_sizes[s] = seg_bp_bounds[s+1] - seg_bp_bounds[s]
			seg_sizes.append(seg_bp_bounds[s+1] - seg_bp_bounds[s])
	seg_sizes = np.array(seg_sizes, dtype=np.float)
	min_seg_size = int(np.min(seg_sizes))
	max_seg_size = int(np.max(seg_sizes))
	mean_seg_size = int(np.mean(seg_sizes))
	median_seg_size = int(np.median(seg_sizes))
	print(">>> plot genomic length")
	print("min = {}, max = {}, mean = {}, median = {}".format(min_seg_size, max_seg_size, mean_seg_size, median_seg_size))
	ax = sns.distplot(
		seg_sizes / 1000000,
		kde=False,
		norm_hist=False,
		# bins=500,
		bins=[i / 1000000 for i in range(0, 5000000, 100000)]
	)
	plt_name = "opt_seg_sizes"
	if drop_zeros:
		plt_name += "_nz"
	else:
		plt_name += "_z"
	plt_name += "_{}".format(eval_split)
	ax.set(
		xlabel="genomic length of segment (Mbp)",
		ylabel="counts",
		xlim=[-0.1,5],
		# title=plt_name
	)
	ax.text(0.75, 0.85, f"total = {sum(num_segs)}", fontsize=10, transform=ax.transAxes)
	plt_path = os.path.join(plot_dir_path,plt_name)
	plt.savefig(plt_path)
	plt.clf()


def plot_opt_naive_muts(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split):
	""" plot the number of distinct mutation positions in each segment """

	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	seg_mut_counts = np.zeros([2,sum(num_segs)], dtype=chrmlib.INT_T)
	# get opt mutation counts - can be grouped
	opt_segs = [chrms[c].get_opt_seg(num_segs[c], eval_split) for c in range(len(chrms))]
	cur_seg_idx = 0
	for c in range(len(chrms)):
		seg_mut_bounds = opt_segs[c].get_mut_bounds()
		assert len(seg_mut_bounds) == num_segs[c]+1
		for s in range(num_segs[c]):
			start_idx, end_idx = seg_mut_bounds[s], seg_mut_bounds[s+1]
			seg_mut_counts[0][cur_seg_idx+s] = np.sum(chrms[c].num_mut_pos_g[start_idx:end_idx])
		cur_seg_idx += num_segs[c]
	assert cur_seg_idx == seg_mut_counts.shape[1]
	# group_by = opt_segs[0].group_by
	# if group_by:
	# 	assert np.all([opt_seg.group_by == group_by for opt_seg in opt_segs])
	# 	seg_mut_counts[0] = group_by*seg_mut_counts[0]
	# get naive mutation counts - never grouped
	naive_segs = [chrm.get_naive_seg(naive_seg_size, eval_split) for chrm in chrms]
	cur_seg_idx = 0
	for c in range(len(chrms)):
		seg_mut_bounds = naive_segs[c].get_mut_bounds(drop_zeros)
		assert len(seg_mut_bounds) == num_segs[c]+1
		for s in range(num_segs[c]):
			start_idx, end_idx = seg_mut_bounds[s], seg_mut_bounds[s+1]
			seg_mut_counts[1][cur_seg_idx+s] = end_idx - start_idx
		cur_seg_idx += num_segs[c]
	assert cur_seg_idx == seg_mut_counts.shape[1]
	# plot the results
	print(">>> plot number of muts")
	assert np.sum(seg_mut_counts[0]) == np.sum(seg_mut_counts[1]), np.sum(seg_mut_counts,axis=1)
	seg_types = ["opt", "naive"]
	for i in range(len(seg_types)):
		print(seg_types[i])
		min_seg_mut_counts = int(np.min(seg_mut_counts[i]))
		max_seg_mut_counts = int(np.max(seg_mut_counts[i]))
		mean_seg_mut_counts = int(np.mean(seg_mut_counts[i]))
		median_seg_mut_counts = int(np.median(seg_mut_counts[i]))
		print("min = {}, max = {}, mean = {}, median = {}".format(min_seg_mut_counts, max_seg_mut_counts, mean_seg_mut_counts, median_seg_mut_counts))
		ax = sns.distplot(
			seg_mut_counts[i],
			kde=False,
			norm_hist=False,
			# bins=500,
			bins=[i for i in range(0,81000,1000)],
			label=seg_types[i]
		)
	plt_name = "opt_vs_naive_seg_mut_counts"
	if drop_zeros:
		plt_name += "_nz"
	else:
		plt_name += "_z"
	plt_name += "_{}".format(eval_split)
	ax.set(
		xlabel="number of mutations in segment",
		ylabel="counts",
		# xlim=[-1000,81000],
		xlim=[-1000,81000],
		ylim=[0,300],
		# title=plt_name
	)
	ax.legend()
	plt_path = os.path.join(plot_dir_path,plt_name)
	plt.savefig(plt_path)
	plt.clf()

	# # plot the 
	# mut_thresh = 1000
	# opt_thresh_segs = np.nonzero(seg_mut_counts[0] <= mut_thresh)[0]
	# opt_thresh_mut_counts = seg_mut_counts[0][opt_thresh_segs]
	# ax = sns.distplot(
	# 	opt_thresh_mut_counts,
	# 	kde=False,
	# 	norm_hist=False,
	# 	bins=10
	# )
	# plt_name = "opt_{}_mut_seg_mut_counts_{}MB".format(mut_thresh, naive_seg_size // 1000000)
	# if drop_zeros:
	# 	plt_name += "_nz"
	# ax.set(
	# 	xlabel="number of distinct mutations in segment",
	# 	ylabel="counts",
	# 	title=plt_name)
	# plt_path = os.path.join(plot_dir_path,plt_name)
	# plt.savefig(plt_path)
	# plt.clf()

	# # plot the real sizes of the minimum optimal segments
	# mut_threshes = [100, 1000]
	# opt_seg_sizes = []
	# for c in range(len(chrms)):
	# 	# seg_sizes = np.zeros(num_segs[c], dtype=chrmlib.INT_T)
	# 	seg_bp_bounds = opt_segs[c].get_bp_bounds()
	# 	for s in range(num_segs[c]):
	# 		# seg_sizes[s] = seg_bp_bounds[s+1] - seg_bp_bounds[s]
	# 		opt_seg_sizes.append(seg_bp_bounds[s+1] - seg_bp_bounds[s])
	# opt_seg_sizes = np.array(opt_seg_sizes, dtype=np.float)
	# for m in range(len(mut_threshes)):
	# 	mut_thresh = mut_threshes[m]
	# 	opt_thresh_segs = np.nonzero(seg_mut_counts[0] <= mut_thresh)[0]
	# 	print(f"mut_thresh = {mut_thresh}, num opt_min_segs = {len(opt_thresh_segs)}")
	# 	opt_thresh_sizes = opt_seg_sizes[opt_thresh_segs]
	# 	print(f"min = {np.min(opt_thresh_sizes)}, max = {np.max(opt_thresh_sizes)}, mean = {np.mean(opt_thresh_sizes)}, median = {np.median(opt_thresh_sizes)}")
	# 	plt_name = "opt_{}_mut_seg_sizes_{}MB".format(mut_thresh, naive_seg_size // 1000000)
	# 	if drop_zeros:
	# 		plt_name += "_nz"
	# 	ax = sns.distplot(
	# 		opt_thresh_sizes / 1000000,
	# 		kde=False,
	# 		norm_hist=False,
	# 		bins=[i / 1000000 for i in range(0, 5000000, 100000)]
	# 	)
	# 	ax.set(
	# 		xlabel="segment size (Mbp)",
	# 		ylabel="counts",
	# 		title=plt_name
	# 	)
	# 	ax.text(0.75, 0.85, f"total = {len(opt_thresh_segs)}", fontsize=10, transform=ax.transAxes)
	# 	plt_path = os.path.join(plot_dir_path,plt_name)
	# 	plt.savefig(plt_path)
	# 	plt.clf()


def make_plots(ana_file_path, plot_dir_path, naive_seg_size, drop_zeros, eval_split):

	assert os.path.isfile(ana_file_path)
	os.makedirs(plot_dir_path, exist_ok=True)
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	plot_opt_sizes(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split)
	plot_opt_naive_muts(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split)


def get_opt_seg_from_bp_bounds(mut_ints, mut_pos, num_segs, bp_bounds, type_to_idx):
	""" all of these arrays are for one chromosome ONLY """

	# assign each mutation to a segment
	num_tumour_types = mut_ints.shape[2]
	seg_mut_ints = np.zeros([2,num_segs,num_tumour_types], dtype=chrmlib.FLOAT_T)
	seg_mut_bounds = np.zeros([num_segs+1], dtype=chrmlib.INT_T)
	cur_seg = 0
	for i in range(mut_pos.shape[0]):
		while mut_pos[i] >= bp_bounds[cur_seg+1]:
			cur_seg += 1
			assert cur_seg < num_segs
		seg_mut_ints[:, cur_seg] += mut_ints[:,i]
		seg_mut_bounds[cur_seg+1] = i
	seg = OptimalSegmentation(type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, bp_bounds, None, None)
	return seg


def save_alt_seg_results(results_dir_path, mc_dir_path, mc_alt_dir_path, ana_alt_file_path, naive_seg_size, num_perms):

	raise NotImplementedError
	# print(results_dir_path)
	# print(mc_dir_path)
	# print(mc_alt_dir_path)
	# print(ana_alt_file_path)
	# assert os.path.isdir(mc_dir_path), mc_dir_path
	# assert os.path.isdir(mc_alt_dir_path), mc_alt_dir_path
	# # assert os.path.isfile(ana_alt_file_path)
	# chrms = chrmlib.load_mc_data(mc_dir_path)
	# alt_chrms = chrmlib.load_mc_data(mc_alt_dir_path)
	# for c in range(chrmlib.NUM_CHRMS):
	# 	S_s_file_name = "S_s_chrm_{}.dat".format(c)
	# 	S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
	# 	E_f_file_name = "E_f_chrm_{}.dat".format(c)
	# 	E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
	# 	chrm = chrms[c]
	# 	alt_chrm = alt_chrms[c]
	# 	for drop_zeros in [False, True]:
	# 		num_segs = alt_chrm.get_num_segs(naive_seg_size, drop_zeros)
	# 		orig_opt_seg = load_seg_results(S_s_file_path, E_f_file_path, chrm.get_mut_array("all"), chrm.get_mut_pos(), num_segs, chrm.group_by, chrm.get_chrm_len(), chrm.type_to_idx)
	# 		orig_bp_bounds = orig_opt_seg.get_bp_bounds()
	# 		alt_mut_array = alt_chrm.get_mut_array("all")
	# 		alt_mut_pos = alt_chrm.get_mut_pos()
	# 		alt_opt_seg = get_opt_seg_from_bp_bounds(alt_mut_array, alt_mut_pos, num_segs, orig_bp_bounds, alt_chrm.type_to_idx)
	# 		alt_chrm.add_opt_seg(num_segs, alt_opt_seg)
	# 		# exits here if num_perms == 0
	# 		perm_bp_bounds = permute_bp_bounds(orig_bp_bounds, num_perms)
	# 		for cur_bp_bounds in perm_bp_bounds:
	# 			cur_perm_seg = get_opt_seg_from_bp_bounds(alt_mut_array, alt_mut_pos, num_segs, cur_bp_bounds, alt_chrm.type_to_idx)
	# 			alt_chrm.add_perm_seg(num_segs, drop_zeros, cur_perm_seg)
	# 	print("finished chrm {}".format(c))
	# for alt_chrm in alt_chrms:
	# 	alt_chrm.delete_non_ana_data() # should actually do nothing in this case
	# with open(ana_alt_file_path, "wb") as pkl_file:
	# 	pickle.dump(alt_chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--run_dir_path", type=str, default="runs_both_red/90/co", help="path to directory that holds: ana file (after seg), results dir, plots dir")
	parser.add_argument("--results_dir_name", type=str, default="results", help="name of directory with S_s, E_f, and E_s files") #"/scratch/q/qmorris/youngad2/results")
	parser.add_argument("--mc_dir_path", type=str, default="mc_data_both", help="path to chromosome data .pkl file, read-only") #"/home/q/qmorris/youngad2/MutSeg/mc_data_mp.pkl")
	parser.add_argument("--ana_file_name", type=str, default="ana_data.pkl", help="path to analyzsis data .pkl file")
	parser.add_argument("--naive_seg_size", type=int, default=1000000, help="size of segments in naive segmentation (in bp)")
	parser.add_argument("--program_mode", type=str, choices=["seg", "cmi", "tmi", "ann", "plt", "seg", "cmi"], default="seg")
	parser.add_argument("--data_type", type=str, choices=["normal", "alt", "sig"], default="normal")
	parser.add_argument("--csv_dir_path", type=str, default="for_adamo", help="only useful in \'ann\' mode, path to directory where csvs that need to be annotated will be")
	parser.add_argument("--num_procs", type=int, default=mp.cpu_count(), help="only useful in \'ann\' mode, number of process to fork")
	parser.add_argument("--drop_zeros", type=lambda x:bool(strtobool(x)), default=True)
	parser.add_argument("--plot_dir_name", type=str, default="plots", help="only useful in \'plt\' mode")
	parser.add_argument("--ana_mode", type=str, choices=["sample_freqs", "counts", "tumour_freqs"], default="counts")
	parser.add_argument("--tumour_set", type=str, choices=["all", "reduced"], default="reduced", help="set of tumour types to use")
	# parser.add_argument("--df_sig_file_path", type=str, default="df_sig.pkl")
	parser.add_argument("--train_split", type=str, choices=["all", "train"], default="all")
	parser.add_argument("--eval_split", type=str, choices=["all", "train", "valid"], default="all")
	# parser.add_argument("--num_perms", type=int, default=0)
	FLAGS = parser.parse_args()
	print(FLAGS)
	np.set_printoptions(threshold=1000)

	assert os.path.isdir(FLAGS.run_dir_path), FLAGS.run_dir_path
	results_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.train_split,FLAGS.results_dir_name)
	mc_dir_path = FLAGS.mc_dir_path
	ana_file_path = os.path.join(FLAGS.run_dir_path,FLAGS.ana_file_name)
	plot_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.plot_dir_name)
	if FLAGS.data_type == "alt": # interpet the alt data for the mutations
		mc_alt_dir_path = mc_dir_path + "_alt"
		ana_file_path = ana_file_path.rstrip(".pkl") + "_alt.pkl"
		plot_dir_path += "_alt"
	if FLAGS.tumour_set == "all":
		tumour_list = sorted(chrmlib.ALL_SET)
	else: # FLAGS.tumour_set == "reduced"
		tumour_list = sorted(chrmlib.REDUCED_SET)
	if FLAGS.train_split != "all":
		assert FLAGS.data_type == "normal"
	ana_file_path = ana_file_path.rstrip(".pkl") + "_{}.pkl".format(FLAGS.train_split)

	if FLAGS.program_mode == "seg":
		# interpret segmentation results and save them in a file of chromosomes
		# does not care about drop_zeros since it automatically computes results for both kinds
		assert FLAGS.data_type == "normal"
		save_seg_results(results_dir_path, mc_dir_path, ana_file_path, FLAGS.naive_seg_size)
	elif FLAGS.program_mode == "cmi":
		# compute conditional mutual information for optimal and naive
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("train split == {}, eval_split == {}".format(FLAGS.train_split, FLAGS.eval_split))
		if FLAGS.data_type == "sig":
			raise NotImplementedError
		else:
			compute_cmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list, FLAGS.eval_split)
	elif FLAGS.program_mode == "tmi":
		# compute total mutual information for optimal and naive
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("train split == {}, eval_split == {}".format(FLAGS.train_split, FLAGS.eval_split))
		if FLAGS.data_type == "sig":
			raise NotImplementedError
		else:
			compute_tmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list, FLAGS.eval_split)
	elif FLAGS.program_mode == "ann":
		# modify input csvs such that each mutation is annotated with an optimal segment
		raise NotImplementedError
	elif FLAGS.program_mode == "plt":
		make_plots(ana_file_path, plot_dir_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.eval_split)
	else:
		raise NotImplementedError

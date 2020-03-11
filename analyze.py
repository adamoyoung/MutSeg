"""
Script for intepreting, saving, and plotting segmentation results.
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
# plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def median(a, b):
	return int(round((a+b)/2))


def nats_to_bits(val):
	return val / np.log(2)


def safelog(val):
	""" perform log operation while ensuring numerical stability """
	return np.log(val + chrmlib.EPS)


def safelog2(val):
	""" perform log_2 operation while ensuring numerical stability """
	return np.log2(val + chrmlib.EPS)


def safedivide(x,y):
	""" perform division while ensuring numerical stability """
	return x / (y + chrmlib.EPS)


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
	for i in range(1,len(seg_mut_bounds)-1):
		beg_pt = mut_pos[seg_mut_bounds[i]-1][1]
		end_pt = mut_pos[seg_mut_bounds[i]][0]
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

	"""
	ints_total = np.sum(ints_array)
	P_of_C_and_B_and_T = ints / ints_total
	P_of_C = np.sum(P_of_C_and_B_and_T, axis=(1,2))
	P_of_B_and_T_given_C = np.sum(P_of_C_and_B_and_T, axis=0) / P_of_C[..., np.newaxis, np.newaxis] 
	P_of_B_given_C = np.sum(P_of_C_and_B_and_T, axis=(0,1)) / P_of_C[..., np.newaxis]
	P_of_T_given_C = np.sum(P_of_C_and_B_and_T, axis=(0,2)) / P_of_C[..., np.newaxis]
	H_of_B_given_C = np.sum(P_of_C * ( -np.sum(P_of_B_given_C * safelog(P_of_B_given_C),axis=1) ), axis=0)
	H_of_T_given_C = np.sum(P_of_C * ( -np.sum(P_of_T_given_C * safelog(P_of_T_given_C),axis=1) ), axis=0)
	H_of_B_and_T_given_C = np.sum(P_of_C * ( -np.sum(P_of_B_and_T_given_C * safelog(P_of_B_and_T_given_C),axis=(1,2)) ), axis=0)
	I_of_B_and_T_given_C = H_of_T_given_C + H_of_B_given_C - H_of_B_and_T_given_C
	cond_vals = {
		"H_of_B_given_C": H_of_B_given_C,
		"H_of_T_given_C": H_of_T_given_C,
		"H_of_B_and_T_given_C": H_of_B_and_T_given_C,
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


def compute_cmis(ana_file_path, naive_seg_size, drop_zeros, ana_mode, tumour_list, eval_split):
	""" 
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
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size, eval_split).get_mut_ints(drop_zeros, ana_mode, tumour_list) for chrm in chrms]
	for c in range(chrmlib.NUM_CHRMS):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	naive_cv = compute_cmi_from_ints_array(ints_array)
	print(">> naive")
	print(naive_cv)
	print(">> optimal")
	print(optimal_cv)
	return None


def compute_tmi_from_ints_array(ints_array, num_segs):

	# get constants
	num_chrms = ints_array.shape[0]
	max_num_segs = ints_array.shape[1]
	T = ints_array.shape[2]
	total_num_segs = sum(num_segs)
	# get arrays
	ints_total = np.sum(ints_array)
	n = int(ints_total)
	p = total_num_segs*T
	print(f"n = {n}, p = {p}")
	ints_B_and_T = np.zeros([total_num_segs,T],dtype=chrmlib.FLOAT_T)
	assert num_chrms == len(num_segs)
	cur_num_segs = 0
	for c in range(num_chrms):
		ints_B_and_T[cur_num_segs:cur_num_segs+num_segs[c]] = ints_array[c][:num_segs[c]]
		cur_num_segs += num_segs[c]
	assert cur_num_segs == total_num_segs
	mle_P_of_B_and_T = ints_B_and_T / ints_total
	mle_P_of_T = np.sum(mle_P_of_B_and_T, axis=0) 
	mle_P_of_B = np.sum(mle_P_of_B_and_T, axis=1)
	mle_H_of_T = -np.sum(mle_P_of_T * safelog(mle_P_of_T), axis=0)
	mle_H_of_B = -np.sum(mle_P_of_B * safelog(mle_P_of_B), axis=0)
	mle_H_of_B_and_T = -np.sum(mle_P_of_B_and_T * safelog(mle_P_of_B_and_T), axis=(0,1))
	mle_I_of_B_and_T = mle_H_of_B + mle_H_of_T - mle_H_of_B_and_T 
	mle_total_vals = {
		"H_of_B": mle_H_of_B,
		"H_of_T": mle_H_of_T,
		"H_of_B_and_T": mle_H_of_B_and_T,
		"I_of_B_and_T": mle_I_of_B_and_T
	}
	# compute JS information estimates
	t_B_and_T = 1. / float(p)
	lambda_B_and_T = (1. - np.sum(mle_P_of_B_and_T**2)) / ((float(n)-1)*np.sum((t_B_and_T-mle_P_of_B_and_T)**2))
	js_P_of_B_and_T = lambda_B_and_T*t_B_and_T + (1-lambda_B_and_T)*mle_P_of_B_and_T
	js_P_of_T = np.sum(js_P_of_B_and_T, axis=0) 
	js_P_of_B = np.sum(js_P_of_B_and_T, axis=1)
	js_H_of_T = -np.sum(js_P_of_T * safelog(js_P_of_T), axis=0)
	js_H_of_B = -np.sum(js_P_of_B * safelog(js_P_of_B), axis=0)
	js_H_of_B_and_T = -np.sum(js_P_of_B_and_T * safelog(js_P_of_B_and_T), axis=(0,1))
	js_I_of_B_and_T = js_H_of_B + js_H_of_T - js_H_of_B_and_T 
	js_total_vals = {
		"H_of_B": js_H_of_B,
		"H_of_T": js_H_of_T,
		"H_of_B_and_T": js_H_of_B_and_T,
		"I_of_B_and_T": js_I_of_B_and_T
	}
	return mle_total_vals, js_total_vals


def compute_tmis(ana_file_path, naive_seg_size, drop_zeros, ana_mode, tumour_list, eval_split):
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
	# compute naive tmi
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros, eval_split) for chrm in chrms]
	max_num_segs = max(num_segs)
	total_num_segs = sum(num_segs)
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size, eval_split).get_mut_ints(drop_zeros,ana_mode,tumour_list) for chrm in chrms]
	for c in range(num_chrms):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	naive_tv = compute_tmi_from_ints_array(ints_array, num_segs)
	print(">> naive")
	print("> mle")
	print(naive_tv[0])
	print("> js")
	print(naive_tv[1])
	print(">> optimal")
	print("> mle")
	print(optimal_tv[0])
	print("> js")
	print(optimal_tv[1])
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
		bins=[i / 1000000 for i in range(0, 5000000, 100000)],
		label="opt"
	)
	plt_name = "opt_seg_sizes"
	if drop_zeros:
		plt_name += "_nz"
	else:
		plt_name += "_z"
	plt_name += "_{}".format(eval_split)
	ax.set(
		xlabel="genomic length of segment (Mb)",
		ylabel="counts",
		xlim=[-0.1,5],
		# title=plt_name
	)
	ax.legend()
	# ax.text(0.75, 0.85, f"total = {sum(num_segs)}", fontsize=10, transform=ax.transAxes)
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


def plot_chrms(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split, ana_mode, tumour_list):

	""" plots naive and optimal segmentations (note zeros are NOT dropped, to keep the naive segments looking nice)"""

	# set plotting settings
	# requires latex to be installed
	# sudo apt install texlive-latex-base, texlive-latex-extra, dvipng
	# warning: > 1GB install
	font_dict = {
		'family':'sans-serif',
		'sans-serif':['Computer Modern'],
		'size': 28 #32
	}
	mpl.rc('font',**font_dict)
	mpl.rc('text',usetex=True)

	for c, chrm in enumerate(chrms):
		
		print("plotting chrm {}".format(c))
		chrm_len = chrm.get_chrm_len()
		chrm_num = chrm.get_chrm_num()
		assert chrm_num == c+1
		
		# prepare naive
		naive_seg = chrm.get_naive_seg(naive_seg_size,eval_split)
		naive_bp_bounds = naive_seg.get_bp_bounds(False)
		naive_ints_t = naive_seg.get_mut_ints(False,ana_mode,tumour_list)
		naive_bp_sizes = np.array([naive_bp_bounds[i+1] - naive_bp_bounds[i] for i in range(len(naive_bp_bounds)-1)])
		naive_ints = np.sum(naive_ints_t, axis=1)
		naive_probs_t = safedivide(naive_ints_t, naive_ints[...,np.newaxis])
		naive_ents = -np.sum(naive_probs_t * safelog2(naive_probs_t), axis=1)
		assert np.sum(naive_bp_sizes) == chrm_len
		
		# prepare opt
		opt_num_segs = chrm.get_num_segs(naive_seg_size, False, eval_split)
		opt_seg = chrm.get_opt_seg(opt_num_segs, eval_split)
		opt_bp_bounds = opt_seg.get_bp_bounds()
		opt_ints_t = opt_seg.get_mut_ints(ana_mode,tumour_list)
		opt_bp_sizes = np.array([opt_bp_bounds[i+1] - opt_bp_bounds[i] for i in range(len(opt_bp_bounds)-1)])
		opt_ints = np.sum(opt_ints_t, axis=1)
		opt_probs_t = safedivide(opt_ints_t, opt_ints[...,np.newaxis])
		opt_ents = -np.sum(opt_probs_t * safelog2(opt_probs_t), axis=1)
		assert np.sum(opt_bp_sizes) == chrm_len
		
		# put them together
		both_bp_sizes = np.concatenate([naive_bp_sizes[...,np.newaxis],opt_bp_sizes[...,np.newaxis]],axis=1)
		both_ints = np.concatenate([naive_ints[...,np.newaxis],opt_ints[...,np.newaxis]],axis=1)
		both_ents = np.concatenate([naive_ents[...,np.newaxis],opt_ents[...,np.newaxis]],axis=1)
		# assert np.max(both_ents) <= 3., np.max(both_ents)
		# assert np.min(both_ents) >= 1., np.min(both_ents)

		# set up ints cmap
		max_int = np.ceil(np.max(both_ints) / 1000.0)*1000.0
		min_int = np.ceil(np.min(both_ints) / 1000.0)*1000.0
		print(min_int, max_int)
		int_cmap = plt.get_cmap("Blues",1000)
		int_norm = mpl.colors.Normalize(vmin=min_int,vmax=max_int)
		int_sm = plt.cm.ScalarMappable(cmap=int_cmap, norm=int_norm)
		both_ints_colors = int_sm.to_rgba(both_ints)
		
		# set up ents cmap
		max_ent = np.max(both_ents)
		min_ent = np.abs(np.min(both_ents)) # absolute value is to prevent -0.
		print(min_ent, max_ent)
		ent_cmap = plt.get_cmap("Reds",1000)
		ent_norm = mpl.colors.Normalize(vmin=min_ent,vmax=max_ent)
		ent_sm = plt.cm.ScalarMappable(cmap=ent_cmap, norm=ent_norm)
		both_ents_colors = ent_sm.to_rgba(both_ents)
		
		# get the plot file path
		assert os.path.isdir(plot_dir_path)
		plot_dp = os.path.join(plot_dir_path,"chrm_comp")
		os.makedirs(plot_dp,exist_ok=True)
		plot_fp = os.path.join(plot_dp,"chrm_comp_{0:02}".format(chrm_num))
		
		# set up plot
		fig, ax = plt.subplots(figsize=(10,10))
		color1 = "green"
		color2 = "#000000" # "black"
		x_pos = [0.5,1.0,1.5,2.25,2.75,3.25]
		width = 0.40
		
		# plot bars
		p_seg_init = ax.bar([x_pos[0],x_pos[3]], both_bp_sizes[0], width, color=color1)
		p_mut_init = ax.bar([x_pos[1],x_pos[4]], both_bp_sizes[0], width, color=both_ints_colors[0])
		p_ent_init = ax.bar([x_pos[2],x_pos[5]], both_bp_sizes[0], width, color=both_ents_colors[0])
		for i in range(1,len(opt_bp_sizes)):
			seg_color = (color1 if i % 2 == 0 else color2)
			p_seg = ax.bar([x_pos[0],x_pos[3]], both_bp_sizes[i], width, color=seg_color, bottom=np.sum(both_bp_sizes[:i],axis=0))
			int_colors = both_ints_colors[i]
			p_mut = ax.bar([x_pos[1],x_pos[4]], both_bp_sizes[i], width, color=int_colors, bottom=np.sum(both_bp_sizes[:i],axis=0))
			ent_colors = both_ents_colors[i]
			p_ent = ax.bar([x_pos[2],x_pos[5]], both_bp_sizes[i], width, color=ent_colors, bottom=np.sum(both_bp_sizes[:i],axis=0))
		
		# add int colorbar
		int_sm.set_array([])
		cbaxes = fig.add_axes([0.9, 0.15, 0.03, 0.3]) 
		int_cbar_bounds = np.linspace(min_int,max_int,1000).astype(np.int)
		int_cbar = plt.colorbar(ax=ax, mappable=int_sm, cax=cbaxes, boundaries=int_cbar_bounds)
		int_cbar.set_ticks(np.linspace(min_int,max_int,5).astype(np.int))
		int_cbar.set_ticklabels([f"{i}k" for i in np.linspace(min_int/1000,max_int/1000,5).astype(np.int)])
		int_cbar.set_label("Muts (count)",rotation=270,labelpad=30)
		
		# add ent colorbar
		ent_sm.set_array([])
		cbaxes = fig.add_axes([0.9, 0.525, 0.03, 0.3])
		ent_cbar_bounds = np.linspace(min_ent,max_ent,1000)
		ent_cbar = plt.colorbar(ax=ax, mappable=ent_sm, cax=cbaxes, boundaries=ent_cbar_bounds)
		ent_cbar.set_ticks(np.linspace(min_ent,max_ent,5))
		ent_cbar.set_ticklabels(np.around(np.linspace(min_ent,max_ent,5),decimals=1))
		ent_cbar.set_label("Entropies (bits)",rotation=270,labelpad=35)
		
		# format axes labels
		ax.set_xticks([np.mean(x_pos[:3]),np.mean(x_pos[3:])])
		ax.set_xticklabels(['Naive', 'Optimal'])
		ax.set_xlim(0.0,x_pos[-1]+0.5)
		ax.tick_params(axis='x', pad=10, length=0)
		max_tick = np.around(chrm_len, decimals=-7)
		ax.set_yticks(np.arange(0, max_tick+1, 20000000))
		ax.set_ylim(0,chrm_len+1)
		ax.set_yticklabels(np.arange(0, int(np.floor(max_tick / 1000000))+1, 20))
		ax.set_ylabel("Position (Mb)",labelpad=20)
		ax.spines["top"].set_visible(False)
		ax.spines["bottom"].set_visible(False)
		ax.spines["right"].set_visible(False)

		# save plot
		plt.savefig(plot_fp+".pdf",bbox_inches="tight",format="pdf")
		plt.savefig(plot_fp+".png",bbox_inches="tight",format="png")
		plt.clf()

def make_plots(ana_file_path, plot_dir_path, naive_seg_size, drop_zeros, eval_split, ana_mode, tumour_list):
	
	assert os.path.isfile(ana_file_path), ana_file_path
	os.makedirs(plot_dir_path, exist_ok=True)
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	#plot_opt_sizes(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split)
	#plot_opt_naive_muts(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split)
	plot_chrms(chrms, plot_dir_path, naive_seg_size, drop_zeros, eval_split, ana_mode, tumour_list)


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


def save_alt_seg_results(results_dir_path, mc_dir_path, mc_alt_dir_path, ana_alt_file_path, naive_seg_size):

	print(results_dir_path)
	print(mc_dir_path)
	print(mc_alt_dir_path)
	print(ana_alt_file_path)
	assert os.path.isdir(mc_dir_path), mc_dir_path
	assert os.path.isdir(mc_alt_dir_path), mc_alt_dir_path
	# assert os.path.isfile(ana_alt_file_path)
	chrms = chrmlib.load_mc_data(mc_dir_path)
	alt_chrms = chrmlib.load_mc_data(mc_alt_dir_path)
	for c in range(chrmlib.NUM_CHRMS):
		S_s_file_name = "S_s_chrm_{}.dat".format(c)
		S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
		E_f_file_name = "E_f_chrm_{}.dat".format(c)
		E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
		chrm = chrms[c]
		alt_chrm = alt_chrms[c]
		for drop_zeros in [False, True]:
			num_segs = alt_chrm.get_num_segs(naive_seg_size, drop_zeros, "all")
			orig_opt_seg = load_seg_results(S_s_file_path, E_f_file_path, chrm.get_mut_array("all"), chrm.get_mut_pos(), num_segs, chrm.group_by, chrm.get_chrm_len(), chrm.type_to_idx)
			orig_bp_bounds = orig_opt_seg.get_bp_bounds()
			alt_mut_array = alt_chrm.get_mut_array("all")
			alt_mut_pos = alt_chrm.get_mut_pos()
			alt_opt_seg = get_opt_seg_from_bp_bounds(alt_mut_array, alt_mut_pos, num_segs, orig_bp_bounds, alt_chrm.type_to_idx)
			alt_chrm.add_opt_seg(num_segs, "all", alt_opt_seg)
		print("finished chrm {}".format(c))
	for alt_chrm in alt_chrms:
		alt_chrm.delete_non_ana_data() # should actually do nothing in this case
	with open(ana_alt_file_path, "wb") as pkl_file:
		pickle.dump(alt_chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--run_dir_path", type=str, default="runs_both_red/90_1.0/co", help="path to directory that holds: ana file (after seg), results dir, plots dir")
	parser.add_argument("--results_dir_name", type=str, default="results", help="name of directory with S_s, E_f, and E_s files")
	parser.add_argument("--mc_dir_path", type=str, default="mc_data", help="path to chromosome data .pkl file, read-only")
	parser.add_argument("--ana_file_name", type=str, default="ana_data.pkl", help="name of analysis data .pkl file")
	parser.add_argument("--naive_seg_size", type=int, default=1000000, help="size of segments in naive segmentation (in bp)")
	parser.add_argument("--program_mode", type=str, choices=["seg", "cmi", "tmi", "plt"], default="seg")
	parser.add_argument("--train_data_type", type=str, choices=["normal", "both"], default="both")
	parser.add_argument("--eval_data_type", type=str, choices=["same", "alt"], default="same")
	parser.add_argument("--drop_zeros", type=lambda x:bool(strtobool(x)), default=True)
	parser.add_argument("--plot_dir_name", type=str, default="plots", help="only useful in \'plt\' mode")
	parser.add_argument("--ana_mode", type=str, choices=["sample_freqs", "counts", "tumour_freqs"], default="counts")
	parser.add_argument("--tumour_set", type=str, choices=["all", "reduced"], default="reduced", help="set of tumour types to use")
	parser.add_argument("--train_split", type=str, choices=["all", "train"], default="all")
	parser.add_argument("--eval_split", type=str, choices=["all", "train", "valid"], default="all")
	FLAGS = parser.parse_args()
	print(FLAGS)
	np.set_printoptions(threshold=1000)

	assert os.path.isdir(FLAGS.run_dir_path), FLAGS.run_dir_path
	results_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.train_split,FLAGS.results_dir_name)
	mc_dir_path = FLAGS.mc_dir_path
	ana_file_path = os.path.join(FLAGS.run_dir_path,FLAGS.ana_file_name)
	plot_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.plot_dir_name)
	if FLAGS.eval_data_type == "alt": # interpet the alt data for the mutations
		mc_alt_dir_path = mc_dir_path + "_alt"
		ana_file_path = ana_file_path.rstrip(".pkl") + "_alt.pkl"
		plot_dir_path += "_alt"
	if FLAGS.train_data_type == "both":
		mc_dir_path = mc_dir_path + "_both"
	if FLAGS.tumour_set == "all":
		tumour_list = sorted(chrmlib.ALL_SET)
	else: # FLAGS.tumour_set == "reduced"
		tumour_list = sorted(chrmlib.REDUCED_SET)
	# if FLAGS.train_split != "all":
	# 	assert FLAGS.data_type == "normal" or FLAGS.data_type == "both"
	ana_file_path = ana_file_path.rstrip(".pkl") + "_{}.pkl".format(FLAGS.train_split)

	if FLAGS.program_mode == "seg":
		# interpret segmentation results and save them in a file of chromosomes
		# does not care about drop_zeros since it automatically computes results for both kinds
		if FLAGS.eval_data_type == "same":
			save_seg_results(results_dir_path, mc_dir_path, ana_file_path, FLAGS.naive_seg_size)
		elif FLAGS.eval_data_type == "alt":
			save_alt_seg_results(results_dir_path, mc_dir_path, mc_alt_dir_path, ana_file_path, FLAGS.naive_seg_size)
		else:
			raise NotImplementedError
	elif FLAGS.program_mode == "cmi":
		# compute conditional mutual information for optimal and naive
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("train split == {}, eval_split == {}".format(FLAGS.train_split, FLAGS.eval_split))
		compute_cmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list, FLAGS.eval_split)
	elif FLAGS.program_mode == "tmi":
		# compute total mutual information for optimal and naive
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("train split == {}, eval_split == {}".format(FLAGS.train_split, FLAGS.eval_split))
		compute_tmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list, FLAGS.eval_split)
	elif FLAGS.program_mode == "plt":
		make_plots(ana_file_path, plot_dir_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.eval_split, FLAGS.ana_mode, tumour_list)
	else:
		raise NotImplementedError

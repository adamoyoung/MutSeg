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


parser = argparse.ArgumentParser()
parser.add_argument("--run_dir_path", type=str, default="runs_kat/jul_31_sf", help="path to directory that holds: ana file (after seg), results dir, plots dir")
parser.add_argument("--results_dir_name", type=str, default="results", help="name of directory with S_s, E_f, and E_s files") #"/scratch/q/qmorris/youngad2/results")
parser.add_argument("--mc_file_path", type=str, default="mc_data_kat.pkl", help="path to chromosome data .pkl file, read-only") #"/home/q/qmorris/youngad2/MutSeg/mc_data_mp.pkl")
parser.add_argument("--ana_file_name", type=str, default="ana_data.pkl", help="path to analyzsis data .pkl file")
parser.add_argument("--naive_seg_size", type=int, default=1000000, help="size of segments in naive segmentation (in bp)")
parser.add_argument("--program_mode", type=str, choices=["seg", "cmi", "tmi", "ann", "plt", "seg", "cmi"], default="seg")
parser.add_argument("--data_type", type=str, choices=["normal", "alt", "sig"], default="normal")
parser.add_argument("--csv_dir_path", type=str, default="for_adamo", help="only useful in \'ann\' mode, path to directory where csvs that need to be annotated will be")
parser.add_argument("--num_procs", type=int, default=mp.cpu_count(), help="only useful in \'ann\' mode, number of process to fork")
parser.add_argument("--drop_zeros", type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument("--plot_dir_name", type=str, default="plots", help="only useful in \'plt\' mode")
parser.add_argument("--ana_mode", type=str, choices=["sample_freqs", "counts", "tumour_freqs"], default="sample_freqs")
parser.add_argument("--tumour_set", type=str, choices=["all", "reduced"], default="all", help="set of tumour types to use")
parser.add_argument("--df_sig_file_path", type=str, default="df_sig.pkl")
parser.add_argument("--seg_split", type=str, choices=["all", "train"], default="all")
parser.add_argument("--mut_split", type=str, choices=["all", "valid"], default="all")


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


def save_seg_results(results_dir_path, mc_file_path, ana_file_path, naive_seg_size, mut_split):
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
		# get chromosome information
		chrm = chrms[c]
		M = chrm.get_unique_pos_count()
		T = chrm.get_num_tumour_types()
		mut_array = chrm.get_mut_array(mut_split)
		mut_pos = chrm.get_mut_pos()
		group_by = chrm.group_by
		chrm_len = chrm.get_chrm_len()
		type_to_idx = chrm.type_to_idx
		for drop_zeros in [False, True]:
			# this implicitly computes the naive segmentation
			num_segs = chrm.get_num_segs(naive_seg_size, drop_zeros)
			# load optimal segmentation information
			opt_seg = load_seg_results(S_s_file_path, E_f_file_path, mut_array, mut_pos, num_segs, group_by, chrm_len, type_to_idx)
			chrms[c].add_opt_seg(num_segs, opt_seg)
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
	return un_I_of_B_and_T_given_C, I_of_B_and_T_given_C

def compute_h_from_ints_array(ints_array):
	"""
	compute H(T|C) from ints_array
	"""
	total_ints_over_B_and_T = np.sum(ints_array, axis=(1,2))
	total_ints_over_B = np.sum(ints_array, axis=1)
	H_of_T_given_C = - np.sum((total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]) * safelog(total_ints_over_B / total_ints_over_B_and_T[..., np.newaxis]), axis=1)
	return H_of_T_given_C


def compute_cmis(ana_data_path, naive_seg_size, drop_zeros, ana_mode, tumour_list):
	""" 
	cmi -- conditional mutual information I(B;T|C)
	T is cancer type
	B is segmentation boundary
	C is chromosome
	I is mutual information
	H is entropy
	"""
	# load chrm data
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	assert len(chrms) == chrmlib.NUM_CHRMS
	print("loaded chrms")
	# set up constants
	T = len(tumour_list)
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
	max_num_segs = max(num_segs)
	# compute optimal cmi
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	# seg_score is equivalent to H(B|C) - H(B,T|C) with log base e
	seg_scores = np.zeros([chrmlib.NUM_CHRMS], dtype=chrmlib.FLOAT_T)
	for c in range(chrmlib.NUM_CHRMS):
		seg = chrms[c].get_opt_seg(num_segs[c])
		seg_scores[c] = seg.final_score
		ints_array[c,:num_segs[c],:] = seg.get_mut_ints(ana_mode, tumour_list)
	un_opt_cmi_1, opt_cmi = compute_cmi_from_ints_array(ints_array)
	# if not drop_zeros:
	# 	# use alternate seg score approach for computing information
	# 	un_opt_cmi_2 = seg_scores + compute_h_from_ints_array(ints_array)
	# 	# verify that they are similar
	# 	assert np.isclose(un_opt_cmi_1, un_opt_cmi_2).all()
	# compute naive cmi
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size).get_mut_ints(drop_zeros, ana_mode, tumour_list) for chrm in chrms]
	for c in range(chrmlib.NUM_CHRMS):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	_, naive_cmi = compute_cmi_from_ints_array(ints_array)
	# # compute naive nz cmi
	# ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	# naive_seg_mut_ints = [chrm.get_naive_nz_seg_mut_ints(naive_seg_size) for chrm in chrms]
	# for c in range(chrmlib.NUM_CHRMS):
	# 	assert naive_seg_mut_ints[c].shape[0] <= num_segs[c]
	# 	ints_array[c,:naive_seg_mut_ints[c].shape[0],:] = naive_seg_mut_ints[c]
	# _, naive_nz_cmi = compute_cmi_from_ints_array(ints_array)
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


def compute_tmis(ana_file_path, naive_seg_size, drop_zeros, ana_mode, tumour_list):
	"""
	tmi -- total mutual information I(B;T)
	I(B;T) = I(B;T|C) - H(C|T) - H(C|B) + H(C|B,T) + H(C)
	does into log space for numerical stability
	"""
	# load chrm data
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	assert len(chrms) == chrmlib.NUM_CHRMS
	print("loaded chrms")
	# get constants
	T = len(tumour_list)
	num_chrms = chrmlib.NUM_CHRMS
	# compute optimal tmi
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
	max_num_segs = max(num_segs)
	total_num_segs = sum(num_segs)
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	for c in range(num_chrms):
		seg = chrms[c].get_opt_seg(num_segs[c])
		ints_array[c,:num_segs[c],:] = seg.get_mut_ints(ana_mode, tumour_list)
	opt_I_of_B_and_T = compute_tmi_from_ints_array(ints_array, num_segs)
	# compute naive tmi
	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
	max_num_segs = max(num_segs)
	total_num_segs = sum(num_segs)
	ints_array = np.zeros([chrmlib.NUM_CHRMS, max_num_segs, T], dtype=chrmlib.FLOAT_T)
	naive_seg_mut_ints = [chrm.get_naive_seg(naive_seg_size).get_mut_ints(drop_zeros,ana_mode,tumour_list) for chrm in chrms]
	for c in range(num_chrms):
		ints_array[c,:num_segs[c],:] = naive_seg_mut_ints[c]
	naive_I_of_B_and_T = compute_tmi_from_ints_array(ints_array, num_segs)
	return opt_I_of_B_and_T, naive_I_of_B_and_T


def annotate_files_func(proc_inputs):
	""" function called by each process """
	proc_id, proc_files_path, sh_arrays = proc_inputs[0], proc_inputs[1], proc_inputs[2]
	num_segs, seg_bp_bounds = sh_arrays[0], sh_arrays[1]
	# local function to find the row
	def find_opt_seg(row):
		chrm_num, mut_pos = row["Chromosome"]-1, row["Start_position"]-1
		chrm_seg_bp_bounds = seg_bp_bounds[chrm_num]
		chrm_num_segs = num_segs[chrm_num]
		prev_num_segs = sum([num_segs[i] for i in range(chrm_num)])
		opt_seg = -1
		for s in range(1,chrm_num_segs+1):
			if mut_pos < chrm_seg_bp_bounds[s]:
				opt_seg = prev_num_segs + s + 1
				break
		return opt_seg
	# read in each file, apply function, overwrite file
	for file_path in proc_files_path:
		df = pd.read_csv(file_path)
		opt_segs = df.apply(find_opt_seg, axis=1)
		assert (opt_segs >= 0).all()
		df["optimal_segment"] = opt_segs
		# df.apply(assert_naive_seg, axis=1)
		df.to_csv(file_path)
	return None


def annotate_segs(ana_file_path, naive_seg_size, csv_dir_path, num_procs, drop_zeros):
	"""
	annotate mutations in original csv files with optimal segment location
	assumes these csv files are valid
	note that both naive and optimal segmentations have 1-based indexing in the files
	"""
	raise NotImplementedError
	# # load chrm data
	# with open(ana_file_path, "rb") as pkl_file:
	# 	chrms = pickle.load(pkl_file)
	# assert len(chrms) == chrmlib.NUM_CHRMS
	# print("loaded chrms")
	# # iterate over files in directory
	# file_paths = []
	# file_count = 0
	# entries = sorted(os.listdir(csv_dir_path))
	# for entry in entries:
	# 	entry_path = os.path.join(csv_dir_path,entry)
	# 	if os.path.isfile(entry_path):
	# 		file_paths.append(entry_path)
	# 		file_count += 1
	# assert file_count == len(entries)
	# # divide up file names equally amongst processes
	# num_per_proc = [len(file_paths) // num_procs for i in range(num_procs)]
	# for i in range(len(file_paths) % num_procs):
	# 	num_per_proc[i] += 1
	# assert np.sum(num_per_proc) == len(file_paths)
	# print("max num_per_proc = %d" % np.max(num_per_proc))
	# # set up inputs
	# num_chrms = len(chrms)
	# num_segs = [chrm.get_num_segs(naive_seg_size) for chrm in chrms]
	# max_num_segs = max(num_segs)
	# # shared read-only arrays
	# sh_num_segs = np.ctypeslib.as_array(mp.Array(ctypes.c_uint, num_chrms))
	# sh_seg_bp_bounds = np.ctypeslib.as_array(mp.Array(ctypes.c_uint, num_chrms*(max_num_segs+1))).reshape(num_chrms,max_num_segs+1)
	# # num_segs = np.array([chrm.get_num_segs(naive_seg_size) for chrm in chrms], dtype=chrmlib.INT_T)
	# # all_seg_bp_bounds = np.zeros([len(chrms),max(num_segs)+1], dtype=chrmlib.INT_T)
	# for c in range(len(chrms)):
	# 	sh_num_segs[c] = num_segs[c]
	# 	sh_seg_bp_bounds[c][0:num_segs[c]+1] = chrms[c].get_opt_seg(naive_seg_size).seg_bp_bounds
	# sh_arrays = [sh_num_segs, sh_seg_bp_bounds]
	# running_total = 0
	# proc_inputs = []
	# for i in range(num_procs):
	# 	proc_inputs.append((i,file_paths[running_total:running_total+num_per_proc[i]],sh_arrays))
	# 	running_total += num_per_proc[i]
	# assert running_total == len(file_paths)
	# # annotate files and save results
	# pool = mp.Pool(num_procs)
	# proc_results = pool.map(annotate_files_func, proc_inputs)
	# # proc_results is a list of None's
	# assert len(proc_results) == num_procs


def plot_opt_sizes(chrms, plot_dir_path, naive_seg_size, drop_zeros):
	""" plots the sizes of the segments in the optimal segmentation (in bp) """

	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
	opt_segs = [chrms[c].get_opt_seg(num_segs[c]) for c in range(len(chrms))]
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
	print("min = {}, max = {}, mean = {}, median = {}".format(min_seg_size, max_seg_size, mean_seg_size, median_seg_size))
	ax = sns.distplot(
		seg_sizes / 1000000,
		kde=False,
		norm_hist=False,
		bins=[i / 1000000 for i in range(0, 5000000, 100000)]
	)
	plt_name = "opt_seg_sizes_{}MB".format(naive_seg_size // 1000000)
	if drop_zeros:
		plt_name += "_nz"
	ax.set(
		xlabel="segment size (Mbp)",
		ylabel="counts",
		title=plt_name
	)
	ax.text(0.75, 0.85, f"total = {sum(num_segs)}", fontsize=10, transform=ax.transAxes)
	plt_path = os.path.join(plot_dir_path,plt_name)
	plt.savefig(plt_path)
	plt.clf()


def plot_opt_naive_muts(chrms, plot_dir_path, naive_seg_size, drop_zeros):
	""" plot the number of distinct mutation positions in each segment """

	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
	seg_mut_counts = np.zeros([2,sum(num_segs)], dtype=chrmlib.INT_T)
	# get opt mutation counts - can be grouped
	opt_segs = [chrms[c].get_opt_seg(num_segs[c]) for c in range(len(chrms))]
	cur_seg_idx = 0
	for c in range(len(chrms)):
		seg_mut_bounds = opt_segs[c].get_mut_bounds()
		assert len(seg_mut_bounds) == num_segs[c]+1
		for s in range(num_segs[c]):
			seg_mut_counts[0][cur_seg_idx+s] = seg_mut_bounds[s+1] - seg_mut_bounds[s]
		cur_seg_idx += num_segs[c]
	assert cur_seg_idx == seg_mut_counts.shape[1]
	group_by = opt_segs[0].group_by
	if group_by:
		assert np.all([opt_seg.group_by == group_by for opt_seg in opt_segs])
		seg_mut_counts[0] = group_by*seg_mut_counts[0]
	# get naive mutation counts - never grouped
	naive_segs = [chrm.get_naive_seg(naive_seg_size) for chrm in chrms]
	cur_seg_idx = 0
	for c in range(len(chrms)):
		seg_mut_bounds = naive_segs[c].get_mut_bounds(drop_zeros)
		assert len(seg_mut_bounds) == num_segs[c]+1
		for s in range(num_segs[c]):
			seg_mut_counts[1][cur_seg_idx+s] = seg_mut_bounds[s+1] - seg_mut_bounds[s]
		cur_seg_idx += num_segs[c]
	assert cur_seg_idx == seg_mut_counts.shape[1]
	# plot the results
	assert np.sum(seg_mut_counts[0]) == np.sum(seg_mut_counts[1]), np.sum(seg_mut_counts,axis=1)
	seg_types = ["opt", "naive"]
	for i in range(len(seg_types)):
		min_seg_mut_counts = int(np.min(seg_mut_counts[i]))
		max_seg_mut_counts = int(np.max(seg_mut_counts[i]))
		mean_seg_mut_counts = int(np.mean(seg_mut_counts[i]))
		median_seg_mut_counts = int(np.median(seg_mut_counts[i]))
		print("min = {}, max = {}, mean = {}, median = {}".format(min_seg_mut_counts, max_seg_mut_counts, mean_seg_mut_counts, median_seg_mut_counts))
		ax = sns.distplot(
			seg_mut_counts[i],
			kde=False,
			norm_hist=False,
			bins=[i for i in range(0,81000,1000)],
			label=seg_types[i]
		)
		# plt_name = "{}_seg_mut_counts_{}MB".format(seg_types[i],naive_seg_size // 1000000)
		# if drop_zeros:with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	with open(mc_alt_file_path, "rb") as pkl_file:
		alt_chrms = pickle.load(pkl_file)
		# 	plt_name += "_nz"
		# ax.set(
		# 	xlabel="number of distinct mutations in segment",
		# 	ylabel="counts",
		# 	xlim=[-1000,81000],
		# 	title=plt_name)
		# plt_path = os.path.join(plot_dir_path,plt_name)
		# plt.savefig(plt_path)
		# plt.clf()
	plt_name = "opt_vs_naive_seg_mut_counts_{}MB".format(naive_seg_size // 1000000)
	if drop_zeros:
		plt_name += "_nz"
	ax.set(
		xlabel="number of distinct mutations in segment",
		ylabel="counts",
		xlim=[-1000,81000],
		title=plt_name)
	ax.legend()
	plt_path = os.path.join(plot_dir_path,plt_name)
	plt.savefig(plt_path)
	plt.clf()

	# plot the 
	mut_thresh = 1000
	opt_thresh_segs = np.nonzero(seg_mut_counts[0] <= mut_thresh)[0]
	opt_thresh_mut_counts = seg_mut_counts[0][opt_thresh_segs]
	ax = sns.distplot(
		opt_thresh_mut_counts,
		kde=False,
		norm_hist=False,
		bins=10
	)
	plt_name = "opt_{}_mut_seg_mut_counts_{}MB".format(mut_thresh, naive_seg_size // 1000000)
	if drop_zeros:
		plt_name += "_nz"
	ax.set(
		xlabel="number of distinct mutations in segment",
		ylabel="counts",
		title=plt_name)
	plt_path = os.path.join(plot_dir_path,plt_name)
	plt.savefig(plt_path)
	plt.clf()

	# plot the real sizes of the minimum optimal segments
	mut_threshes = [100, 1000]
	opt_seg_sizes = []
	for c in range(len(chrms)):
		# seg_sizes = np.zeros(num_segs[c], dtype=chrmlib.INT_T)
		seg_bp_bounds = opt_segs[c].get_bp_bounds()
		for s in range(num_segs[c]):
			# seg_sizes[s] = seg_bp_bounds[s+1] - seg_bp_bounds[s]
			opt_seg_sizes.append(seg_bp_bounds[s+1] - seg_bp_bounds[s])
	opt_seg_sizes = np.array(opt_seg_sizes, dtype=np.float)
	for m in range(len(mut_threshes)):
		mut_thresh = mut_threshes[m]
		opt_thresh_segs = np.nonzero(seg_mut_counts[0] <= mut_thresh)[0]
		print(f"mut_thresh = {mut_thresh}, num opt_min_segs = {len(opt_thresh_segs)}")
		opt_thresh_sizes = opt_seg_sizes[opt_thresh_segs]
		print(f"min = {np.min(opt_thresh_sizes)}, max = {np.max(opt_thresh_sizes)}, mean = {np.mean(opt_thresh_sizes)}, median = {np.median(opt_thresh_sizes)}")
		plt_name = "opt_{}_mut_seg_sizes_{}MB".format(mut_thresh, naive_seg_size // 1000000)
		if drop_zeros:
			plt_name += "_nz"
		ax = sns.distplot(
			opt_thresh_sizes / 1000000,
			kde=False,
			norm_hist=False,
			bins=[i / 1000000 for i in range(0, 5000000, 100000)]
		)
		ax.set(
			xlabel="segment size (Mbp)",
			ylabel="counts",
			title=plt_name
		)
		ax.text(0.75, 0.85, f"total = {len(opt_thresh_segs)}", fontsize=10, transform=ax.transAxes)
		plt_path = os.path.join(plot_dir_path,plt_name)
		plt.savefig(plt_path)
		plt.clf()


def make_plots(ana_file_path, plot_dir_path, naive_seg_size, drop_zeros):

	assert os.path.isfile(ana_file_path)
	os.makedirs(plot_dir_path, exist_ok=True)
	with open(ana_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	plot_opt_sizes(chrms, plot_dir_path, naive_seg_size, drop_zeros)
	plot_opt_naive_muts(chrms, plot_dir_path, naive_seg_size, drop_zeros)


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


def save_alt_seg_results(results_dir_path, mc_file_path, mc_alt_file_path, ana_alt_file_path, naive_seg_size):

	print(results_dir_path)
	print(mc_file_path)
	print(mc_alt_file_path)
	print(ana_alt_file_path)
	assert os.path.isfile(mc_file_path)
	assert os.path.isfile(mc_alt_file_path)
	# assert os.path.isfile(ana_alt_file_path)
	with open(mc_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	with open(mc_alt_file_path, "rb") as pkl_file:
		alt_chrms = pickle.load(pkl_file)
	for c in range(chrmlib.NUM_CHRMS):
		S_s_file_name = "S_s_chrm_{}.dat".format(c)
		S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
		E_f_file_name = "E_f_chrm_{}.dat".format(c)
		E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
		chrm = chrms[c]
		alt_chrm = alt_chrms[c]
		for drop_zeros in [False, True]:
			num_segs = alt_chrm.get_num_segs(naive_seg_size, drop_zeros)
			orig_opt_seg = load_seg_results(S_s_file_path, E_f_file_path, chrm.get_mut_array("all"), chrm.get_mut_pos(), num_segs, chrm.group_by, chrm.get_chrm_len(), chrm.type_to_idx)
			orig_bp_bounds = orig_opt_seg.get_bp_bounds()
			alt_opt_seg = get_opt_seg_from_bp_bounds(alt_chrm.get_mut_array("all"), alt_chrm.get_mut_pos(), num_segs, orig_bp_bounds, alt_chrm.type_to_idx)
			alt_chrm.add_opt_seg(num_segs, alt_opt_seg)
		print("finished chrm {}".format(c))
	for alt_chrm in alt_chrms:
		alt_chrm.delete_non_ana_data() # should actually do nothing in this case
	with open(ana_alt_file_path, "wb") as pkl_file:
		pickle.dump(alt_chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


def get_mut_ints_from_sig_df_and_bp_bounds(sig_df, bp_bounds, sig_type_to_idx):
	""" assumes sig_df is sorted already by chrm, start, end"""

	num_segs = bp_bounds.shape[0]
	num_sig_types = len(sig_type_to_idx.keys())
	straddle_mut_count = 0
	mut_ints = np.zeros([num_segs, num_sig_types], dtype=chrmlib.FLOAT_T)
	starts = sig_df["start"].to_numpy()
	ends = sig_df["end"].to_numpy()
	sig_types = sig_df["sig_typ"].to_numpy()
	cur_seg_idx = 0
	for i in range(len(starts)):
		while starts[i] >= bp_bounds[cur_seg_idx+1]:
			cur_seg_idx += 1
			assert cur_seg_idx < num_segs
		if ends[i] < bp_bounds[cur_seg_idx+1]:
			mut_ints[cur_seg_idx,sig_type_to_idx[sig_types[i]]] += ends[i] - starts[i]
		else: # ends[i] >= chrm_bp_bounds[cur_seg_idx+1]
			assert cur_seg_idx < num_segs-1, (starts[i], ends[i], bp_bounds[cur_seg_idx+1])
			straddle_mut_count += 1
			mut_ints[cur_seg_idx,sig_type_to_idx[sig_types[i]]] += bp_bounds[cur_seg_idx+1] - starts[i]
			mut_ints[cur_seg_idx+1,sig_type_to_idx[sig_types[i]]] += ends[i] - bp_bounds[cur_seg_idx+1]
	print("number of straddle mutations = {}".format(straddle_mut_count))
	return mut_ints


def save_sig_seg_results(results_dir_path, mc_file_path, df_sig_file_path, ana_sig_file_path, naive_seg_size):
	""" gets results for drop_zeros and not drop_zeros """

	print(mc_file_path)
	print(df_sig_file_path)
	print(ana_sig_file_path)
	assert os.path.isfile(mc_file_path)
	assert os.path.isfile(df_sig_file_path)
	with open(mc_file_path, "rb") as pkl_file:
		chrms = pickle.load(pkl_file)
	sig_df = pd.read_pickle(df_sig_file_path)
	print("total number of sigs = {}".format(sig_df.shape[0]))
	sig_dfs = [sig_df[sig_df["chrm"] == c].sort_values(["start","end"]) for c in range(chrmlib.NUM_CHRMS)]
	sig_chrms = [chrmlib.Chromosome(c) for c in range(chrmlib.NUM_CHRMS)]
	# get type_to_idx
	sig_types = sorted(set(sig_df["sig_typ"]))
	sig_type_to_idx = {}
	for idx in range(len(sig_types)):
		sig_type_to_idx[sig_types[idx]] = idx
	# get naive sig segs
	for c in range(chrmlib.NUM_CHRMS):
		chrm = chrms[c]
		sig_chrm = sig_chrms[c]
		naive_bp_bounds = chrm.get_default_naive_bp_bounds(naive_seg_size)
		sig_naive_mut_ints = get_mut_ints_from_sig_df_and_bp_bounds(sig_dfs[c], naive_bp_bounds, sig_type_to_idx)
		num_segs = chrm._get_num_segs(naive_seg_size)
		nz_seg_idx = np.nonzero(np.sum(sig_naive_mut_ints,axis=1))[0]
		sig_naive_seg = NaiveSigSegmentation(sig_type_to_idx, num_segs, sig_naive_mut_ints, naive_bp_bounds, nz_seg_idx, len(nz_seg_idx))
		sig_chrm.naive_segmentations[naive_seg_size] = sig_naive_seg
		print("finished chrm {}".format(c))
	print("done sig naive segs")
	# get optimal sig segs
	for c in range(chrmlib.NUM_CHRMS):
		S_s_file_name = "S_s_chrm_{}.dat".format(c)
		S_s_file_path = os.path.join(results_dir_path, S_s_file_name)
		E_f_file_name = "E_f_chrm_{}.dat".format(c)
		E_f_file_path = os.path.join(results_dir_path, E_f_file_name)
		chrm = chrms[c]
		sig_chrm = sig_chrms[c]
		for drop_zeros in [False, True]:
			# all naive segmentations have been computed by this point
			num_segs = sig_chrm.get_num_segs(naive_seg_size, drop_zeros)
			# load orig results from file
			orig_opt_seg = load_seg_results(S_s_file_path, E_f_file_path, chrm.get_mut_array("all"), chrm.get_mut_pos(), num_segs, chrm.group_by, chrm.get_chrm_len(), chrm.type_to_idx)
			orig_bp_bounds = orig_opt_seg.get_bp_bounds()
			# compute the new segmentation using the orig boundaries
			sig_opt_mut_ints = get_mut_ints_from_sig_df_and_bp_bounds(sig_dfs[c], orig_bp_bounds, sig_type_to_idx)
			sig_opt_seg = OptimalSigSegmentation(sig_type_to_idx, num_segs, sig_opt_mut_ints, orig_bp_bounds)
			sig_chrm.add_opt_seg(num_segs, sig_opt_seg)
		print("finished chrm {}".format(c))
	print("done sig opt segs")
	for sig_chrm in sig_chrms:
		sig_chrm.delete_non_ana_data() # doesn't actually do anything here
	with open(ana_sig_file_path, "wb") as pkl_file:
		pickle.dump(sig_chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


# def save_alt_seg_results(mc_alt_file_path, ana_file_path, ana_alt_file_path, ana_mode, naive_seg_size, drop_zeros):
# 	""" computes cmi and tmi for optimal and naive segmentations using the alt data"""

# 	assert os.path.isfile(mc_alt_file_path)
# 	assert os.path.isfile(ana_file_path)
# 	assert os.path.isfile(ana_alt_file_path)
# 	assert ana_mode == "sample_freqs"
# 	with open(ana_file_path, "rb") as pkl_file:
# 		chrms = pickle.load(pkl_file)
# 	with open(mc_alt_file_path, "rb") as pkl_file:
# 		alt_chrms = pickle.load(pkl_file)
# 	num_segs = [chrm.get_num_segs(naive_seg_size, drop_zeros) for chrm in chrms]
# 	naive_bp_bounds = [chrm.get_naive_seg(naive_seg_size).get_bp_bounds(drop_zeros) for chrm in chrms]
# 	opt_bp_bounds = [chrms[c].get_opt_seg(num_segs[c]).get_bp_bounds() for c in range(len(chrms))]
# 	alt_mut_ints = [alt_chrm.mut_array[1] for alt_chrm in alt_chrms]
# 	alt_mut_pos = [alt_chrm.get_mut_pos() for alt_chrm in alt_chrms]
# 	num_tumour_types = alt_chrms[0].get_num_tumour_types()
# 	assert all([alt_chrm.get_num_tumour_types() == num_tumour_types for alt_chrm in alt_chrms])
# 	naive_mut_ints = get_mut_ints_from_bp_bounds(alt_mut_ints, alt_mut_pos, num_segs, naive_bp_bounds)
# 	_, naive_cmi = compute_cmi_from_ints_array(naive_mut_ints)
# 	naive_tmi = compute_tmi_from_ints_array(naive_mut_ints, num_segs)
# 	opt_mut_ints = get_mut_ints_from_bp_bounds(alt_mut_ints, alt_mut_pos, num_segs, opt_bp_bounds)
# 	_, opt_cmi = compute_cmi_from_ints_array(opt_mut_ints)
# 	opt_tmi = compute_tmi_from_ints_array(opt_mut_ints, num_segs)
# 	return opt_cmi, naive_cmi, opt_tmi, naive_tmi


if __name__ == "__main__":

	FLAGS = parser.parse_args()
	print(FLAGS)

	assert os.path.isdir(FLAGS.run_dir_path)
	results_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.seg_split,FLAGS.results_dir_name)
	mc_file_path = FLAGS.mc_file_path
	ana_file_path = os.path.join(FLAGS.run_dir_path,FLAGS.ana_file_name)
	plot_dir_path = os.path.join(FLAGS.run_dir_path,FLAGS.plot_dir_name)
	if FLAGS.data_type == "alt": # interpet the alt data for the mutations
		mc_alt_file_path = mc_file_path.rstrip(".pkl") + "_alt.pkl"
		ana_file_path = ana_file_path.rstrip(".pkl") + "_alt.pkl"
		plot_dir_path += "_alt"
	elif FLAGS.data_type == "sig":
		ana_file_path = ana_file_path.rstrip(".pkl") + "_sig.pkl"
		plot_dir_path += "_sig"
	if FLAGS.tumour_set == "all":
		tumour_list = sorted(chrmlib.ALL_SET)
	else: # FLAGS.tumour_set == "reduced"
		tumour_list = sorted(chrmlib.REDUCED_SET)
	if FLAGS.mut_split != "all":
		assert FLAGS.program_mode != "plt"
		assert FLAGS.data_type == "normal"
		assert FLAGS.tumour_set == "all"
		ana_file_path = ana_file_path.rstrip(".pkl") + "_{}.pkl".format(FLAGS.mut_split)
	if FLAGS.program_mode == "seg":
		# interpret segmentation results and save them in a file of chromosomes
		# does not care about drop_zeros since it automatically computes results for both kinds
		if FLAGS.data_type == "alt":
			save_alt_seg_results(results_dir_path, mc_file_path, mc_alt_file_path, ana_file_path, FLAGS.naive_seg_size)
		elif FLAGS.data_type == "sig":
			save_sig_seg_results(results_dir_path, mc_file_path, FLAGS.df_sig_file_path, ana_file_path, FLAGS.naive_seg_size)
		else: # FLAGS.data_type == "normal"
			save_seg_results(results_dir_path, mc_file_path, ana_file_path, FLAGS.naive_seg_size, FLAGS.mut_split)
	elif FLAGS.program_mode == "cmi":
		# compute conditional mutual information for optimal and naive
		optimal_cmi, naive_cmi = compute_cmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list)
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("optimal cmi", nats_to_bits(optimal_cmi))
		print("naive cmi", nats_to_bits(naive_cmi))
		print("cmi difference", nats_to_bits(optimal_cmi-naive_cmi))
	elif FLAGS.program_mode == "tmi":
		# compute total mutual information for optimal and naive
		optimal_tmi, naive_tmi = compute_tmis(ana_file_path, FLAGS.naive_seg_size, FLAGS.drop_zeros, FLAGS.ana_mode, tumour_list)
		print("drop_zeros ==", FLAGS.drop_zeros)
		print("optimal tmi", nats_to_bits(optimal_tmi))
		print("naive tmi", nats_to_bits(naive_tmi))
		print("tmi difference", nats_to_bits(optimal_tmi-naive_tmi))
	# elif FLAGS.program_mode == "ann":
	# 	# modify input csvs such that each mutation is annotated with an optimal segment
	# 	annotate_segs(ana_file_path, FLAGS.naive_seg_size, FLAGS.csv_dir_path, FLAGS.num_procs, FLAGS.drop_zeros)
	elif FLAGS.program_mode == "plt":
		make_plots(ana_file_path, plot_dir_path, FLAGS.naive_seg_size, FLAGS.drop_zeros)
	else:
		raise NotImplementedError

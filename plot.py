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
from analyze import *

FIG_DIR = "/home/adamo/Documents/MutSeg/figures/sept_18"
small_K = 1000000
large_K = 100000

def hier_cluster_opt(mc_data, b_data):

	# only look at "L" mode

	typs_list = mc_data["typs_list"]
	mut_freqs = b_data["mut_ints"][1][-1] #[k_modes,folds,C,max(K),T]
	Ks = b_data["Ks"][1]
	seg_size = b_data["seg_size"][0]
	C = mut_freqs.shape[0]
	T = mut_freqs.shape[2]
	seg_mut_freqs = np.zeros([np.sum(Ks),T], dtype=np.float64)
	chrm_start_locations = np.zeros([C], dtype=np.int)

	for c in range(C):
		prev_K = sum(Ks[0:c])
		chrm_start_locations[c] = prev_K
		for k in range(Ks[c]):
			seg_mut_freqs[prev_K+k] = mut_freqs[c,k]

	# plot unnormalized
	pd_seg_mut_freqs = pd.DataFrame(seg_mut_freqs, columns=typs_list)
	g = sns.clustermap(pd_seg_mut_freqs, figsize=(10,15), metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('segment')
	g.ax_heatmap.set_yticks(chrm_start_locations)
	g.ax_heatmap.set_yticklabels(["chrm {}".format(c+1) for c in range(C)])
	for c in range(C):
		g.ax_heatmap.axhline(chrm_start_locations[c],linewidth=1,color="grey")
	if seg_size == small_K:
		g.fig.suptitle('Optimal Segmentation (small K, unnormalized)')
		plt.savefig("{}/opt_hhmap_small_un.png".format(FIG_DIR))
	else:
		g.fig.suptitle('Optimal Segmentation (large K, unnormalized)')
		plt.savefig("{}/opt_hhmap_large_un.png".format(FIG_DIR))
	plt.show()

	# normalize across columns
	means = np.reshape(np.mean(seg_mut_freqs, axis=1), [sum(Ks),1]) #[1,T])
	stds = np.reshape(np.std(seg_mut_freqs, axis=1), [sum(Ks),1]) #[1,T])
	seg_mut_freqs = (seg_mut_freqs - means) / stds
	#print(np.max(np.sum(seg_mut_freqs,axis=1)))

	# plot row normalized
	pd_seg_mut_freqs = pd.DataFrame(seg_mut_freqs, columns=typs_list)
	g = sns.clustermap(pd_seg_mut_freqs, figsize=(10,15), metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('segment')
	g.ax_heatmap.set_yticks(chrm_start_locations)
	g.ax_heatmap.set_yticklabels(["chrm {}".format(c+1) for c in range(C)])
	for c in range(C):
		g.ax_heatmap.axhline(chrm_start_locations[c],linewidth=1,color="grey")
	if seg_size == small_K:
		g.fig.suptitle('Optimal Segmentation (small K, row normalized)')
		plt.savefig("{}/opt_hhmap_small_rn.png".format(FIG_DIR))
	else:
		g.fig.suptitle('Optimal Segmentation (large K, row normalized)')
		plt.savefig("{}/opt_hhmap_large_rn.png".format(FIG_DIR))
	plt.show()


def hier_cluster_naive(mc_data,b_data):

	typs_list = mc_data["typs_list"]
	seg_size = b_data["seg_size"][0]
	mut_freqs, mut_bounds, bp_bounds, Ks = naive_traceback(mc_data,seg_size)
	Ks = b_data["Ks"][1]
	C = mut_freqs.shape[0]
	T = mut_freqs.shape[2]
	seg_mut_freqs = np.zeros([np.sum(Ks),T], dtype=np.float64)
	chrm_start_locations = np.zeros([C], dtype=np.int)

	for c in range(C):
		prev_K = sum(Ks[0:c])
		chrm_start_locations[c] = prev_K
		for k in range(Ks[c]):
			seg_mut_freqs[prev_K+k] = mut_freqs[c,k]
	
	# plot unnormalized
	pd_seg_mut_freqs = pd.DataFrame(seg_mut_freqs, columns=typs_list)
	g = sns.clustermap(pd_seg_mut_freqs, figsize=(10,15), metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('segment')
	g.ax_heatmap.set_yticks(chrm_start_locations)
	g.ax_heatmap.set_yticklabels(["chrm {}".format(c+1) for c in range(C)])
	for c in range(C):
		g.ax_heatmap.axhline(chrm_start_locations[c],linewidth=1,color="grey")
	if seg_size == small_K:
		g.fig.suptitle('Naive Segmentation (small K, unnormalized)')
		plt.savefig("{}/naive_hhmap_small_un.png".format(FIG_DIR))
	else:
		g.fig.suptitle('Naive Segmentation (large K, unnormalized)')
		plt.savefig("{}/naive_hhmap_large_un.png".format(FIG_DIR))
	plt.show()

	# normalize across columns
	means = np.reshape(np.mean(seg_mut_freqs, axis=1), [sum(Ks),1]) #[1,T])
	stds = np.reshape(np.std(seg_mut_freqs, axis=1), [sum(Ks),1]) #[1,T])
	seg_mut_freqs = (seg_mut_freqs - means) / stds
	seg_mut_freqs = np.nan_to_num(seg_mut_freqs)
	#print(np.max(np.sum(seg_mut_freqs,axis=1)))

	# plot normalized
	pd_seg_mut_freqs = pd.DataFrame(seg_mut_freqs, columns=typs_list)
	g = sns.clustermap(pd_seg_mut_freqs, figsize=(10,15), metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('segment')
	g.ax_heatmap.set_yticks(chrm_start_locations)
	g.ax_heatmap.set_yticklabels(["chrm {}".format(c+1) for c in range(C)])
	for c in range(C):
		g.ax_heatmap.axhline(chrm_start_locations[c],linewidth=1,color="grey")
	if seg_size == small_K:
		g.fig.suptitle('Naive Segmentation (small K, row normalized)')
		plt.savefig("{}/naive_hhmap_small_rn.png".format(FIG_DIR))
	else:
		g.fig.suptitle('Naive Segmentation (large K, normalized)')
		plt.savefig("{}/naive_hhmap_large_rn.png".format(FIG_DIR))
	plt.show()

def hist_seg_size_bp(mc_data, b_data):

	bp_bounds = b_data["bp_bounds"][1][-1] # [2,1,22,max(Ks)]
	Ks = b_data["Ks"][1] # [2,22]
	seg_size = b_data["seg_size"][0]
	total_num_segs = np.sum(Ks)
	seg_length_bp = np.zeros([total_num_segs], dtype=np.float64)

	for c in range(Ks.shape[0]):
		prev_K = sum(Ks[0:c])
		for k in range(Ks[c]):
			seg_length_bp[prev_K+k] = (bp_bounds[c,k+1] - bp_bounds[c,k])
	seg_length_bp //= 1000
	print("min = {}, max = {}".format(np.min(seg_length_bp), np.max(seg_length_bp)) )
	g = sns.distplot(seg_length_bp, kde=False, bins=30)
	g.set_yscale('log')
	g.set_ylabel("log Frequency")
	g.set_xlabel("Segment Size (kbp)")
	if seg_size == small_K:
		g.set_title('Segment Size (small K)')
		plt.savefig("{}/segment_size_dist_small.png".format(FIG_DIR))
	else:
		g.set_title('Segment Size (large K)')
		plt.savefig("{}/segment_size_dist_large.png".format(FIG_DIR))
	plt.show()

def mutual_info_per_segment(mc_data, b_data):
	""" uses "L" mode """

	I_of_seg_and_T_given_C, H_of_seg_given_C = classify_segs_2(mc_data,b_data)
	Ks = b_data["Ks"][1]
	seg_size = b_data["seg_size"][0]
	C = I_of_seg_and_T_given_C.shape[0]
	T = I_of_seg_and_T_given_C.shape[1]
	seg_test_results = np.zeros([np.sum(Ks)], dtype=np.float64)

	for c in range(C):
		prev_K = sum(Ks[0:c])
		for k in range(Ks[c]):
			seg_test_results[prev_K+k] = I_of_seg_and_T_given_C[1,c,k] / H_of_seg_given_C[1,c,k]
	
	# plot here
	g = sns.distplot(seg_test_results, kde=False, bins=30)
	g.set_yscale('log')
	g.set_ylabel("log Frequency")
	g.set_xlabel("Conditional Mutual Information / Conditional Entropy")
	if seg_size == small_K:
		g.set_title("Conditional Mutual Information Relative to Conditional Entropy (small K)")
		plt.savefig("{}/segment_conditional_mutual_info_small.png".format(FIG_DIR))
	else:
		g.set_title("Conditional Mutual Information Relative to Conditional Entropy (large K)")
		plt.savefig("{}/segment_conditional_mutual_info_large.png".format(FIG_DIR))
	plt.show()


if __name__ == "__main__":

	mc_data = np.load("/home/adamo/Documents/MutSeg/data/mc_100_f.npz")
	b_data = np.load("/home/adamo/Documents/MutSeg/data/b_100_f_10.npz")
	#hier_cluster_opt(mc_data,b_data)
	hier_cluster_naive(mc_data,b_data)
	#hist_seg_size_bp(mc_data,b_data)
	#mutual_info_per_segment(mc_data,b_data)
	#opt_vs_naive_information_2(mc_data,b_data)

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

# def plot(mc_data, b_data):

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



import numpy as np
import struct
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

float_t = np.float64
eps = np.finfo(float_t).eps

SEG_SIZE = 1000000 # (1 MB)
NUM_CHRMS = 22
# chromosome lens (for 22 autosomal chromosomes and Y, X not included)
# got these from cytoBand.txt online
#chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]
# got these from UCSC genome browser (hg38)
chrm_lens = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,57227415 ]
typs = ['BLCA','BOCA','BRCA','BTCA','CESC','CLLE','CMDI','COAD','DLBC','EOPC','ESAD','GACA','GBM','HNSC','KICH','KIRC','KIRP','LAML','LGG','LICA','LIHC','LINC','LIRI','LUAD','LUSC','MALY','MELA','ORCA','OV','PACA','PAEN','PBCA','PRAD','READ','RECA','SARC','SKCM','STAD','THCA','UCEC']

def median(a, b):

	return int(round((a+b)/2))

def myround(x, base):

	return int(base*round(float(x)/base))

def traceback(S_s_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]
	chrm_mut_pos = mut_pos[chrm]

	# read in contents of S_s file
	S_s_file = open(S_s_file_name, 'rb')
	S_s_bytes = S_s_file.read(4*M*K)
	S_s = []
	for i in range(len(S_s_bytes) // 4):
		number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
		S_s.append(number)

	# get segmentation
	final_path = []
	final_path.insert(0,M)
	k = K-1
	col = M-1
	while k > 0:
		col = S_s[ k*M+col ]
		final_path.insert(0,col+1)
		k -= 1
	final_path.insert(0,0)

	# find the number of mutations in each segment
	num_muts = np.zeros([len(final_path)-1, T], dtype=float_t)
	for i in range(len(final_path)-1):
		cur_num_muts = np.sum(array[ chrm_beg+final_path[i] : chrm_beg+final_path[i+1] ], axis=0)
		num_muts[i] = cur_num_muts

	# get the acutal bp positions of the segment boundaries
	bounds = []
	bounds.append(0)
	for i in range(len(final_path)-2):
		beg_pt = chrm_mut_pos[final_path[i]][1]
		end_pt = chrm_mut_pos[final_path[i+1]][0]
		bounds.append(median(beg_pt,end_pt))
	bounds.append(chrm_lens[chrm])
	bounds = np.array(bounds, dtype=np.uint32)

	return num_muts, bounds

	# # save files as csv
	# print(bounds)
	# np.savetxt("chrm_{}_bounds.csv".format(chrm+1), bounds, "%u", delimiter=",")
	# np.savetxt("chrm_{}_num_muts.csv".format(chrm+1), num_muts, "%f", delimiter=",")

def scores(E_f_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	num_muts = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_num_muts = sum(num_muts)
	Ks = [ round((num_muts[i]/total_num_muts)*total_K) for i in range(len(num_muts)) ]

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = num_muts[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]

	E_f_file = open(E_f_file_name, 'rb')
	E_f_bytes = E_f_file.read(8*K)

	initial_score = struct.unpack('d', E_f_bytes[:8])[0]
	final_score = struct.unpack('d', E_f_bytes[-8:])[0]

	total_Ts = np.zeros([len(chrm_begs), T], dtype=float_t)
	for i in range(len(chrm_begs) - 1):
		total_Ts[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ], axis=0)
	total_Ts[-1] = np.sum(array[ chrm_begs[-1] : ], axis=0)
	total_Ms = np.reshape(np.sum(total_Ts, axis=1), [len(chrm_begs), 1])
	term_threes = - np.sum( (total_Ts / total_Ms) * np.log((total_Ts / total_Ms)), axis=1 )

	return term_threes[chrm], term_threes[chrm] + final_score
	#print( "Entropy of T = {}".format(term_threes[chrm]) )
	#print( "Initial: score = {}, information = {}".format(initial_score, initial_score + term_threes[chrm]) )
	#print( "Optimal: score = {}, information = {}".format(final_score, final_score + term_threes[chrm]) )

def naive_traceback(mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	T = array.shape[1]
	bounds = [i*SEG_SIZE for i in range(chrm_lens[chrm] // SEG_SIZE)] + [chrm_lens[chrm]]
	num_muts = np.zeros([chrm_lens[chrm] // SEG_SIZE, T], dtype=float_t)

	chrm_beg = chrm_begs[chrm]
	chrm_mut_pos = mut_pos[chrm]
	cur_pos = 0
	for j in range(1, (chrm_lens[chrm] // SEG_SIZE) - 1):
		beg_pt = (j-1)*SEG_SIZE
		end_pt = j*SEG_SIZE
		while cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] < end_pt:
			num_muts[j-1] += array[ chrm_beg+cur_pos ]
			cur_pos += 1
		if cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] >= end_pt:
			if chrm_mut_pos[cur_pos][1] - end_pt < end_pt - chrm_mut_pos[cur_pos][0]:
				# put it in this segment
				num_muts[j-1] += array[ chrm_beg+cur_pos ]
				cur_pos += 1
			else:
				# put it in the next segment
				chrm_mut_pos[cur_pos][0] = chrm_mut_pos[cur_pos][1]
		# if not num_muts[j].any():
		# 	# can't have any segments with 0 mutations, set them to random small numbers
		# 	num_muts[j] = np.random.uniform(low=10*eps,high=100*eps,size=[T])
	if chrm == len(chrm_begs) - 1:
		num_muts[-1] = np.sum( array[ chrm_begs[chrm] + cur_pos : ], axis=0 )
	else:
		num_muts[-1] = np.sum( array[ chrm_begs[chrm] + cur_pos : chrm_begs[chrm+1] ], axis=0)

	return num_muts, bounds

def hier_preproc(seg_c, seg_b):

	total_seg_c = np.reshape( np.sum(seg_c, axis=0), [1,seg_c.shape[1]] )
	seg_r = seg_c / total_seg_c
	row_means = np.reshape(np.mean(seg_r, axis=1), [seg_r.shape[0], 1])
	row_stds = np.reshape(np.std(seg_r, axis=1), [seg_r.shape[0], 1])
	norm_seg_r = ( seg_r - row_means ) / row_stds
	for i in range(norm_seg_r.shape[0]):
		if np.isnan(norm_seg_r[i]).any():
			assert( np.isnan(norm_seg_r[i]).all() )
			norm_seg_r[i, :] = 0.
	clipped_b = [ bound // 1000000 for bound in seg_b[:-1] ]
	p_norm_seg_r = pd.DataFrame(norm_seg_r, index=clipped_b, columns=typs)
	return p_norm_seg_r

def plot(S_s_file_name, mc_data, total_K, chrm):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	mut_totals = [ mut_pos[i].shape[0] for i in range(len(mut_pos)) ]
	total_mut_totals = sum(mut_totals)
	Ks = [ round((mut_totals[i]/total_mut_totals)*total_K) for i in range(len(mut_totals)) ]
	naive_seg_c, naive_seg_b = naive_traceback(mc_data, total_K, chrm)
	opt_seg_c, opt_seg_b = traceback(S_s_file_name, mc_data, total_K, chrm)

	# important constants/arrays for the current chromosome
	M, K, T, chrm_beg = mut_totals[chrm], Ks[chrm], array.shape[1], chrm_begs[chrm]

	# # read in contents of S_s file
	# S_s_file = open(S_s_file_name, 'rb')
	# S_s_bytes = S_s_file.read(4*M*K)
	# S_s = []
	# for i in range(len(S_s_bytes) // 4):
	# 	number = S_s_bytes[4*i] + (16**2)*S_s_bytes[4*i+1] + (16**4)*S_s_bytes[4*i+2] + (16**6)*S_s_bytes[4*i+3]
	# 	S_s.append(number)

	# # get segmentation
	# final_path = []
	# final_path.insert(0,M)
	# k = K-1
	# col = M-1
	# while k > 0:
	# 	col = S_s[ k*M+col ]
	# 	final_path.insert(0,col+1)
	# 	k -= 1
	# final_path.insert(0,0)

	# seg_counts = np.zeros([len(final_path)-1,T], dtype=float_t)
	# for i in range(len(final_path)-1):
	# 	seg_counts[i] = np.sum(array[ chrm_beg+final_path[i] : chrm_beg+final_path[i+1] ], axis=0)
	
	sns.set(style="white")

	# hierarchical clustering (real data)

	opt_norm_seg_r = hier_preproc(opt_seg_c, opt_seg_b)

	g = sns.clustermap(opt_norm_seg_r, metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.fig.suptitle('Chromosome {} (Optimal Segmentation)'.format(chrm+1))
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('chromosome position (Mbp from the start)')
	plt.savefig("figures/chrm_{}_opt_hhmap.png".format(chrm+1))
	#plt.show()
	plt.clf()


	# hierachical clustering (naive)

	naive_norm_seg_r = hier_preproc(naive_seg_c, naive_seg_b)

	g = sns.clustermap(naive_norm_seg_r, metric="sqeuclidean", row_cluster=False, col_cluster=True)
	g.fig.suptitle('Chromosome {} (Naive Segmentation)'.format(chrm+1))
	g.ax_heatmap.set_xlabel('tumour type')
	g.ax_heatmap.set_ylabel('chromosome position (Mbp from the start)')
	plt.savefig("figures/chrm_{}_naive_hhmap.png".format(chrm+1))
	#plt.show()
	plt.clf()


	# mutation histogram

	g = sns.distplot(np.sum(opt_seg_c,axis=1), kde=False, bins=10) #, hist_kws={ "range": [0,myround(np.max(num_muts), 10000)] })
	g.set_title('Chromosome {} Segment Sizes (mutations)'.format(chrm+1))
	g.set_xlabel("mutation count")
	plt.savefig("figures/chrm_{}_hist_muts.png".format(chrm+1))
	#plt.show()
	plt.clf()

	# bp histogram

	opt_seg_b_sizes = [np.log10(opt_seg_b[i+1] - opt_seg_b[i]) for i in range(len(opt_seg_b[0:-1]))]
	ax = sns.distplot(opt_seg_b_sizes, kde=False, bins=50) #, hist_kws={ "range": [0,myround(np.max(num_muts), 10000)] })
	ax.set_title('Chromosome {} Segment Sizes (genomic DNA)'.format(chrm+1))
	ax.set_xlabel("log10 bp count")
	plt.savefig("figures/chrm_{}_hist_bp_log.png".format(chrm+1))
	#plt.show()
	plt.clf()

	# # correlation matrix
	# corr = np.corrcoef(seg_counts, rowvar=False)
	# print(np.min(corr))
	# print(np.max(corr))
	# p_corr = pd.DataFrame(corr, index=typs, columns=typs)
	# mask = np.zeros_like(p_corr, dtype=np.bool)
	# mask[np.triu_indices_from(mask)] = True
	# cmap = sns.diverging_palette(220, 10, as_cmap=True)
	# sns.heatmap(p_corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
	# plt.show()


def compare_everything():

	sexes = ["male","female","both"]
	sex_abbrevs = ["m", "f", "b"]
	total_K = 2865

	seg_scores = np.zeros([len(sexes), NUM_CHRMS, 5], dtype=float_t)
	total_seg_scores = np.zeros([len(sexes)], dtype=float_t)
	total_naive_seg_scores = np.zeros([len(sexes)], dtype=float_t)

	for s in range(len(sexes)):
		mc_data = np.load("data/mc_100_{}.npz".format(sex_abbrevs[s]))
		naive_data = np.load("results/naive_100_{}.npz".format(sex_abbrevs[s]) )
		naive_seg_scores = naive_data["seg_scores"]
		#print(naive_seg_scores[0])
		for i in range(NUM_CHRMS):
			E_f_file_name = "results/{}/E_f_chrm_{}.dat".format(sexes[s],i+1)
			seg_scores[s,i,0], seg_scores[s,i,1] = scores(E_f_file_name, mc_data, total_K, i)
			seg_scores[s,i,2] = np.sum(naive_seg_scores[i]) + seg_scores[s,i,0]
			seg_scores[s,i,3] = seg_scores[s,i,1] - seg_scores[s,i,2]
			seg_scores[s,i,4] = seg_scores[s,i,3] / seg_scores[s,i,1]
		print( "{}:".format(sexes[s]) )
		#print( "entropy of T | opt | naive | opt-naive | (opt-naive)/naive")
		#print( seg_scores[s] )

		array = mc_data["array"]
		chrm_begs = list(mc_data["chrm_begs"]) + [array.shape[0]]
		total_Ms = np.zeros([NUM_CHRMS], dtype=float_t)
		for i in range(NUM_CHRMS):
			total_Ms[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ])
		total_M = np.sum(total_Ms) # total_M == np.sum(array) (although there is some fp error)
		ratio_Ms = total_Ms / total_M

		total_seg_scores[s] = np.sum(ratio_Ms * seg_scores[s,:,1])
		total_naive_seg_scores[s] = np.sum(ratio_Ms * seg_scores[s,:,2] )
		print( "Total seg score = {}".format(total_seg_scores[s]) )
		print( "Total naive score = {}".format(total_naive_seg_scores[s]) )
		print( "% improvement = {}".format( 100*(total_seg_scores[s]-total_naive_seg_scores[s]) / total_naive_seg_scores[s] ) )

	# data_dict = {
	# 	"chromosome": 3*[i+1 for i in range(NUM_CHRMS)], 
	# 	"sex": NUM_CHRMS*["male"] + NUM_CHRMS*["female"] + NUM_CHRMS*["both"], 
	# 	"% improvement": list(100*seg_scores[0,:,-1]) + list(100*seg_scores[1,:,-1]) + list(100*seg_scores[2,:,-1]) 
	# }
	# p_data = pd.DataFrame(data_dict)
	# g = sns.factorplot(data=p_data, kind="bar", x="chromosome", y="% improvement", hue="sex")
	# g.fig.suptitle("% Improvement Over Naive Seg for Every Chromosome")
	# plt.show()
	# plt.clf()

		# xs = ["chrm " + str(i+1) for i in range(NUM_CHRMS)]
		# ys = pd.DataFrame(seg_scores[s,:,-1])
		# ax = sns.barplot(, )


if __name__ == "__main__":

	if ( len(sys.argv) != 6 and (len(sys.argv) == 2 and sys.argv[1] != "c") ):
		print("Error: Usage")
		quit()

	mode = sys.argv[1]
	if mode == "c":
		compare_everything()
		quit()
	mc_data_file_name = sys.argv[2]
	file_name = sys.argv[3] # could be S_s or E_f
	total_K = int(sys.argv[4])
	chrm = int(sys.argv[5])-1

	mc_data = np.load(mc_data_file_name)

	if mode == "s":
		traceback(file_name, mc_data, total_K, chrm)
	elif mode == "e":
		scores(file_name, mc_data, total_K, chrm)
	elif mode == "p":
		plot(file_name, mc_data, total_K, chrm)
	else:
		print("Error: Usage")
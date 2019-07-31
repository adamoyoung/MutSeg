"""
Important data structures for the other scripts in the project.
"""

import numpy as np
import pickle
import os

CHRM_LENS = [ 
	249250621, 
	243199373, 
	198022430, 
	191154276, 
	180915260, 
	171115067, 
	159138663, 
	146364022, 
	141213431, 
	135534747, 
	135006516, 
	133851895, 
	115169878, 
	107349540, 
	102531392, 
	90354753, 
	81195210, 
	78077248, 
	59128983, 
	63025520, 
	48129895, 
	51304566 
]

NUM_CHRMS = len(CHRM_LENS)

INT_T = np.uint32
FLOAT_T = np.float64 #np.float32
EPS = np.finfo(FLOAT_T).eps # machine epsilon, for entropy calculations

CFILE_BASE = "chrm"

ALL_SET = {'Biliary-AdenoCA', 'Bladder-TCC', 'Bone-Cart', 'Bone-Epith', 'Bone-Leiomyo', 'Bone-Osteosarc', 'Breast-AdenoCA', 'Breast-DCIS', 'Breast-LobularCA', 'CNS-GBM', 'CNS-Medullo', 'CNS-Oligo', 'CNS-PiloAstro', 'Cervix-AdenoCA', 'Cervix-SCC', 'ColoRect-AdenoCA', 'Eso-AdenoCA', 'Head-SCC', 'Kidney-ChRCC', 'Kidney-RCC', 'Liver-HCC', 'Lung-AdenoCA', 'Lung-SCC', 'Lymph', 'Lymph-NOS', 'Myeloid', 'Myeloid-MDS', 'Ovary-AdenoCA', 'Panc-AdenoCA', 'Panc-Endocrine', 'Prost-AdenoCA', 'Skin-Melanoma', 'Stomach-AdenoCA', 'Thy-AdenoCA', 'Uterus-AdenoCA'}
REDUCED_SET = {'Bone-Osteosarc', 'Breast-AdenoCA', 'ColoRect-AdenoCA', 'Eso-AdenoCA', 'Kidney-RCC', 'Liver-HCC', 'Ovary-AdenoCA', 'Panc-AdenoCA', 'Panc-Endocrine', 'Prost-AdenoCA', 'Skin-Melanoma'}
# removed from REDUCED_SET: 'CNS-Medullo',
SMALL_SET = {"Breast-AdenoCA", "CNS-Oligo", "CNS-PiloAstro", "Liver-HCC", "Ovary-AdenoCA", "Panc-Endocrine"}

def save_mc_data(mc_dir_path, chrms):
	assert len(chrms) == NUM_CHRMS
	os.makedirs(mc_dir_path, exist_ok=True)
	for chrm_id, chrm in enumerate(chrms):
		mc_file_path = os.path.join(mc_dir_path,"{}_{}.pkl".format(CFILE_BASE,chrm_id))
		with open(mc_file_path, "wb") as pkl_file:
			pickle.dump(chrm, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

def load_mc_data(mc_dir_path):
	chrms = []
	assert os.path.isdir(mc_dir_path)
	for chrm_id in range(NUM_CHRMS):
		mc_file_path = os.path.join(mc_dir_path,"{}_{}.pkl".format(CFILE_BASE,chrm_id))
		with open(mc_file_path, "rb") as pkl_file:
			chrm = pickle.load(pkl_file)
			chrms.append(chrm)
	return chrms

# Maps labels to meaning according to roadmap chromatin state annotations for the 18 State model
class TF_MAP_18():
    MAP = {
        "1_TssA": "Active TSS",
        "2_TssFlnk": "Flanking Active TSS",
        "3_TssFlnkU": "Flanking Active TSS Upstream",
        "4_TssFlnkD": "Flanking Active TSS DownStream",
        "5_Tx": "Strong Transcription",
        "6_TxWk": "Weak Transcription",
        "7_EnhG1": "Genic enhancer1",
        "8_EnhG2": "Genic enhancer2",
        "9_EnhA1": "Active enhancer1",
        "10_EnhA2": "Active enhancer2",
        "11_EnhWk": "Weak Enhancer",
        "12_ZNF/Rpts": "ZNF genes and repeats",
        "13_Het": "Heterochromatin",
        "14_TssBiv": "Bivalent-Poised TSS",
        "15_EnhBiv": "Bivalent Enhancer",
        "11_BivFlnk": "Flaking Bivalent TSS-Enh",
        "16_ReprPC": "Repressed PolyComb",
        "17_ReprPCWk": "Weak Repressed PolyComb",
        "18_Quies": "Quiescent-Low",
    }

class Segmentation:
	"""basically just a struct for segmentation-related data"""

	def __init__(self, type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds):
		self.type_to_idx = type_to_idx
		self.num_segs = num_segs
		self.mut_ints = seg_mut_ints # 2 x M x T
		self.mut_bounds = seg_mut_bounds
		self.bp_bounds = seg_bp_bounds

	def get_mut_ints(self):
		raise NotImplementedError

	def get_mut_bounds(self):
		raise NotImplementedError

	def get_seg_bp_bounds(self):
		raise NotImplementedError

	def get_num_segs(self):
		raise NotImplementedError

	def _interpret_mode(self, mode):
		if mode == "counts" or mode == "tumour_freqs":
			return 0
		elif mode == "sample_freqs":
			return 1
		else:
			raise ValueError("invalid mode")

	def _convert_cos_to_freqs(self, mut_ints_co):
		totals = np.sum(mut_ints_co, axis=0)
		return mut_ints_co / totals[np.newaxis, ...]


class NaiveSegmentation(Segmentation):

	def __init__(self, type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, nz_seg_idx, num_nz_segs):
		super(NaiveSegmentation, self).__init__(type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds)
		self.nz_seg_idx = nz_seg_idx # only used for naive
		self.num_nz_segs = num_nz_segs # only used for naive

	def get_mut_ints(self, drop_zeros, ana_mode, tumour_list):
		mode = self._interpret_mode(ana_mode)
		tumour_idx = np.zeros([len(tumour_list)], dtype=INT_T)
		for t in range(len(tumour_list)):
			tumour_idx[t] = self.type_to_idx[tumour_list[t]]
		if drop_zeros:
			nz_mut_ints = self.mut_ints[mode][self.nz_seg_idx][:,tumour_idx]
			mut_ints = nz_mut_ints
		else:
			mut_ints = self.mut_ints[mode][:][:,tumour_idx]
		if ana_mode == "tumour_freqs":
			mut_ints = self._convert_cos_to_freqs(mut_ints)
		return mut_ints

	def get_mut_bounds(self, drop_zeros):
		if drop_zeros:
			nz_mut_bounds = np.zeros([self.num_nz_segs+1], dtype=self.mut_bounds.dtype)
			for i in range(self.num_nz_segs):
				nz_mut_bounds[i+1] = self.mut_bounds[self.nz_seg_idx[i]+1]
			return nz_mut_bounds
		else:
			return self.mut_bounds

	def get_bp_bounds(self, drop_zeros):
		if drop_zeros:
			raise NotImplementedError
		# 	nz_bp_bounds = np.zeros([self.num_nz_segs+1], dtype=self.bp_bounds.dtype)
		# 	for i in range(self.num_nz_segs):
		# 		nz_bp_bounds[i+1] = self.bp_bounds[self.nz_seg_idx[i]+1]
		# 	return nz_bp_bounds
		else:
			return self.bp_bounds

	def get_num_segs(self, drop_zeros):
		if drop_zeros:
			return self.num_nz_segs
		else:
			return self.num_segs


class OptimalSegmentation(Segmentation):

	def __init__(self, type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, final_score, group_by):
		super(OptimalSegmentation, self).__init__(type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds)
		self.final_score = final_score # might not always be there
		self.group_by = group_by # might not always be there

	def get_mut_ints(self, ana_mode, tumour_list):
		mode = self._interpret_mode(ana_mode)
		tumour_idx = np.zeros([len(tumour_list)], dtype=INT_T)
		for t in range(len(tumour_list)):
			tumour_idx[t] = self.type_to_idx[tumour_list[t]]
		mut_ints = self.mut_ints[mode][:][:,tumour_idx]
		if ana_mode == "tumour_freqs":
			mut_ints = self._convert_cos_to_freqs(mut_ints)
		return mut_ints

	def get_mut_bounds(self):
		return self.mut_bounds

	def get_bp_bounds(self):
		return self.bp_bounds

	def get_num_segs(self):
		return self.num_segs


class NaiveSigSegmentation(NaiveSegmentation):

	def __init__(self, type_to_idx, num_segs, seg_mut_ints, seg_bp_bounds, nz_seg_idx, num_nz_segs):
		super(NaiveSigSegmentation, self).__init__(type_to_idx, num_segs, seg_mut_ints, None, seg_bp_bounds, nz_seg_idx, num_nz_segs)

	def get_mut_ints(self, drop_zeros, ana_mode, sig_list):
		""" mut ints is only M x T, not 2 x M x T"""
		assert ana_mode != "sample_freqs"
		sig_idx = np.zeros([len(sig_list)], dtype=INT_T)
		for s in range(len(sig_list)):
			sig_idx[s] = self.type_to_idx[sig_list[s]]
		if drop_zeros:
			nz_mut_ints = self.mut_ints[self.nz_seg_idx][:,tumour_idx]
			mut_ints = nz_mut_ints
		else:
			mut_ints = self.mut_ints[:][:,sig_idx]
		if ana_mode == "tumour_freqs":
			mut_ints = self._convert_cos_to_freqs(mut_ints)
		return mut_ints


class OptimalSigSegmentation(OptimalSegmentation):

	def __init__(self, type_to_idx, num_segs, seg_mut_ints, seg_bp_bounds):
		super(OptimalSigSegmentation, self).__init__(type_to_idx, num_segs, seg_mut_ints, None, seg_bp_bounds, None, None)

	def get_mut_ints(self, ana_mode, sig_list):
		assert ana_mode != "sample_freqs"
		sig_idx = np.zeros([len(sig_list)], dtype=INT_T)
		for s in range(len(sig_list)):
			tumour_idx[s] = self.type_to_idx[tumour_list[s]]
		mut_ints = self.mut_ints[:][:,sig_idx]
		if ana_mode == "tumour_freqs":
			mut_ints = self._convert_cos_to_freqs(mut_ints)
		return mut_ints

	def get_mut_bounds(self):
		raise NotImplementedError


class Chromosome:

	def __init__(self,chrm_id):
		assert chrm_id in range(NUM_CHRMS)
		self.chrm_id = chrm_id # int, starts at 1
		self.length = CHRM_LENS[chrm_id] # int
		self.tumour_types = None # set
		self.type_to_idx = None # cancer type to index in mut_array
		self.unique_pos_count = None
		self.mut_array = None # numpy array
		self.mut_pos = None # numpy array
		self.pos_to_idx = None # numpy array
		self.group_by = None # int
		self.opt_segmentations = {} # dict of optimal segmentation objects, indexed by num_segs
		self.naive_segmentations = {} # dict of naive segmentation objects, indexed by naive_seg_size
		# self.tumour_totals = None
		self.valid_frac = None

	def get_chrm_len(self):
		return self.length

	def get_chrm_id(self):
		return self.chrm_id

	def get_chrm_num(self):
		return self.chrm_id + 1

	def get_unique_pos_count(self):
		if self.group_by:
			return self.unique_pos_count_g
		else:
			return self.unique_pos_count

	def get_num_tumour_types(self):
		return len(self.tumour_types)

	def _get_num_segs(self, naive_seg_size):
		""" helper function, does not consider zero segments """
		num_segs = self.get_chrm_len() // naive_seg_size
		if self.get_chrm_len() % naive_seg_size > 0:
			num_segs += 1
		return num_segs

	def get_num_segs(self, naive_seg_size, drop_zeros):
		""" assumes that the naive segmentation has been computed already if drop_zeros is true """
		# num_segs = self._get_num_segs(naive_seg_size)
		naive_seg = self.get_naive_seg(naive_seg_size)
		return naive_seg.get_num_segs(drop_zeros)

	def set_mut_arrays(self, typ_set, mut_pos_set, valid_frac):
		""" """
		self.tumour_types = typ_set
		self.type_to_idx = {}
		for idx, typ in enumerate(sorted(self.tumour_types)):
			self.type_to_idx[typ] = idx
		sorted_set = sorted(mut_pos_set)
		self.unique_pos_count = len(sorted_set)
		self.valid_frac = valid_frac
		if self.valid_frac > 0.:
			# train/test split
			self.mut_array = np.zeros( [2, 2, self.unique_pos_count, len(self.tumour_types)], dtype=FLOAT_T )
		else:
			self.mut_array = np.zeros( [1, 2, self.unique_pos_count, len(self.tumour_types)], dtype=FLOAT_T )
		self.mut_pos = np.array( sorted_set, dtype=INT_T )
		self.pos_to_idx = {}
		for i in range(self.unique_pos_count):
			self.pos_to_idx[sorted_set[i]] = i

	def get_mut_array(self, tv_split):
		if self.group_by:
			mut_array = self.mut_array_g
		else:
			mut_array = self.mut_array
		if self.valid_frac == 0.:
			assert tv_split == "all"
			return mut_array[0]
		else:
			if tv_split == "all":
				return np.sum(mut_array, axis=0)
			elif tv_split == "train":
				assert self.valid_frac > 0.
				return mut_array[0]
			else: # tv_split == "valid"
				assert self.valid_frac > 0.
				return mut_array[1]

	def get_mut_pos(self):
		if self.group_by:
			mut_pos = self.mut_pos_g
		else:
			mut_pos = self.mut_pos
		return mut_pos

	def _interpret_mode(self, mode):
		if mode == "counts" or mode == "tumour_freqs":
			return 0
		elif mode == "sample_freqs":
			return 1
		else:
			raise ValueError("invalid mode")
		

	# def update(self, pos, typ, ints):
	# 	self.mut_array[self.pos_to_idx[pos]][self.type_to_idx[typ]] += ints

	def update(self, dfs, valid_idx):
		"""  """
		# # assign dfs to train or valid
		# num_valid_dfs = int(np.round(self.valid_frac*len(dfs)))
		# num_train_dfs = len(dfs) - num_valid_dfs
		# df_sizes = np.array([df.shape[0] for df in dfs])
		# total = np.sum(df_sizes)		
		# sorted_dfs = np.argsort(df_sizes)
		# valid_idx, train_idx = [], []
		# cur_valid_total, cur_train_total = 0, 0
		# valid_total = int(np.round(self.valid_frac*total))
		# train_total = total - valid_total_frac
		# for d in range(sorted_dfs):
		# 	if cur_valid_total <= valid_total_frac

		tumour_types_list = sorted(self.tumour_types)
		if valid_idx:
			assert self.valid_frac > 0.
		for df_idx, df in enumerate(dfs):
			sample_ints = df["ints"].sum()
			typ = df["typ"].iloc[0]
			split_idx = 0
			if self.valid_frac > 0. and df_idx in valid_idx[tumour_types_list.index(typ)]:
				split_idx = 1
			# compute counts/freqs
			cdf = df[df["chrm"] == self.chrm_id]
			pos = cdf["pos"].to_numpy()
			ints = cdf["ints"].to_numpy()
			sample_freqs = ints / sample_ints
			for i in range(cdf.shape[0]):
				self.mut_array[split_idx][0][self.pos_to_idx[pos[i]]][self.type_to_idx[typ]] += ints[i]
				self.mut_array[split_idx][1][self.pos_to_idx[pos[i]]][self.type_to_idx[typ]] += sample_freqs[i]
		# self.tumour_totals = np.sum(self.mut_array[0], axis=0)[np.newaxis, ...]
		assert np.sum(self.mut_array, axis=(0,3))[0].all()
		assert np.sum(self.mut_array, axis=(0,3))[1].all()

	def group(self, group_by):
		""" 
		remainder mutations are discarded 
		mut_pos_g is has inclusive boundaries
		"""
		self.group_by = group_by
		self.unique_pos_count_g = self.unique_pos_count // group_by
		split_dim = 1
		if self.valid_frac > 0.:
			split_dim += 1
		self.mut_pos_g = np.zeros( [self.unique_pos_count_g, 2], dtype=INT_T )
		self.mut_array_g = np.zeros( [split_dim, 2, self.unique_pos_count_g, len(self.tumour_types)], dtype=FLOAT_T )
		for i in range(self.unique_pos_count_g):
			self.mut_array_g[0,:,i] = np.sum(self.mut_array[0,:,(i*group_by):((i+1)*group_by)], axis=1)
			if self.valid_frac > 0.:
				self.mut_array_g[1,:,i] = np.sum(self.mut_array[1,:,(i*group_by):((i+1)*group_by)], axis=1)
			self.mut_pos_g[i][0] = self.mut_pos[i*group_by]
			self.mut_pos_g[i][1] = self.mut_pos[(i+1)*group_by-1]
		assert np.sum(self.mut_array_g, axis=(0,3))[0].all()
		assert np.sum(self.mut_array_g, axis=(0,3))[1].all()

	def mut_array_to_bytes(self, mode, tumour_list, tv_split):
		itemsize = np.dtype(FLOAT_T).itemsize
		mode_idx = self._interpret_mode(mode)
		mut_array = self.get_mut_array(tv_split)[mode_idx]
		assert len(mut_array.shape) == 2
		# assert mut_array.shape[1] <= len(tumour_list)
		tumour_idx = np.zeros([len(tumour_list)], dtype=INT_T)
		for t in range(len(tumour_list)):
			tumour_idx[t] = self.type_to_idx[tumour_list[t]]
		mut_array = mut_array[:,tumour_idx]
		if mode == "tumour_freqs":
			mut_array = mut_array / np.sum(mut_array, axis=0)[np.newaxis,...]
		barray = bytes(mut_array)
		if self.group_by:
			assert( len(barray) == self.unique_pos_count_g*len(tumour_list)*itemsize )
		else:
			assert( len(barray) == self.unique_pos_count*len(tumour_list)*itemsize )
		return barray

	def add_opt_seg(self, num_segs, segmentation):
		# only allows one segmentation per num_segs
		self.opt_segmentations[num_segs] = segmentation

	def get_opt_seg(self, num_segs):
		return self.opt_segmentations[num_segs]

	def get_default_naive_bp_bounds(self, naive_seg_size):
		num_segs = self._get_num_segs(naive_seg_size)
		bp_bounds = np.zeros([num_segs+1], dtype=INT_T)
		# set bp bounds
		for k in range(num_segs):
			bp_bounds[k] = k*naive_seg_size
		bp_bounds[-1] = self.get_chrm_len()
		return bp_bounds

	def get_naive_seg(self, naive_seg_size, tv_split="all"):
		""" computes the naive segmentation (with and without zero segments), 
		or fetches it if it's already been computed """
		if naive_seg_size in self.naive_segmentations:
			return self.naive_segmentations[naive_seg_size]
		# get necessary constants and arrays
		num_segs = self._get_num_segs(naive_seg_size)
		T = self.get_num_tumour_types()
		# get the right split
		if tv_split == "all":
			mut_array = np.sum(self.mut_array, axis=0) # not mut_array_g
		elif tv_split == "train":
			mut_array = self.mut_array[0]
		else: # tv_split == "valid"
			mut_array = self.mut_array[1]
		mut_pos = self.mut_pos # not mut_pos_g
		assert mut_array.shape[1] == mut_pos.shape[0]
		max_mut_idx = mut_array.shape[1]
		# remove the last few mutations that were not included in the grouping operation
		if self.group_by:
			max_mut_idx = (max_mut_idx // self.group_by) * self.group_by
		# set up new arrays
		seg_mut_ints = np.zeros([2,num_segs, T], dtype=FLOAT_T)
		seg_mut_bounds = np.zeros([num_segs+1], dtype=INT_T)
		seg_bp_bounds = self.get_default_naive_bp_bounds(naive_seg_size)
		# compute seg_mut_bounds and seg_mut_ints from seg_bp_bounds
		seg_mut_bounds[0] = 0
		cur_idx = 0
		for k in range(num_segs):
			prev_idx = cur_idx
			end_pt = seg_bp_bounds[k+1]
			while cur_idx < max_mut_idx and mut_pos[cur_idx] < end_pt:
				cur_idx += 1
			if prev_idx != cur_idx:
				seg_mut_ints[:,k] = np.sum(mut_array[:, prev_idx:cur_idx], axis=1)
			else: # prev_idx == cur_idx
				# there are no mutations in this segment
				pass
			seg_mut_bounds[k+1] = cur_idx
		total_seg_mut_ints = np.sum(seg_mut_ints,axis=(1,2))
		total_mut_array = np.sum(mut_array[:,:max_mut_idx],axis=(1,2))
		assert np.all(np.isclose(total_seg_mut_ints, total_mut_array, atol=0.1)),  "chrm {}: {} vs {}".format(self.chrm_id, total_seg_mut_ints, total_mut_array)
		nz_seg_idx = np.nonzero(np.sum(seg_mut_ints[0], axis=1))[0]
		assert np.all(nz_seg_idx == np.nonzero(np.sum(seg_mut_ints[1], axis=1))[0])
		num_nz_segs = len(nz_seg_idx)
		assert num_nz_segs <= num_segs
		assert num_nz_segs > 0
		naive_seg = NaiveSegmentation(self.type_to_idx, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, nz_seg_idx, num_nz_segs)
		self.naive_segmentations[naive_seg_size] = naive_seg
		return naive_seg

	def delete_non_ana_data(self):
		del self.mut_array
		del self.mut_pos
		if self.group_by:
			del self.mut_array_g
			del self.mut_pos_g

	def delete_non_run_data(self):
		# del self.unique_pos_count
		del self.mut_array
		del self.mut_pos
		del self.pos_to_idx
		if self.group_by:
			# del self.unique_pos_count_g
			del self.mut_array_g
			del self.mut_pos_g
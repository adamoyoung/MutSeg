"""
Important data structures for the oother scripts in the project.
"""

import numpy as np

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
FLOAT_T = np.float64
EPS = np.finfo(FLOAT_T).eps # machine epsilon, for entropy calculations

CFILE_BASE = "all_chrm"


class Segmentation:
	"""basically just a struct for segmentation-related data"""

	def __init__(self, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds):
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

	def _interpret_ana_mode(self, ana_mode):
		if ana_mode == "sample_freqs":
			return 1
		elif ana_mode == "counts":
			return 0
		else:
			raise ValueError("invalid ana_mode")


class NaiveSegmentation(Segmentation):

	def __init__(self, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, nz_seg_idx, num_nz_segs):
		super(NaiveSegmentation, self).__init__(num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds)
		self.nz_seg_idx = nz_seg_idx # only used for naive
		self.num_nz_segs = num_nz_segs # only used for naive

	def get_mut_ints(self, drop_zeros, ana_mode):
		mode = self._interpret_ana_mode(ana_mode)
		if drop_zeros:
			nz_mut_ints = self.mut_ints[mode][self.nz_seg_idx]
			return nz_mut_ints
		else:
			return self.mut_ints[mode]

	def get_mut_bounds(self, drop_zeros):
		if drop_zeros:
			nz_mut_bounds = np.zeros([self.num_nz_segs+1], dtype=self.mut_bounds.dtype)
			for i in range(self.num_nz_segs):
				nz_mut_bounds[i+1] = self.mut_bounds[self.nz_seg_idx[i]+1]
			return nz_mut_bounds
		else:
			return self.mut_bounds

	def get_num_segs(self, drop_zeros):
		if drop_zeros:
			return self.num_nz_segs
		else:
			return self.num_segs


class OptimalSegmentation(Segmentation):

	def __init__(self, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, final_score, group_by):
		super(OptimalSegmentation, self).__init__(num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds)
		self.final_score = final_score
		self.group_by = group_by

	def get_mut_ints(self, ana_mode):
		mode = self._interpret_ana_mode(ana_mode)
		return self.mut_ints[mode]

	def get_mut_bounds(self):
		return self.mut_bounds

	def get_bp_bounds(self):
		return self.bp_bounds

	def get_num_segs(self):
		return self.num_segs


class Chromosome:

	def __init__(self,chrm_id,cancer_types):
		assert chrm_id in range(NUM_CHRMS)
		self.chrm_id = chrm_id # int, starts at 1
		self.length = CHRM_LENS[chrm_id] # int
		self.cancer_types = cancer_types # set
		self.type_to_idx = {} # cancer type to index in mut_array
		for idx, typ in enumerate(sorted(self.cancer_types)):
			self.type_to_idx[typ] = idx
		self.mut_array = None # numpy array
		self.mut_pos = None # numpy array
		self.pos_to_idx = None # numpy array
		self.group_by = None # int
		self.opt_segmentations = {} # dict of optimal segmentation objects, indexed by num_segs
		self.naive_segmentations = {} # dict of naive segmentation objects, indexed by naive_seg_size

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

	def get_num_cancer_types(self):
		return len(self.cancer_types)

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

	def set_mut_arrays(self, mut_pos_set):
		""" """
		sorted_set = sorted(mut_pos_set)
		self.unique_pos_count = len(sorted_set)
		self.mut_array = np.zeros( [2, self.unique_pos_count, len(self.cancer_types)], dtype=FLOAT_T )
		self.mut_pos = np.array( sorted_set, dtype=INT_T )
		self.pos_to_idx = {}
		for i in range(self.unique_pos_count):
			self.pos_to_idx[sorted_set[i]] = i

	def get_mut_array(self):
		# _mode = self.interpret_mode(mode)
		if self.group_by:
			return self.mut_array_g
		else:
			return self.mut_array

	def get_mut_pos(self):
		# _mode = self.interpret_mode(mode)
		if self.group_by:
			return self.mut_pos_g
		else:
			return self.mut_pos

	# def interpret_mode(self, mode):
	# 	if mode == "seg":
	# 		return self.seg_mode
	# 	elif mode == "ana":
	# 		return self.ana_mode
	# 	else:
	# 		raise ValueError

	# def update(self, pos, typ, ints):
	# 	self.mut_array[self.pos_to_idx[pos]][self.type_to_idx[typ]] += ints

	def update(self, dfs):
		for d in range(len(dfs)):
			df = dfs[d]
			sample_ints = df["ints"].sum()
			df = df[df["chrm"] == self.chrm_id]
			pos = df["pos"].to_numpy()
			typ = df["typ"].to_numpy()
			ints = df["ints"].to_numpy()
			freqs = ints / sample_ints
			s_1 = df["ints"].sum()
			s_2 = np.sum(df["ints"].to_numpy())
			s_3 = np.sum(df["ints"].values)
			for i in range(df.shape[0]):
				self.mut_array[0][self.pos_to_idx[pos[i]]][self.type_to_idx[typ[i]]] += ints[i]
				self.mut_array[1][self.pos_to_idx[pos[i]]][self.type_to_idx[typ[i]]] += freqs[i]
		assert np.sum(self.mut_array[0], axis=1).all()
		assert np.sum(self.mut_array[1], axis=1).all()

	def group(self, group_by):
		""" 
		remainder mutations are discarded 
		mut_pos_g is has inclusive boundaries
		"""
		self.group_by = group_by
		self.unique_pos_count_g = self.unique_pos_count // group_by
		self.mut_pos_g = np.zeros( [self.unique_pos_count_g, 2], dtype=INT_T )
		self.mut_array_g = np.zeros( [2, self.unique_pos_count_g, len(self.cancer_types)], dtype=FLOAT_T )
		for i in range(self.unique_pos_count_g):
			self.mut_array_g[:,i] = np.sum(self.mut_array[:,(i*group_by):((i+1)*group_by)], axis=1)
			self.mut_pos_g[i][0] = self.mut_pos[i*group_by]
			self.mut_pos_g[i][1] = self.mut_pos[(i+1)*group_by-1]
		assert np.sum(self.mut_array_g[0], axis=1).all()
		assert np.sum(self.mut_array_g[1], axis=1).all()

	def mut_array_to_bytes(self):
		itemsize = np.dtype(FLOAT_T).itemsize
		if self.group_by:
			barray = bytes(self.mut_array_g[self.mode])
			assert( len(barray) == self.unique_pos_count_g*len(self.cancer_types)*itemsize )
		else:
			barray = bytes(self.mut_array[self.mode])
			assert( len(barray) == self.unique_pos_count*len(self.cancer_types)*itemsize )
		return barray

	def add_opt_seg(self, num_segs, segmentation):
		# only allows one segmentation per num_segs
		self.opt_segmentations[num_segs] = segmentation

	def get_opt_seg(self, num_segs):
		return self.opt_segmentations[num_segs]

	def get_naive_seg(self, naive_seg_size):
		""" computes the naive segmentation (with and without zero segments), 
		or fetches it if it's already been computed """
		if naive_seg_size in self.naive_segmentations:
			return self.naive_segmentations[naive_seg_size]
		# get necessary constants and arrays
		num_segs = self._get_num_segs(naive_seg_size)
		T = self.get_num_cancer_types()
		mut_array = self.mut_array # not mut_array_g
		mut_pos = self.mut_pos # not mut_pos_g
		assert mut_array.shape[1] == mut_pos.shape[0]
		max_mut_idx = mut_array.shape[1]
		# remove the last few mutations that were not included in the grouping operation
		if self.group_by:
			max_mut_idx = (max_mut_idx // self.group_by) * self.group_by
		# set up new arrays
		seg_mut_ints = np.zeros([2,num_segs, T], dtype=FLOAT_T)
		seg_mut_bounds = np.zeros([num_segs+1], dtype=INT_T)
		seg_bp_bounds = np.zeros([num_segs+1], dtype=INT_T)
		# set bp bounds
		for k in range(num_segs):
			seg_bp_bounds[k] = k*naive_seg_size
		seg_bp_bounds[-1] = self.get_chrm_len()
		# compute seg_mut_bounds and seg_mut_ints from seg_bp_bounds
		# # saddle_mut_count should be very low, but it's not currently checked
		# saddle_mut_count = 0
		seg_mut_bounds[0] = 0
		cur_idx = 0
		for k in range(num_segs):
			prev_idx = cur_idx
			end_pt = seg_bp_bounds[k+1]
			# if self.group_by:
			# 	while cur_idx < max_mut_idx and mut_pos[cur_idx][1] < end_pt:
			# 		cur_idx += 1
			# 	if cur_idx < max_mut_idx and mut_pos[cur_idx][0] < end_pt and mut_pos[cur_idx][1] >= end_pt:
			# 		# mutation is saddling boundaries
			# 		saddle_mut_count += 1
			# 		if end_pt - mut_pos[cur_idx][0] > mut_pos[cur_idx][1] - end_pt + 1:
			# 			# mutation overlaps more with current segment, add it to the current seg_mut_ints
			# 			cur_idx += 1
			# 	if prev_idx != cur_idx:
			# 		seg_mut_ints[k] = np.sum(mut_array[prev_idx:cur_idx], axis=0)
			# 	else: # prev_idx == cur_idx
			# 		# there are no mutations in this segment
			# 		pass
			# 	seg_mut_bounds[k+1] = cur_idx
			# else:
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
		naive_seg = NaiveSegmentation(num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, nz_seg_idx, num_nz_segs)
		self.naive_segmentations[naive_seg_size] = naive_seg
		return naive_seg

	def delete_non_ana_data(self):
		if self.group_by:
			del self.mut_array
			del self.mut_pos

	# def get_naive_seg_mut_ints(self, naive_seg_size, drop_zeros):

	# 	naive_seg = self.get_naive_seg(naive_seg_size)
	# 	naive_seg_mut_ints = naive_seg.seg_mut_ints
	# 	if drop_zeros:
	# 		naive_seg_mut_ints = naive_seg_mut_ints[naive_seg.nz_seg_idx]
	# 	assert naive_seg_mut_ints.shape[0] == self.get_num_segs(naive_seg_size, drop_zeros)
	# 	return naive_seg_mut_ints

	# def get_naive_nz_seg_mut_ints(self, naive_seg_size):

	# 	naive_seg = self.get_naive_seg(naive_seg_size)
	# 	num_segs = self.get_num_segs(naive_seg_size)
	# 	nz_seg_idx = np.nonzero(np.sum(naive_seg.seg_mut_ints, axis=1))[0]
	# 	naive_num_nz_segs = len(nz_seg_idx)
	# 	assert naive_num_nz_segs <= num_segs
	# 	assert naive_num_nz_segs > 0
	# 	return naive_seg.seg_mut_ints[nz_seg_idx]
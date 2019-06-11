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

	def __init__(self, num_segs, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, final_score):

		self.num_segs = num_segs
		self.seg_mut_ints = seg_mut_ints
		self.seg_mut_bounds = seg_mut_bounds
		self.seg_bp_bounds = seg_bp_bounds
		self.final_score = final_score


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
		self.segmentations = {} # dict of segmentation objects

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

	def get_num_segs(self, naive_seg_size):
		num_segs = self.get_chrm_len() // naive_seg_size
		if self.get_chrm_len() % naive_seg_size > 0:
			num_segs += 1
		return num_segs

	def set_mut_arrays(self, mut_pos_set):
		""" """
		sorted_set = sorted(mut_pos_set)
		self.unique_pos_count = len(sorted_set)
		self.mut_array = np.zeros( [self.unique_pos_count, len(self.cancer_types)], dtype=FLOAT_T )
		self.mut_pos = np.array( sorted_set, dtype=INT_T )
		self.pos_to_idx = {}
		for i in range(self.unique_pos_count):
			self.pos_to_idx[sorted_set[i]] = i

	def get_mut_array(self):
		if self.group_by:
			return self.mut_array_g
		else:
			return self.mut_array

	def get_mut_pos(self):
		if self.group_by:
			return self.mut_pos_g
		else:
			return self.mut_pos

	# def update(self, pos, typ, ints):
	# 	self.mut_array[self.pos_to_idx[pos]][self.type_to_idx[typ]] += ints

	def update(self, dfs):
		for df in dfs:
			_df = df[df["chrm"] == self.chrm_id]
			_pos = _df["pos"].values
			_typ = _df["typ"].values
			_ints = _df["ints"].values
			for i in range(_df.shape[0]):
				self.mut_array[self.pos_to_idx[_pos[i]]][self.type_to_idx[_typ[i]]] += _ints[i]
		assert np.sum(self.mut_array, axis=1).all()

	def group(self, group_by):
		""" 
		remainder mutations are discarded 
		mut_pos_g is has inclusive boundaries
		"""
		self.group_by = group_by
		self.unique_pos_count_g = self.unique_pos_count // group_by
		self.mut_pos_g = np.zeros( [self.unique_pos_count_g, 2], dtype=INT_T )
		self.mut_array_g = np.zeros( [self.unique_pos_count_g, len(self.cancer_types)], dtype=FLOAT_T )
		for i in range(self.unique_pos_count_g):
			self.mut_array_g[i] = np.sum(self.mut_array[(i*group_by):((i+1)*group_by)], axis=0)
			self.mut_pos_g[i][0] = self.mut_pos[i*group_by]
			self.mut_pos_g[i][1] = self.mut_pos[(i+1)*group_by-1]
		assert np.sum(self.mut_array_g, axis=1).all()

	def mut_array_to_bytes(self):
		itemsize = np.dtype(FLOAT_T).itemsize
		if self.group_by:
			barray = bytes(self.mut_array_g)
			assert( len(barray) == self.unique_pos_count_g*len(self.cancer_types)*itemsize )
		else:
			barray = bytes(self.mut_array)
			assert( len(barray) == self.unique_pos_count*len(self.cancer_types)*itemsize )
		return barray

	def add_seg(self, num_segs, segmentation):
		# only allows one segmentation per num_segs
		self.segmentations[num_segs] = segmentation

	def get_seg(self, naive_seg_size):
		num_segs = self.get_num_segs(naive_seg_size)
		return self.segmentations[num_segs]

	def get_naive_seg_arrays(self, naive_seg_size):
		# get necessary constants and arrays
		T = self.get_num_cancer_types()
		num_segs = self.get_num_segs(naive_seg_size)
		mut_array = self.mut_array # not mut_array_g
		mut_pos = self.mut_pos # not mut_pos_g
		assert mut_array.shape[0] == mut_pos.shape[0]
		# remove the last few mutations that were not included in the grouping operation
		max_mut_idx = mut_array.shape[0]
		if self.group_by:
			max_mut_idx = (self.mut_array.shape[0] // self.group_by) * self.group_by
		# set up new arrays
		seg_mut_ints = np.zeros([num_segs, T], dtype=FLOAT_T)
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
				seg_mut_ints[k] = np.sum(mut_array[prev_idx:cur_idx], axis=0)
			else: # prev_idx == cur_idx
				# there are no mutations in this segment
				pass
			seg_mut_bounds[k+1] = cur_idx
		total_seg_mut_ints = np.sum(seg_mut_ints)
		total_mut_array = np.sum(mut_array)
		assert np.isclose(total_seg_mut_ints, total_mut_array, atol=0.1),  "{} vs {}, sm = {}".format(total_seg_mut_ints, total_mut_array, saddle_mut_count)
		return Segmentation(naive_seg_size, seg_mut_ints, seg_mut_bounds, seg_bp_bounds, None)
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

CFILE_BASE = "all_chrm"

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

	def get_num_naive_segs(self, seg_size):
		num_segs = self.get_chrm_len() // seg_size
		if self.get_chrm_len() % seg_size > 0:
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
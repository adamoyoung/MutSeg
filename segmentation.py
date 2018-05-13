import sys
import numpy as np
from k_seg_hdf5 import compute_counts, score, k_seg

def find_segmentation(mut_file_path="mutation_counts.npz", output_file_path="seg_output.npz", K=1000, min_size=100, seed=[], data_file_name="k_seg_data"):

	mut_array = np.load(mut_file_path)["array"]

	score, segs = k_seg(mut_array,K,min_size,seed,data_file_name)

	print(score)
	print(segs)

if __name__ == "__main__":

	find_segmentation()
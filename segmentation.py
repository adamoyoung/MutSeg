import sys
import numpy as np
from k_seg import compute_counts, score, k_seg

def find_segmentation(mut_file_path="mutations.npy", pos_file_path="positions.npy", output_file_path="seg_output.npz", K=3000, min_size=100):

	mut_array = np.load(mut_file_path)

	score, segs = k_seg(mut_array,[],K,min_size)

	print(score)
	print(segments)

	np.savez(output_file_path, score=score, segs=segments)

if __name__ == "__main__":

	find_segmentation()
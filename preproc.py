import sys
import numpy as np
import time
# from convert_to_C import convert
import os
import argparse
import chromosome as chrmlib
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir_path", type=str)
parser.add_argument("--mc_file_path", type=str, default="mc_data.pkl")
parser.add_argument("--group_by", type=int, default=100)
parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--cfile_dir_path", type=str, default="cfiles")
parser.add_argument("--mode", type=str, default="freqs")
parser.add_argument("--random_seed", type=int, default=373)


def preproc(dir_name, mode, group_by):

	beg_time = time.time()
	print("[0] starting preproc, file = {}, group_by = {}".format(dir_name,group_by) )

	# set of unique positions
	mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
	# # list of dicts that completely define a mutation
	# mut_entries = []
	# set of cancer types encountered
	typs_set = set()
	# list of dataframes
	file_entries = []

	file_paths = []
	file_count = 0
	entries = sorted(os.listdir(dir_name))
	for entry in entries:
		entry_path = os.path.join(dir_name,entry)
		if os.path.isfile(entry_path):
			file_paths.append(entry_path)
			file_count += 1
	assert( file_count == len(entries) )

	file_count = 0
	for file_path in file_paths:
		file_mut_entries = []
		desired_entries = ["Chromosome", "Start_position", "cancer_type", "Count"]
		df = pd.read_csv(file_path)
		if "Count" not in df.columns:
			cd = pd.DataFrame({"Count": np.ones(df.shape[0], dtype=chrmlib.FLOAT_T)})
			df = pd.concat([df,cd], axis=1)
		assert (df["Variant_Type"] == "SNP").all()
		assert (df["Start_position"] == df["End_position"]).all()
		assert (df["Chromosome"] >= 1).all()
		assert (df["Chromosome"] <= chrmlib.NUM_CHRMS).all()
		df = df[desired_entries]
		# chrom_cond = df.Chromsome not in ["X", "Y"]
		# df.where(chrom_cond, inplace=True).dropna(axis=0, inplace=True)
		df["Chromosome"] = df["Chromosome"] - 1
		df["Start_position"] = df["Start_position"] - 1
		if mode == "freqs":
			file_ints = df["Count"].sum()
			df["Count"] = df["Count"] / file_ints
		df.rename(index=str, columns={"Chromosome": "chrm", 
										"Start_position": "pos", 
										"cancer_type": "typ", 
										"Count": "ints"}, inplace=True)
		typs_set = typs_set.union(set(df["typ"]))
		for c in range(chrmlib.NUM_CHRMS):
			mut_pos[c] = mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
		file_entries.append(df)
		file_count += 1
		if file_count % 1000 == 0:
			print(".", end="",flush=True)

	cur_time = time.time()
	print( "[{0:.0f}] read in all of the files".format(cur_time-beg_time) )

	chrms = []
	for c in range(chrmlib.NUM_CHRMS):
		chrms.append(chrmlib.Chromosome(c,typs_set))
		chrms[c].set_mut_arrays(mut_pos[c])
	del mut_pos
	del typs_set

	cur_time = time.time()
	print( "[{0:.0f}] initialized chromosomes with new mut arrays".format(cur_time-beg_time) )

	for c in range(chrmlib.NUM_CHRMS):
		chrms[c].update(file_entries)

	cur_time = time.time()
	print( "[{0:.0f}] filled mut arrays with mut entry data".format(cur_time-beg_time) )	

	if group_by > 1:
		for chrm in chrms:
			chrm.group(group_by)
		cur_time = time.time()
		print( "[{0:.0f}] grouped mutations".format(cur_time-beg_time) )

	return chrms


if __name__ == "__main__":

	FLAGS = parser.parse_args()
	if FLAGS.random_seed:
		np.random.seed(FLAGS.random_seed)
	np.set_printoptions(threshold=1000)

	chrms = preproc(FLAGS.src_dir_path, FLAGS.mode, FLAGS.group_by)
	with open(FLAGS.mc_file_path, "wb") as pkl_file:
		pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

	if not os.path.exists(FLAGS.cfile_dir_path):
		os.mkdir(FLAGS.cfile_dir_path)
	assert os.path.isdir(FLAGS.cfile_dir_path)
	for c in range(chrmlib.NUM_CHRMS):
		barray = chrms[c].mut_array_to_bytes()
		cfile_path = "{}/{}_{}.dat".format(FLAGS.cfile_dir_path,chrmlib.CFILE_BASE,c)
		with open(cfile_path, 'wb') as file:
			file.write(barray)
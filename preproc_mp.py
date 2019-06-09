import sys
import numpy as np
import time
# from convert_to_C import convert
import os
import argparse
import chromosome as chrmlib
import pickle
import pandas as pd
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir_path", type=str, default="for_adamo")
parser.add_argument("--mc_file_path", type=str, default="mc_data.pkl")
parser.add_argument("--group_by", type=int, default=100)
# parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--cfile_dir_path", type=str, default="cfiles")
parser.add_argument("--mode", type=str, default="freqs")
parser.add_argument("--random_seed", type=int, default=373)
parser.add_argument("--num_procs", type=int, default=mp.cpu_count())


def proc_files_func(proc_input):
	proc_start = time.time()
	proc_id, mode, proc_file_paths = proc_input[0], proc_input[1], proc_input[2]
	proc_file_count = 0
	proc_mut_entries = []
	proc_typs_set = set()
	proc_mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
	for file_path in proc_file_paths:
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
		proc_typs_set = proc_typs_set.union(set(df["typ"]))
		for c in range(chrmlib.NUM_CHRMS):
			proc_mut_pos[c] = proc_mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
		proc_mut_entries.append(df)
		proc_file_count += 1
	proc_end = time.time()
	# print("proc id {0} terminating, {1} files processed, {2:.0f} s elapsed".format(proc_id,proc_file_count,proc_end-proc_start))
	return proc_file_count, proc_mut_entries, proc_typs_set, proc_mut_pos


def preproc(dir_name, mode, group_by, num_procs):

	beg_time = time.time()
	print("[0] starting preproc, file = {}, group_by = {}".format(dir_name,group_by) )

	# # set of unique positions
	# mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
	# # list of dicts that completely define a mutation
	# mut_entries = []
	# # set of cancer types encountered
	# typs_set = set()
	# # list of dataframes
	# file_entries = []

	file_paths = []
	file_count = 0
	entries = sorted(os.listdir(dir_name))
	for entry in entries:
		entry_path = os.path.join(dir_name,entry)
		if os.path.isfile(entry_path):
			file_paths.append(entry_path)
			file_count += 1
	assert( file_count == len(entries) )

	num_per_proc = [len(file_paths) // num_procs for i in range(num_procs)]
	for i in range(len(file_paths) % num_procs):
		num_per_proc[i] += 1
	assert np.sum(num_per_proc) == len(file_paths)
	print("max num_per_proc = %d" % np.max(num_per_proc))
	running_total = 0
	proc_inputs = []
	for i in range(num_procs):
		proc_inputs.append((i,mode,file_paths[running_total:running_total+num_per_proc[i]]))
		running_total += num_per_proc[i]
	assert running_total == len(file_paths)
	# file_count = mp.Manager.Value('d',0)
	# mut_entries = mp.Manager.list()
	# types_set = mp.Manager.dict()
	# # mut_pos = [mp.Manager.dict() for i in range(chrmlib.NUM_CHRMS)]
	# mut_pos_list = mp.Manager.list()

	cur_time = time.time()
	print( "[{0:.0f}] read in all of the files".format(cur_time-beg_time) )

	pool = mp.Pool(num_procs)
	proc_results = pool.map(proc_files_func, proc_inputs)
	assert len(proc_results) == num_procs
	# aggregate results
	file_count = 0
	mut_entries = []
	typs_set = set()
	mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
	for proc_result in proc_results:
		file_count += proc_result[0]
		mut_entries.extend(proc_result[1])
		typs_set = typs_set.union(proc_result[2])
		for c in range(len(mut_pos)):
			mut_pos[c] = mut_pos[c].union(proc_result[3][c])

	cur_time = time.time()
	print( "[{0:.0f}] aggregated all the results".format(cur_time-beg_time) )
	print("number of distinct positions", [len(mut_pos[i]) for i in range(len(mut_pos))])
	print("number of cancer types", len(typs_set))

	chrms = []
	for c in range(chrmlib.NUM_CHRMS):
		chrms.append(chrmlib.Chromosome(c,typs_set))
		chrms[c].set_mut_arrays(mut_pos[c])
	del mut_pos
	del typs_set

	cur_time = time.time()
	print( "[{0:.0f}] initialized chromosomes with new mut arrays".format(cur_time-beg_time) )

	for c in range(chrmlib.NUM_CHRMS):
		chrms[c].update(mut_entries)

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
	print(FLAGS)

	chrms = preproc(FLAGS.src_dir_path, FLAGS.mode, FLAGS.group_by, FLAGS.num_procs)
	with open(FLAGS.mc_file_path, "wb") as pkl_file:
		pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

	os.makedirs(FLAGS.cfile_dir_path, exist_ok=True)
	for c in range(chrmlib.NUM_CHRMS):
		barray = chrms[c].mut_array_to_bytes()
		cfile_path = "{}/{}_{}.dat".format(FLAGS.cfile_dir_path,chrmlib.CFILE_BASE,c)
		with open(cfile_path, 'wb') as file:
			file.write(barray)
"""
Script for preprocessing .csv files and saving the results in a chromosome .pkl file and mutliple .dat files for the C program.
This script should be run on boltz
"""


import sys
import numpy as np
import time
import os
import argparse
import chromosome as chrmlib
import pickle
import pandas as pd
import multiprocessing as mp
from distutils.util import strtobool

pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir_path", type=str, default="for_adamo")
parser.add_argument("--kat_file_path", type=str, default="kataegis/snvs.tsv")
parser.add_argument("--mc_file_path", type=str, default="mc_data.pkl")
parser.add_argument("--group_by", type=int, default=100)
# parser.add_argument("--num_folds", type=int, default=5)
parser.add_argument("--cfile_dir_path", type=str, default="cfiles", help="directory for .dat files for C program")
# parser.add_argument("--mode", type=str, choices=["sample_freqs", "counts", "tumour_freqs"], default="sample_freqs", help="normalization method (counts is no normalization)")
parser.add_argument("--random_seed", type=int, default=373)
parser.add_argument("--num_procs", type=int, default=mp.cpu_count())
parser.add_argument("--overwrite", type=lambda x:bool(strtobool(x)), default=False)


def get_pseudo_median_fn(col_name):
	""" getter function for accumulating grouped tables by col"""
	def pseudo_median(grouped_df):
		sorted_pos = np.sort(grouped_df[col_name].to_numpy())
		return sorted_pos[len(sorted_pos) // 2]
	return pseudo_median

def copy_new_pos(row):
	if not np.isnan(row["new_pos"]):
		row["pos"] = row["new_pos"]
	return row

def proc_files_func(proc_input):
	proc_start = time.time()
	proc_id, proc_file_paths, kat_df = proc_input[0], proc_input[1], proc_input[2]
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
		df["Chromosome"] = df["Chromosome"] - 1
		df["Start_position"] = df["Start_position"] - 1
		df = df.groupby(by=["Chromosome", "Start_position", "cancer_type"]).sum()
		df = df.reset_index().rename(index=str, columns={"Chromosome": "chrm", 
										"Start_position": "pos", 
										"cancer_type": "typ", 
										"Count": "ints"})
		# correct for kataegis
		orig_num = df.shape[0]
		file_name = os.path.splitext(os.path.basename(os.path.normpath(file_path)))[0]
		# print(f">>> {file_name}")
		sk = kat_df[kat_df["sample"] == file_name]
		kat_num = sk.shape[0]
		if kat_num > 0:
			sk.drop(columns=["sample"], inplace=True)
			assert len(set(sk["typ"])) == 1 
			# and set(sk["typ"]) == set(df["typ"]), \
			# 	"{}: {} {}".format(file_name,set(sk["typ"]),set(df["typ"]))
			sk.drop(columns=["typ"], inplace=True)
			skdf = df.merge(sk, how="inner", on=["chrm","pos"]) # chrm, pos, typ, ints, sample, start, end
			kat_num_2 = skdf.shape[0]
			pseudo_median = get_pseudo_median_fn("pos")
			skdfg = skdf[["chrm", "start", "end", "pos"]].groupby(by=["chrm","start","end"]).apply(pseudo_median)
			skdfg = skdfg.reset_index().rename(index=str, columns={0: "new_pos"}) # chrm, start, end, new_pos
			group_num = skdfg.shape[0]
			skdf = skdf.merge(skdfg, how="inner", on=["chrm", "start", "end"])
			df = df.merge(skdf[["chrm", "pos", "new_pos"]], how="left", on=["chrm", "pos"])
			df = df.apply(copy_new_pos, axis=1)
			df.drop(columns=["new_pos"], inplace=True)
			# take the mean ints for kataegic mutations
			df = df.groupby(by=["chrm","pos","typ"]).mean()
			df = df.reset_index()
			if df.shape[0] != (orig_num-kat_num_2+group_num):
				print(f"{file_name}: {df.shape[0]} vs {orig_num-kat_num_2+group_num} | {orig_num} {kat_num_2} {group_num}")
			# assert df.shape[0] == (orig_num-kat_num+group_num), f"{file_name}: {df.shape[0]} {orig_num} {kat_num} {group_num}"
		# assert df["ints"].min() == 1., df["ints"].min()
		# assert df["ints"].max() == 1., df["ints"].max()
		# update stuff
		proc_typs_set = proc_typs_set.union(set(df["typ"]))
		for c in range(chrmlib.NUM_CHRMS):
			proc_mut_pos[c] = proc_mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
		proc_mut_entries.append(df)
		proc_file_count += 1
	proc_end = time.time()
	# print("proc id {0} terminating, {1} files processed, {2:.0f} s elapsed".format(proc_id,proc_file_count,proc_end-proc_start))
	return proc_file_count, proc_mut_entries, proc_typs_set, proc_mut_pos

def read_kataegis(kat_file_path):
	assert os.path.isfile(kat_file_path)
	df = pd.read_csv(kat_file_path, sep="\t")
	desired_entries = ["sample", "chr", "pos", "start", "end", "histology"]
	df = df[desired_entries]
	df.drop(df[df["chr"] == "X"].index, inplace=True)
	df.drop(df[df["chr"] == "Y"].index, inplace=True)
	df = df.astype({"chr": int, "pos": int, "start": int, "end": int})
	assert (df["chr"] >= 1).all()
	assert (df["chr"] <= chrmlib.NUM_CHRMS).all()
	# make everything 0-indexed for consistency
	df["chr"] = df["chr"] - 1
	df["pos"] = df["pos"] - 1
	df["start"] = df["start"] - 1
	df["end"] = df["end"] - 1
	assert set(df["chr"]) == set(range(chrmlib.NUM_CHRMS))
	# rename certain columns for consistency
	df.rename(index=str, columns={"chr": "chrm", "histology": "typ"}, inplace=True)
	return df

def preproc(src_dir_path, kat_file_path, group_by, num_procs):

	beg_time = time.time()
	print("[0] starting preproc, directory = {}, group_by = {}".format(src_dir_path,group_by) )

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
	entries = sorted(os.listdir(src_dir_path))
	for entry in entries:
		entry_path = os.path.join(src_dir_path,entry)
		if os.path.isfile(entry_path):
			file_paths.append(entry_path)
			file_count += 1
	assert file_count == len(entries)

	cur_time = time.time()
	print( "[{0:.0f}] read in kataegis file".format(cur_time-beg_time) )
	kat_df = read_kataegis(kat_file_path)

	num_per_proc = [len(file_paths) // num_procs for i in range(num_procs)]
	for i in range(len(file_paths) % num_procs):
		num_per_proc[i] += 1
	assert np.sum(num_per_proc) == len(file_paths)
	print("max num_per_proc = %d" % np.max(num_per_proc))
	running_total = 0
	proc_inputs = []
	for i in range(num_procs):
		proc_inputs.append((i,file_paths[running_total:running_total+num_per_proc[i]],kat_df))
		running_total += num_per_proc[i]
	assert running_total == len(file_paths)

	cur_time = time.time()
	print( "[{0:.0f}] read in all of the files".format(cur_time-beg_time) )
	if num_procs > 1:
		pool = mp.Pool(num_procs)
		proc_results = pool.map(proc_files_func, proc_inputs)
		assert len(proc_results) == num_procs
	else:
		proc_results = [proc_files_func(proc_inputs[0])]
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
	if not FLAGS.overwrite and os.path.isfile(FLAGS.mc_file_path):
		print("loading previous mc_data")
		with open(FLAGS.mc_file_path, "rb") as pkl_file:
			chrms = pickle.load(pkl_file)
	else:
		print("creating new mc_data (with possible overwrite)")
		chrms = preproc(FLAGS.src_dir_path, FLAGS.kat_file_path, FLAGS.group_by, FLAGS.num_procs)
		with open(FLAGS.mc_file_path, "wb") as pkl_file:
			pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
	modes = ["counts", "sample_freqs", "tumour_freqs"]
	for mode in modes:
		cfile_dir_path = os.path.join(FLAGS.cfile_dir_path,mode)
		os.makedirs(cfile_dir_path, exist_ok=True)
		for c in range(chrmlib.NUM_CHRMS):
			barray = chrms[c].mut_array_to_bytes(mode)
			cfile_path = "{}/{}_{}.dat".format(cfile_dir_path,chrmlib.CFILE_BASE,c)
			with open(cfile_path, 'wb') as file:
				file.write(barray)
	for chrm in chrms:
			chrm.delete_non_run_data()
	for mode in modes:
		run_file_path = os.path.join(FLAGS.cfile_dir_path,mode,"run_data.pkl")
		with open(run_file_path, "wb") as pkl_file:
			pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
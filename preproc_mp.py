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
	proc_id, proc_file_paths, kat_df, donor_df, alt_typs_set = proc_input[0], proc_input[1], proc_input[2], proc_input[3], proc_input[4]
	proc_file_count = 0
	proc_mut_entries = []
	proc_typs = set()
	proc_mut_pos = [set() for c in range(chrmlib.NUM_CHRMS)]
	for file_path in proc_file_paths:
		df = pd.read_csv(file_path)
		if alt_typs_set and df["cancer_type"].iloc[0] not in alt_typs_set:
			# skip this entry
			continue
		desired_entries = ["Chromosome", "Start_position", "cancer_type", "Count"]
		if "Count" not in df.columns:
			df["Count"] = np.ones(df.shape[0], dtype=chrmlib.FLOAT_T)
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
		# TBD implicitly assumes all mutations at one location are the same, regardless of type (i.e. A->C)
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
		# add donor information
		donor = donor_df[donor_df["fname"] == file_name]["donor"]
		assert len(donor) == 1
		df["donor"] = np.array(df.shape[0]*[donor.iloc[0]])
		# update stuff
		proc_typs.add(df["typ"].iloc[0])
		for c in range(chrmlib.NUM_CHRMS):
			proc_mut_pos[c] = proc_mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
		proc_mut_entries.append(df)
		proc_file_count += 1
	proc_end = time.time()
	return proc_file_count, proc_mut_entries, proc_typs, proc_mut_pos


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


def read_donor(donor_file_path):

	assert os.path.isfile(donor_file_path)
	df = pd.read_csv(donor_file_path, sep="\t")
	df.drop(columns=["Proj_Code", "newLabel"], inplace=True)
	df.rename(index=str, columns={"Tumor_Sample_Barcode": "fname", "Donor_ID": "donor", "Hist_Abbr": "hist"}, inplace=True)
	return df


def train_valid_split(typs_set, mut_entries, valid_frac):

	tumour_types_list = sorted(typs_set)
	# do split
	train_num, valid_num = [], []
	train_idx, valid_idx = [], []
	for tumour_type in tumour_types_list:
		donors_num = {}
		donors_idx = {}
		total_num = 0
		for idx, df in enumerate(mut_entries):
			if df["typ"].iloc[0] != tumour_type:
				continue
			total_num += 1
			donor = df["donor"].iloc[0]
			if donor not in donors_num:
				assert donor not in donors_idx
				donors_num[donor] = 1
				donors_idx[donor] = [idx]
			else:
				assert donor in donors_idx
				donors_num[donor] += 1
				donors_idx[donor].append(idx)
		donors_id, donors_num = zip(*donors_num.items())
		donors_id_2, donors_idx = zip(*donors_idx.items())
		assert donors_id == donors_id_2, (donors_id, donors_id_2)
		# shuffle first for tiebreakers
		shuffle_idx = np.arange(len(donors_num))
		np.random.shuffle(shuffle_idx) # shuffles inplace
		donors_id = np.array(donors_id)[shuffle_idx]
		donors_num = np.array(donors_num)[shuffle_idx]
		donors_idx = np.array(donors_idx)[shuffle_idx]
		# sort based on number
		donors_id_argsort = np.argsort(donors_num)
		donors_id = donors_id[donors_id_argsort]
		donors_num = donors_num[donors_id_argsort]
		donors_idx = donors_idx[donors_id_argsort]
		# compute valid and train number according to proportion
		final_valid_num = int(np.round(valid_frac*total_num))
		final_train_num = total_num - final_valid_num
		assert final_valid_num >= 0 and final_train_num > 0, (final_valid_num, final_train_num)
		if final_valid_num == 0:
			print(tumour_type, "final_valid_num is 0")
		# assign stuff to train and valid
		cur_train_num, cur_valid_num = 0, 0
		cur_train_idx, cur_valid_idx = [], []
		cur_idx = 0
		while cur_idx < len(donors_id):
			assert cur_valid_num + cur_train_num < total_num
			next_num = donors_num[-1-cur_idx]
			if cur_valid_num + next_num <= final_valid_num and cur_valid_num < cur_train_num:
				# train always gets the big ones first
				cur_valid_idx.extend(donors_idx[-1-cur_idx])
				cur_valid_num += next_num
			else: 
				cur_train_idx.extend(donors_idx[-1-cur_idx])
				cur_train_num += next_num
			cur_idx += 1
		assert cur_valid_num + cur_train_num == total_num
		if cur_valid_num == 0:
			print(tumour_type, "cur_valid_num is 0")
		train_num.append(cur_train_num)
		valid_num.append(cur_valid_num)
		train_idx.append(np.array(cur_train_idx))
		valid_idx.append(np.array(cur_valid_idx))
	return train_num, valid_num, train_idx, valid_idx


def preproc(src_dir_path, kat_file_path, donor_file_path, proc_file_path, df_alt_file_path, group_by, num_procs, valid_frac, max_group_dist):

	beg_time = time.time()
	print("[0] starting preproc")
	alt_typs_set = None
	use_alt = os.path.isfile(df_alt_file_path)
	if use_alt:
		cur_time = time.time()
		print("[{0:.0f}] loading alt data {1}".format(cur_time-beg_time,df_alt_file_path))
		alt_df = pd.read_pickle(df_alt_file_path)
		alt_df["donor"] = alt_df["id"]
		alt_typs_set = set(alt_df["typ"])
		proc_file_path = proc_file_path.rstrip(".pkl") + "_alt.pkl"
	if os.path.isfile(proc_file_path):
		cur_time = time.time()
		print("[{0:.0f}] loading proc results {1}".format(cur_time-beg_time, proc_file_path))
		with open(proc_file_path, "rb") as pkl_file:
			proc_results = pickle.load(pkl_file)
		print(len(proc_results))
	else:
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
		print("[{0:.0f}] read in donor file".format(cur_time-beg_time))
		donor_df = read_donor(donor_file_path)
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
			proc_inputs.append((i,file_paths[running_total:running_total+num_per_proc[i]],kat_df,donor_df,alt_typs_set))
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
		cur_time = time.time()
		print( "[{0:.0f}] saving proc file {1}".format(cur_time-beg_time, proc_file_path) )
		with open(proc_file_path, "wb") as pkl_file:
			pickle.dump(proc_results, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
	# aggregate results
	file_count = 0
	mut_entries = []
	typs_set = set()
	if use_alt:
		alt_typs_set = set(alt_df["typ"])
		# print(alt_typs_set)
	skip_count = 0
	mut_pos = [set() for c in range(chrmlib.NUM_CHRMS)]
	for proc_result in proc_results:	
		file_count += proc_result[0]
		mut_entries.extend(proc_result[1])
		typs_set = typs_set.union(proc_result[2])
		for c in range(chrmlib.NUM_CHRMS):
			mut_pos[c] = mut_pos[c].union(proc_result[3][c])
		print(file_count, end=",")
	cur_time = time.time()
	print( "[{0:.0f}] aggregated all the results".format(cur_time-beg_time) )
	print("number skipped", skip_count)
	print("number of distinct positions", [len(mut_pos[i]) for i in range(len(mut_pos))])
	print("number of cancer types", len(typs_set))
	#######
	temp_data = {
		"file_count": file_count,
		"mut_pos": mut_pos,
		"mut_entries": mut_entries,
		"typs_set": typs_set,
		"alt_typs_set": alt_typs_set
	}
	with open("tempdump.pkl", "wb") as pkl_file:
		pickle.dump(temp_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
	#######
	if use_alt:
		assert typs_set.issubset(alt_typs_set)
		alt_ids = sorted(set(alt_df["id"]))
		print(len(alt_ids))
		# mut_entries.extend(alt_mut_entries)
		alt_mut_entries = []
		for alt_id in alt_ids:
			alt_entry = alt_df[alt_df["id"] == alt_id]
			if alt_entry["typ"].iloc[0] not in typs_set:
				continue
			for c in range(len(mut_pos)):
				cur_mut_pos = set(alt_entry[alt_entry["chrm"] == c]["pos"])
				mut_pos[c] = mut_pos[c].union(cur_mut_pos)
			alt_mut_entries.append(alt_entry)
			print(alt_id, end=",")
	cur_time = time.time()
	print( "[{0:.0f}] aggregated all the alt results".format(cur_time-beg_time) )
	print("number of distinct positions", [len(mut_pos[i]) for i in range(len(mut_pos))])
	print("number of cancer types", len(typs_set))
	#######
	temp_data = {
		"file_count": file_count,
		"mut_pos": mut_pos,
		"mut_entries": mut_entries,
		"alt_mut_entries": alt_mut_entries,
		"typs_set": typs_set,
		"alt_typs_set": alt_typs_set
	}
	with open("tempdumpalt.pkl", "wb") as pkl_file:
		pickle.dump(temp_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
	#######
	chrms = []
	for c in range(chrmlib.NUM_CHRMS):
		chrms.append(chrmlib.Chromosome(c))
		chrms[c].set_mut_arrays(typs_set,mut_pos[c],valid_frac)
	del mut_pos
	cur_time = time.time()
	print( "[{0:.0f}] initialized chromosomes with new mut arrays".format(cur_time-beg_time) )
	if valid_frac > 0.:
		train_num, valid_num, train_idx, valid_idx = train_valid_split(typs_set,mut_entries,valid_frac)
		assert len(train_num) == len(typs_set)
		if use_alt:
			_train_num, _valid_num, _train_idx, _valid_idx = train_valid_split(typs_set,alt_mut_entries,valid_frac)
			for t in range(len(train_num)):
				train_idx[t] = np.concatenate([train_idx[t], train_num[t] + _train_idx[t]])
				valid_idx[t] = np.concatenate([valid_idx[t], valid_num[t] + _valid_idx[t]])
				train_num[t] += _train_num[t]
				valid_num[t] += _valid_num[t]
			mut_entries.extend(alt_mut_entries)
		# print(train_num)
		# print(train_idx)
		# print(valid_num)
		# print(valid_idx)
	else:
		valid_idx = None
	for c in range(chrmlib.NUM_CHRMS):
		chrms[c].update(mut_entries,valid_idx)
	cur_time = time.time()
	print( "[{0:.0f}] filled mut arrays with mut entry data".format(cur_time-beg_time) )	
	if group_by > 1:
		for chrm in chrms:
			chrm.group(group_by, max_group_dist)
		cur_time = time.time()
		print( "[{0:.0f}] grouped mutations".format(cur_time-beg_time) )
		print([chrm.get_unique_pos_count() for chrm in chrms])
	return chrms


def get_convert_chrm_fn():

	GOOD_SET = set([str(i) for i in range(1,23)])
	CONV_SET = set(["chr{}".format(i) for i in range(1,23)])
	def convert_chrm(chrm):
		if chrm in GOOD_SET:
			return chrm
		elif chrm in CONV_SET:
			return chrm[3:]
		else:
			return np.nan
	return convert_chrm


def preproc_alt(alt_file_path, df_alt_file_path):

	beg_time = time.time()
	print("[0] starting preproc_alt")
	if os.path.isfile(df_alt_file_path):
		print("[0] found preprocessed alt dataframe, loading")
		df = pd.read_pickle(df_alt_file_path)
		cur_time = time.time()
		print( "[{0:.0f}] loaded alt dataframe".format(cur_time-beg_time) )
	else:
		df = pd.read_csv(alt_file_path, low_memory=False)
		cur_time = time.time()
		print( "[{0:.0f}] read alt df".format(cur_time-beg_time) )
		# fix df attributes
		df.rename(index=str, columns={"chr": "chrm", "hist": "typ", "ID": "id"}, inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] rename chr -> chrm, hist -> typ".format(cur_time-beg_time))
		df.dropna(subset=["typ"], inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] drop nan typ".format(cur_time-beg_time))
		df.drop(df[df["typ"] == "Others"].index, inplace=True)
		# df.drop(df[df["typ"] == "Lymph-CLL"].index, inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] drop Others typ".format(cur_time-beg_time))
		df["new_chrm"] = df["chrm"].map(get_convert_chrm_fn())
		cur_time = time.time()
		print("[{0:.0f}] convert new_chrm".format(cur_time-beg_time))
		df.drop(columns=["chrm"], inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] drop chrm".format(cur_time-beg_time))
		df.rename(index=str, columns={"new_chrm": "chrm"}, inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] rename new_chrm -> chrm".format(cur_time-beg_time))
		df.dropna(subset=["chrm"], inplace=True)
		cur_time = time.time()
		print("[{0:.0f}] rename new_chrm -> chrm, drop nan chrm".format(cur_time-beg_time))
		good_chrms = set([str(i) for i in range(1,23)])
		assert set(df["chrm"]) == good_chrms
		assert not df.isnull().any().any() # no null entries
		df = df.astype({"chrm": int, "pos": int, "typ": str, "id": str})
		assert (df["chrm"] >= 1).all()
		assert (df["chrm"] <= chrmlib.NUM_CHRMS).all()
		df["chrm"] = df["chrm"] - 1
		df["pos"] = df["pos"] - 1
		df["ints"] = np.ones(df.shape[0], dtype=chrmlib.FLOAT_T)
		df = df.groupby(by=["chrm", "pos", "typ", "id"]).sum()
		df = df.reset_index()
		cur_time = time.time()
		print("[{0:.0f}] saved dataframe".format(cur_time-beg_time))
		df.to_pickle(df_alt_file_path)
	cur_time = time.time()
	print("[{0:.0f}] aggregating results".format(cur_time-beg_time))
	# aggregate results
	typs_set = set(df["typ"])
	mut_pos = [set(df[df["chrm"] == c]["pos"]) for c in range(chrmlib.NUM_CHRMS)]
	sample_ids = sorted(set(df["id"]))
	mut_entries = []
	for sample_id in sample_ids:
		sample_df = df[df["id"] == sample_id]
		mut_entries.append(sample_df)
	cur_time = time.time()
	print( "[{0:.0f}] aggregated all the results".format(cur_time-beg_time) )
	print("number of distinct positions", [len(mut_pos[i]) for i in range(len(mut_pos))])
	print("number of cancer types", len(typs_set))
	chrms = []
	for c in range(chrmlib.NUM_CHRMS):
		chrms.append(chrmlib.Chromosome(c))
		chrms[c].set_mut_arrays(typs_set,mut_pos[c],0.)
	del typs_set
	del mut_pos
	cur_time = time.time()
	print( "[{0:.0f}] initialized chromosomes with new mut arrays".format(cur_time-beg_time) )
	for c in range(chrmlib.NUM_CHRMS):
		chrms[c].update(mut_entries, None)
	cur_time = time.time()
	print( "[{0:.0f}] filled mut arrays with mut entry data".format(cur_time-beg_time) )
	return chrms


def read_cell_type(ct_file_path):

	assert os.path.isfile(ct_file_path)
	df = pd.read_csv(ct_file_path, skiprows=[0,1])[["Unnamed: 1", "Unnamed: 6"]]
	df.dropna(inplace=True)
	df.rename(index=str, columns={"Unnamed: 1": "ct_id", "Unnamed: 6": "ct_name"}, inplace=True)
	def valid_id(ct_id):
		if not isinstance(ct_id, str) or not (ct_id[0:2] == "E0" or ct_id[0:2] == "E1"):
			return False
		else:
			return True
	df["valid_id"] = df["ct_id"].map(valid_id)
	df = df[df["valid_id"]]
	df.drop(columns=["valid_id"], inplace=True)
	return df


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--src_dir_path", type=str, default="dsets/for_adamo")
	parser.add_argument("--kat_file_path", type=str, default="dsets/kataegis/snvs.tsv")
	parser.add_argument("--donor_file_path", type=str, default="dsets/for_adamo_donors.tsv")
	parser.add_argument("--alt_file_path", type=str, default="dsets/alt_muts/snvs.csv")
	parser.add_argument("--mc_dir_path", type=str, default="mc_data_kat")
	parser.add_argument("--mc_alt_dir_path", type=str, default="mc_data_kat_alt")
	parser.add_argument("--proc_file_path", type=str, default="proc_kat.pkl")
	parser.add_argument("--df_alt_file_path", type=str, default="df_alt.pkl")
	parser.add_argument("--group_by", type=int, default=100)
	parser.add_argument("--cfile_dir_path", type=str, default="cfiles_kat", help="directory for .dat files for C program")
	parser.add_argument("--random_seed", type=int, default=373)
	parser.add_argument("--num_procs", type=int, default=mp.cpu_count())
	parser.add_argument("--overwrite", type=lambda x:bool(strtobool(x)), default=False)
	parser.add_argument("--tumour_set", type=str, choices=["all", "reduced", "small"], default="all", help="set of tumour types to use, only relevant for cfiles")
	parser.add_argument("--create_cfiles", type=lambda x:bool(strtobool(x)), default=True)
	parser.add_argument("--valid_frac", type=float, default=0.3)
	parser.add_argument("--ct_file_path", type=str, default="dsets/cell_type.csv")
	parser.add_argument("--max_group_dist", type=int, default=chrmlib.MAX_INT)
	FLAGS = parser.parse_args()
	if FLAGS.random_seed:
		np.random.seed(FLAGS.random_seed)
	print(FLAGS)
	# printing options
	np.set_printoptions(threshold=1000)
	pd.options.mode.chained_assignment = None
	# do PCAWG mutation data
	if not FLAGS.overwrite and os.path.isdir(FLAGS.mc_dir_path):
		if FLAGS.create_cfiles:
			print(">>> mc_data exists, loading previous mc_data")
			chrms = chrmlib.load_mc_data(FLAGS.mc_dir_path)
		else:
			print(">>> mc_data exists, no need to load")
			pass
	else:
		print(">>> creating new mc_data (with possible overwrite)")
		chrms = preproc(FLAGS.src_dir_path, FLAGS.kat_file_path, FLAGS.donor_file_path, FLAGS.proc_file_path, FLAGS.df_alt_file_path, FLAGS.group_by, FLAGS.num_procs, FLAGS.valid_frac, FLAGS.max_group_dist)
		chrmlib.save_mc_data(FLAGS.mc_dir_path, chrms)
	# do alternate mutation data
	if not FLAGS.overwrite and os.path.isdir(FLAGS.mc_alt_dir_path):
		print(">>> alt data exists, no need to load")
		pass
	else:
		print(">>> creating new alt data (with possible overwrite)")
		alt_chrms = preproc_alt(FLAGS.alt_file_path, FLAGS.df_alt_file_path)
		chrmlib.save_mc_data(FLAGS.mc_alt_dir_path, alt_chrms)
	# do the cfiles directory
	if FLAGS.create_cfiles:
		# if FLAGS.group_by > 1:
		# 	assert all([chrm.group_by == FLAGS.group_by for chrm in chrms])
		# 	assert all([chrm.max_group_dist == FLAGS.max_group_dist for chrm in chrms])
		cfile_dir_path_base = FLAGS.cfile_dir_path
		if FLAGS.tumour_set == "all":
			tumour_list = sorted(chrmlib.ALL_SET)
		elif FLAGS.tumour_set == "reduced":
			tumour_list = sorted(chrmlib.REDUCED_SET)
			cfile_dir_path_base += "_red"
		else: # FLAGS.tumour_set == "small"
			tumour_list = sorted(chrmlib.SMALL_SET)
		modes = ["counts", "sample_freqs", "tumour_freqs"]
		splits = ["all"]
		if FLAGS.valid_frac > 0.:
			# assert FLAGS.tumour_set == "all"
			splits.extend(["train", "valid"])
		for mode in modes:
			for split in splits:
				cfile_dir_path = os.path.join(cfile_dir_path_base,mode,split)
				os.makedirs(cfile_dir_path, exist_ok=True)
				for c in range(chrmlib.NUM_CHRMS):
					barray = chrms[c].mut_array_to_bytes(mode, tumour_list, split)
					cfile_path = "{}/{}_{}.dat".format(cfile_dir_path,chrmlib.CFILE_BASE,c)
					with open(cfile_path, 'wb') as file:
						file.write(barray)
		for chrm in chrms:
				chrm.delete_non_run_data()
		for mode in modes:
			for split in splits:
				run_file_path = os.path.join(cfile_dir_path_base,mode,split,"run_data.pkl")
				with open(run_file_path, "wb") as pkl_file:
					pickle.dump(chrms, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
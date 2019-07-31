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

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir_path", type=str, default="dsets/for_adamo")
parser.add_argument("--kat_file_path", type=str, default="dsets/kataegis/snvs.tsv")
parser.add_argument("--donor_file_path", type=str, default="dsets/for_adamo_donors.tsv")
parser.add_argument("--alt_file_path", type=str, default="dsets/alt_muts/snvs.csv")
parser.add_argument("--sig_dir_path", type=str, default="dsets/chrm_sigs")
parser.add_argument("--mc_dir_path", type=str, default="mc_data_kat")
parser.add_argument("--mc_alt_dir_path", type=str, default="mc_data_kat_alt")
parser.add_argument("--df_sig_file_path", type=str, default="df_sig.pkl")
parser.add_argument("--group_by", type=int, default=100)
parser.add_argument("--cfile_dir_path", type=str, default="cfiles_kat", help="directory for .dat files for C program")
parser.add_argument("--random_seed", type=int, default=373)
parser.add_argument("--num_procs", type=int, default=mp.cpu_count())
parser.add_argument("--overwrite", type=lambda x:bool(strtobool(x)), default=False)
parser.add_argument("--tumour_set", type=str, choices=["all", "reduced", "small"], default="all", help="set of tumour types to use")
parser.add_argument("--create_cfiles", type=lambda x:bool(strtobool(x)), default=True)
parser.add_argument("--valid_frac", type=float, default=0.3)
# parser.add_argument("--tv_split", type=str, choices=["all", "valid"], default="all")


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
	proc_id, proc_file_paths, kat_df, donor_df = proc_input[0], proc_input[1], proc_input[2], proc_input[3]
	proc_file_count = 0
	proc_mut_entries = []
	proc_typs_set = set()
	proc_mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
	for file_path in proc_file_paths:
		desired_entries = ["Chromosome", "Start_position", "cancer_type", "Count"]
		df = pd.read_csv(file_path)
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
		proc_typs_set = proc_typs_set.union(set(df["typ"]))
		for c in range(chrmlib.NUM_CHRMS):
			proc_mut_pos[c] = proc_mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
		proc_mut_entries.append(df)
		proc_file_count += 1
	proc_end = time.time()
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


def read_donor(donor_file_path):

	assert os.path.isfile(donor_file_path)
	df = pd.read_csv(donor_file_path, sep="\t")
	df.drop(columns=["Proj_Code", "newLabel"], inplace=True)
	df.rename(index=str, columns={"Tumor_Sample_Barcode": "fname", "Donor_ID": "donor", "Hist_Abbr": "hist"}, inplace=True)
	return df


def preproc(src_dir_path, kat_file_path, donor_file_path, group_by, num_procs, valid_frac):

	beg_time = time.time()
	print("[0] starting preproc")
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
		proc_inputs.append((i,file_paths[running_total:running_total+num_per_proc[i]],kat_df,donor_df))
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
	mut_pos = [set() for c in range(chrmlib.NUM_CHRMS)]
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
		chrms.append(chrmlib.Chromosome(c))
		chrms[c].set_mut_arrays(typs_set,mut_pos[c],valid_frac)
	del mut_pos
	cur_time = time.time()
	print( "[{0:.0f}] initialized chromosomes with new mut arrays".format(cur_time-beg_time) )
	if valid_frac > 0.:
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
			# sort based on number
			donors_id_argsort = np.argsort(donors_num)
			donors_id = np.array(donors_id)[donors_id_argsort]
			donors_num = np.array(donors_num)[donors_id_argsort]
			donors_idx = np.array(donors_idx)[donors_id_argsort]
			# compute valid and train number according to proportion
			final_valid_num = int(np.round(valid_frac*total_num))
			final_train_num = total_num - final_valid_num
			assert final_valid_num >= 0 and final_train_num > 0, (final_valid_num, final_train_num)
			if final_valid_num == 0:
				print(tumour_type)
			# assign stuff to train and valid
			cur_train_num, cur_valid_num = 0, 0
			cur_train_idx, cur_valid_idx = [], []
			cur_idx = 0
			while cur_idx < len(donors_id):
				assert cur_valid_num + cur_train_num < total_num
				next_num = donors_num[-1-cur_idx]
				if cur_valid_num + next_num < final_valid_num and cur_valid_num < cur_train_num:
					cur_valid_idx.append(donors_idx[-1-cur_idx])
					cur_valid_num += next_num
				else: 
					cur_train_idx.append(donors_idx[-1-cur_idx])
					cur_train_num += next_num
				cur_idx += 1
			assert cur_valid_num + cur_train_num == total_num
			train_num.append(cur_train_num)
			valid_num.append(cur_valid_num)
			train_idx.append(cur_train_idx)
			valid_idx.append(cur_valid_idx)
	else:
		valid_idx = None
	for c in range(chrmlib.NUM_CHRMS):
		chrms[c].update(mut_entries,valid_idx)
	cur_time = time.time()
	print( "[{0:.0f}] filled mut arrays with mut entry data".format(cur_time-beg_time) )	
	if group_by > 1:
		for chrm in chrms:
			chrm.group(group_by)
		cur_time = time.time()
		print( "[{0:.0f}] grouped mutations".format(cur_time-beg_time) )
	return chrms


# def proc_files_func_alt(proc_input):
	
# 	proc_start = time.time()
# 	proc_id, proc_dfs = proc_input[0], proc_input[1]
# 	proc_df_count = 0
# 	proc_mut_entries = []
# 	proc_typs_set = set()
# 	proc_mut_pos = [set() for i in range(chrmlib.NUM_CHRMS)]
# 	for df in proc_dfs:
# 		desired_entries = ["chrm", "pos", "typ"]
# 		assert (df["chrm"] >= 1).all()
# 		assert (df["chrm"] <= chrmlib.NUM_CHRMS).all()
# 		df = df[desired_entries]
# 		cd = pd.DataFrame({"ints": np.ones(df.shape[0], dtype=chrmlib.FLOAT_T)})
# 		df = pd.concat([df,cd], axis=1)
# 		df["chrm"] = df["chrm"] - 1
# 		df["pos"] = df["pos"] - 1
# 		df = df.groupby(by=["chrm", "pos", "typ"]).sum()
# 		df = df.reset_index()
# 		# update stuff
# 		proc_typs_set = proc_typs_set.union(set(df["typ"]))
# 		for c in range(chrmlib.NUM_CHRMS):
# 			proc_mut_pos[c] = proc_mut_pos[c].union(set(df[df["chrm"] == c]["pos"]))
# 		proc_mut_entries.append(df)
# 		proc_df_count += 1
# 	proc_end = time.time()
# 	return proc_df_count, proc_mut_entries, proc_typs_set, proc_mut_pos


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


def preproc_alt(alt_file_path):

	beg_time = time.time()
	print("[0] starting preproc_alt")
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
	# accumulate results
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
		chrms[c].update(mut_entries)
	cur_time = time.time()
	print( "[{0:.0f}] filled mut arrays with mut entry data".format(cur_time-beg_time) )
	return chrms


def proc_sig_files_func(proc_input):

	proc_start = time.time()
	proc_id, proc_file_paths = proc_input[0], proc_input[1]
	proc_file_count = 0
	proc_sig_entries = []
	col_names = ["chrm", "start", "end", "sig_typ", "rand1", "rand2", "rand3", "rand4", "rand5"]
	desired_cols = ["chrm", "start", "end", "sig_typ"]
	for file_path in proc_file_paths:
		df = pd.read_csv(file_path, sep="\t", skiprows=1, header=None, names=col_names)
		df = df[desired_cols]
		df["new_chrm"] = df["chrm"].map(get_convert_chrm_fn())
		df.dropna(subset=["new_chrm"], axis=0, inplace=True)
		df.drop(columns=["chrm"], inplace=True)
		df.rename(index=str, columns={"new_chrm": "chrm"}, inplace=True)
		df = df.astype({"chrm": int, "start": int, "end": int, "sig_typ": str})
		assert (df["chrm"] >= 1).all()
		assert (df["chrm"] <= chrmlib.NUM_CHRMS).all()
		# make chrms and positions 0-based
		df["chrm"] = df["chrm"] - 1
		df["start"] = df["start"] - 1
		df["end"] = df["end"] - 1
		proc_sig_entries.append(df)
		proc_file_count += 1
	proc_end = time.time()
	return proc_file_count, proc_sig_entries


def preproc_sig(sig_dir_path, num_procs):

	assert os.path.isdir(sig_dir_path)
	beg_time = time.time()
	print("[0] starting preproc_sig")
	file_paths = []
	file_count = 0
	entries = sorted(os.listdir(sig_dir_path))
	for entry in entries:
		entry_path = os.path.join(sig_dir_path,entry)
		if os.path.isfile(entry_path):
			file_paths.append(entry_path)
			file_count += 1
	assert file_count == len(entries)
	cur_time = time.time()
	num_per_proc = [len(file_paths) // num_procs for i in range(num_procs)]
	for i in range(len(file_paths) % num_procs):
		num_per_proc[i] += 1
	assert np.sum(num_per_proc) == len(file_paths)
	print("max num_per_proc = %d" % np.max(num_per_proc))
	running_total = 0
	proc_inputs = []
	for i in range(num_procs):
		proc_inputs.append((i,file_paths[running_total:running_total+num_per_proc[i]]))
		running_total += num_per_proc[i]
	assert running_total == len(file_paths)
	cur_time = time.time()
	print( "[{0:.0f}] read in all of the files".format(cur_time-beg_time) )
	if num_procs > 1:
		pool = mp.Pool(num_procs)
		proc_results = pool.map(proc_sig_files_func, proc_inputs)
		assert len(proc_results) == num_procs
	else:
		proc_results = [proc_sig_files_func(proc_inputs[0])]
	# aggregate results
	file_count = 0
	sig_entries = []
	for proc_result in proc_results:
		file_count += proc_result[0]
		sig_entries.extend(proc_result[1])
	cur_time = time.time()
	print( "[{0:.0f}] aggregated all the results".format(cur_time-beg_time) )
	sig_df = pd.concat(sig_entries, axis=0)
	cur_time = time.time()
	print( "[{0:.0f}] concatenated dataframes".format(cur_time-beg_time) )
	sig_df.sort_values(["chrm","start","end"], inplace=True)
	cur_time = time.time()
	print( "[{0:.0f}] sorting big dataframe".format(cur_time-beg_time) )
	return sig_df


if __name__ == "__main__":

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
		chrms = preproc(FLAGS.src_dir_path, FLAGS.kat_file_path, FLAGS.donor_file_path, FLAGS.group_by, FLAGS.num_procs, FLAGS.valid_frac)
		chrmlib.save_mc_data(FLAGS.mc_dir_path, chrms)
	quit()
	# do alternate mutation data
	if not FLAGS.overwrite and os.path.isdir(FLAGS.mc_alt_dir_path):
		print(">>> alt data exists, no need to load")
		pass
	else:
		print(">>> creating new alt data (with possible overwrite)")
		alt_chrms = preproc_alt(FLAGS.alt_file_path)
		chrmlib.save_mc_data(FLAGS.mc_alt_dir_path, alt_chrms)
	# do chromatin signature data
	if not FLAGS.overwrite and os.path.isfile(FLAGS.df_sig_file_path):
		print(">>> sig data exists, no need to load")
		pass
	else:
		print(">>> creating new sig data (with possible overwrite)")
		sig_df = preproc_sig(FLAGS.sig_dir_path, FLAGS.num_procs)
		sig_df.to_pickle(FLAGS.df_sig_file_path, protocol=pickle.HIGHEST_PROTOCOL)
	# do the cfiles directory
	if FLAGS.create_cfiles:
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
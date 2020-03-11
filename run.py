""" 
Script for creating bash scripts and then running them on scinet with slurm.
If maximum parallelism is used (80 logical cores) then the longest segmentation (chromosome 2) takes ~1h 40 min.
Before running on scinet, run the command "module load $PY_MOD" to load the necessary python modules.
Must make sure that the executable (usally called "segmentation", indicated by exe_path) has been compiled.
To compile "segmentation" on niagara, run "module load gcc" then type make in the directory with k_seg.c, k_seg.h, segmentation.c, and the Makefile
"""

import pickle
import argparse
import os
import stat
import numpy as np
import chromosome as chrmlib
import subprocess as sp
import shlex
from datetime import date
from distutils.util import strtobool


def today_date():
	today = date.today()
	today_str = str.lower(today.strftime("%b_%d"))
	return today_str


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--run_name", type=str, default="test")
	parser.add_argument("--cfile_dp", type=str, default="/home/q/qmorris/youngad2/MutSeg/cfiles_both_red")
	parser.add_argument("--output_dp", type=str, default="/scratch/q/qmorris/youngad2/{}".format(today_date()))
	parser.add_argument("--tmp_script_dp", type=str, default="/home/q/qmorris/youngad2/MutSeg/scripts", help="temporary director for bash scripts")
	parser.add_argument("--mode", type=str, choices=["counts", "sample_freqs", "tumour_freqs"], default="counts")
	parser.add_argument("--chrm_id", type=int, default=-1, choices=list(range(-1,22)), help="chromosome id (starts at 0)")
	parser.add_argument("--naive_seg_size", type=int, default=1000000) # 1 Megabase
	parser.add_argument("--num_cores", type=int, default=80, choices=list(range(1,81)), help="number of logical cores required on scinet, should be <=80")
	parser.add_argument("--max_time", type=int, default=12, choices=list(range(1,25)), help="wall time in hours, should be <=24")
	parser.add_argument("--exe_path", type=str, default="/home/q/qmorris/youngad2/MutSeg/segmentation", help="path to segmentation executable")
	parser.add_argument("--overwrite", type=lambda x:bool(strtobool(x)), default=False)
	parser.add_argument("--tumour_set", type=str, choices=["all", "reduced"], default="reduced", help="set of tumour types to use")
	parser.add_argument("--train_split", type=str, choices=["all", "train"], default="all")
	parser.add_argument("--min_size", type=int, default=1)
	parser.add_argument("--h_pen", type=float, default=1.0)
	FLAGS = parser.parse_args()
	
	if FLAGS.tumour_set == "reduced":
		assert FLAGS.cfile_dp.rstrip("/").endswith("_red"), FLAGS.cfile_dp
	cfile_dp = os.path.join(FLAGS.cfile_dp,FLAGS.mode,FLAGS.train_split)
	print(cfile_dp)
	assert os.path.isdir(cfile_dp)
	run_fp = os.path.join(cfile_dp,"run_data.pkl")
	assert os.path.isfile(run_fp)
	with open(run_fp, "rb") as pkl_file:
		run_data = pickle.load(pkl_file)
	assert len(run_data) == chrmlib.NUM_CHRMS

	if FLAGS.chrm_id == -1:
		cfps = []
		chrms = run_data
		for c in range(chrmlib.NUM_CHRMS):
			cfile_name = "{}_{}.dat".format(chrmlib.CFILE_BASE,c)
			cfps.append(os.path.join(cfile_dp,cfile_name))
	else:
		# do a specific chromosome
		cfile_name = "{}_{}.dat".format(chrmlib.CFILE_BASE,FLAGS.chrm_id)
		cfps = [os.path.join(cfile_dp,cfile_name)]
		chrms = [run_data[FLAGS.chrm_id]]
	
	# attributes that are always the same
	assert os.path.isfile(FLAGS.exe_path)
	exe_path = FLAGS.exe_path
	quick_test = 0
	mp = FLAGS.num_cores
	seg_size = FLAGS.naive_seg_size
	prev_k = 0
	min_size = FLAGS.min_size
	h_pen = FLAGS.h_pen
	# if not FLAGS.overwrite:
	# 	assert not os.path.exists(FLAGS.output_dp)
	# output_dp = FLAGS.output_dp
	# if FLAGS.tumour_set != "all":
	# 	output_dp += "_" + FLAGS.tumour_set[:3]
	# output_dp += "_" + str(FLAGS.min_size)
	# output_dp += "_" + str(FLAGS.h_pen)
	# if FLAGS.mode == "counts":
	# 	output_dp += "_co"
	# elif FLAGS.mode == "sample_freqs":
	# 	output_dp += "_sf"
	# elif FLAGS.mode == "tumour_freqs":
	# 	output_dp += "_tf"
	
	# set up directories
	output_dp = os.path.join(FLAGS.output_dp,FLAGS.run_name)
	os.makedirs(output_dp, exist_ok=True)
	results_dp = os.path.join(output_dp,"results")
	os.makedirs(results_dp, exist_ok=True)
	stdout_dp = os.path.join(output_dp,"stdouts")
	os.makedirs(stdout_dp, exist_ok=True)
	stderr_dp = os.path.join(output_dp,"stderrs")
	os.makedirs(stderr_dp, exist_ok=True)
	script_dp = os.path.join(output_dp,"scripts")
	os.makedirs(script_dp, exist_ok=True)
	
	# set up temporary directory for scripts
	tmp_script_dp = os.path.join(FLAGS.tmp_script_dp,FLAGS.run_name)
	os.makedirs(tmp_script_dp, exist_ok=True)

	# dump the flags
	flags_fp = os.path.join(output_dp,"flags.txt")
	with open(flags_fp, "w") as flags_file:
		print(str(FLAGS), end="\n", file=flags_file)
	
	assert FLAGS.max_time >= 1 and FLAGS.max_time <= 24
	time = "{}:00:00".format(FLAGS.max_time)
	assert FLAGS.num_cores <= 80
	num_cores = FLAGS.num_cores
	
	# if FLAGS.tumour_set != "all":
	# 	script_dp += "_" + FLAGS.tumour_set[:3]
	# script_dp += "_" + str(FLAGS.min_size)
	# script_dp += "_" + str(FLAGS.h_pen)
	# if FLAGS.mode == "counts":
	# 	script_dp += "_co"
	# elif FLAGS.mode == "sample_freqs":
	# 	script_dp += "_sf"
	# elif FLAGS.mode == "tumour_freqs":
	# 	script_dp += "_tf"
	# script_dp += "_" + FLAGS.train_split
	# os.makedirs(script_dp, exist_ok=True)
	
	if FLAGS.tumour_set == "all":
		tumour_list = sorted(chrmlib.ALL_SET)
	else: # FLAGS.tumour_set == "reduced"
		tumour_list = sorted(chrmlib.REDUCED_SET)
	
	for i in range(len(cfps)):
		# create the script
		cfp = cfps[i]
		assert os.path.isfile(cfp)
		muts_file_name = cfps[i]
		m = chrms[i].get_unique_pos_count()
		t = len(tumour_list)
		k = chrms[i]._get_num_segs(seg_size)
		chrm_id = chrms[i].get_chrm_id()
		e_f_fp = os.path.join(results_dp,"E_f_chrm_{}.dat".format(chrm_id))
		s_s_fp = os.path.join(results_dp,"S_s_chrm_{}.dat".format(chrm_id))
		e_s_fp = os.path.join(results_dp,"E_s_chrm_{}.dat".format(chrm_id))
		cmd = f"{exe_path} {quick_test} {muts_file_name} {m} {t} {k} {mp} {e_f_fp} {s_s_fp} {e_s_fp} {prev_k} {min_size} {h_pen}"
		# copy of the script for reference
		script_fp = os.path.join(script_dp,"chrm_{}.sh".format(chrm_id))
		with open(script_fp, "w") as script_file:
			print("#!/bin/bash\n\n" + cmd + "\n", end="", file=script_file)
		# the actual script that is run, then deleted
		tmp_script_fp = os.path.join(tmp_script_dp,"chrm_{}.sh".format(chrm_id))
		with open(tmp_script_fp, "w") as script_file:
			print("#!/bin/bash\n\n" + cmd + "\n", end="", file=script_file)
		os.chmod(tmp_script_fp, os.stat(tmp_script_fp).st_mode | stat.S_IEXEC) # chmod +x
		# run it
		job_name = "chrm_{}".format(chrm_id)
		stdout_fp = os.path.join(stdout_dp, "chrm_{}.out".format(chrm_id))
		stderr_fp = os.path.join(stderr_dp, "chrm_{}.err".format(chrm_id))
		srun_cmd = f"srun --nodes=1 --cpus-per-task={num_cores} --time={time} --job-name={job_name} --output={stdout_fp} --error={stderr_fp} {tmp_script_fp}"
		print(f"submitting job {job_name}")
		sp.Popen(shlex.split(srun_cmd))
		# # delete the temporary script file
		# os.remove(tmp_script_fp)
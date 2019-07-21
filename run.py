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


parser = argparse.ArgumentParser()
parser.add_argument("--cfile_dir_path", type=str, default="/home/q/qmorris/youngad2/MutSeg/cfiles_kat")
parser.add_argument("--mode", type=str, choices=["counts", "sample_freqs", "tumour_freqs"], default="sample_freqs")
parser.add_argument("--chrm_id", type=int, default=-1, choices=list(range(-1,22)), help="chromosome id (starts at 0)")
parser.add_argument("--naive_seg_size", type=int, default=1000000) # 1 Megabase
parser.add_argument("--num_cores", type=int, default=80, choices=list(range(1,81)), help="number of logical cores required on scinet, should be <=80")
parser.add_argument("--output_dir_path", type=str, default="/scratch/q/qmorris/youngad2/{}".format(today_date()))
parser.add_argument("--max_time", type=int, default=24, choices=list(range(1,25)), help="wall time in hours, should be <=24")
parser.add_argument("--exe_path", type=str, default="/home/q/qmorris/youngad2/MutSeg/segmentation", help="path to segmentation executable")
parser.add_argument("--script_dir_path", type=str, default="/home/q/qmorris/youngad2/MutSeg/scripts", help="directory for creating bash scripts")
parser.add_argument("--overwrite", type=lambda x:bool(strtobool(x)), default=False)

if __name__ == "__main__":

	FLAGS = parser.parse_args()
	assert os.path.isdir(FLAGS.cfile_dir_path)
	mode_dir_path = os.path.join(FLAGS.cfile_dir_path,FLAGS.mode)
	assert os.path.isdir(mode_dir_path)
	mc_file_path = os.path.join(mode_dir_path,"run_data.pkl")
	assert os.path.isfile(mc_file_path)
	with open(mc_file_path, "rb") as pkl_file:
		mc_data = pickle.load(pkl_file)
	assert len(mc_data) == chrmlib.NUM_CHRMS

	if FLAGS.chrm_id == -1:
		cfile_paths = []
		chrms = mc_data
		for c in range(chrmlib.NUM_CHRMS):
			cfile_name = "{}_{}.dat".format(chrmlib.CFILE_BASE,c)
			cfile_paths.append(os.path.join(mode_dir_path,cfile_name))
	else:
		# do a specific chromosome
		cfile_name = "{}_{}.dat".format(chrmlib.CFILE_BASE,FLAGS.chrm_id)
		cfile_paths = [os.path.join(mode_dir_path,cfile_name)]
		chrms = [mc_data[FLAGS.chrm_id]]
	
	# attributes that are always the same
	assert os.path.isfile(FLAGS.exe_path)
	exe_path = FLAGS.exe_path
	quick_test = 0
	mp = FLAGS.num_cores
	#total_ks = [chrm_len // FLAGS.seg_size + chrm_len % FLAGS.seg_size for chrm_len in chrmlib.CHRM_LENS]
	#total_m = sum([chrm.get_unique_pos_count() for chrm in mc_data])
	seg_size = FLAGS.naive_seg_size
	prev_k = 0
	if not FLAGS.overwrite:
		assert not os.path.exists(FLAGS.output_dir_path)
	output_dir_path = FLAGS.output_dir_path
	if FLAGS.mode == "counts":
		output_dir_path += "_co"
	elif FLAGS.mode == "sample_freqs":
		output_dir_path += "_sf"
	elif FLAGS.mode == "tumour_freqs":
		output_dir_path += "_tf"
	os.makedirs(output_dir_path, exist_ok=True)
	results_dir_path = os.path.join(output_dir_path,"results")
	os.makedirs(results_dir_path, exist_ok=True)
	stdout_dir_path = os.path.join(output_dir_path,"stdouts")
	os.makedirs(stdout_dir_path, exist_ok=True)
	stderr_dir_path = os.path.join(output_dir_path,"stderrs")
	os.makedirs(stderr_dir_path, exist_ok=True)
	assert FLAGS.max_time >= 1 and FLAGS.max_time <= 24
	time = "{}:00:00".format(FLAGS.max_time)
	assert FLAGS.num_cores <= 80
	num_cores = FLAGS.num_cores
	script_dir_path = FLAGS.script_dir_path
	os.makedirs(script_dir_path, exist_ok=True)
	
	for i in range(len(cfile_paths)):
		# create the script
		cfile_path = cfile_paths[i]
		assert os.path.isfile(cfile_path)
		muts_file_name = cfile_paths[i]
		m = chrms[i].get_unique_pos_count()
		t = chrms[i].get_num_cancer_types()
		k = chrms[i]._get_num_segs(seg_size)
		chrm_id = chrms[i].get_chrm_id()
		e_f_fp = os.path.join(results_dir_path,"E_f_chrm_{}.dat".format(chrm_id))
		s_s_fp = os.path.join(results_dir_path,"S_s_chrm_{}.dat".format(chrm_id))
		e_s_fp = os.path.join(results_dir_path,"E_s_chrm_{}.dat".format(chrm_id))
		cmd = f"{exe_path} {quick_test} {muts_file_name} {m} {t} {k} {mp} {e_f_fp} {s_s_fp} {e_s_fp} {prev_k}"
		script_file_path = os.path.join(script_dir_path,"chrm_{}.sh".format(chrm_id))
		with open(script_file_path, "w") as script_file:
			print("#!/bin/bash\n\n" + cmd + "\n", end="", file=script_file)
		os.chmod(script_file_path, os.stat(script_file_path).st_mode | stat.S_IEXEC) # chmod +x
		# run it
		job_name = "chrm_{}".format(chrm_id)
		stdout_fp = os.path.join(stdout_dir_path, "chrm_{}.out".format(chrm_id))
		stderr_fp = os.path.join(stderr_dir_path, "chrm_{}.err".format(chrm_id))
		srun_cmd = f"srun --nodes=1 --cpus-per-task={num_cores} --time={time} --job-name={job_name} --output={stdout_fp} --error={stderr_fp} {script_file_path}"
		print(f"submitting job {job_name}")
		#print(srun_cmd)
		sp.Popen(shlex.split(srun_cmd))

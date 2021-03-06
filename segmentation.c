#include "k_seg.h"

int main(int argc, char *argv[]) {
	// argv[1] == quick test (0/1, whether to run a quick test or not)
	// argv[2] == muts_file_name (str, path for a .dat file that contains a MxT array for a specific chromosome)
	// argv[3] == M (int, number of unique mutation positions)
	// argv[4] == T (int, number of cancer types)
	// argv[5] == K (int, number of segments)
	// argv[6] == mp (0/1, whether or not to use multiprocessing)
	// argv[7] == E_f file name (str)
	// argv[8] == S_s file name (str)
	// argv[9] == E_s file name (str)
	// argv[10] == prev_K (int)
	// argv[11] == min_size(int)
	// argv[12] == h_pen (double)

	int expected_argc = 13;
	if (argc != expected_argc) {
		fprintf(stderr, "Error: argc = %d, should be %d\n", argc, expected_argc);
		return -1;
	}

	// get commandline arguments
	int quick_test = atoi(argv[1]);
	char *muts_file_name = argv[2];
	int M = atoi(argv[3]);
	int T = atoi(argv[4]);
	int K = atoi(argv[5]);
	int mp = atoi(argv[6]);
	char *E_f_file_name = argv[7];
	char *S_s_file_name = argv[8];
	char *E_s_file_name = argv[9];
	int prev_K = atoi(argv[10]);
	int min_size = atoi(argv[11]);
	double h_pen = atof(argv[12]);

	// these are not passed as arguments... for now
	// int min_size = 1;
	int seeds[10];
	int num_seeds = 0;

	double *muts = calloc( M*T, sizeof(double) );
	if (!muts) {
		perror("muts calloc");
		return -1;
	}

	if (quick_test) {
		int random_seed = 303;
		srand(random_seed);
		int num_muts, offset, mut_ptr;

		for (int i = 0; i < M; i++) {
			num_muts = 1 + (rand() % T);
			//printf("muts[%d] = %d\n", i, num_muts);
			offset = rand() % T;
			for (int j = 0; j < num_muts; j++) {
				mut_ptr = (offset + j) % T;
				assert(mut_ptr < T);
				muts[ i*T+mut_ptr ] = 1.;
			}
		}
	} else {
		FILE *muts_fp = fopen(muts_file_name, "r");
		if (!muts_fp) {
			perror("muts file open");
			return -1;
		}
		int num_read = fread( muts, sizeof(double), M*T, muts_fp );
		if (num_read != M*T) {
			perror("muts file read");
			return -1;
		}
		if (fclose(muts_fp)) {
			perror("muts file close");
			return -1;
		}

	}

	double final_score;
	int ret_val;
	ret_val = k_seg(muts, M, T, K, min_size, seeds, num_seeds, E_f_file_name, S_s_file_name, E_s_file_name, &final_score, mp, prev_K, h_pen);
	if (ret_val < 0) {
		fprintf(stderr, "Error: program terminated early\n");
		return -1;
	} else {
		if (!prev_K) {
			printf("final_score = %f\n", final_score);
		} else {
			printf("intermediate score = %f\n", final_score);
		}
		
	}

	uint32_t *final_seg = malloc((K+prev_K)*sizeof(uint32_t));
	if (!final_seg) {
		perror("final_seg malloc");
		return -1;
	}
	ret_val = traceback(S_s_file_name, M, K+prev_K, final_seg);
	if (ret_val < 0) {
		fprintf(stderr, "Error: program terminated early\n");
		return -1;
	} else {
		print_path(final_seg, K+prev_K);
	}

	return 0;

}
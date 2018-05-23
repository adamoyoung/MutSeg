#include "k_seg.h"

int main(int argc, char *argv[]) {
	// argv[1] == quick test
	// argv[2] == muts_file_name (not important if the thing is a quick test)
	// argv[3] == M
	// argv[4] == T
	// argv[5] == K
	// argv[6] == mp

	if (argc != 7) {
		printf("Error: usage\n");
		return -1;
	}

	int quick_test = atoi(argv[1]);
	char *muts_file_name = argv[2];
	int M = atoi(argv[3]);
	int T = atoi(argv[4]);
	int K = atoi(argv[5]);
	int mp = atoi(argv[6]);
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
	
	// these are not passed as arguments... for now
	int min_size = 1;
	int seeds[10];
	int num_seeds = 0;
	char *E_f_file_name = "E_f_file.dat";
	char *S_s_file_name = "S_s_file.dat";

	double final_score;
	int ret_val;
	ret_val = k_seg(muts, M, T, K, min_size, seeds, num_seeds, E_f_file_name, S_s_file_name, &final_score, mp);
	if (ret_val < 0) {
		fprintf(stderr, "Error: program terminated early\n");
		return -1;
	} else {
		printf("final_score = %f\n", final_score);
	}

	uint32_t *final_seg = malloc(K*sizeof(uint32_t));
	if (!final_seg) {
		perror("final_seg malloc");
		return -1;
	}
	ret_val = traceback(S_s_file_name, M, K, final_seg);
	if (ret_val < 0) {
		fprintf(stderr, "Error: program terminated early\n");
		return -1;
	} else {
		print_path(final_seg, K);
	}

	return 0;

}
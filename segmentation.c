#include "k_seg.h"

int main(int argc, char *argv[]) {
	// argv[1] == muts_file_name
	// argv[2] == M
	// argv[3] == T
	// argv[4] == K

	if (argc != 5) {
		printf("Error: usage\n");
		return -1;
	}

	FILE *muts_fp = fopen(argv[1], "r");
	if (!muts_fp) {
		perror("muts file open");
		return -1;
	}

	int M = atoi(argv[2]);
	int T = atoi(argv[3]);
	int K = atoi(argv[4]);
	double *muts = malloc( M*T*sizeof(double) );
	if (!muts) {
		perror("muts malloc");
		return -1;
	}

	int num_read = fread( muts, sizeof(double), M*T, muts_fp );
	if (num_read != M*T) {
		perror("muts file read");
		return -1;
	}

	// these are not passed as arguments... for now
	int min_size = 100;
	int seeds[10];
	int num_seeds = 0;
	char *E_f_file_name = "E_f_file.dat";
	char *S_s_file_name = "S_s_file.dat";

	double final_score;
	int ret_val = k_seg(muts, M, T, K, min_size, seeds, num_seeds, E_f_file_name, S_s_file_name, &final_score);
	if (ret_val < 0) {
		printf("Error: program terminated early\n");
		return -1;
	} else {
		printf("final_score = %f\n", final_score);
	}

	return 0;

}
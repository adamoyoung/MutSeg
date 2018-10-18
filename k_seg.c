#include "k_seg.h"

int compute_counts(double *muts, int M, int T, double **C_ptr) {
	// muts is (M,T)

	double *C = calloc( (M+1)*T, sizeof(double) );
	if (!C) {
		perror("Error");
		return -1;
	}

	for (int i = 1; i < M+1; i++) {
		for (int j = 0; j < T; j++) {
			C[ i*T+j ] = C[ (i-1)*T+j ] + muts[ (i-1)*T+j ];
		}
	}

	*C_ptr = C;

	return 0;

}

double score(int start, int end, double *C, double M_total, int T, int *seeds, int num_seeds) {

	double tumour_C[T];
	double total_C = 0.;

	// check that segmentation does not violate seeds
	for (int i = 0; i < num_seeds; i++) {
		if (seeds[i] > start && seeds[i] < end) {
			return -INFINITY;
		}	
	}

	for (int j = 0; j < T; j++) {
		tumour_C[j] = C[ end*T+j ] - C[ start*T+j ];
		total_C += tumour_C[j];
	}

	double term_one = 0.;
	double term_two = - (total_C / M_total) * log( (total_C / M_total) + EPS );

	for (int j = 0; j < T; j++) {
		term_one += (tumour_C[j] / M_total) * log( (tumour_C[j] / M_total) + EPS );
	}

	return term_one + term_two;

}

double compute_M_total(double *C, int M, int T) {
	// C is (M+1,T)

	double M_total = 0.;

	for (int j = 0; j < T; j++) {
		M_total += C[ M*T+j ];
	}

	return M_total;

}

int evict(uint32_t *S_w, int k, int M, FILE *S_s_fp) {
	// remove all rows from memory except for the last one
	// S_w is (W,M)
	
	assert(k > 0);
	int num_rows = (k+1) % W;
	if (!num_rows) {
		// (k+1) is a multiple of W, evict entire bank
		num_rows = W;
	}
	int num_elements = num_rows*M;
	int num_written = fwrite( S_w, sizeof(uint32_t), num_elements, S_s_fp );
	if (num_written != num_elements) {
		perror("Evict S_w write");
		return -1;
	}

	printf("M=%d, num_rows=%d, num_elements=%d, num_written=%d\n\n", M, num_rows, num_elements, num_written);

	return 0;

}

void print_double_array(double *array, int dim_1, int dim_2) {

	for (int i = 0; i < dim_1; i++) {
		for (int j = 0; j < dim_2; j++) {
			printf("%.2f ", array[ i*dim_2+j ]);
		}
		printf("\n");
	}

}

void print_uint32_array(uint32_t *array, int dim_1, int dim_2) {

	for (int i = 0; i < dim_1; i++) {
		for (int j = 0; j < dim_2; j++) {
			printf("%d ", array[ i*dim_2+j ]);
		}
		printf("\n");
	}
}

void update_table(double *E_cur, uint32_t *S_cur, double *E_prev, int mp, int M, int k, int min_size, double *C, double M_total, int T, int *seeds, int num_seeds) {
	
	if (mp) {

		#pragma omp parallel for num_threads(mp)
		for (int i = 0; i < M; i++) {
			double max_score = -INFINITY;
			uint32_t max_seg = 0;
			double temp_score;
			if (i >= k) {
				for (int j = k; j < i+1; j++) {
					if (i-(j-1) < min_size) {
						break;
					}
					temp_score = E_prev[ j-1 ] + score(j,i+1,C,M_total,T,seeds,num_seeds);
					if (temp_score > max_score) {
						max_score = temp_score;
						max_seg = j-1;
					}
				}
			}
			E_cur[ 0*M+i ] = max_score;
			S_cur[ 0*M+i ] = max_seg;
		}

	} else {
		
		for (int i = 0; i < M; i++) {
			double max_score = -INFINITY;
			uint32_t max_seg = 0;
			double temp_score;
			if (i >= k) {
				for (int j = k; j < i+1; j++) {
					if (i-(j-1) < min_size) {
						break;
					}
					temp_score = E_prev[ j-1 ] + score(j,i+1,C,M_total,T,seeds,num_seeds);
					if (temp_score > max_score) {
						max_score = temp_score;
						max_seg = j-1;
					}
				}
			}
			E_cur[ i ] = max_score;
			S_cur[ i ] = max_seg;
		}

	}

}

int k_seg(double *muts, int M, int T, int K, int min_size, int *seeds, int num_seeds, const char *E_f_file_name, const char *S_s_file_name, const char *E_s_file_name, double *final_score, int mp, int prev_K) {

	assert(M < MAX_UINT32);
	assert(prev_K >= 0);
	
	// timekeeping variables
	double begin, end;
	double time_spent;

	// open data files
	char *fp_mode;
	if (!prev_K) {
		fp_mode = "w";
	} else {
		fp_mode = "a+";
	}
	FILE *E_f_fp = fopen(E_f_file_name, fp_mode);
	if (!E_f_fp) {
		perror("E_f_open");
		return -1;
	}
	FILE *S_s_fp = fopen(S_s_file_name, fp_mode);
	if (!S_s_fp) {
		perror("S_s_open");
		return -1;
	}
	FILE *E_s_fp = fopen(E_s_file_name, fp_mode);
	if (!E_s_fp) {
		perror("E_s_open");
		return -1;
	}

	double *C = NULL;
	compute_counts(muts,M,T,&C);
	assert(C);

	double M_total = compute_M_total(C,M,T);

	free(muts);
	muts = NULL;

	// working sets
	double *E_w = calloc( W*M, sizeof(double) ); // (W,M)
	if (!E_w) {
		perror("E_w_calloc");
		return -1;
	}
	uint32_t *S_w = calloc( W*M, sizeof(uint32_t) ); // (W,M)
	if (!S_w) {
		perror("S_w_calloc");
		return -1;
	}

	// final scores to save on disk
	double *E_f = calloc( K, sizeof(double) ); // (K,)
	if (!E_f) {
		perror("E_f_calloc");
		return -1;
	}

	// previous scores
	double *E_l = calloc( M, sizeof(double) ); // (M,)
	if (!E_l) {
		perror("E_l calloc");
		return -1;
	}


	if (prev_K) {
		// load previous state and store in S_s[0] and E_s[0]
		printf(">>> load k = %d\n", prev_K);
		begin = omp_get_wtime();
		int seek, read;
		// E_s file
		seek = fseek(E_s_fp, -M*sizeof(double), SEEK_END);
		if (seek) {
			perror("E_s seek");
			return -1;
		}
		read = fread(E_l, sizeof(double), M, E_s_fp);
		if (read != M) {
			perror("E_s read");
			return -1;
		}
		printf("E_f[%d] = %f\n", prev_K, E_l[M-1]);
		// // move the E_s pointer back to the end of the file
		// seek = fseek(E_s_fp, 0, SEEK_END);
		// if (seek) {
		// 	perror("E_s seek");
		// 	return -1;
		// }
		// S_s file
		// seek = fseek(S_s_fp, -M*sizeof(uint32_t), SEEK_END);
		// if (seek) {
		// 	perror("S_s seek");
		// 	return -1;
		// }
		// read = fread(S_w, sizeof(uint32_t), M, S_s_fp);
		// if (read != M) {
		// 	perror("S_s seek");
		// 	return -1;
		// }
		// // move the S_s pointer back so that the last row gets overwritten in eviction
		// seek = fseek(S_s_fp, -M*sizeof(uint32_t), SEEK_END);
		// if (seek) {
		// 	perror("S_s seek");
		// 	return -1;
		// }
		end = omp_get_wtime();
		time_spent = (end-begin);
		printf("time spent = %fs\n\n", time_spent);
	}

	if (mp) {
		printf("Using %d threads\n\n", mp);
	}
	
	if (!prev_K) {
		// initial run
		printf(">>> init k = 1\n");
		begin = omp_get_wtime();
		for (int i = 0; i < M; i++) {
			if (i+1 < min_size) {
				E_w[ 0*M+i ] = -INFINITY;
			} else {
				E_w[ 0*M+i ] = score(0,i+1,C,M_total,T,seeds,num_seeds);
				S_w[ 0*M+i ] = 0;
			}
		}
		E_f[0] = E_w[ 0*M+(M-1) ];
		printf("E_f[1] = %f\n", E_f[0]);
	} else {
		// initial run uses E_l array
		printf(">>> k = %d\n", prev_K+1);
		begin = omp_get_wtime();
		update_table(&E_w[0],&S_w[0],&E_l[0],mp,M,prev_K,min_size,C,M_total,T,seeds,num_seeds);
		E_f[0] = E_w[ 0*M+(M-1) ];
		printf("E_f[%d] = %f\n", prev_K+1, E_w[M-1]);
	}
	end = omp_get_wtime();
	time_spent = (end-begin);
	printf("time spent = %fs\n\n", time_spent);

	// indexing
	uint32_t I_w = 0;
	uint32_t I_w_prev = -1;

	for (int k = 1; k < K; k++) {
		printf(">>> k = %d\n",prev_K+k+1);
		begin = omp_get_wtime();
		I_w_prev = I_w;
		I_w = (I_w+1) % W;

		update_table(&E_w[I_w*M],&S_w[I_w*M],&E_w[I_w_prev*M],mp,M,prev_K+k,min_size,C,M_total,T,seeds,num_seeds);
		
		// update the final array
		E_f[k] = E_w[ I_w*M+(M-1) ];
		printf("E_f[%d] = %f\n", prev_K+k+1, E_f[k]);
		//print_double_array(E_w, W, M);

		end = omp_get_wtime();
		time_spent = (end-begin);
		printf("time spent = %fs\n\n", time_spent);

		// push modifications onto disk if necessary
		if ( (I_w+1) % W == 0 || (k+1) == K ) {
			printf(">>> eviction!\n");
			begin = omp_get_wtime();
			if (evict(S_w,k,M,S_s_fp) < 0) {
				return -1;
			}
			end = omp_get_wtime();
			time_spent = (end-begin);
			printf("time spent = %fs\n\n", time_spent);
		}
		
	}

	*final_score = E_f[K-1];
	fwrite( E_f, sizeof(double), K, E_f_fp );
	double *E_s = &E_w[I_w*M];
	fwrite( E_s, sizeof(double), M, E_s_fp );
	//print_double_array(E_f, K, 1);
	//print_uint32_array(S_w, K, M);

	// perform cleanup
	free(E_f);
	free(E_w);
	free(S_w);
	free(E_l);
	if (fclose(E_f_fp)) {
		perror("E_f close");
		return -1;
	}
	if (fclose(S_s_fp)) {
		perror("S_w close");
		return -1;
	}
	if (fclose(E_s_fp)) {
		perror("E_s close");
		return -1;
	}

	return 0;

}

int traceback(char *S_s_file_name, int M, int K, uint32_t *final_seg) {
	// note: K is the column that you want to start the traceback from

	assert(M >= 0 && K >= 0);

	FILE *S_s_fp = fopen(S_s_file_name, "r");
	if (!S_s_fp) {
		perror("S_s open");
		return -1;
	}

	// uint32_t S_s[K*M];
	// if (fread(&S_s, sizeof(uint32_t), K*M, S_s_fp) != K*M) {
	// 	perror("S_s read");
	// 	return -1;
	// }
	// print_uint32_array(S_s, K, M);

	final_seg[K-1] = M;
	uint32_t k = K-1;
	uint32_t col = M-1;

	while (k > 0) {
		// row = S_s[row,k]
		if (fseek(S_s_fp, (k*M + col)*sizeof(uint32_t), SEEK_SET)) {
			perror("S_s seek");
			return -1;
		}
		if ( fread(&col, sizeof(uint32_t), 1, S_s_fp) != 1 ) {
			perror("S_s read");
			return -1;
		}
		final_seg[k] = col+1;
		k--;
	}
	final_seg[0] = 0;

	if (fclose(S_s_fp)) {
		perror("S_s close");
		return -1;
	}

	return 0;
}

void print_path(uint32_t *final_seg, int K) {

	printf("[");
	for (int i = 0; i < K-1; i++) {
		printf("%d, ", final_seg[i]);
	}
	printf("%d]\n", final_seg[K-1]);

}

// int main(int argc, char *argv[]) {

// 	int M = 1000;
// 	int T = 40;
// 	int K = 10;
// 	int min_size = 1;
// 	int num_seeds = 0;
// 	int seeds[M];
// 	char * S_s_file_name = "S_s_file.dat";
// 	char * E_f_file_name = "E_f_file.dat";
// 	int mp = 1;

// 	double *muts = calloc( M*T, sizeof(double) ); // (M,T)
// 	if (!muts) {
// 		perror("Error");
// 		return -1;
// 	}

// 	int random_seed = 303;
// 	srand(random_seed);
// 	int num_muts, offset, mut_ptr;

// 	for (int i = 0; i < M; i++) {
// 		num_muts = 1 + (rand() % T);
// 		//printf("muts[%d] = %d\n", i, num_muts);
// 		offset = rand() % T;
// 		for (int j = 0; j < num_muts; j++) {
// 			mut_ptr = (offset + j) % T;
// 			assert(mut_ptr < T);
// 			muts[ i*T+mut_ptr ] = 1.;
// 		}
// 	}

// 	//print_double_array(muts, M, T);
// 	double final_score;
// 	int ret_val = k_seg(muts, M, T, K, min_size, seeds, num_seeds, E_f_file_name, S_s_file_name, &final_score, mp);
// 	if (ret_val < 0) {
// 		fprintf(stderr, "Error: program terminated early\n");
// 	} else {
// 		printf("final_score = %f\n", final_score);
// 	}
	
// 	uint32_t *final_seg = malloc(K*sizeof(uint32_t));
// 	if (!final_seg) {
// 		perror("final_seg malloc");
// 		return -1;
// 	}
// 	// ret_val = traceback(S_s_file_name, M, K, final_seg);
// 	// if (ret_val < 0) {
// 	// 	fprintf(stderr, "Error: program terminated early\n");
// 	// 	return -1;
// 	// } else {
// 	// 	print_path(final_seg, K);
// 	// }

// 	return 0;

// }
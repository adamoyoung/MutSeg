#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#define EPS 2.2204460492503131e-16
#define MAX_UINT32 4294967295
#define W 2

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

double score(int start, int end, double *C, int M_total, int T, int *seeds, int num_seeds) {

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

int evict(uint32_t *S_w, int index, int M, FILE *S_w_fp) {

	int size_write = fwrite( &S_w[ index*M ], sizeof(uint32_t), M, S_w_fp);
	if (size_write != M) {
		perror("Evict S_w write");
		return -1;
	}
	return 0;

}

void print_double_array(double *array, int dim_1, int dim_2) {

	for (int i = 0; i < dim_1; i++) {
		for (int j = 0; j < dim_2; j++) {
			printf("%f ", array[ i*dim_2+j ]);
		}
		printf("\n");
	}

}

int k_seg(double *muts, int M, int T, int K, int min_size, int *seeds, int num_seeds, const char *E_f_file_name, const char *S_w_file_name) {

	assert(M < MAX_UINT32);

	// open data files and write array dimensions
	FILE *E_f_fp = fopen(E_f_file_name, "w");
	if (!E_f_fp) {
		perror("E_f_open");
		return 1;
	}
	if (fprintf(E_f_fp,"%d\n",K) < 0) {
		perror("E_f_write");
		return 1;
	}
	FILE *S_w_fp = fopen(S_w_file_name, "w");
	if (!S_w_fp) {
		perror("S_w_open");
		return 1;
	}
	if (fprintf(S_w_fp,"%d,%d\n",M,K) < 0) {
		perror("S_w_write");
		return 1;
	}

	double *C = NULL;
	compute_counts(muts,M,T,&C);
	assert(C);

	double M_total = compute_M_total(C,M,T);

	free(muts);
	muts = NULL;

	// working sets
	double *E_w = malloc( W*M*sizeof(double) ); // (W,M)
	if (!E_w) {
		perror("E_w_malloc");
		return -1;
	}
	uint32_t *S_w = malloc( W*M*sizeof(uint32_t) ); // (W,M)
	if (!S_w) {
		perror("S_w_malloc");
		return 1;
	}

	// save on disk
	double *E_f = malloc( K*sizeof(double) ); // (K,)
	if (!E_f) {
		perror("E_f_malloc");
		return 1;
	}
	// can't declare S_s array -- too big

	// indexing
	uint32_t *I_w = malloc( K*sizeof(uint32_t) ); // (K,)
	if (!I_w) {
		perror("I_w_malloc");
		return 1;
	}


	for (int i = 0; i < M; i++) {
		if (i+1 < min_size) {
			E_w[ 0*M+i ] = -INFINITY;
		} else {
			E_w[ 0*M+i ] = score(0,i+1,C,M_total,T,seeds,num_seeds);
			// S_w[ i*W+0 ] = 0;
		}
	}

	I_w[0] = 0;
	E_f[0] = E_w[ 0*M+(M-1) ];
	// int usage_count = 0;
	// int most_recently_used = 0;

	for (int k = 1; k < K; k++) {
		I_w[k] = (I_w[k-1] + 1) % W;
		printf("I_w[%d] = %d\n", k, I_w[k]);
		for (int i = k; i < M; i++) {
			double max_score = -INFINITY;
			uint32_t max_seg = 0;
			double temp_score;
			for (int j = k; j < i+1; j++) {
				if (i-(j-1) < min_size) {
					break;
				}
				temp_score = E_w[ I_w[k-1]*M+(j-1) ] + score(j,i+1,C,M_total,T,seeds,num_seeds);
				if (temp_score > max_score) {
					max_score = temp_score;
					max_seg = j-1;
				}
			}
			E_w[ I_w[k]*M+i ] = max_score;
			S_w[ I_w[k]*M+i ] = max_seg;
		}
		// update the final array
		E_f[k] = E_w[ I_w[k]*M+(M-1) ];
		// push modifications onto disk
		if (k > 1) { // we don't actually want S_w[i,0] on disk
			// evict previous index
			if (evict(S_w,I_w[k-1],M,S_w_fp) < 0) {
				return 1;
			}
		}
	}

	double final_score = E_f[K-1];
	print_double_array(E_f, K, 1);
	fwrite( E_f, sizeof(double), K, E_f_fp );
	free(E_f);

	return final_score;

}

int main(int argc, char *argv[]) {

	int M = 10;
	int T = 2;
	int K = 4;
	int min_size = 1;
	int num_seeds = 0;
	int seeds[10];
	char * S_w_file_name = "S_w_file.dat";
	char * E_f_file_name = "E_f_file.dat";

	double *muts = calloc( M*T, sizeof(double) ); // (M,T)
	if (!muts) {
		perror("Error");
		return 1;
	}

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

	print_double_array(muts, M, T);
	double final_score = k_seg(muts, M, T, K, min_size, seeds, num_seeds, E_f_file_name, S_w_file_name);
	printf("final_score = %f\n", final_score);

	return 0;

}
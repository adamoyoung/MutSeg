#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define EPS 2.2204460492503131e-16
#define MAX_UINT32 4294967295
#define W 100

int compute_counts(double *muts, int M, int T, double **C_ptr);

double score(int start, int end, double *C, int M_total, int T, int *seeds, int num_seeds);

double compute_M_total(double *C, int M, int T);

int evict(uint32_t *S_w, uint32_t *I_w, int k, int M, int K, FILE *S_w_fp);

void print_double_array(double *array, int dim_1, int dim_2);

int k_seg(double *muts, int M, int T, int K, int min_size, int *seeds, int num_seeds, const char *E_f_file_name, const char *S_w_file_name, double *final_score);
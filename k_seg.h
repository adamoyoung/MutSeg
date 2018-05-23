#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <sys/sysinfo.h>

#define EPS 2.2204460492503131e-16
#define MAX_UINT32 4294967295
#define W 500

int compute_counts(double *muts, int M, int T, double **C_ptr);

double score(int start, int end, double *C, int M_total, int T, int *seeds, int num_seeds);

double compute_M_total(double *C, int M, int T);

int evict(uint32_t *S_w, uint32_t *I_w, int k, int M, int K, FILE *S_w_fp);

void print_double_array(double *array, int dim_1, int dim_2);

void print_uint32_array(uint32_t *array, int dim1, int dim2);

int k_seg(double *muts, int M, int T, int K, int min_size, int *seeds, int num_seeds, const char *E_f_file_name, const char *S_w_file_name, double *final_score, int mp);

int traceback(char *S_s_file_name, int M, int K, uint32_t *final_seg);

void print_path(uint32_t *final_seg, int K);
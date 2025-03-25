#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#ifdef SIMD
#include <arm_neon.h>
#endif

#include "common.h"
#include "compute.h"

void zero_z() {
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            Z[i][j] = 0;
        }
    }
}

void compute_row_major_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_mkn() {
    zero_z();
    for (int i = 0; i != m; ++i) { 
        for (int l = 0; l != k; ++l) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_kmn() {
    zero_z();
    for (int l = 0; l != k; ++l) {
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_nmk() {
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int i = 0; i != m; ++i) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_nkm() {
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int l = 0; l != k; ++l) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_row_major_knm() {
    zero_z();
    for (int l = 0; l != k; ++l) {
        for (int j = 0; j != n; ++j) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
    // TODO: task 1
}

void compute_y_transpose_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * YP[j][l];
            }
        }
    }
}

void compute_row_major_mnkkmn_generic(int B) {
    zero_z();
    for (int kk = 0; kk < k; kk += B) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Z[i][j];
                for (int k_ = kk; k_ < kk + B && k_ < k; k_++) {
                    sum += X[i][k_] * Y[k_][j];
                }
                Z[i][j] = sum;
            }
        }
    }
}
void compute_row_major_mnkkmn_b32() {
    compute_row_major_mnkkmn_generic(32);
}

void compute_row_major_mnkkmn_b16() {
    compute_row_major_mnkkmn_generic(16);
}
void compute_row_major_mnkkmn_b64() {
    compute_row_major_mnkkmn_generic(64);
}
void compute_row_major_mnk_lu2() {
    zero_z();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            int k_;
            for (k_ = 0; k_ < k - 1; k_ += 2) {
                sum += X[i][k_] * Y[k_][j];
                sum += X[i][k_ + 1] * Y[k_ + 1][j];
            }
            // Handle odd k
            if (k_ < k) {
                sum += X[i][k_] * Y[k_][j];
            }
            Z[i][j] = sum;
        }
    }
}
void compute_row_major_mnkkmn_b64_lu4() {
    zero_z();
    int B = 64;
    for (int kk = 0; kk < k; kk += B) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Z[i][j];
                int k_end = (kk + B < k) ? (kk + B) : k;
                int k_idx;
                for (k_idx = kk; k_idx <= k_end - 4; k_idx += 4) {
                    sum += X[i][k_idx]   * Y[k_idx][j] +
                           X[i][k_idx+1] * Y[k_idx+1][j] +
                           X[i][k_idx+2] * Y[k_idx+2][j] +
                           X[i][k_idx+3] * Y[k_idx+3][j];
                }
                for (; k_idx < k_end; k_idx++) {
                    sum += X[i][k_idx] * Y[k_idx][j];
                }
                Z[i][j] = sum;
            }
        }
    }
}
void compute_simd() {
#ifdef SIMD
    // TODO: task 3
#endif
}

uint64_t elapsed(const struct timespec start, const struct timespec end) {
    struct timespec result;
    result.tv_sec = end.tv_sec - start.tv_sec;
    result.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += SEC;
    }
    uint64_t res = result.tv_sec * SEC + result.tv_nsec;
    return res;

}

uint64_t compute() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    switch (COMPUTE_SELECT) {
        case COMPUTE_ROW_MAJOR_MNK:
            //printf("COMPUTE_ROW_MAJOR_MNK\n");
            compute_row_major_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MKN:
            //printf("COMPUTE_ROW_MAJOR_MKN\n");
            compute_row_major_mkn();
            break;
        case COMPUTE_ROW_MAJOR_KMN:
            //printf("COMPUTE_ROW_MAJOR_KMN\n");
            compute_row_major_kmn();
            break;
        case COMPUTE_ROW_MAJOR_NMK:
            //printf("COMPUTE_ROW_MAJOR_NMK\n");
            compute_row_major_nmk();
            break;
        case COMPUTE_ROW_MAJOR_NKM:
            //printf("COMPUTE_ROW_MAJOR_NKM\n");
            compute_row_major_nkm();
            break;
        case COMPUTE_ROW_MAJOR_KNM:
            //printf("COMPUTE_ROW_MAJOR_KNM\n");
            compute_row_major_knm();
            break;
        case COMPUTE_Y_TRANSPOSE_MNK:
            //printf("COMPUTE_Y_TRANSPOSE_MNK\n");
            compute_y_transpose_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B32:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B32\n");
            compute_row_major_mnkkmn_b32();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B16:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B16\n");
            compute_row_major_mnkkmn_b16();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B64:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B16\n");
            compute_row_major_mnkkmn_b64();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B64_LU4:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B16\n");
            compute_row_major_mnkkmn_b64_lu4();
            break;
        case COMPUTE_ROW_MAJOR_MNK_LU2:
            //printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
            compute_row_major_mnk_lu2();
            break;
        
        case COMPUTE_SIMD:
            //printf("COMPUTE_SIMD\n");
            compute_simd();
            break;
        default:
            printf("Unreachable!");
            return 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed(start, end);
}

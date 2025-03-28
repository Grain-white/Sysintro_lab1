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
    
void compute_row_major_mnkkmn_b32() {
    zero_z();
    int B=32;
    
    for (int mm = 0; mm < m; mm += B) { 
    for (int nn = 0; nn < n; nn += B) { 
    for (int kk = 0; kk < k; kk += B) { 
        for (int l = kk; l < kk + B && l < k; ++l) {
            for (int i = mm; i < mm + B && i < m; ++i) {
                for (int j = nn; j < nn + B && j < n; ++j) { 
                     Z[i][j]+= X[i][l] * Y[l][j];
                }
            }
        }      
    }
    }
    }
}
    
void compute_row_major_mnk_lu2() {
    zero_z();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (k % 2 == 1) {
            Z[i][j] += X[i][k - 1] * Y[k - 1][j];
            }
            for (int k_ = 0; k_ < k-1; k_ += 2) { 
                Z[i][j] += X[i][k_] * Y[k_][j];
                Z[i][j] += X[i][k_+1] * Y[k_+1][j];
            }

            
        
        }
    }
}

void compute_knmknm_b64_lu4(){
    zero_z();
    //int B=64;
    for (int kk = 0; kk < k; kk += 64) { 
    for (int nn = 0; nn < n; nn += 64) { 
    for (int mm = 0; mm < m; mm += 64) { 
        for (int l = kk; l < kk + 64 && l<k; ++l) {
            for (int j = nn; j < nn + 64 && j<n; ++j) {   
                for (int i = mm; i < mm + 64 && i<m-3; i+=4) {
                    Z[i][j] += X[i][l] * Y[l][j];
                    Z[i+1][j] += X[i+1][l] * Y[l][j];
                    Z[i+2][j] += X[i+2][l] * Y[l][j];
                    Z[i+3][j] += X[i+3][l] * Y[l][j];
                /*if (kk == k-1){
                    int r = i - k;
                    switch (r) {
                        case 4:
                        case 1:
                            Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                            Z[i][j] += X[i][k - 2] * Y[k - 2][j];
                        case 2:Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                        case 0:
                            Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                            Z[i][j] += X[i][k - 2] * Y[k - 2][j];
                            Z[i][j] += X[i][k - 1] * Y[k - 3][j];       
                    } 
                } */                 
                }
            }
        }      
    }
    }
    }
}
void compute_knmknm_b16_lu2(){
    zero_z(); 
    for (int kk = 0; kk < k; kk += 8) {
    for (int mm = 0; mm < m; mm += 8) { 
    for (int nn = 0; nn < n; nn += 8) { 
        for (int l = kk; l < kk + 8 && l<k; ++l) { 
            for (int i = mm; i < mm + 8 && i<m; ++i) {
                for (int j = nn; j < nn + 8 && j<n; ++j) {   
                    Z[i][j] += X[i][l] * Y[l][j];
                    //Z[i][j] += X[i][l+1] * Y[l+1][j];         
                }
            }
        }      
    }
    }
    }
}
void compute_y_t_mnk_lu4() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l < k-3; l+=4) {
                Z[i][j] += X[i][l] * YP[j][l];
                Z[i][j] += X[i][l+1] * YP[j][l+1];
                Z[i][j] += X[i][l+2] * YP[j][l+2];
                Z[i][j] += X[i][l+3] * YP[j][l+3];
            }
        }
    }
}
void compute_y_t_mnk_b64_lu4() {
    zero_z();
    const int B=64;
    for (int mm = 0; mm < m; mm += B) { 
            for (int nn = 0; nn < n; nn += B) {
                for (int kk = 0; kk < k; kk += B) { 
                for (int i = mm; i != mm + B &&i<m; ++i) {
                    for (int j = nn; j != nn +B && j<n; ++j) {
                        for (int l = kk; l < kk+B-3 && l<k-3; l+=4) {
                            Z[i][j] += X[i][l] * YP[j][l];
                            Z[i][j] += X[i][l+1] * YP[j][l+1];
                            Z[i][j] += X[i][l+2] * YP[j][l+2];
                            Z[i][j] += X[i][l+3] * YP[j][l+3];
                        }
                    }
                }
            }
        }
    }
}
void compute_knmmnk_b64_lu4(){
    zero_z();
    int B=64;
    /*for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Z[i][j] = 0; 
        }
    }*/
    for (int kk = 0; kk < k; kk += B) { 
    for (int nn = 0; nn < n; nn += B) { 
    for (int mm = 0; mm < m; mm += B) { 
        for (int i = mm; i < mm + B && i < m; ++i) {
            for (int j = nn; j < nn + B && j < n; ++j) { 
                if (kk == k-1){
                    int r = k % 4;
                    switch (r) {
                        case 0:
                        case 2:
                            Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                            Z[i][j] += X[i][k - 2] * Y[k - 2][j];
                        case 1:Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                        case 3:
                            Z[i][j] += X[i][k - 1] * Y[k - 1][j];
                            Z[i][j] += X[i][k - 2] * Y[k - 2][j];
                            Z[i][j] += X[i][k - 1] * Y[k - 3][j];       
                    } 
                }  
                for (int l = kk; l < kk + B && l < k; l+=4) {
                    Z[i][j] += X[i][l] * Y[l][j];
                    Z[i][j] += X[i][l+1 ] * Y[l + 1][j];
                    Z[i][j] += X[i][l+2] * Y[l + 2][j];
                    Z[i][j] += X[i][l+3] * Y[l + 3][j];           
                }
            }
        }      
    }
    }
    }
}
void compute_simd() {
    #ifdef SIMD
        int i, j, l;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                uint64_t sum = 0;
                if (k >= 8) {
                    for (l = 0; l <= k - 8; l += 8) {

                        uint16x8_t vx16 = vld1q_u16(&X16[i][l]);
                        uint16x8_t vy16 = vld1q_u16(&YP16[j][l]);

                        uint16x4_t vx16_low = vget_low_u16(vx16);
                        uint16x4_t vy16_low = vget_low_u16(vy16);
                        uint16x4_t vx16_high = vget_high_u16(vx16);
                        uint16x4_t vy16_high = vget_high_u16(vy16);

                        uint32x4_t prod_low  = vmull_u16(vx16_low, vy16_low);
                        uint32x4_t prod_high = vmull_u16(vx16_high, vy16_high);

                        uint32x2_t prod_low_low   = vget_low_u32(prod_low);
                        uint32x2_t prod_low_high  = vget_high_u32(prod_low);
                        uint32x2_t prod_high_low  = vget_low_u32(prod_high);
                        uint32x2_t prod_high_high = vget_high_u32(prod_high);
                        
                        uint64x2_t prod_ll = vmovl_u32(prod_low_low);
                        uint64x2_t prod_lh = vmovl_u32(prod_low_high);
                        uint64x2_t prod_hl = vmovl_u32(prod_high_low);
                        uint64x2_t prod_hh = vmovl_u32(prod_high_high);
                        
                        uint64x2_t sum_vec = vaddq_u64(prod_ll, prod_lh);
                        sum_vec = vaddq_u64(sum_vec, prod_hl);
                        sum_vec = vaddq_u64(sum_vec, prod_hh);
                        
                        uint64_t tmp[2];
                        vst1q_u64(tmp, sum_vec);
                        sum += tmp[0] + tmp[1];
                    }
                } else {
                    l = 0;
                }
                for (; l < k; l++) {
                    uint64_t prod = (uint64_t)X16[i][l] * (uint64_t)YP16[j][l];
                    sum += prod;
                }
                Z[i][j] = sum;
            }
        }
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
        case COMPUTE_ROW_MAJOR_KNMKNM_B64_LU4:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B16\n");
            compute_knmknm_b64_lu4();
            break;
        case COMPUTE_ROW_MAJOR_KNMKNM_B16_LU2:
            compute_knmknm_b16_lu2();
            break;
        case COMPUTE_ROW_MAJOR_MNK_LU2:
            //printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
            compute_row_major_mnk_lu2();
            break;
        case COMPUTE_T_MNK_LU4:
            compute_y_t_mnk_lu4();
            break;
        case COMPUTE_T_MNK_B64_LU4:
            compute_y_t_mnk_b64_lu4();
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

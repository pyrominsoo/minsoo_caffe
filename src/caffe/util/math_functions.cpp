#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <stdlib.h>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/Fixed.h"
#include "caffe/util/log_mult.hpp"
#include "caffe/util/DRUM_mult.hpp"
#include "caffe/util/mitchk_mult.hpp"
#include "caffe/util/mitchk_bias.hpp"
#include "caffe/util/mitchk_bias_lg.hpp"
#include "caffe/util/mitchk_bias_c1.hpp"
#include "caffe/util/mitchk_c1.hpp"
#include "caffe/util/asm_mult.hpp"
#include "minsoo/fixed.hpp"
#include <fstream>


// The global variable to turn on value reporting
bool gemm_report(false);
bool log_report(false);

// The global variable to decide on mult type
// 1: float, 2: fixed, 3: mitch, 4: iterlog
unsigned int mult_type;
// k value for drum
unsigned int drum_k;


namespace caffe {


using namespace numeric; // From Fixed.h


typedef Fixed<INTBITS,FRACBITS> fixed_f_t;
typedef Fixed<INTBITS,FRACBITS> fixed_d_t;

//typedef Fixed<35,29> fixed_f_t;
//typedef Fixed<35,29> fixed_d_t;

//typedef Fixed<19,13> fixed_f_t;
//typedef Fixed<19,13> fixed_d_t;

//typedef Fixed<11,5> fixed_f_t;
//typedef Fixed<11,5> fixed_d_t;





void minsoo_sgemm_mitchk_bias_lg(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchk_bias_lg.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchk_bias_lg.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitchk_bias_lg(&temp, &op2,drum_k);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_mitchk_bias_lg(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitchk_bias_lg(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitchk_bias_lg(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchk_bias_lg.\n");
        exit(EXIT_FAILURE);
    }
}






































void minsoo_sgemm_mitchk_c1(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchk_c1.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchk_c1.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitchk_c1(&temp, &op2,drum_k);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_mitchk_c1(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitchk_c1(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitchk_c1(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchk_c1.\n");
        exit(EXIT_FAILURE);
    }
}































void asm_sgemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                asm_mult(&temp, &op2);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void asm_sgemv(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                asm_mult(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                asm_mult(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value.\n");
        exit(EXIT_FAILURE);
    }
}
























void minsoo_sgemm_mitchk_bias_c1(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchk_bias_c1.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchk_bias_c1.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitchk_bias_c1(&temp, &op2,drum_k);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_mitchk_bias_c1(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitchk_bias_c1(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitchk_bias_c1(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchk_bias_c1.\n");
        exit(EXIT_FAILURE);
    }
}




















void minsoo_sgemm_mitchk_bias(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchk_bias.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchk_bias.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitchk_bias(&temp, &op2,drum_k);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_mitchk_bias(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitchk_bias(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitchk_bias(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchk_bias.\n");
        exit(EXIT_FAILURE);
    }
}


















void minsoo_sgemm_mitchk(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchk.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchk.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitchk_mult(&temp, &op2,drum_k,INTBITS+FRACBITS);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_mitchk(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitchk_mult(&temp, &temp2,drum_k, INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitchk_mult(&temp, &temp2,drum_k, INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchk.\n");
        exit(EXIT_FAILURE);
    }
}

















void minsoo_sgemm_drum(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_drum.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_drum.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                DRUM_mult(&temp, &op2,drum_k);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}


void minsoo_sgemv_drum(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                DRUM_mult(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                DRUM_mult(&temp, &temp2,drum_k);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_drum.\n");
        exit(EXIT_FAILURE);
    }
}



// Fixed Point + Mitchell log mult
void minsoo_sgemm_mitchell(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
            
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_mitchell.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_mitchell.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t accum = 0;
            fixed_f_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                mitch_mult(&temp, &op2,INTBITS+FRACBITS);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}

void minsoo_dgemm_mitchell(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta, double* C) {

    printf("ERROR: minsoo_dgemm_mitchell used.\n");
    exit(EXIT_FAILURE);

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_d_t op_A[M][K];
    // fixed_d_t op_B[K][N];
    fixed_d_t** op_A = new fixed_d_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_d_t[K];
    }
    fixed_d_t** op_B = new fixed_d_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_d_t[N];
    }


    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemm_mitchell.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_dgemm_mitchell.\n");
        exit(EXIT_FAILURE);
    }

    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_d_t accum = 0;
            fixed_d_t temp;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_d_t op2 = op_B[k_index][col];
                mitch_mult(&temp, &op2,INTBITS+FRACBITS);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_double();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

}


void minsoo_sgemv_mitchell(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;
        
    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitch_mult(&temp, &temp2,INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitch_mult(&temp, &temp2,INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_mitchell.\n");
        exit(EXIT_FAILURE);
    }
}


void minsoo_dgemv_mitchell(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const double alpha, const double* A, const double* x, const double beta, double* y) {
    
    fixed_d_t temp;
    fixed_d_t temp2;
    fixed_d_t accum;

    printf("ERROR: minsoo_dgemv_mitchell used.\n");
    exit(EXIT_FAILURE);

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                mitch_mult(&temp, &temp2,INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                mitch_mult(&temp, &temp2,INTBITS+FRACBITS);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemv_mitchell.\n");
        exit(EXIT_FAILURE);
    }
}





// Fixed Point + Log multiplication
void minsoo_sgemm_logm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }
    

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_logm.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_logm.\n");
        exit(EXIT_FAILURE);
    }

    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for default(shared) collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            fixed_f_t temp;
            fixed_f_t accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                fixed_f_t op2 = op_B[k_index][col];
                log_mult2(&temp, &op2);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();
}

void minsoo_dgemm_logm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta, double* C) {

    printf("ERROR: minsoo_dgemm_logm used.\n");
    exit(EXIT_FAILURE);

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_d_t op_A[M][K];
    // fixed_d_t op_B[K][N];
    fixed_d_t** op_A = new fixed_d_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_d_t[K];
    }
    fixed_d_t** op_B = new fixed_d_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_d_t[N];
    }
    
    fixed_d_t temp;
    fixed_d_t accum;

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemm_logm.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_dgemm_logm.\n");
        exit(EXIT_FAILURE);
    }

    // Fill up C with alpha*op(A)*op(B) + beta*C
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                log_mult2(&temp, &op_B[k_index][col]);
                //temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_double();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

}


void minsoo_sgemv_logm(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {

    fixed_f_t temp;
    fixed_f_t temp2;
    fixed_f_t accum;

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                log_mult2(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                log_mult2(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_logm.\n");
        exit(EXIT_FAILURE);
    }
}


void minsoo_dgemv_logm(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const double alpha, const double* A, const double* x, const double beta, double* y) {
    
    fixed_d_t temp;
    fixed_d_t temp2;
    fixed_d_t accum;


    printf("ERROR: minsoo_dgemv_logm used.\n");
    exit(EXIT_FAILURE);

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp2 = x[col];
                log_mult2(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp2 = x[col];
                log_mult2(&temp, &temp2);
                //temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemv_logm.\n");
        exit(EXIT_FAILURE);
    }
}




// Fixed Point
// 1. Change temporary variables to fixed point
// 2. Change calculation part
// 3. Use to_float() or to_double() to save the value in float or double
// Minsoo Matrix multiplication
void minsoo_sgemm_fixed(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_f_t op_A[M][K];
    // fixed_f_t op_B[K][N];
    fixed_f_t** op_A = new fixed_f_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_f_t[K];
    }
    fixed_f_t** op_B = new fixed_f_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_f_t[N];
    }

    fixed_f_t temp;
    fixed_f_t accum;

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_fixed.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_fixed.\n");
        exit(EXIT_FAILURE);
    }


    // If needed, prepare for the file output
    std::ofstream fout;
    fout.open("gemm.log",ios::out | ios::app);


    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for collapse(2) default(shared) private(accum,temp)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_float();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

    fout.close();


}

void minsoo_dgemm_fixed(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta, double* C) {

    printf("ERROR: minsoo_dgemm_fixed used.\n");
    exit(EXIT_FAILURE);

    // Array in stack memory causes stack overflow. Allocate from heap.
    // fixed_d_t op_A[M][K];
    // fixed_d_t op_B[K][N];
    fixed_d_t** op_A = new fixed_d_t*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new fixed_d_t[K];
    }
    fixed_d_t** op_B = new fixed_d_t*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new fixed_d_t[N];
    }
    
    fixed_d_t temp;
    fixed_d_t accum;

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemm_fixed.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_dgemm_fixed.\n");
        exit(EXIT_FAILURE);
    }

    // Fill up C with alpha*op(A)*op(B) + beta*C
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                temp = op_A[row][k_index];
                temp *= op_B[k_index][col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= C[row*N+col];
            temp += accum;
            C[row*N+col] = temp.to_double();
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

}


void minsoo_sgemv_fixed(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {
    
    fixed_f_t temp;
    fixed_f_t accum;

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_float();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_fixed.\n");
        exit(EXIT_FAILURE);
    }
}


void minsoo_dgemv_fixed(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const double alpha, const double* A, const double* x, const double beta, double* y) {
    
    printf("ERROR: minsoo_dgemv_fixed used.\n");
    exit(EXIT_FAILURE);
    
    fixed_d_t temp;
    fixed_d_t accum;

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                temp = A[col*N+row];
                temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                temp = A[row*N+col];
                temp *= x[col];
                accum += temp;
            }
            accum *= alpha;
            temp = beta;
            temp *= y[row];
            temp += accum;
            y[row] = temp.to_double();
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemv_fixed.\n");
        exit(EXIT_FAILURE);
    }
}

// Floating Point
// Minsoo Matrix multiplication
void minsoo_sgemm_float(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K, 
    const float alpha, const float* A, const float* B, const float beta, float* C) {

    // Array in stack memory causes stack overflow. Allocate from heap.
    // float op_A[M][K];
    // float op_B[K][N];
    float** op_A = new float*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new float[K];
    }
    float** op_B = new float*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new float[N];
    }

    float accum;

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemm_float.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_sgemm_float.\n");
        exit(EXIT_FAILURE);
    }

    // Fill up C with alpha*op(A)*op(B) + beta*C
    #pragma omp parallel for collapse(2) default(shared) private(accum)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                accum = accum + op_A[row][k_index] * op_B[k_index][col];
            }
            C[row*N+col] = alpha * accum + beta * C[row*N+col];
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

}

void minsoo_dgemm_float(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
    const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta, double* C) {

    printf("ERROR: minsoo_dgemm_float used.\n");
    exit(EXIT_FAILURE);

    // Array in stack memory causes stack overflow. Allocate from heap.
    // double op_A[M][K];
    // double op_B[K][N];
    double** op_A = new double*[M];
    for (int i = 0; i < M; i++) {
        op_A[i] = new double[K];
    }
    double** op_B = new double*[K];
    for (int i = 0; i < K; i++) {
        op_B[i] = new double[N];
    }

    double accum;

    // Assumes RowMajor Order

    // Make op(A), which is M x K
    if (TransA == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[row + col*M];
            }
        }
    } else if (TransA == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < K; col++) {
                op_A[row][col] = A[col + row*K];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemm_float.\n");
        exit(EXIT_FAILURE);
    }

    // Make op(B), which is K x N
    if (TransB == CblasTrans) {
        // Perform transpose
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[row + col*K];
            }
        }
    } else if (TransB == CblasNoTrans) {
        // No need to change
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < N; col++) {
                op_B[row][col] = B[col + row*N];
            }
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransB value for minsoo_dgemm_float.\n");
        exit(EXIT_FAILURE);
    }

    // Fill up C with alpha*op(A)*op(B) + beta*C
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            accum = 0;
            for (int k_index = 0; k_index < K; k_index++) {
                accum = accum + op_A[row][k_index] * op_B[k_index][col];
            }
            C[row*N+col] = alpha * accum + beta * C[row*N+col];
        }
    }

    // Free the allocated memory
    for (int i = 0; i < M; i++) {
        delete[] op_A[i];
    }
    delete[] op_A;

    for (int i = 0; i < K; i++) {
        delete[] op_B[i];
    }
    delete[] op_B;

}


void minsoo_sgemv_float(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const float alpha, const float* A, const float* x, const float beta, float* y) {
    
    float accum;

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                accum = accum + A[col*N+row] * x[col];
            }
            y[row] = alpha * accum + beta * y[row];
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                accum = accum + A[row*N+col] * x[col];
            }
            y[row] = alpha * accum + beta * y[row];
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_sgemv_float.\n");
        exit(EXIT_FAILURE);
    }
}


void minsoo_dgemv_float(const CBLAS_TRANSPOSE TransA, 
    const int M, const int N, 
    const double alpha, const double* A, const double* x, const double beta, double* y) {
    
    double accum;
    
    printf("ERROR: minsoo_dgemv_float used.\n");
    exit(EXIT_FAILURE);

    // Assumes RowMajor Order

    if (TransA == CblasTrans) {
        // Transpose case
        for (int row = 0; row < N; row++) {
            accum = 0;
            for (int col = 0; col < M; col++) {
                accum = accum + A[col*N+row] * x[col];
            }
            y[row] = alpha * accum + beta * y[row];
        }
    } else if (TransA == CblasNoTrans) {
        // No transpose case
        for (int row = 0; row < M; row++) {
            accum = 0;
            for (int col = 0; col < N; col++) {
                accum = accum + A[row*N+col] * x[col];
            }
            y[row] = alpha * accum + beta * y[row];
        }
    } else {
        // Invalid value. Trap
        printf("ERROR: Invalid TransA value for minsoo_dgemv_float.\n");
        exit(EXIT_FAILURE);
    }
}




template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // int lda = (TransA == CblasNoTrans) ? K : M;
  // int ldb = (TransB == CblasNoTrans) ? N : K;
  // cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    // ldb, beta, C, N);

    switch (mult_type) {
        case 1 : 
            minsoo_sgemm_float(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 2 :
            minsoo_sgemm_fixed(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 3 :
            minsoo_sgemm_mitchell(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 4 :            
            minsoo_sgemm_logm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 5 :
            minsoo_sgemm_drum(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 6 :
            minsoo_sgemm_mitchk(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 7 :
            minsoo_sgemm_mitchk_bias(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 8 :
            minsoo_sgemm_mitchk_bias_c1(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 9 :
            asm_sgemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 10 :
            minsoo_sgemm_mitchk_c1(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        case 11 :
            minsoo_sgemm_mitchk_bias_lg(TransA, TransB, M, N, K, alpha, A, B, beta, C);
            break;
        default :
            std::cout << "undefined mult_type: " << mult_type << std::endl;
            exit(1);
            break;
    }
  
  // minsoo_sgemm_float(TransA, TransB, M, N, K, alpha, A, B, beta, C);
  // minsoo_sgemm_fixed(TransA, TransB, M, N, K, alpha, A, B, beta, C);
  // minsoo_sgemm_logm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
  // minsoo_sgemm_mitchell(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
    
    printf("ERROR: caffe_cpu_gemm<double> used.\n");
    exit(EXIT_FAILURE);
  // int lda = (TransA == CblasNoTrans) ? K : M;
  // int ldb = (TransB == CblasNoTrans) ? N : K;
  // cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
     // ldb, beta, C, N);
    // minsoo_dgemm_float(TransA, TransB, M, N, K, alpha, A, B, beta, C);
    minsoo_dgemm_fixed(TransA, TransB, M, N, K, alpha, A, B, beta, C);
    // minsoo_dgemm_logm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
    // minsoo_dgemm_mitchell(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
    
    switch (mult_type) {
        case 1 : 
            minsoo_sgemv_float(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 2 :
            minsoo_sgemv_fixed(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 3 :
            minsoo_sgemv_mitchell(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 4 :            
            minsoo_sgemv_logm(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 5 :
            minsoo_sgemv_drum(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 6 :
            minsoo_sgemv_mitchk(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 7 :
            minsoo_sgemv_mitchk_bias(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 8 :
            minsoo_sgemv_mitchk_bias_c1(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 9 :
            asm_sgemv(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 10 :
            minsoo_sgemv_mitchk_c1(TransA, M, N, alpha, A, x, beta, y); 
            break;
        case 11 :
            minsoo_sgemv_mitchk_bias_lg(TransA, M, N, alpha, A, x, beta, y); 
            break;
        default :
            std::cout << "undefined mult_type: " << mult_type << std::endl;
            exit(1);
            break;
    }
  // cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    // minsoo_sgemv_float(TransA, M, N, alpha, A, x, beta, y); 
    // minsoo_sgemv_fixed(TransA, M, N, alpha, A, x, beta, y); 
    // minsoo_sgemv_logm(TransA, M, N, alpha, A, x, beta, y); 
    // minsoo_sgemv_mitchell(TransA, M, N, alpha, A, x, beta, y); 

}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
    
    printf("ERROR: caffe_cpu_gemv<double> used.\n");
    exit(EXIT_FAILURE);
  // cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    // minsoo_dgemv_float(TransA, M, N, alpha, A, x, beta, y);
    minsoo_dgemv_fixed(TransA, M, N, alpha, A, x, beta, y);
    // minsoo_dgemv_logm(TransA, M, N, alpha, A, x, beta, y);
    // minsoo_dgemv_mitchell(TransA, M, N, alpha, A, x, beta, y);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}








}  // namespace caffe

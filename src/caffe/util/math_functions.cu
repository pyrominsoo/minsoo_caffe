#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "hyunjin/custom_maths.cuh"
// for generate random number
#include <time.h>
#include "minsoo/fixed.hpp"
#include <iostream>

// global variables
//_MODE_TYPE   mult_type;         // multiplier mode
extern unsigned int mult_type;         // multiplier mode
extern unsigned int drum_k;
/* unsigned int allnumbits;        // #all bits in format  */
/* unsigned int mantissa_numbits;  // #mantissa bits in format */
/* unsigned int fixed_numbits;     // #fixed width in format */
//unsigned int stage1_k;          // #MSBs in mantissa of first stage 
//unsigned int stage2_k;          // #MSBs in mantissa of second stage 
//unsigned int stage3_k;          // #MSBs in mantissa of third stage 
//_RMODE_TYPE  stage1_rmode;      // rounding mode in first stage 
//_RMODE_TYPE  stage2_rmode;      // rounding mode in second stage 
//_RMODE_TYPE  stage3_rmode;      // rounding mode in third stage 
//_RMODE_TYPE  acc_rmode;         // rounding mode after accumulation

//_DMODE_TYPE  data_mode;         // data mode of logarithmic representation
//unsigned int numbitssampling;   // 2 ^ numbitssampling in logarithmic stochastic rounding 
//unsigned int numbits_lsr;       // bits used as weights in logarithmic stochastic rounding 
#define MULT_SWITCH 12 
#define BLOCK_SIZE 32
#define DRUM_K 4
#define ALLNUMBITS INTBITS+FRACBITS
#include <cstddef> 

namespace caffe {
// Original commented out under this fold
//{{{
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}
//}}}



template <>
void caffe_gpu_gemm_approx<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
//{{{
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  if (mult_type == 1) // FLOAT
  {
    CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    return;
  } 

  // memory for storing results 
  float *dop_A;
  cudaMalloc((void **)&dop_A, sizeof(float) * M * K);

  float *dop_B;
  cudaMalloc((void **)&dop_B, sizeof(float) * K * N);

  const float alpha_scale = 1;
  const float beta_scale = 0;

  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransA,
      K, M, &alpha_scale, A, lda, &beta_scale, A, lda, dop_A, K));
 
//  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransB, cuTransB,
      N, K, &alpha_scale, B, ldb, &beta_scale, B, ldb, dop_B, N));

  // calling matrix multiplication kernal

  dim3 threadsPerBlock(M, N); // x, y
  dim3 blocksPerGrid(1, 1);
  if (N*M > BLOCK_SIZE*BLOCK_SIZE)
  {
    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = BLOCK_SIZE;
    blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
  }

  switch (mult_type) 
  {
    case 2: // FIXED
      mult_bfloat16<<<blocksPerGrid,threadsPerBlock>>>
        (dop_B, dop_A, C, N, M, K, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);     
      break;
    case 3: // FIXED
      mult_bfloat16_ILM1<<<blocksPerGrid,threadsPerBlock>>>
        (dop_B, dop_A, C, N, M, K, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    default :
      std::cout << "undefined mult_type: " << mult_type << std::endl;
      exit(1);
    break;
  }

  cudaDeviceSynchronize();

  // free memory
  cudaFree(dop_A);
  cudaFree(dop_B);

  return;
}
//}}}


template <>
void caffe_gpu_gemm_approx<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {

    std::cout << "ERROR: caffe_gpu_gemm_approx<double> called" << std::endl;
    throw;
}


template <>
void caffe_gpu_gemv_approx<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
//{{{
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

  if (mult_type == 1) // float
  {
    CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
    return;
  }

  // memory for storing results 
  float *dop_A;
  float *dop_B;
  cudaMalloc((void **)&dop_A, sizeof(float) * M * N);

  if (TransA == CblasTrans) 
    cudaMalloc((void **) &dop_B, sizeof(float)*M);
  else if (TransA == CblasNoTrans) 
    cudaMalloc((void **) &dop_B, sizeof(float)*N);

  //int lda = (TransA == CblasNoTrans) ? M : N;

  const float alpha_scale = 1;
  const float beta_scale = 0;

  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransA,
      M, N, &alpha_scale, A, N, &beta_scale, A, N, dop_A, M));

  if (TransA == CblasTrans) 
  {
    CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), M, x, 1, dop_B, 1));
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), M, &alpha_scale, dop_B, 1));
  }
  else if (TransA == CblasNoTrans) 
  {
    CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), N, x, 1, dop_B, 1));
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha_scale, dop_B, 1));
  }

  unsigned int row, col;
  if (TransA == CblasTrans) 
  {
    col = 1;
    row = N;
  }
  else if (TransA == CblasNoTrans) 
  {
    col = 1;
    row = M;
  }

  // calling matrix multiplication kernal
  dim3 threadsPerBlock(col, row); // x, y
  dim3 blocksPerGrid(1, 1);
  if (col*row > BLOCK_SIZE*BLOCK_SIZE)
  {
    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = BLOCK_SIZE;
    blocksPerGrid.x = ceil(double(col)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(row)/double(threadsPerBlock.y));
  }
 
  switch (mult_type) 
  {
    case 2: // FIXED
      mult_bfloat16<<<blocksPerGrid,threadsPerBlock>>>
        (dop_A, dop_B, y, row, col, N, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    case 3: // FIXED
      mult_bfloat16_ILM1<<<blocksPerGrid,threadsPerBlock>>>
//      mult_bfloat16<<<blocksPerGrid,threadsPerBlock>>>
        (dop_A, dop_B, y, row, col, N, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    default :
      std::cout << "undefined mult_type: " << mult_type << std::endl;
      exit(1);
    break;
  }

  cudaDeviceSynchronize();
  // free memory
  cudaFree(dop_A);
  cudaFree(dop_B);
  
  return;
}
//}}}

template <>
void caffe_gpu_gemv_approx<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
    
    std::cout << "ERROR: caffe_gpu_gemv_approx<double> called" << std::endl;
    throw;
    
}

template <>
void caffe_gpu_gemm_approxV2<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
//{{{
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  if (mult_type == 1) // FLOAT
  {
    CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    return;
  } 

  // memory for storing results 
  float *dop_A;
  cudaMalloc((void **)&dop_A, sizeof(float) * M * K);

  float *dop_B;
  cudaMalloc((void **)&dop_B, sizeof(float) * K * N);

  const float alpha_scale = 1;
  const float beta_scale = 0;

  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransA,
      K, M, &alpha_scale, A, lda, &beta_scale, A, lda, dop_A, K));
 
//  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransB, cuTransB,
      N, K, &alpha_scale, B, ldb, &beta_scale, B, ldb, dop_B, N));

  // calling matrix multiplication kernal

  dim3 threadsPerBlock(M, N); // x, y
  dim3 blocksPerGrid(1, 1);
  if (N*M > BLOCK_SIZE*BLOCK_SIZE)
  {
    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = BLOCK_SIZE;
    blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
  }

  switch (mult_type) 
  {
    case 2: // FIXED
      mult_bfloat16<<<blocksPerGrid,threadsPerBlock>>>
        (dop_B, dop_A, C, N, M, K, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    case 3: // FIXED
      mult_bfloat16_ILM2<<<blocksPerGrid,threadsPerBlock>>>
        (dop_B, dop_A, C, N, M, K, drum_k,
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    default :
      std::cout << "undefined mult_type: " << mult_type << std::endl;
      exit(1);
    break;
  }

  cudaDeviceSynchronize();

  // free memory
  cudaFree(dop_A);
  cudaFree(dop_B);

  return;
}
//}}}


template <>
void caffe_gpu_gemm_approxV2<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {

    std::cout << "ERROR: caffe_gpu_gemm_approx<double> called" << std::endl;
    throw;
}


template <>
void caffe_gpu_gemv_approxV2<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
//{{{
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

  if (mult_type == 1) // float
  {
    CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
    return;
  }

  // memory for storing results 
  float *dop_A;
  float *dop_B;
  cudaMalloc((void **)&dop_A, sizeof(float) * M * N);

  if (TransA == CblasTrans) 
    cudaMalloc((void **) &dop_B, sizeof(float)*M);
  else if (TransA == CblasNoTrans) 
    cudaMalloc((void **) &dop_B, sizeof(float)*N);

  //int lda = (TransA == CblasNoTrans) ? M : N;

  const float alpha_scale = 1;
  const float beta_scale = 0;

  // check transposition and scaling 
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransA,
      M, N, &alpha_scale, A, N, &beta_scale, A, N, dop_A, M));

  if (TransA == CblasTrans) 
  {
    CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), M, x, 1, dop_B, 1));
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), M, &alpha_scale, dop_B, 1));
  }
  else if (TransA == CblasNoTrans) 
  {
    CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), N, x, 1, dop_B, 1));
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha_scale, dop_B, 1));
  }

  unsigned int row, col;
  if (TransA == CblasTrans) 
  {
    col = 1;
    row = N;
  }
  else if (TransA == CblasNoTrans) 
  {
    col = 1;
    row = M;
  }

  // calling matrix multiplication kernal
  dim3 threadsPerBlock(col, row); // x, y
  dim3 blocksPerGrid(1, 1);
  if (col*row > BLOCK_SIZE*BLOCK_SIZE)
  {
    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = BLOCK_SIZE;
    blocksPerGrid.x = ceil(double(col)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(row)/double(threadsPerBlock.y));
  }
 
  switch (mult_type) 
  {
    case 2: // Exact 
      mult_bfloat16<<<blocksPerGrid,threadsPerBlock>>>
        (dop_A, dop_B, y, row, col, N, drum_k, 
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    case 3: // ILM
      mult_bfloat16_ILM2<<<blocksPerGrid,threadsPerBlock>>>
        (dop_A, dop_B, y, row, col, N, drum_k, 
        ALLNUMBITS, FRACBITS,  alpha, beta);   
      break;
    default :
      std::cout << "undefined mult_type: " << mult_type << std::endl;
      exit(1);
    break;
  }

  cudaDeviceSynchronize();
  // free memory
  cudaFree(dop_A);
  cudaFree(dop_B);
  
  return;
}
//}}}

template <>
void caffe_gpu_gemv_approxV2<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
    
    std::cout << "ERROR: caffe_gpu_gemv_approx<double> called" << std::endl;
    throw;
    
}
//{{{
template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {

  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {

  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

/*
template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}
*/


DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}
//}}}
}  // namespace caffe

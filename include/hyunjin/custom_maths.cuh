/*****************************************************************************
*   Title      : custom_maths.cuh
*   Desc       : approximate multiplier emulation GPU version
*   Author     : HyunJin Kim
*   Date       : 2019.04.26
*   Ver        : 4.0
*   Description: These class are utilized to describe emulation of multiplier
*                using GPU.
*                M x K matrix and K x N matrix are multiplied, 
*                and then M x N matrix is generated.
*                fixed_f: fixed point with reduced output width
*                mitchk_unbias_c1_f: unbiased truncated Mitchell 
*                                  using one's complement 
*                mitchk_unbias_f: unbiased truncated Mitchell 
*                                  using two's complement 
*                multi_lsr: multiple sampling using logarithmic 
*                           stochastic rounding and float-point format manipulation
****************************************************************************/
#ifndef CUSTOM_MATHS_CUH_ 
#define CUSTOM_MATHS_CUH_ 

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <curand.h>
#include <curand_kernel.h>
__device__ unsigned int xorshift( unsigned int _state);

__global__ void mitchk_unbias_c1_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
      
  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

#ifdef CDEBUG
      printf("ith loop: %d\n", i);
      printf("----------------------------------------\n");
      printf("row: %d, col: %d, i: %d, op_A: %d, op_B: %d\n", row, col, i, _op_A[row * _K + i], _op_B[_N * i + col] );
#endif
      if ((A == 0) || (B == 0))
        continue;

#ifdef CDEBUG
      printf("d%-th loop not zero detected\n", i);
#endif
      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = ~A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = ~B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
#ifdef CDEBUG
      printf("%d-th loop dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
      // End of check & conversion for negativeness, zero, and leading one 
#ifdef CDEBUG
      if ((row ==0) && (col == 0) && (i == 0))
      {
        printf("before truncation dopA: %d, dopB: %d\n", dopA, dopB );
        printf("dopA_lod: %d, dopB_lod: %d\n", dopA_lod, dopB_lod );
      }
#endif
      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _drum_k) 
        dopA = (dopA >> (dopA_lod - _drum_k)); 
      else
        dopA = (dopA << (_drum_k - dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _drum_k) 
      // shift right and left to truncate bits
        dopB = (dopB >> (dopB_lod - _drum_k));
      else
        dopB = (dopB << (_drum_k - dopB_lod));
#ifdef CDEBUG
      printf("%d-th after truncation dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
//      if ((row ==0) && (col == 0) && (i == 0))
//      {
//        printf("after truncation dopA: %d, dopB: %d\n", dopA, dopB );
//      }
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopA + dopB + 1; 
      long long int ma_out;
      if (temp >= __float2ll_rn(exp2f(_drum_k)))   
      {
        if (dopA_lod + dopB_lod + 1 >= _drum_k) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
        else
          ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
      }
      else
      {
        if (dopA_lod + dopB_lod >= _drum_k) 
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) << (dopA_lod + dopB_lod - _drum_k));
        else
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) >> (_drum_k - (dopA_lod + dopB_lod)));
      }
#ifdef CDEBUG
      printf("row: %d, col: %d, i: %d, ma_out: %d\n", row, col, i, ma_out);
#endif
      
      ma_out  = ma_out & maskout;
      // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

      float real_ma_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2float_rn(ma_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2float_rn(~ma_out) / exp2f(_mantissa_numbits* 2);

      sum += real_ma_out; // one's complement conversion
    } // End of for (int i = 0; i < _K; i++)
#ifdef CDEBUG
      printf("sum: %f\n", sum);
#endif
    float accum = sum;
    accum *= _alpha;
    float temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__global__ void mitchk_unbias_c1_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const double _alpha, const double _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __double2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __double2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));


#ifdef CDEBUG
      printf("ith loop: %d\n", i);
      printf("----------------------------------------\n");
      printf("row: %d, col: %d, i: %d, op_A: %d, op_B: %d\n", row, col, i, _op_A[row * _K + i], _op_B[_N * i + col] );
#endif
      if ((A == 0) || (B == 0))
        continue;

#ifdef CDEBUG
      printf("d%-th loop not zero detected\n", i);
#endif
      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = ~A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      // End of make bit width as _allnumbits

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
//        dopB = (B * -1) -1;
        dopB = ~B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
#ifdef CDEBUG
      printf("%d-th loop dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
      // End of check & conversion for negativeness, zero, and leading one 
#ifdef CDEBUG
      if ((row ==0) && (col == 0) && (i == 0))
      {
        printf("before truncation dopA: %d, dopB: %d\n", dopA, dopB );
        printf("dopA_lod: %d, dopB_lod: %d\n", dopA_lod, dopB_lod );
      }
#endif
      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _drum_k) 
        dopA = (dopA >> (dopA_lod - _drum_k)); 
      else
        dopA = (dopA << (_drum_k - dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _drum_k) 
      // shift right and left to truncate bits
        dopB = (dopB >> (dopB_lod - _drum_k));
      else
        dopB = (dopB << (_drum_k - dopB_lod));
#ifdef CDEBUG
      printf("%d-th after truncation dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
//      if ((row ==0) && (col == 0) && (i == 0))
//      {
//        printf("after truncation dopA: %d, dopB: %d\n", dopA, dopB );
//      }
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopA + dopB + 1; 
      long long int ma_out;
      if (temp >= __float2ll_rn(exp2f(_drum_k)))   
      {
        if (dopA_lod + dopB_lod + 1 >= _drum_k) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
        else
          ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
      }
      else
      {
        if (dopA_lod + dopB_lod >= _drum_k) 
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) << (dopA_lod + dopB_lod - _drum_k));
        else
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) >> (_drum_k - (dopA_lod + dopB_lod)));
      }
#ifdef CDEBUG
      printf("row: %d, col: %d, i: %d, ma_out: %d\n", row, col, i, ma_out);
#endif
      
      ma_out = ma_out & maskout;

      double real_ma_out;
//      if (dopA_neg ^ dopB_neg == 0)
//        real_ma_out =__ll2float_rz(ma_out) / (1 << (_mantissa_numbits* 2));
//      else
//        real_ma_out = __ll2float_rz(~ma_out) / (1 << (_mantissa_numbits* 2));
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2double_rn(ma_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2double_rn(~ma_out) / exp2f(_mantissa_numbits* 2);

      sum += real_ma_out; // one's complement conversion
    } // End of for (int i = 0; i < _K; i++)
#ifdef CDEBUG
      printf("sum: %f\n", sum);
#endif
    double accum = sum;
    accum *= _alpha;
    double temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__global__ void mitchk_unbias_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
      
  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

#ifdef CDEBUG
      printf("ith loop: %d\n", i);
      printf("----------------------------------------\n");
      printf("row: %d, col: %d, i: %d, op_A: %d, op_B: %d\n", row, col, i, _op_A[row * _K + i], _op_B[_N * i + col] );
#endif
      if ((A == 0) || (B == 0))
        continue;

#ifdef CDEBUG
      printf("d%-th loop not zero detected\n", i);
#endif
      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = -B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
#ifdef CDEBUG
      printf("%d-th loop dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
      // End of check & conversion for negativeness, zero, and leading one 
#ifdef CDEBUG
      if ((row ==0) && (col == 0) && (i == 0))
      {
        printf("before truncation dopA: %d, dopB: %d\n", dopA, dopB );
        printf("dopA_lod: %d, dopB_lod: %d\n", dopA_lod, dopB_lod );
      }
#endif
      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _drum_k) 
        dopA = (dopA >> (dopA_lod - _drum_k)); 
      else
        dopA = (dopA << (_drum_k - dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _drum_k) 
      // shift right and left to truncate bits
        dopB = (dopB >> (dopB_lod - _drum_k));
      else
        dopB = (dopB << (_drum_k - dopB_lod));
#ifdef CDEBUG
      printf("%d-th after truncation dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
//      if ((row ==0) && (col == 0) && (i == 0))
//      {
//        printf("after truncation dopA: %d, dopB: %d\n", dopA, dopB );
//      }
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopA + dopB + 1; 
      long long int ma_out;
      if (temp >= __float2ll_rn(exp2f(_drum_k)))   
      {
        if (dopA_lod + dopB_lod + 1 >= _drum_k) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
        else
          ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
      }
      else
      {
        if (dopA_lod + dopB_lod >= _drum_k) 
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) << (dopA_lod + dopB_lod - _drum_k));
        else
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) >> (_drum_k - (dopA_lod + dopB_lod)));
      }
#ifdef CDEBUG
      printf("row: %d, col: %d, i: %d, ma_out: %d\n", row, col, i, ma_out);
#endif
      
      ma_out  = ma_out & maskout;
      // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

      float real_ma_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2float_rn(ma_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2float_rn(-ma_out) / exp2f(_mantissa_numbits* 2);

      sum += real_ma_out; // one's complement conversion
    } // End of for (int i = 0; i < _K; i++)
#ifdef CDEBUG
      printf("sum: %f\n", sum);
#endif
    float accum = sum;
    accum *= _alpha;
    float temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__global__ void mitchk_unbias_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const double _alpha, const double _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __double2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __double2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));


#ifdef CDEBUG
      printf("ith loop: %d\n", i);
      printf("----------------------------------------\n");
      printf("row: %d, col: %d, i: %d, op_A: %d, op_B: %d\n", row, col, i, _op_A[row * _K + i], _op_B[_N * i + col] );
#endif
      if ((A == 0) || (B == 0))
        continue;

#ifdef CDEBUG
      printf("d%-th loop not zero detected\n", i);
#endif
      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      // End of make bit width as _allnumbits

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
//        dopB = (B * -1) -1;
        dopB = -B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
#ifdef CDEBUG
      printf("%d-th loop dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
      // End of check & conversion for negativeness, zero, and leading one 
#ifdef CDEBUG
      if ((row ==0) && (col == 0) && (i == 0))
      {
        printf("before truncation dopA: %d, dopB: %d\n", dopA, dopB );
        printf("dopA_lod: %d, dopB_lod: %d\n", dopA_lod, dopB_lod );
      }
#endif
      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _drum_k) 
        dopA = (dopA >> (dopA_lod - _drum_k)); 
      else
        dopA = (dopA << (_drum_k - dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _drum_k) 
      // shift right and left to truncate bits
        dopB = (dopB >> (dopB_lod - _drum_k));
      else
        dopB = (dopB << (_drum_k - dopB_lod));
#ifdef CDEBUG
      printf("%d-th after truncation dopA: %d, dopB: %d\n", i, dopA, dopB );
#endif
//      if ((row ==0) && (col == 0) && (i == 0))
//      {
//        printf("after truncation dopA: %d, dopB: %d\n", dopA, dopB );
//      }
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopA + dopB + 1; 
      long long int ma_out;
      if (temp >= __float2ll_rn(exp2f(_drum_k)))   
      {
        if (dopA_lod + dopB_lod + 1 >= _drum_k) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
        else
          ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
      }
      else
      {
        if (dopA_lod + dopB_lod >= _drum_k) 
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) << (dopA_lod + dopB_lod - _drum_k));
        else
          ma_out = ((temp + __float2ll_rn(exp2f(_drum_k))) >> (_drum_k - (dopA_lod + dopB_lod)));
      }
#ifdef CDEBUG
      printf("row: %d, col: %d, i: %d, ma_out: %d\n", row, col, i, ma_out);
#endif
      
      ma_out = ma_out & maskout;

      double real_ma_out;
//      if (dopA_neg ^ dopB_neg == 0)
//        real_ma_out =__ll2float_rz(ma_out) / (1 << (_mantissa_numbits* 2));
//      else
//        real_ma_out = __ll2float_rz(-ma_out) / (1 << (_mantissa_numbits* 2));
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2double_rn(ma_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2double_rn(-ma_out) / exp2f(_mantissa_numbits* 2);

      sum += real_ma_out; // one's complement conversion
    } // End of for (int i = 0; i < _K; i++)
#ifdef CDEBUG
      printf("sum: %f\n", sum);
#endif
    double accum = sum;
    accum *= _alpha;
    double temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__global__ void fixed_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;

    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));
     if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = -B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      long long int temp = dopA * dopB; 

      temp = temp & maskout;
    
      float real_fixed_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_fixed_out = __ll2float_rn(temp) / exp2f(_mantissa_numbits* 2);
      else
        real_fixed_out = __ll2float_rn(-temp) / exp2f(_mantissa_numbits* 2);
      sum += real_fixed_out; 
    } // End of for (int i = 0; i < _K; i++)
    float accum = sum;
    accum *= _alpha;
    float temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__global__ void fixed_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  unsigned int _fixed_numbits, 
                  const double _alpha, const double _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + (1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  long long int maskout = 0;
  for (int index = (2 * _allnumbits - _fixed_numbits); 
       index < (2 * _allnumbits); index++)
    maskout = maskout + (1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;

    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __double2ll_rn(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __double2ll_rn(_op_B[_K * col + i] * exp2f(_mantissa_numbits));
     if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = -B & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      long long int temp = dopA * dopB; 

      temp = temp & maskout;

      double real_fixed_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_fixed_out = __ll2double_rn(temp) / exp2f(_mantissa_numbits* 2);
      else
        real_fixed_out = __ll2double_rn(-temp) / exp2f(_mantissa_numbits* 2);
      sum += real_fixed_out; 
    } // End of for (int i = 0; i < _K; i++)
    double accum = sum;
    accum *= _alpha;
    double temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

// to be updated: stage1_rmode, acc_rmode, num_sampling, numbits_lsr
// Note: 32-bit fix or log_2(32)=5-bit logarithmic data is assumed.
__global__ void multi_lsr_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  _DMODE_TYPE _data_mode,
                  unsigned int _numbitssampling, unsigned int _numbits_lsr, 
                  int _seed,
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  long long int dopA;
  long long int dopB;
  int dopA_neg;
  int dopB_neg;
  int dopA_lod;
  int dopB_lod;

  // same seed, and differenct sequence in thread
  // shared in one pseudo-random sequence
  //curandState state;
  //if (_stage1_rmode == STC)
  //  curand_init(_seed, row * _N + col, 0, &state);

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
    float sum = 0;
    unsigned int rstate1 = _seed;
    unsigned int rstate2 = _seed;
    //unsigned int rstate1 = 1234567;
    //unsigned int rstate2 = 1234567;

    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rn(_op_A[i * _M + row] * exp2f(16)); 
      long long int B = __float2ll_rn(_op_B[_K * col + i] * exp2f(16));

      if ((A == 0) || (B == 0))
        continue;

      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A;
      }
      else
      {
        dopA_neg = 0;
        dopA = A;
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = -B;
      }
      else
      {
        dopB_neg = 0;
        dopB = B;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
      // End of check & conversion for negativeness, zero, and leading one 

      if (_stage1_rmode == RFLOAT) // Emulation of floating points when #mantissa bits is _numbits_lsr
      {
         
        if ((_data_mode != W_LOG) && (dopA_lod >= _numbits_lsr))
        {
          long long int mask = 0;  
          for (unsigned int mask_index = (dopA_lod-_numbits_lsr); mask_index < (dopA_lod+1); mask_index++)
            mask = mask + __float2ll_rn(exp2f(mask_index));
          
          dopA = dopA & mask; 
        }

        if ((_data_mode != IN_LOG) && (dopB_lod >= _numbits_lsr))
        {
          long long int mask = 0;  
          for (unsigned int mask_index = (dopB_lod-_numbits_lsr); mask_index < (dopB_lod+1); mask_index++)
            mask = mask + __float2ll_rn(exp2f(mask_index));
          
          dopB = dopB & mask; 
        }

      } // End of if (_stage1_rmode == FLOAT)
      else if (_stage1_rmode == STC)
      {
        long long int sumroundedAB= 0;

        // extract weight random of _numbits_lsr bits
        long long int Arw = dopA ^ (1 << dopA_lod);
        long long int Brw = dopB ^ (1 << dopB_lod);

        if (dopA_lod >= _numbits_lsr)
        {
            // rounded to nearest
            if (dopA_lod - _numbits_lsr >= 1)
            {
              long long int rmask = 0;
              rmask = 1 << (dopA_lod - _numbits_lsr - 1);
              if (Arw & rmask)
                Arw = (Arw >> (dopA_lod - _numbits_lsr)) + 1;
              else
                Arw = (Arw >> (dopA_lod - _numbits_lsr));
            }
            else
            Arw = Arw >> (dopA_lod - _numbits_lsr);
        }
        else          
          Arw = Arw << (_numbits_lsr - dopA_lod);

        //////////////////////////////////////////////////////////////////////
        if (dopB_lod >= _numbits_lsr)
        {
            // rounded to nearest
            if (dopB_lod - _numbits_lsr >= 1)
            {
              long long int rmask = 0;
              rmask = 1 << (dopB_lod - _numbits_lsr - 1);
              if (Brw & rmask)
                Brw = (Brw >> (dopB_lod - _numbits_lsr)) + 1;
              else
                Brw = (Brw >> (dopB_lod - _numbits_lsr));
            }
            else
              Brw = Brw >> (dopB_lod - _numbits_lsr);
        }
        else          
          Brw = Brw << (_numbits_lsr - dopB_lod);


        for (unsigned int index = 0; index < __float2uint_rn(exp2f(_numbitssampling)); 
             index++)
        {

          long long int roundedA = 0;
          long long int roundedB = 0;
          // depending on modulo value of randomly generated value
          //unsigned int randAwr = curand(&state) % __float2uint_rn(exp2f(_numbits_lsr));
          rstate1 = xorshift(rstate1);
//          unsigned int randAwr = rstate1 % __float2uint_rn(exp2f(_numbits_lsr)+1);
          unsigned int randAwr = rstate1 % __float2uint_rn(exp2f(_numbits_lsr));

          unsigned int randboundA = (rstate1 >> _numbits_lsr) & 1; 

          if (randAwr > Arw)
            roundedA = __float2ll_rn(exp2f(dopA_lod)); 
          else if ((randAwr == Arw) && randboundA)
            roundedA = __float2ll_rn(exp2f(dopA_lod)); 
          else
            roundedA = __float2ll_rn(exp2f(dopA_lod+1)); 


          // depending on modulo value of randomly generated value
          //unsigned int randBwr = curand(&state) % __float2uint_rn(exp2f(_numbits_lsr));
          // shared random sequence, so comment below line
          rstate2 = xorshift(rstate2);
//          unsigned int randBwr = rstate2 % __float2uint_rn(exp2f(_numbits_lsr)+1);
          unsigned int randBwr = rstate2 % __float2uint_rn(exp2f(_numbits_lsr));

          unsigned int randboundB = (rstate2 >> _numbits_lsr) & 1; 

          if (randBwr > Brw)
            roundedB = __float2ll_rn(exp2f(dopB_lod)); 
          else if ((randBwr == Brw) && randboundB)
            roundedB = __float2ll_rn(exp2f(dopB_lod)); 
          else
            roundedB = __float2ll_rn(exp2f(dopB_lod+1)); 

          //////////////////////////////////////////////////////////////////////
					// when data_mode is W_LOG, input (A) are normal rep..
					if (_data_mode != W_LOG)
					{
						dopA = roundedA;
					}

					// when data_mode is IN_LOG, weight (B) are normal rep..
					if (_data_mode != IN_LOG)
					{
						dopB = roundedB;
					}

					long long int ma_out = dopA * dopB;  

          sumroundedAB +=ma_out;
 
        } // End of for (unsigned int index = 0; index <__float2uint_rn(exp2(_numbitssampling)); index++)

        float real_ma_out;
        if (dopA_neg ^ dopB_neg == 0)
          real_ma_out = __ll2float_rn(sumroundedAB) / exp2f(16* 2);
        else
          real_ma_out = __ll2float_rn(-sumroundedAB) / exp2f(16* 2);

        sum += real_ma_out;
 
      } // End of if (_stage1_rmode == STC)
      else if (_stage1_rmode == R_NEAREST)
      { 
        // when data_mode is not normal, input are logarithmic rep.
        if (_data_mode != W_LOG)
        {
					// rounding depending on next bit to leading one 
					if (dopA_lod == 0)
						dopA = 1 << dopA_lod;
					else if (dopA - (1 << dopA_lod) -(1 << (dopA_lod-1)) < 0)
						dopA = (1 << (dopA_lod)); 
					else
						dopA = (1 << (dopA_lod+1)); 
        }
        
        // when data_mode is not normal, weights are logarithmic rep.
        if (_data_mode != IN_LOG) 
        { 
					if (dopB_lod == 0)
						dopB = 1 << dopB_lod;
					else if (dopB - (1 << dopB_lod) -(1 << (dopB_lod-1)) < 0)
						dopB = (1 << dopB_lod); 
					else
						dopB = (1 << (dopB_lod+1)); 
        }
      }
      else if (_stage1_rmode == R_UP)
      {
        if (_data_mode != W_LOG)
          dopA = (1 << (dopA_lod + 1));

        if (_data_mode != IN_LOG) 
          dopB = (1 << (dopB_lod + 1));
      }
      else if (_stage1_rmode == R_DOWN)
      {
        if (_data_mode != W_LOG)
          dopA = (1 << dopA_lod);
        if (_data_mode != IN_LOG) 
          dopB = (1 << dopB_lod);
      }
      else if (_stage1_rmode == ADD_UNBIAS)
      {
        if (_data_mode != W_LOG)
        {
					dopA = (1 << dopA_lod);
					if (dopA_lod != 0)
						dopA = dopA + (1 << (dopA_lod-1));       
        }

        if (_data_mode != IN_LOG) 
        {
					dopB = (1 << dopB_lod);
					if (dopB_lod != 0)
						dopB = dopB + (1 << (dopB_lod-1));       
        }
      }
      
      if (_stage1_rmode != STC)
      {       
				long long int ma_out = dopA * dopB;  

				float real_ma_out;
				if (dopA_neg ^ dopB_neg == 0)
					real_ma_out = __ll2float_rn(ma_out) / exp2f(16* 2);
				else
					real_ma_out = __ll2float_rn(-ma_out) / exp2f(16* 2);

				sum += real_ma_out; // two's complement conversion
      }
    } // End of for (int i = 0; i < _K; i++)

    if (_stage1_rmode == STC) 
      sum = sum / exp2f(_numbitssampling);    

    float accum = sum;
    accum *= _alpha;
    float temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

// to be updated: stage1_rmode, acc_rmode, num_sampling, numbits_lsr
// Note: 32-bit fix or log_2(32)=5-bit logarithmic data is assumed.
__global__ void multi_lsr_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  _RMODE_TYPE _stage1_rmode, _RMODE_TYPE _acc_rmode,  
                  _DMODE_TYPE _data_mode,
                  unsigned int _numbitssampling, unsigned int _numbits_lsr, 
                  int _seed,
                  const double _alpha, const double _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  long long int dopA;
  long long int dopB;
  int dopA_neg;
  int dopB_neg;
  int dopA_lod;
  int dopB_lod;

  // same seed, and differenct sequence in thread
  // shared in one pseudo-random sequence
  //curandState state;
  //if (_stage1_rmode == STC)
  //  curand_init(_seed, row * _N + col, 0, &state);

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
    double sum = 0;
    unsigned int rstate1 = _seed;
    unsigned int rstate2 = _seed;

    for (int i = 0; i < _K; i++)
    {
      long long int A = __double2ll_rn(_op_A[i * _M + row] * exp2f(16)); 
      long long int B = __double2ll_rn(_op_B[_K * col + i] * exp2f(16));

      if ((A == 0) || (B == 0))
        continue;

      if (A < 0) 
      {
        dopA_neg = 1;
        dopA = -A;
      }
      else
      {
        dopA_neg = 0;
        dopA = A;
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopA >> lod + 1) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = -B;
      }
      else
      {
        dopB_neg = 0;
        dopB = B;
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dopB >> lod + 1) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }
      // End of check & conversion for negativeness, zero, and leading one 

      if (_stage1_rmode == RFLOAT) // Emulation of floating points when #mantissa bits is _numbits_lsr
      {
        if ((_data_mode != W_LOG) && (dopA_lod >= _numbits_lsr))
        {
          long long int mask = 0;  
          for (unsigned int mask_index = (dopA_lod-_numbits_lsr); mask_index < (dopA_lod+1); mask_index++)
            mask = mask + __double2ll_rn(exp2f(mask_index));

          dopA = dopA & mask; 
        }

        if ((_data_mode != IN_LOG) && (dopB_lod >= _numbits_lsr))
        {
          long long int mask = 0;  
          for (unsigned int mask_index = (dopB_lod-_numbits_lsr); mask_index < (dopB_lod+1); mask_index++)
            mask = mask + __double2ll_rn(exp2f(mask_index));
          
          dopB = dopB & mask; 
        }

      } // End of if (_stage1_rmode == FLOAT)
      else if (_stage1_rmode == STC)
      {
        long long int sumroundedAB = 0;

				// extract weight random of _numbits_lsr bits
				long long int Arw = dopA - (1 << dopA_lod);
				long long int Brw = dopB - (1 << dopB_lod);
			 
          if (dopA_lod >= _numbits_lsr)
          {
            // rounded to nearest
            if (dopA_lod - _numbits_lsr >= 1)
            {
              long long int rmask = 0;
              rmask = 1 << (dopA_lod - _numbits_lsr - 1);
              if (Arw & rmask)
                Arw = (Arw >> (dopA_lod - _numbits_lsr)) + 1;
              else
                Arw = (Arw >> (dopA_lod - _numbits_lsr));
            }
            else
						Arw = Arw >> (dopA_lod - _numbits_lsr);
				}
				else          
					Arw = Arw << (_numbits_lsr - dopA_lod);


				if (dopB_lod >= _numbits_lsr)
				{
            // rounded to nearest
            if (dopB_lod - _numbits_lsr >= 1)
            {
              long long int rmask = 0;
              rmask = 1 << (dopB_lod - _numbits_lsr - 1);
              if (Brw & rmask)
                Brw = (Brw >> (dopB_lod - _numbits_lsr)) + 1;
              else
                Brw = (Brw >> (dopB_lod - _numbits_lsr));
            }
            else
						Brw = Brw >> (dopB_lod - _numbits_lsr);
				}
				else          
					Brw = Brw << (_numbits_lsr - dopB_lod);

        for (unsigned int index = 0; index <__float2uint_rn(exp2f(_numbitssampling)); 
             index++)
        {
          long long int roundedA = 0;
          long long int roundedB = 0;

          // depending on modulo value of randomly generated value
          //unsigned int randAwr = curand(&state) % __float2uint_rn(exp2f(_numbits_lsr));

          rstate1 = xorshift(rstate1);
          unsigned int randAwr = rstate1 % __float2uint_rn(exp2f(_numbits_lsr));

          unsigned int randboundA = (rstate1 >> _numbits_lsr) & 1; 

          if (randAwr > Arw)
            roundedA = __float2ll_rn(exp2f(dopA_lod)); 
          else if ((randAwr == Arw) && randboundA)
            roundedA = __float2ll_rn(exp2f(dopA_lod)); 
          else
            roundedA = __float2ll_rn(exp2f(dopA_lod+1)); 

          //////////////////////////////////////////////////////////////////////
          // depending on modulo value of randomly generated value
          //unsigned int randBwr = curand(&state) % __float2uint_rn(exp2f(_numbits_lsr));
          rstate2 = xorshift(rstate2);
          unsigned int randBwr = rstate2 % __float2uint_rn(exp2f(_numbits_lsr));

          unsigned int randboundB = (rstate2 >> _numbits_lsr) & 1; 

          if (randBwr > Brw)
            roundedB = __float2ll_rn(exp2f(dopB_lod)); 
          else if ((randBwr == Brw) && randboundB)
            roundedB = __float2ll_rn(exp2f(dopB_lod)); 
          else
            roundedB = __float2ll_rn(exp2f(dopB_lod+1)); 

					// when data_mode is W_LOG, input (A) are normal rep..
					if (_data_mode != W_LOG)
					{
						// rounding depending on next bit to leading one 
							dopA = roundedA; 
					}
					
					// when data_mode is IN_LOG, weight (B) are normal rep..
					if (_data_mode != IN_LOG) 
					{ 
							dopB = roundedB; 
					}

					long long int ma_out = dopA * dopB;  
          sumroundedAB +=ma_out;

        } // End of for (unsigned int index = 0; index <__double2uint_rn(exp2(_numbitssampling)); index++)

        float real_ma_out;
        if (dopA_neg ^ dopB_neg == 0)
          real_ma_out = __ll2float_rn(sumroundedAB) / exp2f(16* 2);
        else
          real_ma_out = __ll2float_rn(-sumroundedAB) / exp2f(16* 2);

        sum += real_ma_out;

      } // End of if (_stage1_rmode == STC)
      else if (_stage1_rmode == R_NEAREST)
      { 
				// when data_mode is W_LOG, input (A) are normal rep..
        if (_data_mode != W_LOG)
        {
					// rounding depending on next bit to leading one 
					if (dopA_lod == 0)
						dopA = 1 << dopA_lod;
					else if (dopA - (1 << dopA_lod) -(1 << (dopA_lod-1)) < 0)
						dopA = (1 << dopA_lod); 
					else
						dopA = (1 << (dopA_lod+1)); 
        }
        
				// when data_mode is IN_LOG, weight (B) are normal rep..
        if (_data_mode != IN_LOG) 
        { 
					if (dopB_lod == 0)
						dopB = 1 << dopB_lod;
					else if (dopB - (1 << dopB_lod) -(1 << (dopB_lod-1)) < 0)
						dopB = (1 << dopB_lod); 
					else
						dopB = (1 << (dopB_lod+1)); 
        }
      }
      else if (_stage1_rmode == R_UP)
      {
        if (_data_mode != W_LOG)
          dopA = (1 << (dopA_lod + 1));

        if (_data_mode != IN_LOG) 
          dopB = (1 << (dopB_lod + 1));
      }
      else if (_stage1_rmode == R_DOWN)
      {
        if (_data_mode != W_LOG)
          dopA = (1 << dopA_lod);
        if (_data_mode != IN_LOG) 
          dopB = (1 << dopB_lod);
      }
      else if (_stage1_rmode == ADD_UNBIAS)
      {
        if (_data_mode != W_LOG)
        {
					dopA = (1 << dopA_lod);
					if (dopB_lod != 0)
						dopB = dopB + (1 << (dopB_lod-1));       
        }

        if (_data_mode != IN_LOG) 
        {
					dopB = (1 << dopB_lod);
					if (dopB_lod != 0)
						dopB = dopB + (1 << (dopB_lod-1));       
        }
      }
 
      if (_stage1_rmode != STC)
      {
				long long int ma_out = dopA * dopB;  

				double real_ma_out;
				if (dopA_neg ^ dopB_neg == 0)
					real_ma_out = __ll2double_rn(ma_out) / exp2f(16* 2);
				else
					real_ma_out = __ll2double_rn(-ma_out) / exp2f(16* 2);

				sum += real_ma_out; // two's complement conversion
      }
    } // End of for (int i = 0; i < _K; i++)

    if (_stage1_rmode == STC) 
      sum = sum / exp2f(_numbitssampling);    
 
    double accum = sum;
    accum *= _alpha;
    double temp = _beta; 
    temp *= _C[col *_M + row]; // column major order for CUBLAS
    temp += accum;
    _C[col*_M + row] = temp;
     
  } // End of if(row < _M && col < _N) 

  __syncthreads();
  
  return;

}
//}}}

__device__ unsigned int xorshift( unsigned int _state) 
{

	unsigned int x = _state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
  return x;
}

#endif
//{{{
/*
#include <iostream>
#include <Cuda.h>
#include<curand.h>
#include<curand_kernel.h>


int n = 200;
using namespace std;

__device__ float generate( curandState* globalState, int ind ) 
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(float* N, curandState* globalState, int n)
{
    // generate random numbers
    for(int i=0;i<40000;i++)
    {
        int k = generate(globalState, i) * 100000;
        while(k > n*n-1)
        {
            k-=(n*n-1);
        }
        N[i] = k;
    }
}

int main() 
{
    int N=40000;

    curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );

    // setup seeds
    setup_kernel <<< 1, N >>> ( devStates,unsigned(time(NULL)) );

    float N2[40000];
    float* N3;
    cudaMalloc((void**) &N3, sizeof(float)*N);

    kernel<<<1,1>>> (N3, devStates, n);

    cudaMemcpy(N2, N3, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
    {
        cout<<N2[i]<<endl;
    }

    return 0;
} */

//}}

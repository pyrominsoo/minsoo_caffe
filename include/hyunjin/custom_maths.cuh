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

#ifndef NDEBUG
#include <stdio.h>
#define debug(M,...) fprintf(stderr,"DEBUG: %s: %d: " M "\n", __FILE__,__LINE__,##__VA_ARGS__)
#else
#define debug(M,...)
#endif

__device__ unsigned int xorshift( unsigned int _state);

__device__ void float2bfloat(const float src, float& dst) {
	const uint16_t* p = reinterpret_cast<const uint16_t*>(&src);
	uint16_t* q = reinterpret_cast<uint16_t*>(&dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  q[0] = p[0];
  q[1] = 0;
#else
	q[0] = 0;
	q[1] = p[1];
#endif
}
	





__global__ void ma_2stage_c1_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _k1, unsigned int _k2, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits,
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 *_allnumbits - _fixed_numbits
  // no fixed-width multiplication: _fixed_numbits = 2 * _allnumbits
  long long int maskout = 0;
  for (int index = 0; index < (2 * _allnumbits); index++)
    maskout = maskout + ((long long int)1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
      

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
    long long int dopA;
    long long int dopB;
    long long int dopAtrunc;
    long long int dopBtrunc;
    int dopA_neg;
    int dopB_neg;
    int dopA_lod;
    int dopB_lod;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    float sum = 0;
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
      // End of check & conversion for negativeness, zero, and leading one 

      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _k1) 
        dopAtrunc = (dopA >> (dopA_lod - _k1)); 
      else
        dopAtrunc = (dopA << (_k1- dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _k1) 
      // shift right and left to truncate bits
        dopBtrunc = (dopB >> (dopB_lod - _k1));
      else
        dopBtrunc = (dopB << (_k1- dopB_lod));
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopAtrunc + dopBtrunc + 1; //unbias 
      long long int ma_out;
      long long int dop2A;
      long long int dop2B;
      if (temp >= (1 << _k1))   
      {
        if (dopA_lod + dopB_lod + 1 >= _k1) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _k1));
        else
          ma_out = (temp >> (_k1 - (dopA_lod + dopB_lod + 1)));

        dop2A = (1 << dopA_lod) - dopA -1;
        dop2B = (1 << dopB_lod) - dopB -1;
      }
      else
      {
        if (dopA_lod + dopB_lod >= _k1) 
          ma_out = ((temp + (1 << _k1)) << (dopA_lod + dopB_lod - _k1));
        else
          ma_out = ((temp + (1 << _k1)) >> (_k1 - (dopA_lod + dopB_lod)));
    
        dop2A = dopA;
        dop2B = dopB;
      }

      ma_out  = ma_out & maskout;

      unsigned int is_zero_2stage;
			// check any operand of zero
			if ((dop2A == 0) || (dop2B == 0))
				is_zero_2stage = 1;
			else
				is_zero_2stage = 0;

      // 1st stage is finished!!
      int dop2A_lod;
      int dop2B_lod;
      long long int dop2Atrunc;
      long long int dop2Btrunc;

      dop2A_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dop2A >> lod + 1) == 0)
        {
          dop2A_lod = lod;
          break;
        }
      }

      dop2B_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dop2B >> lod + 1) == 0)
        {
          dop2B_lod = lod;
          break;
        }
      }
      // End of check & conversion for negativeness, zero, and leading one 
     
      
      // truncate by leading one and remove leading one
      // remove leading one
      dop2A ^= (1 << dop2A_lod); 

      // shift right and left to truncate bits
      if (dop2A_lod >= _k2) 
        dop2Atrunc = (dop2A >> (dop2A_lod - _k2)); 
      else
        dop2Atrunc = (dop2A << (_k2- dop2A_lod)); 

      // truncation by leading one 
      // remove leading one
      dop2B ^= (1 << dop2B_lod); 

      if (dop2B_lod >= _k2) 
      // shift right and left to truncate bits
        dop2Btrunc = (dop2B >> (dop2B_lod - _k2));
      else
        dop2Btrunc = (dop2B << (_k2- dop2B_lod));
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp2 = dop2Atrunc + dop2Btrunc + 1; //unbias 
      long long int ma2_out;
      if (temp2 >= (1 << _k2))   
      {
        if (dop2A_lod + dop2B_lod + 1 >= _k2) 
          ma2_out = (temp2 << (dop2A_lod + dop2B_lod + 1 - _k2));
        else
          ma2_out = (temp2 >> (_k2 - (dop2A_lod + dop2B_lod + 1)));
      }
      else
      {
        if (dop2A_lod + dop2B_lod >= _k2) 
          ma2_out = ((temp2 + (1 << _k2)) << (dop2A_lod + dop2B_lod - _k2));
        else
          ma2_out = ((temp2 + (1 << _k2)) >> (_k2 - (dop2A_lod + dop2B_lod)));
      }

      // check any operand of zero
      if (is_zero_2stage) 
        ma2_out  = 0;
      else
        ma2_out  = ma2_out & maskout;

      long long int ma12_out = ma_out + ma2_out;

      float real_ma_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2float_rn(ma12_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2float_rn(-ma12_out) / exp2f(_mantissa_numbits* 2);

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

__global__ void ma_2stage_c1_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _k1, unsigned int _k2, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits,
                  const double _alpha, const double _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)1 << index);
  // End of make bit width as _allnumbits

  // make bit width as 2 *_allnumbits - _fixed_numbits
  // no fixed-width multiplication: _fixed_numbits = 2 * _allnumbits
  long long int maskout = 0;
  for (int index = 0; index < (2 * _allnumbits); index++)
    maskout = maskout + ((long long int)1 << index);
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
      

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
    long long int dopA;
    long long int dopB;
    long long int dopAtrunc;
    long long int dopBtrunc;
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
      // End of check & conversion for negativeness, zero, and leading one 

      // truncate by leading one and remove leading one
      // remove leading one
      dopA ^= (1 << dopA_lod); 

      // shift right and left to truncate bits
      if (dopA_lod >= _k1) 
        dopAtrunc = (dopA >> (dopA_lod - _k1)); 
      else
        dopAtrunc = (dopA << (_k1- dopA_lod)); 

      // truncation by leading one 
      // remove leading one
      dopB ^= (1 << dopB_lod); 

      if (dopB_lod >= _k1) 
      // shift right and left to truncate bits
        dopBtrunc = (dopB >> (dopB_lod - _k1));
      else
        dopBtrunc = (dopB << (_k1- dopB_lod));
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp = dopAtrunc + dopBtrunc + 1; //unbias 
      long long int ma_out;
      long long int dop2A;
      long long int dop2B;
      if (temp >= (1 << _k1))   
      {
        if (dopA_lod + dopB_lod + 1 >= _k1) 
          ma_out = (temp << (dopA_lod + dopB_lod + 1 - _k1));
        else
          ma_out = (temp >> (_k1 - (dopA_lod + dopB_lod + 1)));

        dop2A = (1 << dopA_lod) - dopA -1;
        dop2B = (1 << dopB_lod) - dopB -1;
      }
      else
      {
        if (dopA_lod + dopB_lod >= _k1) 
          ma_out = ((temp + (1 << _k1)) << (dopA_lod + dopB_lod - _k1));
        else
          ma_out = ((temp + (1 << _k1)) >> (_k1 - (dopA_lod + dopB_lod)));
    
        dop2A = dopA;
        dop2B = dopB;
      }

      ma_out  = ma_out & maskout;

      unsigned int is_zero_2stage;
			// check any operand of zero
			if ((dop2A == 0) || (dop2B == 0))
				is_zero_2stage = 1;
			else
				is_zero_2stage = 0;

      // 1st stage is finished!!
      int dop2A_lod;
      int dop2B_lod;
      long long int dop2Atrunc;
      long long int dop2Btrunc;


      dop2A_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dop2A >> lod + 1) == 0)
        {
          dop2A_lod = lod;
          break;
        }
      }

      dop2B_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < 32; lod++)
      {
        if ((dop2B >> lod + 1) == 0)
        {
          dop2B_lod = lod;
          break;
        }
      }
      // End of check & conversion for negativeness, zero, and leading one 
     
      
      // truncate by leading one and remove leading one
      // remove leading one
      dop2A ^= (1 << dop2A_lod); 

      // shift right and left to truncate bits
      if (dop2A_lod >= _k2) 
        dop2Atrunc = (dop2A >> (dop2A_lod - _k2)); 
      else
        dop2Atrunc = (dop2A << (_k2- dop2A_lod)); 

      // truncation by leading one 
      // remove leading one
      dop2B ^= (1 << dop2B_lod); 

      if (dop2B_lod >= _k2) 
      // shift right and left to truncate bits
        dop2Btrunc = (dop2B >> (dop2B_lod - _k2));
      else
        dop2Btrunc = (dop2B << (_k2- dop2B_lod));
      // remove leading one, add two truncated mantissa with additional unbiasing value
      // declare device register called "temp"  
      long long int temp2 = dop2Atrunc + dop2Btrunc + 1; //unbias 
      long long int ma2_out;
      if (temp2 >= (1 << _k2))   
      {
        if (dop2A_lod + dop2B_lod + 1 >= _k2) 
          ma2_out = (temp2 << (dop2A_lod + dop2B_lod + 1 - _k2));
        else
          ma2_out = (temp2 >> (_k2 - (dop2A_lod + dop2B_lod + 1)));
      }
      else
      {
        if (dop2A_lod + dop2B_lod >= _k2) 
          ma2_out = ((temp2 + (1 << _k2)) << (dop2A_lod + dop2B_lod - _k2));
        else
          ma2_out = ((temp2 + (1 << _k2)) >> (_k2 - (dop2A_lod + dop2B_lod)));
      }

      // check any operand of zero
      if (is_zero_2stage) 
        ma2_out  = 0;
      else
        ma2_out  = ma2_out & maskout;

      long long int ma12_out = ma_out + ma2_out;

      double real_ma_out;
      if (dopA_neg ^ dopB_neg == 0)
        real_ma_out = __ll2double_rn(ma12_out) / exp2f(_mantissa_numbits* 2);
      else
        real_ma_out = __ll2double_rn(-ma12_out) / exp2f(_mantissa_numbits* 2);

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








__global__ void mult_bfloat16(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
 
  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
			float A = _op_A[i * _M + row];
			float B = _op_B[_K * col + i];
			float tempA = 0;
			float tempB = 0;
			float2bfloat(A, tempA);
			float2bfloat(B, tempB);
			float mult = tempA * tempB;
			float real_ma_out = 0;
			float2bfloat(mult,real_ma_out);
			sum += real_ma_out;
    } // End of for (int i = 0; i < _K; i++)
    //printf("Here \n");
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







__global__ void iterlog2_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : truncate low and high separately to model fixed-point quantization
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }

 
  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;
    int dopA_lod_e;
    int dopB_lod_e;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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
        dopA = ((-1)*A) & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }
      
      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = ((-1)*B) & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

      
      dopA &= ~((long long int)(1) << dopA_lod);
      dopB &= ~((long long int)(1) << dopB_lod);
			
      long long int temp = ((long long int)(1) << (dopA_lod+dopB_lod));
      temp += dopA << dopB_lod;
      temp += dopB << dopA_lod;

      temp = temp & maskout;

      // Error comp
      if (dopA == 0) {
        dopA = 1;
      }

      if (dopB == 0) {
        dopB = 1;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod_e = lod;
          break;
        }
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod_e = lod;
          break;
        }
      }

      dopA &= ~((long long int)(1) << dopA_lod_e);
      dopB &= ~((long long int)(1) << dopB_lod_e);

      long long int temp_err = ((long long int)(1) << (dopA_lod_e + dopB_lod_e));
      temp_err += dopA << dopB_lod_e;
      temp_err += dopB << dopA_lod_e;

      temp_err = temp_err & maskout;

      temp += temp_err;

			long long int ma_out;

			ma_out = temp & maskout; // for 2's comp, this happens before sign conversion
			// for 1's comp, truncation happens after the sign conversion
      ma_out = temp & maskhigh;

			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) != 0) {
				ma_out = ((-1)*ma_out);
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out;
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




__global__ void drum_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
 
  // For unbiasing
  long long int maskdrum = 1;

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;
		int dopA_neg;
		int dopB_neg;
		int dopA_lod;
		int dopB_lod;
    unsigned int shift_back1;
    unsigned int shift_back2;
    unsigned int shift_back;

#ifdef CDEBUG
    printf("row: %d, col: %d\n", row, col);
#endif
    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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
        dopA = ((-1)*A) & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }
      
      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = ((-1)*B) & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

      if (dopA_lod >= _drum_k) {
        shift_back1 = dopA_lod - _drum_k + 1;
        dopA = dopA >> shift_back1;
        dopA = dopA | maskdrum;
      } else {
        shift_back1 = 0;
      }

      if (dopB_lod >= _drum_k) {
        shift_back2 = dopB_lod - _drum_k + 1;
        dopB = dopB >> shift_back2;
        dopB = dopB | maskdrum;
      } else {
        shift_back2 = 0;
      }

      shift_back = shift_back1 + shift_back2;


			long long int temp = dopA * dopB;
			long long int ma_out;

      ma_out = temp << shift_back;
		
			ma_out = ma_out & maskout; // for 2's comp, this happens before sign conversion
			// for 1's comp, truncation happens after the sign conversion
      ma_out = ma_out & maskhigh;

			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) != 0) {
				ma_out = ((-1)*ma_out);
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out;
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





__global__ void mitchk_unbias_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
 
  // For unbiasing
  long long int maskdrum = 1;
  long long int bias;
  if (_drum_k >= 4) {
    bias = __float2ll_rd(exp2f(_drum_k - 4));
  } else {
    bias = 0;
  }

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
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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
        dopA = ((-1)*A) & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }
      
      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = ((-1)*B) & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

		  // truncate by leading one and remove leading one
			// remove leading one
			dopA ^= ((long long int)(1) << dopA_lod); 

			// shift right and left to truncate bits
			if (dopA_lod >= _drum_k) 
				dopA = (dopA >> (dopA_lod - _drum_k)); 
			else
				dopA = (dopA << (_drum_k - dopA_lod)); 

			// truncation by leading one 
			// remove leading one
			dopB ^= ((long long int)(1) << dopB_lod); 

			if (dopB_lod >= _drum_k) 
			// shift right and left to truncate bits
				dopB = (dopB >> (dopB_lod - _drum_k));
			else
				dopB = (dopB << (_drum_k - dopB_lod));
		

      // insert drum bias
      dopA = dopA | maskdrum;
      dopB = dopB | maskdrum;


			long long int temp = dopA + dopB + bias; // include log bias
			long long int ma_out;


			if (temp >= __float2ll_rd(exp2f(_drum_k+1)))   // this can happen because of bias
			{
        temp = temp - __float2ll_rd(exp2f(_drum_k));
				if (dopA_lod + dopB_lod + 2 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 2 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 2)));
				}
			}
			else if (temp >= __float2ll_rd(exp2f(_drum_k)))   // carry happened
			{
				if (dopA_lod + dopB_lod + 1 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
				}
			}
			else  // carry did not happen
			{
				if (dopA_lod + dopB_lod >= _drum_k)  {
					ma_out = ((temp + (1 << _drum_k)) << (dopA_lod + dopB_lod - _drum_k));
				}
				else {
					ma_out = ((temp + (1 << _drum_k)) >> (_drum_k - (dopA_lod + dopB_lod)));
				}
			}
		
			ma_out = ma_out & maskout; // for 2's comp, this happens before sign conversion
			// for 1's comp, truncation happens after the sign conversion
      ma_out = ma_out & maskhigh;

			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) != 0) {
				ma_out = ((-1)*ma_out);
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out;
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



__global__ void mitchk_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
  

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
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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
        dopA = ((-1)*A) & maskAB;
      }
      else
      {
        dopA_neg = 0;
        dopA = A & maskAB;
      }
      
      if (B < 0)
      { 
        dopB_neg = 1;
        dopB = ((-1)*B) & maskAB;
      }
      else
      {
        dopB_neg = 0;
        dopB = B & maskAB;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

		  // truncate by leading one and remove leading one
			// remove leading one
			dopA ^= ((long long int)(1) << dopA_lod); 

			// shift right and left to truncate bits
			if (dopA_lod >= _drum_k) 
				dopA = (dopA >> (dopA_lod - _drum_k)); 
			else
				dopA = (dopA << (_drum_k - dopA_lod)); 

			// truncation by leading one 
			// remove leading one
			dopB ^= ((long long int)(1) << dopB_lod); 

			if (dopB_lod >= _drum_k) 
			// shift right and left to truncate bits
				dopB = (dopB >> (dopB_lod - _drum_k));
			else
				dopB = (dopB << (_drum_k - dopB_lod));
		

			long long int temp = dopA + dopB; 
			long long int ma_out;
			if (temp >= __float2ll_rd(exp2f(_drum_k)))   // carry happened
			{
				if (dopA_lod + dopB_lod + 1 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
				}
			}
			else  // carry did not happen
			{
				if (dopA_lod + dopB_lod >= _drum_k)  {
					ma_out = ((temp + (1 << _drum_k)) << (dopA_lod + dopB_lod - _drum_k));
				}
				else {
					ma_out = ((temp + (1 << _drum_k)) >> (_drum_k - (dopA_lod + dopB_lod)));
				}
			}
		
			ma_out = ma_out & maskout; // for 2's comp, this happens before sign conversion
			// for 1's comp, truncation happens after the sign conversion
      ma_out = ma_out & maskhigh;

			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) != 0) {
				ma_out = ((-1)*ma_out);
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out;
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


__global__ void mitchk_c1_new_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
 
  long long int maskdrum = 1;


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
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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

      // MINSOO : match the behavior to verilog code
      if (dopA == 0) {
        dopA = 1;
      }
      if (dopB == 0) {
        dopB = 1;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

		  // truncate by leading one and remove leading one
			// remove leading one
			dopA ^= ((long long int)(1) << dopA_lod); 

			// shift right and left to truncate bits
			if (dopA_lod >= _drum_k) 
				dopA = (dopA >> (dopA_lod - _drum_k)); 
			else
				dopA = (dopA << (_drum_k - dopA_lod)); 

			// truncation by leading one 
			// remove leading one
			dopB ^= ((long long int)(1) << dopB_lod); 

			if (dopB_lod >= _drum_k) 
			// shift right and left to truncate bits
				dopB = (dopB >> (dopB_lod - _drum_k));
			else
				dopB = (dopB << (_drum_k - dopB_lod));
		
            // insert drum bias
            dopA = dopA | maskdrum;
            dopB = dopB | maskdrum;

			long long int temp = dopA + dopB; 
			long long int ma_out;
			if (temp >= __float2ll_rd(exp2f(_drum_k)))   // carry happened
			{
				if (dopA_lod + dopB_lod + 1 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
				}
			}
			else  // carry did not happen
			{
				if (dopA_lod + dopB_lod >= _drum_k)  {
					ma_out = ((temp + (1 << _drum_k)) << (dopA_lod + dopB_lod - _drum_k));
				}
				else {
					ma_out = ((temp + (1 << _drum_k)) >> (_drum_k - (dopA_lod + dopB_lod)));
				}
			}
		
      ma_out = ma_out & maskhigh;


			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) == 0) {
				ma_out = ma_out & maskout;
			}
			else {
				ma_out = (~ma_out) & maskout;
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out; // one's complement conversion
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


__global__ void mitchk_c1_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
  

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
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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

      // MINSOO : match the behavior to verilog code
      if (dopA == 0) {
        dopA = 1;
      }
      if (dopB == 0) {
        dopB = 1;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

		  // truncate by leading one and remove leading one
			// remove leading one
			dopA ^= ((long long int)(1) << dopA_lod); 

			// shift right and left to truncate bits
			if (dopA_lod >= _drum_k) 
				dopA = (dopA >> (dopA_lod - _drum_k)); 
			else
				dopA = (dopA << (_drum_k - dopA_lod)); 

			// truncation by leading one 
			// remove leading one
			dopB ^= ((long long int)(1) << dopB_lod); 

			if (dopB_lod >= _drum_k) 
			// shift right and left to truncate bits
				dopB = (dopB >> (dopB_lod - _drum_k));
			else
				dopB = (dopB << (_drum_k - dopB_lod));
		

			long long int temp = dopA + dopB; 
			long long int ma_out;
			if (temp >= __float2ll_rd(exp2f(_drum_k)))   // carry happened
			{
				if (dopA_lod + dopB_lod + 1 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
				}
			}
			else  // carry did not happen
			{
				if (dopA_lod + dopB_lod >= _drum_k)  {
					ma_out = ((temp + (1 << _drum_k)) << (dopA_lod + dopB_lod - _drum_k));
				}
				else {
					ma_out = ((temp + (1 << _drum_k)) >> (_drum_k - (dopA_lod + dopB_lod)));
				}
			}
		
      ma_out = ma_out & maskhigh;


			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) == 0) {
				ma_out = ma_out & maskout;
			}
			else {
				ma_out = (~ma_out) & maskout;
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out; // one's complement conversion
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



__global__ void mitchk_unbias_c1_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const float _alpha, const float _beta)
{
//{{{
  unsigned int row =  blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int col =  blockIdx.x*blockDim.x + threadIdx.x ;
  // make bit width as _allnumbits
  long long int maskAB = 0;
  for (int index = 0; index < _allnumbits; index++)
    maskAB = maskAB + ((long long int)(1) << index);
  // End of make bit width as _allnumbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = 0;
  for (int index = 0; index < (_allnumbits+_mantissa_numbits); index++) {
    maskhigh = maskhigh + ((long long int)(1) << index);
  }
 
  long long int maskdrum = 1;
  long long int bias;
  if (_drum_k >= 4) {
    bias = __float2ll_rd(exp2f(_drum_k - 4));
  } else {
    bias = 0;
  }

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
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

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

      // MINSOO : match the behavior to verilog code
      if (dopA == 0) {
        dopA = 1;
      }
      if (dopB == 0) {
        dopB = 1;
      }

      // 32bit fixed-point format is assumed
      // MINSOO : changed 32 to _allnumbits
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopA >> (lod + 1)) == 0)
        {
          dopA_lod = lod;
          break;
        }
      }

      dopB_lod = 0;
      // 32bit fixed-point format is assumed
      for (int lod = 0; lod < _allnumbits; lod++)
      {
        if ((dopB >> (lod + 1)) == 0)
        {
          dopB_lod = lod;
          break;
        }
      }

		  // truncate by leading one and remove leading one
			// remove leading one
			dopA ^= ((long long int)(1) << dopA_lod); 

			// shift right and left to truncate bits
			if (dopA_lod >= _drum_k) 
				dopA = (dopA >> (dopA_lod - _drum_k)); 
			else
				dopA = (dopA << (_drum_k - dopA_lod)); 

			// truncation by leading one 
			// remove leading one
			dopB ^= ((long long int)(1) << dopB_lod); 

			if (dopB_lod >= _drum_k) 
			// shift right and left to truncate bits
				dopB = (dopB >> (dopB_lod - _drum_k));
			else
				dopB = (dopB << (_drum_k - dopB_lod));
	
      // insert drum bias
      dopA = dopA | maskdrum;
      dopB = dopB | maskdrum;

			long long int temp = dopA + dopB + bias;  // includes log bias
			long long int ma_out;

			if (temp >= __float2ll_rd(exp2f(_drum_k+1)))   // carry happened
			{
        temp = temp - __float2ll_rd(exp2f(_drum_k));
				if (dopA_lod + dopB_lod + 2 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 2 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 2)));
				}
			}
			else if (temp >= __float2ll_rd(exp2f(_drum_k)))   // carry happened
			{
				if (dopA_lod + dopB_lod + 1 >= _drum_k) {
					ma_out = (temp << (dopA_lod + dopB_lod + 1 - _drum_k));
				}
				else {
					ma_out = (temp >> (_drum_k - (dopA_lod + dopB_lod + 1)));
				}
			}
			else  // carry did not happen
			{
				if (dopA_lod + dopB_lod >= _drum_k)  {
					ma_out = ((temp + (1 << _drum_k)) << (dopA_lod + dopB_lod - _drum_k));
				}
				else {
					ma_out = ((temp + (1 << _drum_k)) >> (_drum_k - (dopA_lod + dopB_lod)));
				}
			}
		
      ma_out = ma_out & maskhigh;

			float real_ma_out;
			if ((dopA_neg ^ dopB_neg) == 0) {
				ma_out = ma_out & maskout;
			}
			else {
				ma_out = (~ma_out) & maskout;
			}
			real_ma_out = __ll2float_rd(ma_out) / exp2f(_mantissa_numbits* 2);
			sum += real_ma_out; // one's complement conversion
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


__global__ void mitchk_c1_d(
                  const double*_op_A, const double*_op_B, double* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _drum_k, 
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
                  const double _alpha, const double _beta)
{
  printf("ERROR: mitchk_c1_d called");
}


__global__ void fixed_f(
                  const float*_op_A, const float*_op_B, float* _C, 
                  unsigned int _M, unsigned int _N, unsigned int _K,
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
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
  /* long long int maskout = 0; */
  /* for (int index = (2 * _allnumbits - _fixed_numbits);  */
  /*      index < (2 * _allnumbits); index++) */
  /*   maskout = maskout + (1 << index); */
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = exp2f(_mantissa_numbits+_allnumbits);

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;

    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __float2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __float2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));
     /* if (A < 0)  */
     /*  { */
     /*    dopA_neg = 1; */
     /*    dopA = -A & maskAB; */
     /*  } */
     /*  else */
     /*  { */
     /*    dopA_neg = 0; */
     /*    dopA = A & maskAB; */
     /*  } */
     /*  */
     /*  if (B < 0) */
     /*  {  */
     /*    dopB_neg = 1; */
     /*    dopB = -B & maskAB; */
     /*  } */
     /*  else */
     /*  { */
     /*    dopB_neg = 0; */
     /*    dopB = B & maskAB; */
     /*  } */

      dopA = A & maskAB;
      dopB = B & maskAB;

      long long int temp = dopA * dopB; 

      temp = temp & maskout;
      temp = temp % maskhigh;
    
      float real_fixed_out;
      real_fixed_out = __ll2float_rd(temp) / exp2f(_mantissa_numbits* 2);
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
                  unsigned int _allnumbits, unsigned int _mantissa_numbits, 
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
  /* long long int maskout = 0; */
  /* for (int index = (2 * _allnumbits - _fixed_numbits);  */
  /*      index < (2 * _allnumbits); index++) */
  /*   maskout = maskout + (1 << index); */
  // End of make bit width as 2 * _allnumbits: 2 *_allnumbits - _fixed_numbits
  
  // MINSOO : modify so that MSBs are truncated as well as LSBs
  long long int maskout = 0;
  for (int index = _mantissa_numbits; index < (2 * _allnumbits); index++) {
    maskout = maskout + ((long long int)(1) << index);
  }
  long long int maskhigh = exp2f(_mantissa_numbits+_allnumbits);

  // check & conversion for negativeness, zero, and leading one 
  if(row < _M && col < _N) 
  {
		long long int dopA;
		long long int dopB;

    double sum = 0;
    for (int i = 0; i < _K; i++)
    {
      long long int A = __double2ll_rd(_op_A[i * _M + row] * exp2f(_mantissa_numbits)); 
      long long int B = __double2ll_rd(_op_B[_K * col + i] * exp2f(_mantissa_numbits));

      dopA = A & maskAB;
      dopB = B & maskAB;

      long long int temp = dopA * dopB; 

      temp = temp & maskout;
      temp = temp % maskhigh;

      double real_fixed_out;
      real_fixed_out = __ll2double_rd(temp) / exp2f(_mantissa_numbits* 2);
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








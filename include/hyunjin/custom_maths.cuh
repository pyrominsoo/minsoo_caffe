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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>


#define FLOAT_MANT_BITS    (23)
#define FLOAT_EXPO_BITS    (8)
#define FLOAT_EXPO_BIAS    (127)
#define FLOAT_MANT_MASK    (~((~0u) << (FLOAT_MANT_BITS+1))) /* incl. integer bit */
#define EXPO_ADJUST        (1)   /* adjustment for performance reasons */
#define MIN_NORM_EXPO      (1)   /* minimum biased exponent of normals */
#define MAX_NORM_EXPO      (254) /* maximum biased exponent of normals */
#define INF_EXPO           (255) /* biased exponent of infinities */
#define EXPO_MASK          (~((~0u) << FLOAT_EXPO_BITS))
#define FLOAT_SIGN_MASK    (0x80000000u)
#define FLOAT_IMPLICIT_BIT (1 << FLOAT_MANT_BITS)
#define RND_BIT_SHIFT      (31)
#define RND_BIT_MASK       (1u << RND_BIT_SHIFT)
#define FLOAT_INFINITY     (0x7f800000)
#define FLOAT_INDEFINITE   (0xffc00000u)
#define MANT_LSB           (0x00000001)
#define FLOAT_QNAN_BIT     (0x00400000)
#define MAX_SHIFT          (FLOAT_MANT_BITS + 2)

#ifndef NDEBUG
#include <stdio.h>
#define debug(M,...) fprintf(stderr,"DEBUG: %s: %d: " M "\n", __FILE__,__LINE__,##__VA_ARGS__)
#else
#define debug(M,...)
#endif

__device__ unsigned int xorshift( unsigned int _state);
__device__ uint8_t LOD(uint8_t val);
__device__ uint16_t ILM(uint8_t a, uint8_t b, uint8_t iter);
__device__ uint32_t fp32_mul_core (uint32_t a, uint32_t b, uint8_t iter);
__device__  uint32_t uint_as_floatV2 (float a);
__device__ float uint_as_floatV2 (uint32_t a);
__device__ float fp32_mul_ILM (float a, float b, uint8_t iter);


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

__global__ void mult_bfloat16_ILM2(
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
    //printf("Hello from ILM2\n");

    for (int i = 0; i < _K; i++)
    {
			float A = _op_A[i * _M + row];
			float B = _op_B[_K * col + i];
			float tempA = 0;
			float tempB = 0;
			float2bfloat(A, tempA);
			float2bfloat(B, tempB);
			float mult = fp32_mul_ILM(tempA,tempB,2);
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


__global__ void mult_bfloat16_ILM1(
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
    //printf("Hello from ILM1\n");
    for (int i = 0; i < _K; i++)
    {
			float A = _op_A[i * _M + row];
			float B = _op_B[_K * col + i];
			float tempA = 0;
			float tempB = 0;
			float2bfloat(A, tempA);
			float2bfloat(B, tempB);
			float mult = fp32_mul_ILM(tempA,tempB,1);
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




///////// ILM ////////////



__device__ uint8_t LOD(uint8_t val){
    uint32_t n = 0, x;
    x = val;
    if (x <= 0x0000ffff) n += 16, x <<= 16;
    if (x <= 0x00ffffff) n += 8, x <<= 8;
    if (x <= 0x0fffffff) n += 4, x <<= 4;
    if (x <= 0x3fffffff) n += 2, x <<= 2;
    if (x <= 0x7fffffff) n++;
    return 31 - n;
}

__device__ uint16_t ILM(uint8_t a, uint8_t b, uint8_t iter){
    /*
        a, b -> input operands,
        iter -> number of iterations
        only two iterations supported
    */
    if (a == 0 || b == 0) return 0;

    uint8_t Ka, Kb; 
    Ka = LOD(a);
    Kb = LOD(b);

    uint8_t ResA, ResB, Res2B;
    ResA = a ^ (1 << Ka);
    ResB = b ^ (1 << Kb);

    uint16_t prod0, prod1;
    prod0 = a * (1<<Kb) + ResB * (1<<Ka);
    prod1 = 0;
    if(iter == 2){
        if(ResA == 0 || ResB == 0) {
            return prod0;
        }
        Ka = LOD(ResA);
        Kb = LOD(ResB);
        Res2B = ResB ^ (1 << Kb);
        prod1 = ResA * (1<<Kb) + Res2B * (1<<Ka);
    }

    return prod0 + prod1;
}

__device__ uint32_t fp32_mul_core (uint32_t a, uint32_t b, uint8_t iter)
{
    uint64_t prod;
    uint32_t expoa, expob, manta, mantb, shift;
    uint32_t r, signr, expor, mantr_hi, mantr_lo;

    /* split arguments into sign, exponent, significand */
    expoa = ((a >> FLOAT_MANT_BITS) & EXPO_MASK) - EXPO_ADJUST;
    expob = ((b >> FLOAT_MANT_BITS) & EXPO_MASK) - EXPO_ADJUST;
    manta = (a | FLOAT_IMPLICIT_BIT) & FLOAT_MANT_MASK;
    mantb = (b | FLOAT_IMPLICIT_BIT) & FLOAT_MANT_MASK;
    /* result sign bit: XOR sign argument signs */
    signr = (a ^ b) & FLOAT_SIGN_MASK;
    if ((expoa >= (MAX_NORM_EXPO - EXPO_ADJUST)) || /* at least one argument is special */
        (expob >= (MAX_NORM_EXPO - EXPO_ADJUST))) { 
        if ((a & ~FLOAT_SIGN_MASK) > FLOAT_INFINITY) { /* a is NaN */
            /* return quietened NaN */
            return a | FLOAT_QNAN_BIT;
        }
        if ((b & ~FLOAT_SIGN_MASK) > FLOAT_INFINITY) { /* b is NaN */
            /* return quietened NaN */
            return b | FLOAT_QNAN_BIT;
        }
        if ((a & ~FLOAT_SIGN_MASK) == 0) { /* a is zero */
            /* return NaN if b is infinity, else zero */
            return (expob != (INF_EXPO - EXPO_ADJUST)) ? signr : FLOAT_INDEFINITE;
        }
        if ((b & ~FLOAT_SIGN_MASK) == 0) { /* b is zero */
            /* return NaN if a is infinity, else zero */
            return (expoa != (INF_EXPO - EXPO_ADJUST)) ? signr : FLOAT_INDEFINITE;
        }
        if (((a & ~FLOAT_SIGN_MASK) == FLOAT_INFINITY) || /* a or b infinity */
            ((b & ~FLOAT_SIGN_MASK) == FLOAT_INFINITY)) {
            return signr | FLOAT_INFINITY;
        }
        if ((int32_t)expoa < (MIN_NORM_EXPO - EXPO_ADJUST)) { /* a is subnormal */
            /* normalize significand of a */
            manta = a & FLOAT_MANT_MASK;
            expoa++;
            do {
                manta = 2 * manta;
                expoa--;
            } while (manta < FLOAT_IMPLICIT_BIT);
        } else if ((int32_t)expob < (MIN_NORM_EXPO - EXPO_ADJUST)) { /* b is subnormal */
            /* normalize significand of b */
            mantb = b & FLOAT_MANT_MASK;
            expob++;
            do {
                mantb = 2 * mantb;
                expob--;
            } while (mantb < FLOAT_IMPLICIT_BIT);
        }
    }
    /* result exponent: add argument exponents and adjust for biasing */
    expor = expoa + expob - FLOAT_EXPO_BIAS + 2 * EXPO_ADJUST;
    mantb = mantb ; /* preshift to align result signficand */
    /* result significand: multiply argument signficands */
    uint8_t mantA_short = manta >> 16; // Take only 8 bits (1 plus 7 bits of mantissa)
    uint8_t mantB_short = mantb >> 16; // Take only 8 bits (1 plus 7 bits of mantissa)
    uint16_t p_short = ILM(mantA_short,mantB_short,iter);

    prod = (uint64_t)p_short << 32;
    prod = prod << FLOAT_EXPO_BITS;
    mantr_hi = (uint32_t)(prod >> 32);
    mantr_lo = (uint32_t)(prod >>  0);
    /* normalize significand */
    if (mantr_hi < FLOAT_IMPLICIT_BIT) {
        mantr_hi = (mantr_hi << 1) | (mantr_lo >> (32 - 1));
        mantr_lo = (mantr_lo << 1);
        expor--;
    }
    if (expor <= (MAX_NORM_EXPO - EXPO_ADJUST)) { /* normal, may overflow to infinity during rounding */
        /* combine biased exponent, sign and signficand */
        r = (expor << FLOAT_MANT_BITS) + signr + mantr_hi;
        /* round result to nearest or even; overflow to infinity possible */
        r = r + ((mantr_lo == RND_BIT_MASK) ? (mantr_hi & MANT_LSB) : (mantr_lo >> RND_BIT_SHIFT));
    } else if ((int32_t)expor > (MAX_NORM_EXPO - EXPO_ADJUST)) { /* overflow */
        /* return infinity */
        r = signr | FLOAT_INFINITY;
    } else { /* underflow */
        /* return zero, normal, or smallest subnormal */
        shift = 0 - expor;
        if (shift > MAX_SHIFT) shift = MAX_SHIFT;
        /* denormalize significand */
        mantr_lo = mantr_hi << (32 - shift) | (mantr_lo ? 1 : 0);
        mantr_hi = mantr_hi >> shift;
        /* combine sign and signficand; biased exponent known to be zero */
        r = mantr_hi + signr;
        /* round result to nearest or even */
        r = r + ((mantr_lo == RND_BIT_MASK) ? (mantr_hi & MANT_LSB) : (mantr_lo >> RND_BIT_SHIFT));
    }
    return r;
}

__device__  uint32_t uint_as_floatV2 (float a)
{
    uint32_t r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float uint_as_floatV2 (uint32_t a)
{
    float r;
    memcpy (&r, &a, sizeof r);
    return r;
}

__device__ float fp32_mul_ILM (float a, float b, uint8_t iter)
{
    return uint_as_floatV2 (fp32_mul_core (uint_as_floatV2 (a), uint_as_floatV2 (b),iter));
}

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








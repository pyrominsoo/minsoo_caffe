
#ifndef DRUM_MULT_HPP_
#define DRUM_MULT_HPP_


#include "Fixed.h"
#include <math.h>
#include <iostream>
#include <limits.h>

using namespace std;


// op1 and op2 must have same widths
template<size_t I,size_t F>
void DRUM_mult(Fixed<I,F> *op1, Fixed<I,F> *op2, unsigned int k)
{
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> temp;
    Fixed<128,0> leading_one_k1;
    Fixed<128,0> leading_one_k2;
    unsigned int k1_int;
    unsigned int k2_int;
    unsigned int shift_back1;
    unsigned int shift_back2;
    unsigned int shift_back;
    unsigned int shift_amt1;
    unsigned int shift_amt2;
    bool op1_neg;
    bool op2_neg;
    
    
    // Check for 0
    if (*op1 == 0.0) {
        return;
    }
    if (*op2 == 0.0) {
        op1->data_ = 0;
        return;
    }

    // Fix signs here
    if (*op1 < 0) {
        (*op1) *= -1;
        if (*op1 < 0) {
            op1->data_ -= 1;
        }
        op1_neg = true;
    } else {
        op1_neg = false;
    }
    if (*op2 < 0) {
        (*op2) *= -1;
        if (*op2 < 0) {
            op2->data_ -= 1;
        }
        op2_neg = true;
    } else {
        op2_neg = false;
    }

#ifdef DRUMDEBUG
    cout.precision(64);
    cout << "Op1: " << *op1 << "\n";
    cout << "Op2: " << *op2 << "\n";
    cout << "Op1_neg : " << op1_neg << "\n";
    cout << "Op2_neg : " << op2_neg << "\n";
#endif

    N1 = op1->to_raw();
    N2 = op2->to_raw();
    
#ifdef DRUMDEBUG
    cout << "N1 : " << N1 << "\n";
    cout << "N2 : " << N2 << "\n";
#endif

    k1_int = (int)floor(log2(N1.to_double()));
    k2_int = (int)floor(log2(N2.to_double()));

#ifdef DRUMDEBUG
    cout << "k1_int : " << k1_int << "\n";
    cout << "k2_int : " << k2_int << "\n";
#endif

    // shift 
    if (k1_int >= k) {  // need to shift
        shift_back1 = k1_int - k + 1;
        shift_amt1 = shift_back1 + 1;    // perform extra shift to truncate last bit
        N1 >>= shift_amt1;
        N1 <<= 1;
        N1 += 1;
    } else { // no need to do anything for last k bits, but set shift_back to use later
        shift_back1 = 0;
    }

    if (k2_int >= k) {  // need to shift
        shift_back2 = k2_int - k + 1;
        shift_amt2 = shift_back2 + 1;    // perform extra shift to truncate last bit
        N2 >>= shift_amt2;
        N2 <<= 1;
        N2 += 1;
    } else { // no need to do anything for last k bits, but set shift_back to use later
        shift_back2 = 0;
    }

    shift_back = shift_back1 + shift_back2;

#ifdef DRUMDEBUG
    cout << "opN1 : " << N1 << "\n";
    cout << "opN2 : " << N2 << "\n";
    cout << "shift_back1 : " << shift_back1 << "\n";
    cout << "shift_back2 : " << shift_back2 << "\n";
    cout << "shift_back : " << shift_back << "\n";
#endif

    // multiply two shifted numbers
    temp = N1;
    temp *= N2;

#ifdef DRUMDEBUG
    cout << "multed_temp : " << temp << "\n";
#endif

    // shift back
    temp <<= shift_back;

#ifdef DRUMDEBUG
    cout << "shifted_back : " << temp << "\n";
#endif
    
    temp >>= op1->fractional_bits;

    op1->data_ = temp.data_; //>> op1->fractional_bits;
    
    // Return signs
    if (op1_neg) {
        if (op2_neg) {
            (*op2) = -(*op2);
        } else {
            (*op1) = -(*op1);
        }
    } else {
        if (op2_neg) {
            (*op2) = -(*op2);
            (*op1) = -(*op1);
        } else {

            // do nothing
        }
    }
}


#endif

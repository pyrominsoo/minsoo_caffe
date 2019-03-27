#ifndef LOGADD_ULTRA_MOD_HPP_
#define LOGADD_ULTRA_MOD_HPP_


#include "Fixed.h"
#include <cmath>
#include <iostream>
#include <csignal>
//#include <limits.h>

using namespace numeric;

// op1 and op2 must have same widths
template<size_t I,size_t F>
void logadd_ultra_mod(Fixed<I,F> *op1, Fixed<I,F> *op2)
{
    Fixed<I,F> bigger;
    Fixed<I,F> smaller;
    Fixed<128,0> N1;    // bigger
    Fixed<128,0> N2;    // smaller
    unsigned int k1_int;    //bigger
    unsigned int k2_int;    //smaller
    bool op1_neg;
    bool op2_neg;
    bool bigger_neg;
    bool smaller_neg;

#ifdef MDEBUG
    std::cout.precision(64);
#endif

    // Check for 0
    if (*op2 == 0.0) {
        return;
    } else if (*op1 == 0.0) {
        *op1 = *op2;
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

    // Check for maximum (absolute mag)
    if (*op1 < *op2 || *op1 == *op2) {
        bigger = *op2;
        bigger_neg = op2_neg;
        smaller = *op1;
        smaller_neg = op1_neg;
    } else {
        bigger = *op1;
        bigger_neg = op1_neg;
        smaller = *op2;
        smaller_neg = op2_neg;
    }

    N1 = bigger.to_raw();
    N2 = smaller.to_raw();
    
    k1_int = (int)floor(log2(N1.to_double()));
    k2_int = (int)floor(log2(N2.to_double()));

    if (k1_int < k2_int) { // not supposed to happen
        std::cout << "from logaddmod, k1_int < k2_int" << std::endl;
        raise(SIGTERM);
    }

    // Distinguish ADD and SUB by the signs
    if (bigger_neg == smaller_neg) {  // ADD
        if (k1_int == k2_int) { // Case 1
            bigger <<= 1;
        } else { // Case 2
            if (k1_int == k2_int + 1) { // Case 2.1
                // Find the position of first mantissa bit
                double remainder = std::fmod(N1.to_double(), pow(2.0,k1_int));            
                bool lead_mant_one = (remainder >= pow(2.0,k1_int-1))? true:false;
                double corr_amt = pow(2.0,k1_int-1-F);
                if (lead_mant_one) {
                    bigger -= corr_amt;
                    bigger <<= 1;
                } else {
                    bigger += corr_amt;
                }
            } else { // Rest of Case 2
                bigger = bigger;
            }
        }
    } else { // SUB
        if (k1_int == k2_int) { // Case 3
            bigger = 0;
            bigger_neg = false;
        } else { // Case 4
            if (k1_int == k2_int + 1) { // Case 4.1
                bigger >>= 1;
            } else { // Rest of Case 4
                bigger = bigger;
            }
        }
    }

    *op1 = bigger;
    op1_neg = bigger_neg;

    // Return signs
    if (op1_neg) {
        (*op1) = -(*op1);
    } 
    if (op2_neg) {
        (*op2) = -(*op2);
    }
}



#endif

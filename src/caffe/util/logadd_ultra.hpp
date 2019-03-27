#ifndef LOGADD_ULTRA_HPP_
#define LOGADD_ULTRA_HPP_


#include "Fixed.h"
//#include <math.h>
#include <iostream>
//#include <limits.h>

using namespace numeric;

// op1 and op2 must have same widths
template<size_t I,size_t F>
void logadd_ultra(Fixed<I,F> *op1, Fixed<I,F> *op2)
{
    bool op1_neg;
    bool op2_neg;

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
        *op1 = *op2;
        op1_neg = op2_neg;
    }


    // Return signs
    if (op1_neg) {
        (*op1) = -(*op1);
    } 
    if (op2_neg) {
        (*op2) = -(*op2);
    }
}



#endif

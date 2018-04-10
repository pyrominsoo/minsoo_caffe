#ifndef MITCHK_C1_HPP_
#define MITCHK_C1_HPP_


#include "Fixed.h"
#include <math.h>
#include <iostream>
#include <limits.h>

using namespace numeric;

// op1 and op2 must have same widths
template<size_t I,size_t F>
void mitchk_c1(Fixed<I,F> *op1, Fixed<I,F> *op2, unsigned int k)
{
    Fixed<I,F> orig1;
    Fixed<I,F> orig2;
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> K2;
    Fixed<128,0> leading_one;
    Fixed<128,0> temp;
    Fixed<128,0> op1_log;
    unsigned int op1_log_int;
    Fixed<128,0> op2_log;
    unsigned int op2_log_int;
    Fixed<128,0> added_log;
    Fixed<128,0> charac;
    Fixed<128,0> charac_shifted;
    Fixed<128,0> manti;
    Fixed<128,0> result;
    // int total_bits = 2*op1->fractional_bits;
    unsigned int recovery_amt;
    unsigned int shift_amt;
    unsigned int trunc_amt;

    unsigned int total_bits = 32;
    
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

    // Make the leading one reference
    leading_one = 1;
    leading_one <<= total_bits - 1;
    // std::cout << "leading_one : " << std::fixed << leading_one << "\n";


    // std::cout.precision(dbl::max_digits10);
    // std::cout << "Op1: " << *op1 << "\n";
    // std::cout << "Op2: " << *op2 << "\n";
    // std::cout << "Op1_neg : " << op1_neg << "\n";
    // std::cout << "Op2_neg : " << op2_neg << "\n";

    orig1 = *op1;
    orig2 = *op2;

    // Fix signs here
    if (*op1 < 0) {
        (*op1) *= -1;
        op1->data_ -= 1;
        op1_neg = true;
    } else {
        op1_neg = false;
    }
    if (*op2 < 0) {
        (*op2) *= -1;
        op2->data_ -= 1;
        op2_neg = true;
    } else {
        op2_neg = false;
    }
   
    
    // Check for 0 (-1 to 0)
    if (*op1 == 0.0) {
        op1->data_ = 0;
        *op2 = orig2;
        return;    
    }
    if (*op2 == 0.0) {
        op1->data_ = 0;
        *op2 = orig2;
        return;
    }

#ifdef MDEBUG
    std::cout.precision(64);
    std::cout << "Op1: " << *op1 << "\n";
    std::cout << "Op2: " << *op2 << "\n";
    std::cout << "Op1_neg : " << op1_neg << "\n";
    std::cout << "Op2_neg : " << op2_neg << "\n";
#endif


    N1 = op1->to_raw();
    N2 = op2->to_raw();

#ifdef MDEBUG
    std::cout << "N1 : " << N1 << "\n";
    std::cout << "N2 : " << N2 << "\n";
#endif
   

    // Encode op1 and op2 using mitchell algorithm
    if (N1 == 0) {
        op1_log = 0;
    } else {
        temp = N1;
        op1_log = total_bits - 1;
        while (N1 < leading_one) {
            N1 <<= 1;
            op1_log -= 1;
        }

        // Shift right and left again to truncate the bits (workaround since slicing unavailable)
        op1_log_int = op1_log.to_int();
        if (op1_log_int > k) {
            trunc_amt = op1_log_int - k; 
            N1 = temp;
            N1 >>= trunc_amt;
            N1 <<= trunc_amt + (total_bits - 1 - op1_log_int);
        } 

#ifdef MDEBUG
        std::cout << "N1 : " << std::fixed << N1 << "\n";
        std::cout << "op1_log : " << std::fixed << op1_log << "\n";
#endif
        op1_log <<= total_bits - 1;
        N1 -= leading_one;
        op1_log += N1;
    }
    
    if (N2 == 0) {
        op2_log = 0;
    } else {
        temp = N2;
        op2_log = total_bits - 1;
        // std::cout << "N2 : " << std::fixed << N2 << "\n";
        while (N2 < leading_one) {
            N2 <<= 1;
            op2_log -= 1;
        }

        // Shift right and left again to truncate the bits (workaround since slicing unavailable)
        op2_log_int = op2_log.to_int();
        if (op2_log_int > k) {
            trunc_amt = op2_log_int - k; 
            N2 = temp;
            N2 >>= trunc_amt;
            N2 <<= trunc_amt + (total_bits - 1 - op2_log_int);
        }

#ifdef MDEBUG
        std::cout << "N2 : " << std::fixed << N2 << "\n";
        std::cout << "op2_log : " << std::fixed << op2_log << "\n";
#endif
        op2_log <<= total_bits - 1;
        N2 -= leading_one;
        op2_log += N2;
    }

#ifdef MDEBUG
    std::cout << "op1_log : " << std::fixed << op1_log << "\n";
    std::cout << "op2_log : " << std::fixed << op2_log << "\n";
#endif

    // Add encoded op1 and op2
    added_log = op1_log;
    added_log += op2_log;


#ifdef MDEBUG
    std::cout << "added_log : " << std::fixed << added_log << "\n";
#endif

    // Decode into charac and mantissa
    charac = added_log;
    charac >>= total_bits - 1;
    charac_shifted = charac;
    charac_shifted <<= total_bits - 1;
    manti = added_log;
    manti -= charac_shifted;

#ifdef MDEBUG
    std::cout << "charac : " << std::fixed << charac << "\n";
    std::cout << "charac_shifted : " << std::fixed << charac_shifted << "\n";
    std::cout << "manti : " << std::fixed << manti << "\n";
#endif

    // Produce the result from charac and mantissa
    result = manti;
    result += leading_one;
#ifdef MDEBUG
    //std::cout << "result : " << std::fixed << result << "\n";
#endif

    recovery_amt = total_bits + (op1->fractional_bits) - 1;
    if ((unsigned int)charac.to_int() > recovery_amt) {
        std::cout << "Overflow on Mitchell Mult" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    else {
        shift_amt = recovery_amt - (unsigned int)charac.to_int();
        result >>= shift_amt;
    }
    //std::cout << "recovery_amt : " << recovery_amt << "\n";
    //std::cout << "shift_amt : " << shift_amt << "\n";
    //std::cout << "result : " << std::fixed << result << "\n";
    //std::cout << "result : " << result.to_double() << "\n";

    op1->data_ = result.data_;
    
    //std::cout << "op1->data_ : " << std::fixed << op1->data_ << "\n";
#ifdef MDEBUG
    std::cout << "op1 : " << std::fixed << *op1 << "\n";
#endif

    // Return signs
    if (op1_neg) {
        if (op2_neg) {
            *op2 = orig2;
        } else {
            (*op1) = -(*op1);
            op1->data_ -= 1;
        }
    } else {
        if (op2_neg) {
            *op2 = orig2;
            (*op1) = -(*op1);
            op1->data_ -= 1;
        } else {

            // do nothing
        }
    }


}



#endif

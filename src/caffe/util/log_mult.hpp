#include "Fixed.h"
#include <math.h>
#include <iostream>
#include <limits.h>

using namespace numeric;

typedef std::numeric_limits< double > dbl; 

// op1 and op2 must have same widths
template<size_t I,size_t F>
void log_mult(Fixed<I,F> *op1, Fixed<I,F> *op2)
{
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> temp;
    Fixed<128,0> leading_one_k1;
    Fixed<128,0> leading_one_k2;
    unsigned int k1_int;
    unsigned int k2_int;
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


    N1 = op1->to_raw();
    N2 = op2->to_raw();
    

    k1_int = (int)floor(log2(N1.to_double()));
    k2_int = (int)floor(log2(N2.to_double()));
    

    leading_one_k1 = 1;
    leading_one_k1 <<= k1_int;
    leading_one_k2 = 1;
    leading_one_k2 <<= k2_int;

    N1 -= leading_one_k1;
    N2 -= leading_one_k2;
 
    temp = leading_one_k1;
    temp <<= k2_int;
    N1 <<= k2_int;
    N2 <<= k1_int;
   
    temp += N1;
    temp += N2;
   
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



// Performs op1 = op1 * op2; with approximate logarithmic multplication
// op1 and op2 must have same widths
template<size_t I,size_t F>
void log_mult2(Fixed<I,F> *op1, Fixed<I,F> *op2)
{
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> N3;
    Fixed<128,0> N4;
    Fixed<128,0> leading_one_k1;
    Fixed<128,0> leading_one_k2;
    Fixed<128,0> leading_one_k3;
    Fixed<128,0> leading_one_k4;
    Fixed<128,0> N1_shifted;
    Fixed<128,0> N2_shifted;
    Fixed<128,0> temp;
    Fixed<128,0> temp_err;
    unsigned int k1_int;
    unsigned int k2_int;
    unsigned int k3_int;
    unsigned int k4_int;
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
    
    
    
    
    


    N1 = op1->to_raw();
    N2 = op2->to_raw();
   
    
    

    k1_int = (int)floor(log2(N1.to_double()));
    k2_int = (int)floor(log2(N2.to_double()));
    
    
    leading_one_k1 = 1;
    leading_one_k1 <<= k1_int;
    leading_one_k2 = 1;
    leading_one_k2 <<= k2_int;
    

    N1 -= leading_one_k1;
    N2 -= leading_one_k2;
    
    
    temp = leading_one_k1;
    temp <<= k2_int;
    N1_shifted = N1;
    N2_shifted = N2;
    N1_shifted <<= k2_int;
    N2_shifted <<= k1_int;
   
    
    temp += N1_shifted;
    temp += N2_shifted;
  

    temp >>= op1->fractional_bits;

    

    // Error compensation
    //log_mult(&N1, &N2);
    k3_int = (int)floor(log2(N1.to_double()));
    k4_int = (int)floor(log2(N2.to_double()));
    
    
    leading_one_k3 = 1;
    leading_one_k3 <<= k3_int;
    leading_one_k4 = 1;
    leading_one_k4 <<= k4_int;
    
    
    
    N1 -= leading_one_k3;
    N2 -= leading_one_k4;

    
    

    temp_err = leading_one_k3;
    temp_err <<= k4_int;
    N1 <<= k4_int;
    N2 <<= k3_int;

    
    
    

    temp_err += N1;
    temp_err += N2;
    temp_err >>= op1->fractional_bits;

    

    // Combine result and error
    temp += temp_err; 

    

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



// Performs op1 = op1 * op2; with approximate logarithmic multplication
// op1 and op2 must have same widths
template<size_t I,size_t F>
void log_mult3(Fixed<I,F> *op1, Fixed<I,F> *op2)
{
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> N3;
    Fixed<128,0> N4;
    Fixed<128,0> N5;
    Fixed<128,0> N6;
    Fixed<128,0> leading_one_k1;
    Fixed<128,0> leading_one_k2;
    Fixed<128,0> leading_one_k3;
    Fixed<128,0> leading_one_k4;
    Fixed<128,0> leading_one_k5;
    Fixed<128,0> leading_one_k6;
    Fixed<128,0> N1_shifted;
    Fixed<128,0> N2_shifted;
    Fixed<128,0> N3_shifted;
    Fixed<128,0> N4_shifted;
    Fixed<128,0> N5_shifted;
    Fixed<128,0> N6_shifted;
    Fixed<128,0> temp;
    Fixed<128,0> temp_err1;
    Fixed<128,0> temp_err2;
    unsigned int k1_int;
    unsigned int k2_int;
    unsigned int k3_int;
    unsigned int k4_int;
    unsigned int k5_int;
    unsigned int k6_int;
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

   
    N1 = op1->to_raw();
    N2 = op2->to_raw();
    

    k1_int = (int)floor(log2(N1.to_double()));
    k2_int = (int)floor(log2(N2.to_double()));
    
    leading_one_k1 = 1;
    leading_one_k1 <<= k1_int;
    leading_one_k2 = 1;
    leading_one_k2 <<= k2_int;
    

    N1 -= leading_one_k1;
    N2 -= leading_one_k2;
    
    
    temp = leading_one_k1;
    temp <<= k2_int;
    N1_shifted = N1;
    N2_shifted = N2;
    N1_shifted <<= k2_int;
    N2_shifted <<= k1_int;
   
    temp += N1_shifted;
    temp += N2_shifted;
  

    temp >>= op1->fractional_bits;


    // Error compensation
    //log_mult(&N1, &N2);
    k3_int = (int)floor(log2(N1.to_double()));
    k4_int = (int)floor(log2(N2.to_double()));
    
    leading_one_k3 = 1;
    leading_one_k3 <<= k3_int;
    leading_one_k4 = 1;
    leading_one_k4 <<= k4_int;
    
    N1 -= leading_one_k3;
    N2 -= leading_one_k4;
    

    temp_err1 = leading_one_k3;
    temp_err1 <<= k4_int;
    N3_shifted = N1;
    N4_shifted = N2;
    N3_shifted <<= k4_int;
    N4_shifted <<= k3_int;

    temp_err1 += N3_shifted;
    temp_err1 += N4_shifted;
    temp_err1 >>= op1->fractional_bits;

    // Combine result and error
    temp += temp_err1; 

    
    // Error compensation2
    k5_int = (int)floor(log2(N1.to_double()));
    k6_int = (int)floor(log2(N2.to_double()));
    
    leading_one_k5 = 1;
    leading_one_k5 <<= k5_int;
    leading_one_k6 = 1;
    leading_one_k6 <<= k6_int;
    
    N1 -= leading_one_k5;
    N2 -= leading_one_k6;


    temp_err2 = leading_one_k5;
    temp_err2 <<= k6_int;
    N5_shifted = N1;
    N6_shifted = N2;
    N5_shifted <<= k6_int;
    N6_shifted <<= k5_int;
   
    temp_err2 += N5_shifted;
    temp_err2 += N6_shifted;
    temp_err2 >>= op1->fractional_bits;
    temp += temp_err2;


    // Shift and return
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





// Performs op1 = op1 * op2; with approximate logarithmic multplication
// op1 and op2 must have same widths
template<size_t I,size_t F>
void mitch_mult(Fixed<I,F> *op1, Fixed<I,F> *op2, int total_bits)
{
    Fixed<128,0> N1;
    Fixed<128,0> N2;
    Fixed<128,0> leading_one;
    Fixed<128,0> temp;
    Fixed<128,0> op1_log;
    Fixed<128,0> op2_log;
    Fixed<128,0> added_log;
    Fixed<128,0> charac;
    Fixed<128,0> charac_shifted;
    Fixed<128,0> manti;
    Fixed<128,0> result;
    // int total_bits = 2*op1->fractional_bits;
    unsigned int recovery_amt;
    unsigned int shift_amt;
    
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
    
    // std::cout << "Op1: " << *op1 << "\n";
    // std::cout << "Op2: " << *op2 << "\n";
    // std::cout << "Op1_neg : " << op1_neg << "\n";
    // std::cout << "Op2_neg : " << op2_neg << "\n";



    N1 = op1->to_raw();
    N2 = op2->to_raw();

    //std::cout << "N1 : " << std::fixed << N1 << "\n";
    // std::cout << "N2 : " << std::fixed << N2 << "\n";

   

    // Encode op1 and op2 using mitchell algorithm
    if (N1 == 0) {
        op1_log = 0;
    } else {
        op1_log = total_bits - 1;
        while (N1 < leading_one) {
            N1 <<= 1;
            op1_log -= 1;
        }
        // std::cout << "N1 : " << std::fixed << N1 << "\n";
        // std::cout << "op1_log : " << std::fixed << op1_log << "\n";
        op1_log <<= total_bits - 1;
        N1 -= leading_one;
        op1_log += N1;
    }
    
    if (N2 == 0) {
        op2_log = 0;
    } else {
        op2_log = total_bits - 1;
        // std::cout << "N2 : " << std::fixed << N2 << "\n";
        while (N2 < leading_one) {
            N2 <<= 1;
            op2_log -= 1;
        }
        // std::cout << "N2 : " << std::fixed << N2 << "\n";
        // std::cout << "op2_log : " << std::fixed << op2_log << "\n";
        op2_log <<= total_bits - 1;
        N2 -= leading_one;
        op2_log += N2;
    }

    //std::cout << "op1_log : " << std::fixed << op1_log << "\n";
    //std::cout << "op2_log : " << std::fixed << op2_log << "\n";

    // Add encoded op1 and op2
    added_log = op1_log;
    added_log += op2_log;

    //std::cout << "added_log : " << std::fixed << added_log << "\n";

    // Decode into charac and mantissa
    charac = added_log;
    charac >>= total_bits - 1;
    charac_shifted = charac;
    charac_shifted <<= total_bits - 1;
    manti = added_log;
    manti -= charac_shifted;


    //std::cout << "charac : " << std::fixed << charac << "\n";
    //std::cout << "charac_shifted : " << std::fixed << charac_shifted << "\n";
    //std::cout << "manti : " << std::fixed << manti << "\n";


    // Produce the result from charac and mantissa
    result = manti;
    result += leading_one;
    //std::cout << "result : " << std::fixed << result << "\n";

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
    //std::cout << "op1 : " << std::fixed << *op1 << "\n";


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





#include "Fixed.h"
#include <math.h>
#include <iostream>
#include <limits.h>

using namespace numeric;

// weight and multiplier must have same widths
template<size_t I,size_t F>
void asm_mult(Fixed<I,F> *weight, Fixed<I,F> *multiplier)
{
    Fixed<I+F,0> N1,temporaryN1;
    Fixed<I+F,0> N2;
    Fixed<I+F,0> temp;
	Fixed<I+F,0> partial_result;

    bool weight_neg;
    bool multiplier_neg;
	int shift_counter;
    

    // Check for 0
    if (*weight == 0.0) {
        return;
    }
    if (*multiplier == 0.0) {
        weight->data_ = 0;
        return;
    }

    // Fix signs here
    if (*weight < 0) {
        (*weight) *= -1;
        if (*weight < 0) {
            weight->data_ -= 1;
        }
        weight_neg = true;
    } else {
        weight_neg = false;
    }
    if (*multiplier < 0) {
        (*multiplier) *= -1;
        if (*multiplier < 0) {
            multiplier->data_ -= 1;
        }
        multiplier_neg = true;
    } else {
        multiplier_neg = false;
    }


    N1 = weight->to_raw();
	temporaryN1 = weight->to_raw();
    N2 = multiplier->to_raw();

	*weight = 0;
	shift_counter=0;

	//makes sure that N1 will only contain the integer part
	N1>>=F;    
	//loop for the integer part
	while(N1.data_){	//while there's still data, loop
		temp = N1;
		temp &= 15;	//a mask of 1111b(0Fh) is passed in "temp" to make only the least significative nible available
		partial_result = N2;
		switch(temp.data_){
			case 0:
				partial_result = 0;
				break;
			case 1:
				//does nothing
				break;
			case 2:
				partial_result <<= 1;
				break;
			case 3:
			case 4:
			case 5:
				partial_result <<= 2;
				break;
			default:	//if between 6 and 15:
				partial_result <<= 3;
		}
		partial_result <<=(shift_counter*4);	//does the last shifting
		shift_counter++;	//updates the shift counter;
		N1 >>= 4; //"destroy" the last nible of N1 until there's nothing left
		weight->data_ = weight->data_ + partial_result.data_;
	}

	//makes sure that N1 will only contain the fractionary part
	N1 = temporaryN1;	
	N1<<=I;
	int n_bits = I+F;
	//loop for the integer part
	shift_counter=0;
	while(N1.data_){	//while there's still data, loop
		temp = N1;
		Fixed<I+F,0> mask = 15;
		mask <<= n_bits-4;	//4 is the mask size and the last operand is the position
		temp &= mask;
		temp >>= n_bits-4;
		temp &= 15;
		//cout<<"Temp: "<<temp<<endl;

		partial_result = N2;
		//0000(0)
		//0001(1)
		//0010(2)
		//0011(3)
		//0100(4)
		//0101(5)
		//0110(6)
		//0111(7)
		//1000(8)
		

		switch(temp.data_){
			case 0:
				partial_result = 0;
				break;
			case 1:
				partial_result >>= 4;
				break;
			case 2:
				partial_result >>= 3;
				break;
			case 3:
			case 4:
			case 5:
				partial_result >>= 2;
				break;
			default:	//if between 6 and 15:
				partial_result >>= 1;
		}
		partial_result >>=(shift_counter*4);	//does the last shifting
		N1 <<=4;
		shift_counter++;
		weight->data_ = weight->data_ + partial_result.data_;
	}
    // Return signs
    if (weight_neg) {
        if (multiplier_neg) {
            (*multiplier) = -(*multiplier);
        } else {
            (*weight) = -(*weight);
        }
    } else {
        if (multiplier_neg) {
            (*multiplier) = -(*multiplier);
            (*weight) = -(*weight);
        } else {

            // do nothing
        }
    }
	
}

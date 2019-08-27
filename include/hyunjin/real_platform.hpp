/*****************************************************************************
*   Title      : real_platform.hpp
*   Desc       : multiplier in caffe
*                hardware definition class declaration
*   Author     : HyunJin Kim
*   Date       : 2019.04.20
*   Ver        : 4.0
*   Description: These class are adopted to describe hardware module
*                4.0: data_mode and its function are added.
*                5.0: numbitssampling and numbits_lsr are added.
*   Note!!     : 
*       
****************************************************************************/
#ifndef REAL_PLATFORM_H// Avoid redefinition error
#define REAL_PLATFORM_H

#include <string>
#include <iostream>
#include "hyunjin/define.hpp"
#include "hyunjin/platform.hpp"

// Real Class Definition for "Platform" Description
class Real_Platform: public Platform // inheritance from interface class
{

  private:
    // Variables 
    std::string platform_name;  // Is there scheduling description?

  public:
    // virtual destructor
    virtual ~Real_Platform(){ };

    // constructor 
    explicit Real_Platform(const std::string& _platform_name): 
      platform_name(_platform_name) { }

    std::string Get_platform_name() { return platform_name; }     
};
// End of Real Class Definition for "Platform" Description


// Real Class Definition for "Module" Description
class Real_Module: public Module // inheritance from interface class
{
  private:
    // Variables 
    _MODE_TYPE   mode;              // multiplier mode
    unsigned int allnumbits;        // #all bits in format 
    unsigned int mantissa_numbits;  // #mantissa bits in format
    unsigned int fixed_numbits;     // #fixed width in format
    unsigned int stage1_k;          // #MSBs in mantissa of first stage 
    unsigned int stage2_k;          // #MSBs in mantissa of second stage 
    unsigned int stage3_k;          // #MSBs in mantissa of third stage 

    _RMODE_TYPE  stage1_rmode;      // rounding mode in first stage 
    _RMODE_TYPE  stage2_rmode;      // rounding mode in second stage 
    _RMODE_TYPE  stage3_rmode;      // rounding mode in third stage 
    _RMODE_TYPE  acc_rmode;         // rounding mode after accumulation
                       
    _DMODE_TYPE  data_mode;         // data mode in logarithmic representation
    unsigned int numbitssampling;   // 2 ^ numbitssampling in logarithmic stochastic rounding
    unsigned int numbits_lsr;       // bits used as weights in logarithmic stochastic rounding 

  public:
    // virtual destructor
    virtual ~Real_Module(){ };

    // constructor
    explicit Real_Module(_MODE_TYPE   _mode,
                         unsigned int _allnumbits,
                         unsigned int _mantissa_numbits,
                         unsigned int _fixed_numbits,                                        
                         unsigned int _stage1_k,
                         unsigned int _stage2_k,
                         unsigned int _stage3_k,
                         _RMODE_TYPE  _stage1_rmode,
                         _RMODE_TYPE  _stage2_rmode,
                         _RMODE_TYPE  _stage3_rmode,
                         _RMODE_TYPE  _acc_rmode,
                         _DMODE_TYPE  _data_mode,
                         unsigned int _numbitssampling,
                         unsigned int _numbits_lsr)  
      :mode(_mode), allnumbits(_allnumbits), 
       mantissa_numbits(_mantissa_numbits), 
       fixed_numbits(_fixed_numbits), 
       stage1_k(_stage1_k), 
       stage2_k(_stage2_k),
       stage3_k(_stage3_k),
       stage1_rmode(_stage1_rmode), 
       stage2_rmode(_stage2_rmode), 
       stage3_rmode(_stage3_rmode), 
       acc_rmode(_acc_rmode),
       data_mode(_data_mode),
       numbitssampling(_numbitssampling),
       numbits_lsr(_numbits_lsr)
       { }

    _MODE_TYPE   Get_mul_type() { return mode; }        
    unsigned int Get_allnumbits() { return allnumbits; }        
    unsigned int Get_mantissa_numbits() { return mantissa_numbits; }   
    unsigned int Get_fixed_numbits() { return fixed_numbits; }    
 
    unsigned int Get_stage1_k() { return stage1_k; }  
    unsigned int Get_stage2_k() { return stage2_k; }  
    unsigned int Get_stage3_k() { return stage3_k; }  
    _RMODE_TYPE  Get_stage1_rmode() { return stage1_rmode; } 
    _RMODE_TYPE  Get_stage2_rmode() { return stage2_rmode; }
    _RMODE_TYPE  Get_stage3_rmode() { return stage3_rmode; }
    _RMODE_TYPE  Get_acc_rmode() { return acc_rmode; }        

    _DMODE_TYPE  Get_data_mode() { return data_mode; } 
    unsigned int Get_numbitssampling() { return numbitssampling; } 
    unsigned int Get_numbits_lsr() { return numbits_lsr; } 
};
// End of Real Class Definition for "Module" Description


#endif // End of Avoid redefinition error

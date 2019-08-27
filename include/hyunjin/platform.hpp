/*****************************************************************************
*   Title      : platform.hpp
*   Desc       : multiplier in caffe
*                platform description interface class declaration
*   Author     : HyunJin Kim
*   Date       : 2019.04.20
*   Ver        : 4.0
*   Description: These class are utilized to model platform description
*                file
*                4.0: data_mode and its funcation are added
*                5.0: numbitssampling and numbits_lsr are added
*   Note!!     : Platform has its own child class because additional 
*                options would be added up to its own class.
*                      
****************************************************************************/
#ifndef PLATFORM_H // Avoid redefinition error
#define PLATFORM_H 

#include <string>
#include <iostream>
#include <boost/smart_ptr.hpp>
#include "hyunjin/define.hpp"


// Interface Class Declaration for "Platform" Class Declaration 
class Platform
{
  private:

  public:
    // virtual destructor
    virtual ~Platform(){ };
    // virtual constructor

    // pure virtual member function
    virtual std::string Get_platform_name() = 0;     

    // factory function as virtual constructor
    static boost::shared_ptr<Platform> 
       create(const std::string& _platform_name); 
};
// End of Interface Class Declaration for "Platform" Description


// Interface Class Declaration for "Module" Class Declaration 
class Module
{
  private:

  public:
    // virtual destructor
    virtual ~Module(){};
    // virtual constructor

    // pure virtual member function
    virtual _MODE_TYPE   Get_mul_type() = 0;    
    virtual unsigned int Get_allnumbits() = 0;  
    virtual unsigned int Get_mantissa_numbits() = 0;  
    virtual unsigned int Get_fixed_numbits() = 0;  
    virtual unsigned int Get_stage1_k() = 0;  
    virtual unsigned int Get_stage2_k() = 0;  
    virtual unsigned int Get_stage3_k() = 0;  
    virtual _RMODE_TYPE  Get_stage1_rmode() = 0;  
    virtual _RMODE_TYPE  Get_stage2_rmode() = 0;  
    virtual _RMODE_TYPE  Get_stage3_rmode() = 0;  
    virtual _RMODE_TYPE  Get_acc_rmode() = 0;  
    virtual _DMODE_TYPE  Get_data_mode() = 0;  
    virtual unsigned int Get_numbitssampling() = 0;  
    virtual unsigned int Get_numbits_lsr() = 0;  
    
    // factory function as virtual constructor
    static boost::shared_ptr<Module> 
       create(_MODE_TYPE   _mode,
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
              unsigned int _numbits_lsr) ; 
    // end of factory function as virtual constructor

};
// End of Interface Class Declaration for "Module" Description

#endif // End of Avoid redefinition error

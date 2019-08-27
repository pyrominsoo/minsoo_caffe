/*****************************************************************************
*   Title      : real_platform_handle.hpp
*   Desc       : multiplier in caffe description handler
*   Author     : HyunJin Kim
*   Date       : 2018.12.13
*   Ver        : 1.0
*   Description: These class are utilized to handle platform description
*                file
*   Note!!     : Platform_Handler class has platform objects as private members 
*       
****************************************************************************/

#ifndef REAL_PLATFORM_HANDLE_H  // Avoid redefinition error
#define REAL_PLATFORM_HANDLE_H

#include <string>
#include <iostream>
#include <boost/smart_ptr.hpp>
#include <vector>
#include "hyunjin/define.hpp"
#include "hyunjin/platform.hpp"
#include "hyunjin/platform_handle.hpp"

// Real Class Definition for "Platform_Handler" Class Description
class Real_Platform_Handler : public Platform_Handler //inheritance from interface class
{
  private:
    // Variables 
    // platform description xml file
    std::string     platform_filename; 
    // detail platform description vector containers
    // pointer for platform as top property 
    boost::shared_ptr<Platform>                          ptr_platform;
    // pointer for Module
    boost::shared_ptr<Module>                            ptr_module;
    // vector of pointer for memory-based fsms in a module

    unsigned int get_string_uint(const std::string& _string); 

    void do_parse_xml();    

  public:
    // virtual destructor
    virtual ~Real_Platform_Handler(){ };
    // constructor with object initialization
    explicit Real_Platform_Handler(const std::string& _platform_filename):
      platform_filename(_platform_filename)   // platform xml file name initialization
    { 
      do_parse_xml(); 
    } 
    // end of constructor

   boost::shared_ptr<Platform> Get_ptr_platform() 
   { return this-> ptr_platform; }
   
   boost::shared_ptr<Module> Get_ptr_module() 
   { return this-> ptr_module; }

};
// End of Real Class Definition for "Platform_Handler" Class Description

#endif // End of Avoid redefinition error

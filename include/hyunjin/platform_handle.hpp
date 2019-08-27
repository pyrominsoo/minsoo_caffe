/*****************************************************************************
*   Title      : platform_handle.h
*   Desc       : multiplier in caffe description handler 
*   Author     : HyunJin Kim
*   Date       : 2018.12.13
*   Ver        : 1.0
*   Description: These class are utilized to handle platform description
*                file
*   Note!!     : Platform_Handler class has platform objects as private members 
*       
****************************************************************************/
#ifndef PLATFORM_HANDLE_H // Avoid redefinition error
#define PLATFORM_HANDLE_H 

#include <string>
#include <iostream>
#include <boost/smart_ptr.hpp>
#include <vector>
#include "hyunjin/define.hpp"
#include "hyunjin/platform.hpp"

// Interface Class Delaration for "Platform_Handler" Class Declaration 
class Platform_Handler
{
  private:

  public:
   // virtual destructor 
   virtual ~Platform_Handler(){ }; 
   // pure virtual member function
   virtual boost::shared_ptr<Platform> Get_ptr_platform() = 0; 
   virtual boost::shared_ptr<Module> Get_ptr_module() = 0; 

   // factory function as virtual constructor
   static boost::shared_ptr<Platform_Handler> 
       create(const std::string& _platform_filename) ; 
   // end of factory function as virtual constructor
   
};
// Interface Class Delaration for "Platform_Handler" Class Declaration 

#endif // End of Avoid redefinition error

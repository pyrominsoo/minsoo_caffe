/*****************************************************************************
*   Title      : define.h
*   Desc       : multiplier in caffe platform description handler 
*                definition
*   Author     : HyunJin Kim
*   Date       : 2019.05.05
*   Ver        : 2.2
*   Description: 2.0 _DMODE_TYPE is added 
*                2.1 In _RMODE_TYPE, RFLOAT mode is added. 
*                2.2 MITCHK_UNBIAS_F mode is added. 
*       
****************************************************************************/
#ifndef DEFINE_H  // Avoid redefinition error
#define DEFINE_H
// "_" means user-defined data type
// Define user defined data type

#include <string>
#include <vector>
#include <set>
#include <map>

// SS denotes "schedule and stretch"
typedef enum { FALSE = 0, TRUE = 1 } _BOOL;
typedef enum { FLOAT = 0, 
               FIXED = 1, 
               MITCH = 2, 
               ITERLOG = 3,
               DRUM = 4,
               MITCHK = 5,
               MITCHK_BIAS = 6,
               MITCHK_BIAS_C1 = 7,
               ASM = 8,
               MITCHK_C1 = 9,
               MITCHK_BIAS_LG = 10,
               HETERLOG = 11,
               MA_2STAGE_C1 = 12, 
               MITCHK_UNBIAS_C1_F = 13, 
               MITCHK_UNBIAS_F = 14, 
               MULTI_LSR = 15 } _MODE_TYPE;

typedef enum { R_DOWN = 0, 
               R_UP = 1, 
               R_NEAREST = 2, 
               ADD_UNBIAS = 3, 
               STC = 4,
               RFLOAT = 5 } _RMODE_TYPE;

typedef enum { W_LOG = 0,
               IN_LOG = 1, 
               IN_W_LOG = 2, 
               IN_W_ACC_LOG = 3 } _DMODE_TYPE;

#endif // End of Avoid redefinition error





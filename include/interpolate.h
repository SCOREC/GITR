#ifndef _INTERPOLATE_
#define _INTERPOLATE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#define CUDA_CALLABLE_MEMBER_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
#define CUDA_CALLABLE_MEMBER_HOST
#endif

#ifdef __GNUC__ 
#include <stdlib.h>
#endif

struct interpolate { 
  double value;
  
  interpolate(double v) : value{v} {};
};

#endif

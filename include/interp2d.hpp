#ifndef _INTERP2D_
#define _INTERP2D_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include <thrust/device_vector.h>
#include <vector>
#include "math.h"
//using namespace std;
CUDA_CALLABLE_MEMBER

double interp2d ( double x, double z,int nx, int nz,
    double* gridx,double* gridz,double* data );

CUDA_CALLABLE_MEMBER

double interp2dCombined ( double x, double y, double z,int nx, int nz,
    double* gridx,double* gridz,double* data );
CUDA_CALLABLE_MEMBER

double interp3d ( double x, double y, double z,int nx,int ny, int nz,
    double* gridx,double* gridy, double* gridz,double* data );
CUDA_CALLABLE_MEMBER
void interp3dVector (double* field, double x, double y, double z,int nx,int ny, int nz,
        double* gridx,double* gridy,double* gridz,double* datar, double* dataz, double* datat );
CUDA_CALLABLE_MEMBER
void interp2dVector (double* field, double x, double y, double z,int nx, int nz,
double* gridx,double* gridz,double* datar, double* dataz, double* datat );
CUDA_CALLABLE_MEMBER
void interpFieldAlignedVector (double* field, double x, double y, double z,int nx, int nz,
        double* gridx,double* gridz,double* datar, double* dataz, double* datat,
        int nxB, int nzB, double* gridxB,double* gridzB,double* datarB,double* datazB, double* datatB);
CUDA_CALLABLE_MEMBER
double interp1dUnstructured(double samplePoint,int nx, double max_x, double* data,int &lowInd);
CUDA_CALLABLE_MEMBER
double interp1dUnstructured2(double samplePoint,int nx, double *xdata, double* data);
CUDA_CALLABLE_MEMBER
double interp2dUnstructured(double x,double y,int nx,int ny, double *xgrid,double *ygrid, double* data);
#endif

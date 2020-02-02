#ifndef _BOUNDARY_
#define _BOUNDARY_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cstdlib>
#include <stdio.h>
#include "array.h"
#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#else
#include <random>
using namespace std;
#endif

class Boundary 
{
  public:
    int periodic;
    int pointLine;
    int surfaceNumber;
    int surface;
    int inDir;
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
    double a;
    double b;
    double c;
    double d;
    double plane_norm; //16
    #if USE3DTETGEOM > 0
      double x3;
      double y3;
      double z3;
      double area;
    #else
      double slope_dzdx;
      double intercept_z;
    #endif     
    double Z;
    double amu;
    double potential;
    double ChildLangmuirDist;
    #ifdef __CUDACC__
    //curandState streams[7];
    #else
    //std::mt19937 streams[7];
    #endif
	
    double hitWall;
    double length;
    double distanceToParticle;
    double angle;
    double fd;
    double density;
    double ti;
    double ne;
    double te;
    double debyeLength;
    double larmorRadius;
    double flux;
    double startingParticles;
    double impacts;
    double redeposit;

    double midx;
    double midy;
    double midz;

    CUDA_CALLABLE_MEMBER
    void getSurfaceParallel(double A[],double y,double x)
    {
#if USE3DTETGEOM > 0
    double norm = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
    A[1] = (y2 - y1) / norm;
#else
    double norm = sqrt((x2 - x1) * (x2 - x1) + (z2 - z1) * (z2 - z1));
    A[1] = 0.0;
#endif
        //cout << "surf par calc " << x2 << " " << x1 << " " << norm << endl;
        A[0] = (x2-x1)/norm;
        A[2] = (z2-z1)/norm;
#if USE3DTETGEOM > 0
#else
#if USECYLSYMM > 0
    double theta = atan2(y, x);
    double B[3] = {0.0};
    B[0] = cos(theta) * A[0] - sin(theta) * A[1];
    B[1] = sin(theta) * A[0] + cos(theta) * A[1];
    A[0] = B[0];
    A[1] = B[1];
#endif
#endif
    }

  CUDA_CALLABLE_MEMBER
  void getSurfaceNormal(double B[], double y, double x) {
#if USE3DTETGEOM > 0
    B[0] = -a / plane_norm;
    B[1] = -b / plane_norm;
    B[2] = -c / plane_norm;
#else
    double perpSlope = 0.0;
    if (slope_dzdx == 0.0) {
      perpSlope = 1.0e12;
    } else {
      perpSlope = -copysign(1.0, slope_dzdx) / abs(slope_dzdx);
    }
    double Br = 1.0 / sqrt(perpSlope * perpSlope + 1.0);
    double Bt = 0.0;
    B[2] = copysign(1.0,perpSlope) * sqrt(1 - Br * Br);
#if USECYLSYMM > 0
    double theta = atan2(y, x);
    B[0] = cos(theta) * Br - sin(theta) * Bt;
    B[1] = sin(theta) * Br + cos(theta) * Bt;
#else
    B[0] = Br;
    B[1] = Bt;
#endif
//B[0] = -a/plane_norm;
//B[1] = -b/plane_norm;
//B[2] = -c/plane_norm;
//cout << "perp x and z comp " << B[0] << " " << B[2] << endl;
#endif
    }
    CUDA_CALLABLE_MEMBER
        void transformToSurface(double C[],double y, double x)
        {
            double X[3] = {0.0};
            double Y[3] = {0.0};
            double Z[3] = {0.0};
            double tmp[3] = {0.0};
            getSurfaceParallel(X,y,x);
            getSurfaceNormal(Z,y,x);
            Y[0] = Z[1]*X[2] - Z[2]*X[1]; 
            Y[1] = Z[2]*X[0] - Z[0]*X[2]; 
            Y[2] = Z[0]*X[1] - Z[1]*X[0];

            tmp[0] = X[0]*C[0] + Y[0]*C[1] + Z[0]*C[2];
            tmp[1] = X[1]*C[0] + Y[1]*C[1] + Z[1]*C[2];
            tmp[2] = X[2]*C[0] + Y[2]*C[1] + Z[2]*C[2];
            C[0] = tmp[0];
            C[1] = tmp[1];
            C[2] = tmp[2];

        }
//        Boundary(double x1,double y1, double z1, double x2, double y2, double z2,double slope, double intercept, double Z, double amu)
//		{
//    
//		this->x1 = x1;
//		this->y1 = y1;
//		this->z1 = z1;
//        this->x2 = x2;
//        this->y2 = y2;
//        this->z2 = z2;        
//#if USE3DTETGEOM > 0
//#else
//        this->slope_dzdx = slope;
//        this->intercept_z = intercept;
//#endif
//		this->Z = Z;
//		this->amu = amu;
//		this->hitWall = 0.0;
//        array1(amu,0.0);
//        };
};
#endif

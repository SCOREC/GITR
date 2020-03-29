#ifndef HISTORY_REGION_H
#define HISTORY_REGION_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Particles.h"
#include "Boundary.h"

struct history_region { 
  Particles *particlesPointer;
  int subSampleFac = 1;
  double *histX;
  double *histY;
  double *histZ;
  double *histPind;
  double *histTstep;
  int *filled;
  int size = 0;
  int plus = 0; // increment time
 
  double x1 = 0.03, x2 = 0.07;
  double y1 = -0.02, y2 = 0.02;
  double z1 = 0, z2 = 0.02;//0.01275;

  history_region(Particles *particlesPointer, int subSampleFac,
   double *hisX, double *histY, double *histZ,  double *histPind,
    double *histTstep, int *filled, int plus, int size):
   particlesPointer(particlesPointer), subSampleFac(subSampleFac), histX(hisX), 
   histY(histY), histZ(histZ), histPind(histPind), histTstep(histTstep), filled(filled),
   plus(plus), size(size) 
  {}

CUDA_CALLABLE_MEMBER_DEVICE    
  void operator()(size_t indx) const {  
    int tt0 = particlesPointer->tt[indx];
    if(plus)
      particlesPointer->tt[indx] = particlesPointer->tt[indx]+1;
    if (tt0 % subSampleFac == 0) {
      int indexP = particlesPointer->index[indx];
      auto x = particlesPointer->xprevious[indexP];
      auto y = particlesPointer->yprevious[indexP];
      auto z = particlesPointer->zprevious[indexP];  
      if(x>=x1 && x<=x2 && y>=y1 && y<=y2 && z>=z1 && z<=z2) {
        auto n = atomicAdd(filled, 1);
        if(n <size-1) {
          histX[n] = x;
          histY[n] = y;
          histZ[n] = z;
          histPind[n] = indexP;
          histTstep[n] = tt0;
        }
      }
    }
  }
};

#endif

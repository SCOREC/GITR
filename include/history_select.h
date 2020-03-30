#ifndef HISTORY_SELECT_H
#define HISTORY_SELECT_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Particles.h"
#include "Boundary.h"

struct history_select { 
  Particles *particlesPointer;
  int nT = 0;
  int subSampleFac = 1;
  double *histX;
  double *histY;
  double *histZ;
  double *histPind;
  double *histTstep;
  int *filled;
  int size = 0;
  int plus = 0; // increment time
  int debug = 0;

  double x1 = 0.03, x2 = 0.07;
  double y1 = -0.02, y2 = 0.02;
  double z1 = 0, z2 = 0.02;//0.01275;

  history_select(Particles *particlesPointer, int nT, int subSampleFac,
   double *hisX, double *histY, double *histZ,  double *histPind,
    double *histTstep, int *filled, int plus, int size):
   particlesPointer(particlesPointer), nT(nT), subSampleFac(subSampleFac), histX(hisX), 
   histY(histY), histZ(histZ), histPind(histPind), histTstep(histTstep), filled(filled),
   plus(plus), size(size) 
  {}

CUDA_CALLABLE_MEMBER_DEVICE    
  void operator()(size_t indx) const {  
    int tt = particlesPointer->tt[indx];
    int tt0 = (plus > 0) ? tt : tt-1;
    if(plus)
      particlesPointer->tt[indx] = particlesPointer->tt[indx]+1;
    if (tt0 % subSampleFac == 0) {
      int indexP = particlesPointer->index[indx];
      auto x = particlesPointer->xprevious[indexP];
      auto y = particlesPointer->yprevious[indexP];
      auto z = particlesPointer->zprevious[indexP];  
      //bool within = (x>=x1 && x<=x2 && y>=y1 && y<=y2 && z>=z1 && z<=z2);
      int store = particlesPointer->storeRnd[indx];
      if(store > 0) {
        int sid = particlesPointer->storeRndSeqId[indx];
        int pind = (sid >= 0) ? sid : indx;
        int ind = (pind*nT + tt0) /subSampleFac;
        auto n = atomicAdd(filled, 1);
        if(debug)
          printf("n %d tt0 %d ind %d pind %d sid %d size %d \n", 
            n, tt0, ind, pind, sid, size);
        if(n < size) {
          int m = ind; //n
          histX[m] = x;
          histY[m] = y;
          histZ[m] = z;
          histPind[m] = indexP;
          histTstep[m] = tt0;
        }
      }
    }
  }
};
#endif

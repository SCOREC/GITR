#ifndef _HISTORY_
#define _HISTORY_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Particles.h"
#include "Boundary.h"

struct history { 
    Particles *particlesPointer;
    int* tt;
    int nT;
    int subSampleFac;
    int nP;
    double *histX;
    double *histY;
    double *histZ;
    double *histv;
    double *histvx;
    double *histvy;
    double *histvz;
    double *histcharge;
    double *histweight;

    history(Particles *_particlesPointer, int* _tt, int _nT,int _subSampleFac, int _nP, double *_histX,double *_histY,double *_histZ,
          double *_histv,double *_histvx,double *_histvy,double *_histvz, double * _histcharge, double * _histweight) : 
        particlesPointer(_particlesPointer), tt(_tt),nT(_nT),subSampleFac(_subSampleFac), nP(_nP), 
        histX(_histX),histY(_histY),histZ(_histZ),histv(_histv),histvx(_histvx),histvy(_histvy),histvz(_histvz), histcharge(_histcharge), histweight(_histweight) {}

    CUDA_CALLABLE_MEMBER_DEVICE    
void operator()(size_t indx) const 
    {  
       int tt0=particlesPointer->tt[indx];
       particlesPointer->tt[indx] = particlesPointer->tt[indx]+1;
       //cout << "tt subsamplefac indx, nT " << tt0 << " "<< subSampleFac << " " << indx << " " << nT << endl;
       if (tt0 % subSampleFac == 0)
       {
       int indexP = particlesPointer->index[indx];
        int histInd = indexP*(nT/subSampleFac+1) + tt0/subSampleFac;
       //cout << "histInd " << histInd << endl;
        if(histInd <= (nP*(nT/subSampleFac+1)) && histInd >= 0 && indexP < nP)
        {
	  //cout << "inside history " << indx << " " << histInd << " " << particlesPointer->zprevious[indx] << endl;
          histX[histInd] = particlesPointer->xprevious[indexP];
          histY[histInd] = particlesPointer->yprevious[indexP];
          histZ[histInd] = particlesPointer->zprevious[indexP];
          histv[histInd] = particlesPointer->v[indexP];
          histvx[histInd] = particlesPointer->vx[indexP];
          histvy[histInd] = particlesPointer->vy[indexP];
          histvz[histInd] = particlesPointer->vz[indexP];
          histcharge[histInd] = particlesPointer->charge[indexP];
          histweight[histInd] = particlesPointer->weight[indexP];
        }
        //else
        //{
        //    if(particlesPointer->test[indx] == 0.0)
        //    {
        //        particlesPointer->test[indx] = 1.0;
        //        particlesPointer->test0[indx] =     histInd;
        //       particlesPointer->test1[indx] = indx;
        //       particlesPointer->test2[indx]=nT;
        //       particlesPointer->test3[indx]=subSampleFac;
        //      particlesPointer->test4[indx] =tt;
        //    }
        //} 
       }
    }
};

#endif

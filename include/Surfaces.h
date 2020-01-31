#ifndef _SURFACES_
#define _SURFACES_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
using namespace std;
#endif

#include "array.h"
#include <cstdlib>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#endif

#include <random>

//CUDA_CALLABLE_MEMBER

class Surfaces : public ManagedAllocation {
public: 
  int nSurfaces;  
  int nE;
  int nA;
  double E0;
  double E;
  double A0;
  double A;
  double dE;
  double dA;
  sim::Array<int> sumParticlesStrike;
  sim::Array<double> gridE;
  sim::Array<double> gridA;
  sim::Array<double> sumWeightStrike;
  sim::Array<double> grossDeposition;
  sim::Array<double> grossErosion;
  sim::Array<double> aveSputtYld;
  sim::Array<int> sputtYldCount;
  sim::Array<double> energyDistribution;
  sim::Array<double> sputtDistribution;
  sim::Array<double> reflDistribution;

  CUDA_CALLABLE_MEMBER
  
  void setSurface(int nE, double E0, double E, int nA, double A0, double A) {
    this->nE = nE;
    this->E0 = E0;
    this->E = E;
    this->nA = nA;
    this->A0 = A0;
    this->A = A;
    this->dE = (E - E0) / static_cast<double>(nE);
    this->dA = (A - A0) / static_cast<double>(nA);
    for (int i = 0; i < nE; i++) {
      this->gridE[i] = E0 + static_cast<double>(i) * dE;
    }

    for (int i = 0; i < nA; i++) {
      this->gridA[i] = A0 + static_cast<double>(i) * dA;
    }
  };

  CUDA_CALLABLE_MEMBER
  Surfaces(size_t nS,size_t nE, size_t nA) :
   sumParticlesStrike{nS,0}, gridE{nE,0.0}, gridA{nA,0.0},
   sumWeightStrike{nS,0.0}, grossDeposition{nS,0.0},
    grossErosion{nS,0.0}, aveSputtYld{nS,0.0}, sputtYldCount{nS,0},
   energyDistribution{nS*nE*nA,0.0},sputtDistribution{nS*nE*nA,0.0},
   reflDistribution{nS*nE*nA,0.0} {};   

};

#endif

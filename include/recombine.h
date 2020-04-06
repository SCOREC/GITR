#ifndef _RECOMBINE_
#define _RECOMBINE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#define CUDA_CALLABLE_MEMBER_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
#define CUDA_CALLABLE_MEMBER_HOST
using namespace std;
#endif

#include "Particles.h"
#ifdef __CUDACC__
#include <thrust/random.h>
#include <curand_kernel.h>
#endif

#ifdef __GNUC__ 
#include <random>
#include <stdlib.h>
#endif
#include "interpRateCoeff.hpp"

#ifndef RECOMB_DEBUG_PRINT
#define RECOMB_DEBUG_PRINT 0
#endif

struct recombine { 
  Particles *particlesPointer;
  int nR_Dens;
  int nZ_Dens;
  double* DensGridr;
  double* DensGridz;
  double* ne;
  int nR_Temp;
  int nZ_Temp;
  double* TempGridr;
  double* TempGridz;
  double* te;
  int nTemperaturesRecomb;
  int nDensitiesRecomb;
  double* gridDensity_Recombination;
  double* gridTemperature_Recombination;
  double* rateCoeff_Recombination;
  const double dt;
  double tion;

  int dof_intermediate = 0;
  int idof = -1;
  int nT = -1;
  double* intermediate;
  int select = 0;

//int& tt;
#if __CUDACC__
      curandState *state;
#else
      std::mt19937 *state;
#endif

  recombine(Particles *_particlesPointer, double _dt,
#if __CUDACC__
      curandState *_state,
#else
      std::mt19937 *_state,
#endif
     int _nR_Dens,int _nZ_Dens,double* _DensGridr,
     double* _DensGridz,double* _ne,int _nR_Temp, int _nZ_Temp,
     double* _TempGridr, double* _TempGridz,double* _te,int _nTemperaturesRecomb,
     int _nDensitiesRecomb,double* _gridTemperature_Recombination,double* _gridDensity_Recombination,
     double* _rateCoeff_Recombination,  double* intermediate = nullptr, int nT = 0, int idof = 0, 
     int dof_intermediate = 0, int select = 0): 
    particlesPointer(_particlesPointer),

                                               nR_Dens(_nR_Dens),
                                               nZ_Dens(_nZ_Dens),
                                               DensGridr(_DensGridr),
                                               DensGridz(_DensGridz),
                                               ne(_ne),
                                               nR_Temp(_nR_Temp),
                                               nZ_Temp(_nZ_Temp),
                                               TempGridr(_TempGridr),
                                               TempGridz(_TempGridz),
                                               te(_te),
                                               nTemperaturesRecomb(_nTemperaturesRecomb),
                                               nDensitiesRecomb(_nDensitiesRecomb),
                                               gridDensity_Recombination(_gridDensity_Recombination),
                                               gridTemperature_Recombination(_gridTemperature_Recombination),
                                               rateCoeff_Recombination(_rateCoeff_Recombination),
                                               dt(_dt), // JDL missing tion?
                                               state(_state), intermediate(intermediate),nT(nT),idof(idof), 
                                               dof_intermediate(dof_intermediate), select(select) 
                                               { }
 
  
  CUDA_CALLABLE_MEMBER_DEVICE 
  void operator()(size_t indx) { 
  double P1 = 0.0;
  double t_at =0, n_at=0;
      if(particlesPointer->charge[indx] > 0)
    {
       tion = interpRateCoeff2d ( particlesPointer->charge[indx], particlesPointer->x[indx], particlesPointer->y[indx], particlesPointer->z[indx],nR_Temp,nZ_Temp, TempGridr,TempGridz,te,DensGridr,DensGridz, ne,nTemperaturesRecomb,nDensitiesRecomb,gridTemperature_Recombination,gridDensity_Recombination,rateCoeff_Recombination, &t_at, &n_at);
       //double PrP = particlesPointer->PrecombinationPrevious[indx];
       double P = exp(-dt/tion);
       //particlesPointer->PrecombinationPrevious[indx] = PrP*P;
       P1 = 1.0-P;
    }

  if(particlesPointer->hitWall[indx] == 0.0)
	{        
#if PARTICLESEEDS > 0
        #ifdef __CUDACC__
        double r1 = curand_uniform(&state[indx]);
        #else
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r1=dist(state[indx]);
        #endif
#else
    #if __CUDACC__
    double r1 = curand_uniform(&state[1]);
    #else
            std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r1=dist(state[1]);
    #endif
#endif  

        int nthStep = particlesPointer->tt[indx] - 1;
        auto pindex = particlesPointer->index[indx];
        int beg = -1;

        if(dof_intermediate > 0 && particlesPointer->storeRnd[indx] > 0) {
          auto pind = pindex;
          int rid = particlesPointer->storeRndSeqId[indx]; 
          pind = (rid >= 0) ? rid : pind;
          beg = pind*nT*dof_intermediate + nthStep*dof_intermediate;
          if(!((pind >= 0) && (beg >= 0) && (nthStep>=0)))
            printf("recombSelect :  t %d at %d pind %d rid %d indx %d xyz %g %g %g \n", 
              nthStep, beg+idof, pind, rid, (int)indx,
              particlesPointer->x[indx],particlesPointer->y[indx],particlesPointer->z[indx]);
          assert((pind >= 0) && (beg >= 0) && (nthStep>=0));
          intermediate[beg+idof] = r1;
        }

	if(r1 <= P1)
	{
          particlesPointer->charge[indx] = particlesPointer->charge[indx]-1;
          particlesPointer->PrecombinationPrevious[indx] = 1.0;
	} 
        int selectThis = 1;
        if(select)
          selectThis = particlesPointer->storeRnd[indx];

    #if RECOMB_DEBUG_PRINT > 0
        if(selectThis > 0) {
          auto xx=particlesPointer->x[indx];
          auto yy=particlesPointer->y[indx];
          auto zz=particlesPointer->z[indx];
          printf("recomb: ptcl %d timestep %d charge %f rate %.15e temp %.15e "
              " dens %.15e recrand %.15e pos %.15e %.15e %.15e r1 %.15e r1@ %d\n",
              pindex, nthStep, particlesPointer->charge[indx], tion, t_at, n_at, 
              r1, xx, yy, zz, r1, beg+idof);
        }
    #endif
   }	

  } 
};

#endif

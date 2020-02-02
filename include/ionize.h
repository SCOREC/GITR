#ifndef _IONIZE_
#define _IONIZE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#define CUDA_CALLABLE_MEMBER_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
#define CUDA_CALLABLE_MEMBER_HOST
#endif

#include "Particles.h"
#ifdef __CUDACC__
#include <thrust/random.h>
#include <curand_kernel.h>
#endif

#ifdef __GNUC__ 
#include <random>
#include <stdlib.h>
using namespace std;
#endif
#include "interpRateCoeff.hpp"

#ifndef COMPARE_GITR_PRINT
#define COMPARE_GITR_PRINT 1
#endif

struct ionize { 
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
    int nTemperaturesIonize;
    int nDensitiesIonize;
    double* gridDensity_Ionization;
    double* gridTemperature_Ionization;
    double* rateCoeff_Ionization;
    const double dt;
    double tion;

    int dof_intermediate = 0;
    int idof = -1;
    int nT = -1;
    double* intermediate;
    //int& tt;
#if __CUDACC__
    curandState *state;
#else
    std::mt19937 *state;
#endif

        ionize(Particles *_particlesPointer, double _dt,
#if __CUDACC__
                curandState *_state,
#else
                std::mt19937 *_state,
#endif
                int _nR_Dens,int _nZ_Dens,double* _DensGridr,
    double* _DensGridz,double* _ne,int _nR_Temp, int _nZ_Temp,
    double* _TempGridr, double* _TempGridz,double* _te,int _nTemperaturesIonize,
    int _nDensitiesIonize,double* _gridTemperature_Ionization,double* _gridDensity_Ionization,
    double* _rateCoeff_Ionization, double* intermediate, int nT, int idof, int dof_intermediate
              ) : 
   
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
                                         nTemperaturesIonize(_nTemperaturesIonize),
                                         nDensitiesIonize(_nDensitiesIonize),
                                         gridDensity_Ionization(_gridDensity_Ionization),
                                         gridTemperature_Ionization(_gridTemperature_Ionization),
                                         rateCoeff_Ionization(_rateCoeff_Ionization),
                                         dt(_dt), // JDL missing tion here?
                                         state(_state), intermediate(intermediate),nT(nT),
                                          idof(idof), dof_intermediate(dof_intermediate) {}

        CUDA_CALLABLE_MEMBER_DEVICE 
                void operator()(size_t indx)  { 
	//if(particlesPointer->hitWall[indx] == 0.0){        
        //cout << "interpolating rate coeff at "<< particlesPointer->x[indx] << " " << particlesPointer->y[indx] << " " << particlesPointer->z[indx] << endl;
        double t_at=0, n_at=0;
       tion = interpRateCoeff2d ( particlesPointer->charge[indx], particlesPointer->x[indx], particlesPointer->y[indx], particlesPointer->z[indx],nR_Temp,nZ_Temp, TempGridr,TempGridz,te,DensGridr,DensGridz, ne,nTemperaturesIonize,nDensitiesIonize,gridTemperature_Ionization,gridDensity_Ionization,rateCoeff_Ionization, &t_at, &n_at );	
    //double PiP = particlesPointer->PionizationPrevious[indx];
    double P = expf(-dt/tion);
    //particlesPointer->PionizationPrevious[indx] = PiP*P;
    double P1 = 1.0-P;
    //cout << "tion P P1 " << tion << " " << P << " " << P1 << " " << PiP<< endl;
    if(COMPARE_GITR_PRINT==1 && particlesPointer->hitWall[indx] !=0) {
      printf("Not ionizing %d in timestep %d\n", particlesPointer->index[indx], particlesPointer->tt[indx]);
    }
    if(particlesPointer->hitWall[indx] == 0.0)
    {
        //cout << "calculating r1 " << endl;i
#if PARTICLESEEDS > 0
	#ifdef __CUDACC__
	  //double r1 = 0.5;//curand_uniform(&particlesPointer->streams[indx]);
      double r1 = curand_uniform(&state[indx]);
	#else
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double r1=dist(state[indx]);
	//particlesPointer->test[indx] = r1;
        //cout << " r1 " << r1 << endl;
    #endif
#else
  #if __CUDACC__
    curandState localState = state[thread_id];
    double r1 = curand_uniform(&localState);
    state[thread_id] = localState;
  #else
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r1=dist(state[0]);
  #endif
#endif
    //if(tt == 722)
    //{
      //cout << "r1 " << r1 << " " << P1 << endl;
		//particlesPointer->charge[indx] = particlesPointer->charge[indx]+1;
       //particlesPointer->test[indx] = tion; 
       //particlesPointer->test0[indx] = P; 
       //particlesPointer->test1[indx] = P1; 
       //particlesPointer->test2[indx] = r1; 

      int nthStep = particlesPointer->tt[indx];
      auto pindex = particlesPointer->index[indx];
      int beg = -1;
      if(dof_intermediate > 0) { 
        beg = pindex*nT*dof_intermediate + (nthStep-1)*dof_intermediate;
        intermediate[beg+idof] = r1;
      }
      if(COMPARE_GITR_PRINT==1) {
        auto xx=particlesPointer->x[indx];
        auto yy=particlesPointer->y[indx];
        auto zz=particlesPointer->z[indx];
          printf("ioni: ptcl %d timestep %d rate %g temp %g den %g ionirand %g P1 %g " 
              "pos %g %g %g r1 %g r1@ %d \n", 
              pindex, nthStep-1, tion, t_at, n_at, r1, P1, xx, yy, zz, r1, beg+idof);
      }


      if(r1 <= P1)
      {
		  particlesPointer->charge[indx] = particlesPointer->charge[indx]+1;}
          particlesPointer->PionizationPrevious[indx] = 1.0;
        //cout << "Particle " << indx << " ionized at step " << tt << endl;
       if(particlesPointer->firstIonizationZ[indx] == 0.0)
       {
           particlesPointer->firstIonizationZ[indx] = particlesPointer->z[indx];
       }
	    }
        else
        {
       if(particlesPointer->firstIonizationZ[indx] == 0.0)
       {
           particlesPointer->firstIonizationT[indx] = particlesPointer->firstIonizationT[indx] + dt;
       }

        } 
       
    //} 

	} 
};

#endif

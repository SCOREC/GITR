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

    int dof_intermediate;
    int idof;
    int nT;
    double* intermediate;
    int select;
    int* rndSelectPids;

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
    double* _rateCoeff_Ionization, double* intermediate=nullptr, int nT=0, int idof=-1,
    int dof_intermediate = 0, int select=0, int* rndSelectPids=nullptr): 
   
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
                                          idof(idof), dof_intermediate(dof_intermediate), select(select) {}

        CUDA_CALLABLE_MEMBER_DEVICE 
                void operator()(size_t indx)  { 
	//if(particlesPointer->hitWall[indx] == 0.0){        
        //cout << "interpolating rate coeff at "<< particlesPointer->x[indx] << " " << particlesPointer->y[indx] << " " << particlesPointer->z[indx] << endl;
        double t_at=0, n_at=0;
       tion = interpRateCoeff2d ( particlesPointer->charge[indx], particlesPointer->x[indx], particlesPointer->y[indx], particlesPointer->z[indx],nR_Temp,nZ_Temp, TempGridr,TempGridz,te,DensGridr,DensGridz, ne,nTemperaturesIonize,nDensitiesIonize,gridTemperature_Ionization,gridDensity_Ionization,rateCoeff_Ionization, &t_at, &n_at );	
    //double PiP = particlesPointer->PionizationPrevious[indx];
    double P = exp(-dt/tion);
    //particlesPointer->PionizationPrevious[indx] = PiP*P;
    double P1 = 1.0-P;
  #if  DEBUG_PRINT > 0
    printf("ioni0: ptcl %d  chargeIn %d rate-tion %.15e temp %.15e den %.15e P1 %.15e\n", 
              indx, particlesPointer->charge[indx], tion, t_at, n_at,  P1 );
  #endif
    int selectThis = 1;
    if(select)
      selectThis = particlesPointer->storeRnd[indx];
    //cout << "tion P P1 " << tion << " " << P << " " << P1 << " " << PiP<< endl;
  #if  DEBUG_PRINT > 0
    if(particlesPointer->hitWall[indx] !=0 && selectThis > 0)
      printf("indx %d Not ionizing %d in timestep %d\n", (int)indx,
          particlesPointer->index[indx], particlesPointer->tt[indx]-1);
  #endif
  
    if(particlesPointer->hitWall[indx] == 0.0)
    {
        //cout << "calculating r1 " << endl;i
//#if PARTICLESEEDS > 0
//	#ifdef __CUDACC__
	  //double r1 = 0.5;//curand_uniform(&particlesPointer->streams[indx]);
//      double r1 = curand_uniform(&state[indx]);
//	#else
//	std::uniform_real_distribution<double> dist(0.0, 1.0);
//	double r1=dist(state[indx]);
	//particlesPointer->test[indx] = r1;
        //cout << " r1 " << r1 << endl;
//    #endif
//#else
  #if __CUDACC__
    int id = particlesPointer->index[indx];
    double r1 = curand_uniform(&state[indx]);
    if(false)
      printf("ioni indx %d p %d t %d %.15f\n", (int)indx, id, particlesPointer->tt[indx] - 1, r1);
  #else
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r1=dist(state[0]);
  #endif
//#endif
    //if(tt == 722)
    //{
      //cout << "r1 " << r1 << " " << P1 << endl;
		//particlesPointer->charge[indx] = particlesPointer->charge[indx]+1;
       //particlesPointer->test[indx] = tion; 
       //particlesPointer->test0[indx] = P; 
       //particlesPointer->test1[indx] = P1; 
       //particlesPointer->test2[indx] = r1; 

      int nthStep = particlesPointer->tt[indx] - 1;
      int pindex = indx; //particlesPointer->index[indx];
      int beg = -1;
    
      if(dof_intermediate > 0 && selectThis > 0) {
        int rid = particlesPointer->storeRndSeqId[indx]; 
        int pind = (rid >= 0) ? rid : pindex;
        beg = pind*nT*dof_intermediate + nthStep*dof_intermediate;
        if(pind < 0 || beg < 0 || nthStep < 0)
          printf("ionizeSelect :  t %d at %d pind %d rid %d indx %d r1 %g xyz %g %g %g \n", 
            nthStep, beg+idof, pind, rid, (int)indx, r1,
            particlesPointer->x[indx],particlesPointer->y[indx],particlesPointer->z[indx]);
        assert(pind >= 0 && beg >= 0 && nthStep>=0);
        intermediate[beg+idof] = r1;
        if(select > 0)
          rndSelectPids[pind] = indx;
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
        #if  DEBUG_PRINT > 0 
          if(selectThis >0) {
            auto xx=particlesPointer->x[indx];
            auto yy=particlesPointer->y[indx];
            auto zz=particlesPointer->z[indx];
              printf("ioni: ptcl %d timestep %d  charge %d rate %.15e temp %.15e den %.15e ionirand %g P1 %g " 
                  "pos %.15e %.15e %.15e r1 %.15e\n", 
                  pindex, nthStep, particlesPointer->charge[indx], tion, t_at, n_at, r1, P1, xx, yy, zz, r1);
          }
        #endif
      
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

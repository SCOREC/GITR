#ifndef _PARTICLE_
#define _PARTICLE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
using namespace std;
#endif

#include <cstdlib>
#include <cmath>
#include <stdio.h>
//#include <vector>
#include "array.h"
//#include "managed_allocation.h"

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#else
#include <random>
#endif

//CUDA_CALLABLE_MEMBER

class Particle  {
	public:
	    double x;
	    double y;
	    double z;
        double xprevious;
        double yprevious;
        double zprevious;
      	double vx;
      	double vy;
      	double vz;
      	double Z;
      	double amu;
        double charge;
#if PARTICLESEEDS > 0
    #ifdef __CUDACC__
	curandState streams[7];
	#else
        mt19937 streams[7];
        #endif
#endif
	double hitWall;
    double transitTime;
    int wallIndex;
    double perpDistanceToSurface;
	double seed0;
	void BorisMove(double dt, double xMinV,double xMaxV,double yMin,double yMax,double zMin,double zMax);
	void Ionization(double dt);

	CUDA_CALLABLE_MEMBER
        Particle() {
            x=0.0;
	    y=0.0;
	    z=0.0;
        };
	
	CUDA_CALLABLE_MEMBER
        Particle(double x,double y, double z, double Ex, double Ey, double Ez, double Z, double amu, double charge)
		{
    
		this->xprevious = x;
		this->yprevious = y;
		this->zprevious = z;
        this->x = x;
        this->y = y;
        this->z = z;
		this->Z = Z;
	    this->charge = charge;
        this->amu = amu;
		this->hitWall = 0.0;
        this->wallIndex = 0;
		this->vx = Ex/abs(Ex)*sqrt(2.0*abs(Ex)*1.60217662e-19/(amu*1.6737236e-27));
		this->vy = Ey/abs(Ey)*sqrt(2.0*abs(Ey)*1.60217662e-19/(amu*1.6737236e-27));
		this->vz = Ez/abs(Ez)*sqrt(2.0*abs(Ez)*1.60217662e-19/(amu*1.6737236e-27));
		    
		if(Ex == 0.0) this->vx = 0.0;
		if(Ey == 0.0) this->vy = 0.0;
		if(Ez == 0.0) this->vz = 0.0;
        };
};
#endif

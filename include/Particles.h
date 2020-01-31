#ifndef _PARTICLES_
#define _PARTICLES_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
using namespace std;
#endif

#include "array.h"
#include <cstdlib>
#include <stdio.h>

#ifdef __CUDACC__
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#endif
#include <random>

#if USE_CUDA >0
//#if __CUDA_ARCH__ < 600
__device__ double atomicAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull =
                        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
      do {
             assumed = old;
             old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + 
                                __longlong_as_double(assumed)));
                 // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                      } while (assumed != old);
                 
                          return __longlong_as_double(old);
                          }

__device__ double atomicAdd1(int* address, int val)
{
    unsigned long long int* address_as_ull =
                        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
      do {
             assumed = old;
             old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + 
                                __longlong_as_double(assumed)));
                 // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
                      } while (assumed != old);
                 
                          return __longlong_as_double(old);
                          }

#endif
// CUDA_CALLABLE_MEMBER

class Particles : public ManagedAllocation {
public:
  size_t nParticles;
  sim::Array<int> index;
  sim::Array<double> x;
  sim::Array<double> y;
  sim::Array<double> z;
  sim::Array<double> xprevious;
  sim::Array<double> yprevious;
  sim::Array<double> zprevious;
  sim::Array<double> v;
  sim::Array<double> vx;
  sim::Array<double> vy;
  sim::Array<double> vz;
  sim::Array<double> Z;
  sim::Array<double> amu;
  sim::Array<double> charge;
  sim::Array<double> newVelocity;
  sim::Array<double> nu_s;
  sim::Array<double> vD;
  sim::Array<int> tt;
  sim::Array<int> hasLeaked;
  sim::Array<double> leakZ;
#if PARTICLESEEDS > 0
#ifdef __CUDACC__
  // sim::Array<curandState> streams;
  // sim::Array<curandState> streams_rec;
  // sim::Array<curandState> streams_collision1;
  // sim::Array<curandState> streams_collision2;
  // sim::Array<curandState> streams_collision3;
  // sim::Array<curandState> streams_diff;
  // sim::Array<curandState> streams_surf;
#else
  // sim::Array<mt19937> streams;
  // sim::Array<mt19937> streams_rec;
  // sim::Array<mt19937> streams_collision1;
  // sim::Array<mt19937> streams_collision2;
  // sim::Array<mt19937> streams_collision3;
  // sim::Array<mt19937> streams_diff;
  // sim::Array<mt19937> streams_surf;
#endif
#endif

  sim::Array<double> hitWall;
  sim::Array<int> wallHit;
  sim::Array<int> firstCollision;
  sim::Array<double> transitTime;
  sim::Array<double> distTraveled;
  sim::Array<int> wallIndex;
  sim::Array<double> perpDistanceToSurface;
  sim::Array<double> test;
  sim::Array<double> test0;
  sim::Array<double> test1;
  sim::Array<double> test2;
  sim::Array<double> test3;
  sim::Array<double> test4;
  sim::Array<double> distanceTraveled;
  sim::Array<double> weight;
  sim::Array<double> PionizationPrevious;
  sim::Array<double> PrecombinationPrevious;
  sim::Array<double> firstIonizationZ;
  sim::Array<double> firstIonizationT;

  //  void BorisMove(double dt, double xMinV, double xMaxV, double yMin, double
  //  yMax, double zMin, double zMax);

  //  void Ionization(double dt);
  CUDA_CALLABLE_MEMBER
  void setParticle(int indx, double x, double y, double z, double Ex, double Ey,
                   double Ez, double Z, double amu, double charge) {

    // this->index[indx] = indx;
    this->xprevious[indx] = x;
    this->yprevious[indx] = y;
    this->zprevious[indx] = z;
    this->x[indx] = x;
    this->y[indx] = y;
    this->z[indx] = z;
    this->Z[indx] = Z;
    this->charge[indx] = charge;
    this->amu[indx] = amu;
    this->hitWall[indx] = 0.0;
    this->wallIndex[indx] = 0;
    //        double Ex,Ey,Ez;
    //        Ex = E*cos(theta)*sin(phi);
    //        Ey = E*sin(theta)*sin(phi);
    //        Ez = E*cos(phi);
    this->vx[indx] =
        Ex / abs(Ex) *
        sqrt(2.0 * abs(Ex) * 1.60217662e-19 / (amu * 1.6737236e-27));
    this->vy[indx] =
        Ey / abs(Ey) *
        sqrt(2.0 * abs(Ey) * 1.60217662e-19 / (amu * 1.6737236e-27));
    this->vz[indx] =
        Ez / abs(Ez) *
        sqrt(2.0 * abs(Ez) * 1.60217662e-19 / (amu * 1.6737236e-27));

    if (Ex == 0.0)
      this->vx[indx] = 0.0;
    if (Ey == 0.0)
      this->vy[indx] = 0.0;
    if (Ez == 0.0)
      this->vz[indx] = 0.0;
    // cout << " velocity " << this->vx[indx] << " " << this->vy[indx] << "
    // " << this->vz[indx] << endl;
  };

  CUDA_CALLABLE_MEMBER
  void setParticleV(int indx, double x, double y, double z, double Vx, double Vy,
                    double Vz, double Z, double amu, double charge) {
    int indTmp = indx;
    this->index[indx] = indTmp;
    this->xprevious[indx] = x;
    this->yprevious[indx] = y;
    this->zprevious[indx] = z;
    this->x[indx] = x;
    this->y[indx] = y;
    this->z[indx] = z;
    this->Z[indx] = Z;
    this->charge[indx] = charge;
    this->amu[indx] = amu;
    this->hitWall[indx] = 0.0;
    this->wallIndex[indx] = 0;
    this->vx[indx] = Vx;
    this->vy[indx] = Vy;
    this->vz[indx] = Vz;
    this->v[indx] = sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
  };
  CUDA_CALLABLE_MEMBER
  void swapP(int indx, int n) {
    int iT = this->index[indx];
    double xpT = this->xprevious[indx];
    double ypT = this->yprevious[indx];
    double zpT = this->zprevious[indx];
    double xT = this->x[indx];
    double yT = this->y[indx];
    double zT = this->z[indx];
    double wT = this->weight[indx];
    double ZT = this->Z[indx];
    double cT = this->charge[indx];
    double aT = this->amu[indx];
    double hWT = this->hitWall[indx];
    int wIT = this->wallIndex[indx];
    double vxT = this->vx[indx];
    double vyT = this->vy[indx];
    double vzT = this->vz[indx];
    int wHT = this->wallHit[indx];
    double ttT = this->transitTime[indx];
    double dtT = this->distTraveled[indx];
    double firstIonizationZT = this->firstIonizationZ[indx];
    double firstIonizationTT = this->firstIonizationT[indx];

    this->index[indx] = this->index[n];
    this->xprevious[indx] = this->xprevious[n];
    this->yprevious[indx] = this->yprevious[n];
    this->zprevious[indx] = this->zprevious[n];
    this->x[indx] = this->x[n];
    this->y[indx] = this->y[n];
    this->z[indx] = this->z[n];
    this->weight[indx] = this->weight[n];
    this->Z[indx] = this->Z[n];
    this->charge[indx] = this->charge[n];
    this->amu[indx] = this->amu[n];
    this->hitWall[indx] = this->hitWall[n];
    this->wallIndex[indx] = this->wallIndex[n];
    this->vx[indx] = this->vx[n];
    this->vy[indx] = this->vy[n];
    this->vz[indx] = this->vz[n];
    this->wallHit[indx] = this->wallHit[n];
    this->transitTime[indx] = this->transitTime[n];
    this->distTraveled[indx] = this->distTraveled[n];
    this->firstIonizationZ[indx] = this->firstIonizationZ[n];
    this->firstIonizationT[indx] = this->firstIonizationT[n];

    this->index[n] = iT;
    this->xprevious[n] = xpT;
    this->yprevious[n] = ypT;
    this->zprevious[n] = zpT;
    this->x[n] = xT;
    this->y[n] = yT;
    this->z[n] = zT;
    this->weight[n] = wT;
    this->Z[n] = ZT;
    this->charge[n] = cT;
    this->amu[n] = aT;
    this->hitWall[n] = hWT;
    this->wallIndex[n] = wIT;
    this->vx[n] = vxT;
    this->vy[n] = vyT;
    this->vz[n] = vzT;
    this->wallHit[n] = wHT;
    this->transitTime[n] = ttT;
    this->distTraveled[n] = dtT;
    this->firstIonizationZ[n] = firstIonizationZT;
    this->firstIonizationT[n] = firstIonizationTT;
  };
  CUDA_CALLABLE_MEMBER
  Particles(size_t nP)
      : nParticles{nP}, index{nP, 0}, x{nP}, y{nP}, z{nP}, xprevious{nP},
        yprevious{nP}, zprevious{nP},v{nP, 0.0}, vx{nP}, vy{nP}, vz{nP}, Z{nP},
        amu{nP}, charge{nP}, newVelocity{nP}, nu_s{nP}, vD{nP, 0.0}, tt{nP, 0}, hasLeaked{nP, 0}, leakZ{nP,0.0},
#if PARTICLESEEDS > 0
  //    streams{nP},streams_rec{nP},streams_collision1{nP},streams_collision2{nP},
  //    streams_collision3{nP},streams_diff{nP},streams_surf{nP},
#endif
        hitWall{nP, 0.0},wallHit{nP, 0},firstCollision{nP, 1}, transitTime{nP, 0.0}, distTraveled{nP, 0.0},
        wallIndex{nP},
        perpDistanceToSurface{nP}, test{nP, 0.0}, test0{nP, 0.0},
        test1{nP, 0.0}, test2{nP, 0.0}, test3{nP, 0.0}, test4{nP, 0.0},
        distanceTraveled{nP}, weight{nP, 1.0}, PionizationPrevious{nP, 1.0},
        PrecombinationPrevious{nP, 1.0}, firstIonizationZ{nP, 0.0},
        firstIonizationT{nP, 0.0} {};
};

#endif

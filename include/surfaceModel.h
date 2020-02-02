#ifndef _SURFACE_
#define _SURFACE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include "Particles.h"
#include "Boundary.h"
#include "Surfaces.h"
#include <cmath>
#include "boris.h"
#ifdef __CUDACC__
#include "cuda_runtime_api.h"
#include <thrust/random.h>
#else
#include <random>
using namespace std:
#endif

CUDA_CALLABLE_MEMBER
void getBoundaryNormal(Boundary* boundaryVector,int wallIndex,double surfaceNormalVector[],double x,double y){
  #if USE3DTETGEOM > 0
           double norm_normal = boundaryVector[wallIndex].plane_norm; 
            surfaceNormalVector[0] = boundaryVector[wallIndex].a/norm_normal;
            surfaceNormalVector[1] = boundaryVector[wallIndex].b/norm_normal;
            
            surfaceNormalVector[2] = boundaryVector[wallIndex].c/norm_normal;
  #else
            double tol = 1e12;
            double norm_normal = 0.0;
            if (boundaryVector[wallIndex].slope_dzdx == 0.0)
                {
                 surfaceNormalVector[0] = 0.0;
                 surfaceNormalVector[1] = 0.0;
                 surfaceNormalVector[2] = 1.0;
                }
            else if (abs(boundaryVector[wallIndex].slope_dzdx)>= 0.75*tol)
                {
                    surfaceNormalVector[0] = 1.0;
                    surfaceNormalVector[1] = 0.0;
                    surfaceNormalVector[2] = 0.0;
                }
            else
                {
                    surfaceNormalVector[0] = 1.0;
                    surfaceNormalVector[1] = 0.0;
                    surfaceNormalVector[2] = -1.0 / (boundaryVector[wallIndex].slope_dzdx);
            norm_normal = sqrt(surfaceNormalVector[2]*surfaceNormalVector[2] + 1.0); 
            surfaceNormalVector[0] = surfaceNormalVector[0]/norm_normal;
            surfaceNormalVector[1] = surfaceNormalVector[1]/norm_normal;
            
            surfaceNormalVector[2] = surfaceNormalVector[2]/norm_normal;
                }
#if USECYLSYMM > 0 
            double theta = atan2(y,x);
            double Sr = surfaceNormalVector[0];
            surfaceNormalVector[0] = cos(theta)*Sr;
            surfaceNormalVector[1] = sin(theta)*Sr;
#endif            
#endif
}
CUDA_CALLABLE_MEMBER
double screeningLength ( double Zprojectile, double Ztarget ) {
	double bohrRadius = 5.29177e-11;
	double screenLength;

	screenLength = 0.885341*bohrRadius*pow(pow(Zprojectile,(2/3)) + pow(Ztarget,(2/3)),(-1/2));

	return screenLength;
}

CUDA_CALLABLE_MEMBER
double stoppingPower (Particles * particles,int indx, double Mtarget, double Ztarget, double screenLength) {
	        double E0;
            double Q = 1.60217662e-19;
		double ke2 = 14.4e-10;
		double reducedEnergy;
	double stoppingPower;

	E0 = 0.5*particles->amu[indx]*1.6737236e-27*(particles->vx[indx]*particles->vx[indx] + particles->vy[indx]*particles->vy[indx]+ particles->vz[indx]*particles->vz[indx])/Q;
	reducedEnergy = E0*(Mtarget/(particles->amu[indx]+Mtarget))*(screenLength/(particles->Z[indx]*Ztarget*ke2));
	stoppingPower = 0.5*log(1.0 + 1.2288*reducedEnergy)/(reducedEnergy + 0.1728*sqrt(reducedEnergy) + 0.008*pow(reducedEnergy, 0.1504));

	return stoppingPower;	
}

struct erosion { 
    Particles *particles;
    const double dt;

    erosion(Particles *_particles, double _dt) : particles(_particles), dt(_dt) {} 

CUDA_CALLABLE_MEMBER_DEVICE    
void operator()(size_t indx) const { 
	double screenLength;
	double stopPower;
	double q = 18.6006;
	double lambda = 2.2697;
    double mu = 3.1273;
	double Eth = 24.9885;
	double Y0;
	double Ztarget = 74.0;
	double Mtarget = 183.84;
	double term;
	double E0;

	screenLength = screeningLength(particles->Z[indx], Ztarget);
	stopPower = stoppingPower(particles,indx, Mtarget, Ztarget, screenLength); 
	E0 = 0.5*particles->amu[indx]*1.6737236e-27*(particles->vx[indx]*particles->vx[indx] + particles->vy[indx]*particles->vy[indx]+ particles->vz[indx]*particles->vz[indx])/1.60217662e-19;
	term = pow((E0/Eth - 1),mu);
	Y0 = q*stopPower*term/(lambda + term);
    	}
     
};

struct reflection {
    Particles * particles;
    const double dt;
    int nLines;
    Boundary * boundaryVector;
    Surfaces * surfaces;
    int nE_sputtRefCoeff;
    int nA_sputtRefCoeff;
    double* A_sputtRefCoeff;
    double* Elog_sputtRefCoeff;
    double* spyl_surfaceModel;
    double* rfyl_surfaceModel;
    int nE_sputtRefDistOut; 
    int nE_sputtRefDistOutRef; 
    int nA_sputtRefDistOut;
    int nE_sputtRefDistIn;
    int nA_sputtRefDistIn;
    double* E_sputtRefDistIn;
    double* A_sputtRefDistIn;
    double* E_sputtRefDistOut;
    double* E_sputtRefDistOutRef;
    double* A_sputtRefDistOut;
    double* energyDistGrid01;
    double* energyDistGrid01Ref;
    double* angleDistGrid01;
    double* EDist_CDF_Y_regrid;
    double* ADist_CDF_Y_regrid;
    double* EDist_CDF_R_regrid;
    double* ADist_CDF_R_regrid;
    int nEdist;
    double E0dist;
    double Edist;
    int nAdist;
    double A0dist;
    double Adist;

    int dof_intermediate = 0;
    int idof = -1;
    int nT = -1;
    double* intermediate;
#if __CUDACC__
        curandState *state;
#else
        std::mt19937 *state;
#endif
    reflection(Particles* _particles, double _dt,
#if __CUDACC__
                            curandState *_state,
#else
                            std::mt19937 *_state,
#endif
            int _nLines,Boundary * _boundaryVector,
            Surfaces * _surfaces,
    int _nE_sputtRefCoeff,
    int _nA_sputtRefCoeff,
    double* _A_sputtRefCoeff,
    double* _Elog_sputtRefCoeff,
    double* _spyl_surfaceModel,
    double* _rfyl_surfaceModel,
    int _nE_sputtRefDistOut,
    int _nE_sputtRefDistOutRef,
    int _nA_sputtRefDistOut,
    int _nE_sputtRefDistIn,
    int _nA_sputtRefDistIn,
    double* _E_sputtRefDistIn,
    double* _A_sputtRefDistIn,
    double* _E_sputtRefDistOut,
    double* _E_sputtRefDistOutRef,
    double* _A_sputtRefDistOut,
    double* _energyDistGrid01,
    double* _energyDistGrid01Ref,
    double* _angleDistGrid01,
    double* _EDist_CDF_Y_regrid,
    double* _ADist_CDF_Y_regrid, 
    double* _EDist_CDF_R_regrid,
    double* _ADist_CDF_R_regrid,
    int _nEdist,
    double _E0dist,
    double _Edist,
    int _nAdist,
    double _A0dist,
    double _Adist, double* intermediate, int nT, int idof, int dof_intermediate) :
particles(_particles),
                             dt(_dt),
                             nLines(_nLines),
                             boundaryVector(_boundaryVector),
                             surfaces(_surfaces),
                             nE_sputtRefCoeff(_nE_sputtRefCoeff),
                             nA_sputtRefCoeff(_nA_sputtRefCoeff),
                             A_sputtRefCoeff(_A_sputtRefCoeff),
                             Elog_sputtRefCoeff(_Elog_sputtRefCoeff),
                             spyl_surfaceModel(_spyl_surfaceModel),
                             rfyl_surfaceModel(_rfyl_surfaceModel),
                             nE_sputtRefDistOut(_nE_sputtRefDistOut),
                             nE_sputtRefDistOutRef(_nE_sputtRefDistOutRef),
                             nA_sputtRefDistOut(_nA_sputtRefDistOut),
                             nE_sputtRefDistIn(_nE_sputtRefDistIn),
                             nA_sputtRefDistIn(_nA_sputtRefDistIn),
                             E_sputtRefDistIn(_E_sputtRefDistIn),
                             A_sputtRefDistIn(_A_sputtRefDistIn),
                             E_sputtRefDistOut(_E_sputtRefDistOut),
                             E_sputtRefDistOutRef(_E_sputtRefDistOutRef),
                             A_sputtRefDistOut(_A_sputtRefDistOut),
                             energyDistGrid01(_energyDistGrid01),
                             energyDistGrid01Ref(_energyDistGrid01Ref),
                             angleDistGrid01(_angleDistGrid01),
                             EDist_CDF_Y_regrid(_EDist_CDF_Y_regrid),
                             ADist_CDF_Y_regrid(_ADist_CDF_Y_regrid),
                             EDist_CDF_R_regrid(_EDist_CDF_R_regrid),
                             ADist_CDF_R_regrid(_ADist_CDF_R_regrid),
                             nEdist(_nEdist),
                             E0dist(_E0dist),
                             Edist(_Edist),
                             nAdist(_nAdist),
                             A0dist(_A0dist),
                             Adist(_Adist),
                             state(_state),intermediate(intermediate),nT(nT),
                             idof(idof), dof_intermediate(dof_intermediate) {
  }

CUDA_CALLABLE_MEMBER_DEVICE
void operator()(size_t indx) const {
    
    if (particles->hitWall[indx] == 1.0) {
      double E0 = 0.0;
      double thetaImpact = 0.0;
      double particleTrackVector[3] = {0.0};
      double surfaceNormalVector[3] = {0.0};
      double vSampled[3] = {0.0};
      double norm_part = 0.0;
      int signPartDotNormal = 0;
      double partDotNormal = 0.0;
      double Enew = 0.0;
      double angleSample = 0.0;
      int wallIndex = 0;
      double tol = 1e12;
      double Sr = 0.0;
      double St = 0.0;
      double Y0 = 0.0;
      double R0 = 0.0;
      double totalYR = 0.0;
      double newWeight = 0.0;
      int wallHit = particles->wallHit[indx];
      int surfaceHit = boundaryVector[wallHit].surfaceNumber;
      int surface = boundaryVector[wallHit].surface;
      if (wallHit > 260)
        wallHit = 260;
      if (wallHit < 0)
        wallHit = 0;
      if (surfaceHit > 260)
        surfaceHit = 260;
      if (surfaceHit < 0)
        surfaceHit = 0;
      if (surface > 260)
        surface = 260;
      if (surface < 0)
        surface = 0;
      double eInterpVal = 0.0;
      double aInterpVal = 0.0;
      double weight = particles->weight[indx];
      double vx = particles->vx[indx];
      double vy = particles->vy[indx];
      double vz = particles->vz[indx];
#if FLUX_EA > 0
      double dEdist = (Edist - E0dist) / static_cast<double>(nEdist);
      double dAdist = (Adist - A0dist) / static_cast<double>(nAdist);
      int AdistInd = 0;
      int EdistInd = 0;
#endif
      particles->firstCollision[indx] = 1;
      particleTrackVector[0] = vx;
      particleTrackVector[1] = vy;
      particleTrackVector[2] = vz;
      norm_part = sqrt(particleTrackVector[0] * particleTrackVector[0] + particleTrackVector[1] * particleTrackVector[1] + particleTrackVector[2] * particleTrackVector[2]);
      E0 = 0.5 * particles->amu[indx] * 1.6737236e-27 * (norm_part * norm_part) / 1.60217662e-19;
      if (E0 > 1000.0)
        E0 = 990.0;
      //cout << "Particle hit wall with energy " << E0 << endl;
      //cout << "Particle hit wall with v " << vx << " " << vy << " " << vz<< endl;
      //cout << "Particle amu norm_part " << particles->amu[indx] << " " << vy << " " << vz<< endl;
      wallIndex = particles->wallIndex[indx];
      boundaryVector[wallHit].getSurfaceNormal(surfaceNormalVector, particles->y[indx], particles->x[indx]);
      particleTrackVector[0] = particleTrackVector[0] / norm_part;
      particleTrackVector[1] = particleTrackVector[1] / norm_part;
      particleTrackVector[2] = particleTrackVector[2] / norm_part;

      partDotNormal = vectorDotProduct(particleTrackVector, surfaceNormalVector);
      thetaImpact = acos(partDotNormal);
      if (thetaImpact > 3.14159265359 * 0.5) {
        thetaImpact = abs(thetaImpact - (3.14159265359));
      }
      thetaImpact = thetaImpact * 180.0 / 3.14159265359;
      if (thetaImpact < 0.0)
        thetaImpact = 0.0;
      signPartDotNormal = copysign(1.0,partDotNormal);
      if (E0 == 0.0) {
        thetaImpact = 0.0;
      }
      if (boundaryVector[wallHit].Z > 0.0) {
        Y0 = interp2d(thetaImpact, log10(E0), nA_sputtRefCoeff,
                      nE_sputtRefCoeff, A_sputtRefCoeff,
                      Elog_sputtRefCoeff, spyl_surfaceModel);
        R0 = interp2d(thetaImpact, log10(E0), nA_sputtRefCoeff,
                      nE_sputtRefCoeff, A_sputtRefCoeff,
                      Elog_sputtRefCoeff, rfyl_surfaceModel);
      } else {
        Y0 = 0.0;
        R0 = 0.0;
      }
      //cout << "Particle " << indx << " struck surface with energy and angle " << E0 << " " << thetaImpact << endl;
      //cout << " resulting in Y0 and R0 of " << Y0 << " " << R0 << endl;
      totalYR = Y0 + R0;
//if(particles->test[indx] == 0.0)
//{
//    particles->test[indx] = 1.0;
//    particles->test0[indx] = E0;
//    particles->test1[indx] = thetaImpact;
//    particles->test2[indx] = Y0;
//    particles->test3[indx] = R0;
//}
//cout << "Energy angle yield " << E0 << " " << thetaImpact << " " << Y0 << endl;
#if PARTICLESEEDS > 0
#ifdef __CUDACC__
      double r7 = curand_uniform(&state[indx]);
      double r8 = curand_uniform(&state[indx]);
      double r9 = curand_uniform(&state[indx]);
      double r10 = curand_uniform(&state[indx]);
#else
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      double r7 = dist(state[indx]);
      double r8 = dist(state[indx]);
      double r9 = dist(state[indx]);
      double r10 = dist(state[indx]);
#endif

            #else
              #if __CUDACC__
                double r7 = curand_uniform(&state[6]);
                double r8 = curand_uniform(&state[7]);
                double r9 = curand_uniform(&state[8]);
              #else
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double r7=dist(state[6]);
                double r8=dist(state[7]);
                double r9=dist(state[8]);
              #endif
                //double r7 = 0.0;
            #endif

    
      int nthStep = particles->tt[indx];
      auto pindex = particles->index[indx];
      int beg = -1;
      if(dof_intermediate > 0) { 
        beg = pindex*nT*dof_intermediate + (nthStep-1)*dof_intermediate;
        intermediate[beg+idof] = r7;
        intermediate[beg+idof+1] = r8;
        intermediate[beg+idof+2] = r9;
        intermediate[beg+idof+3] = r10;
      }
    
                //particle either reflects or deposits
            double sputtProb = Y0/totalYR;
	    int didReflect = 0;
            if(totalYR > 0.0)
            {
            if(r7 > sputtProb) //reflects
            {
	          didReflect = 1;
                  aInterpVal = interp3d (r8,thetaImpact,log10(E0),
                          nA_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                                    angleDistGrid01,A_sputtRefDistIn,
                                    E_sputtRefDistIn,ADist_CDF_R_regrid);
                   eInterpVal = interp3d ( r9,thetaImpact,log10(E0),
                           nE_sputtRefDistOutRef,nA_sputtRefDistIn,nE_sputtRefDistIn,
                                         energyDistGrid01Ref,A_sputtRefDistIn,
                                         E_sputtRefDistIn,EDist_CDF_R_regrid );
                   //newWeight=(R0/(1.0-sputtProb))*weight;
		   newWeight = weight*(totalYR);
    #if FLUX_EA > 0
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist))
              {
            #if USE_CUDA > 0
                  atomicAdd1(&surfaces->reflDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd],newWeight);
            #else      
                  surfaces->reflDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->reflDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  newWeight;
            #endif
               }
	       #endif
                  if(surface > 0)
                {

            #if USE_CUDA > 0
                    atomicAdd1(&surfaces->grossDeposition[surfaceHit],weight*(1.0-R0));
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight*(1.0-R0);
            #endif
                }
            }
            else //sputters
            {
                  aInterpVal = interp3d(r8,thetaImpact,log10(E0),
                          nA_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                          angleDistGrid01,A_sputtRefDistIn,
                          E_sputtRefDistIn,ADist_CDF_Y_regrid);
                  eInterpVal = interp3d(r9,thetaImpact,log10(E0),
                           nE_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                           energyDistGrid01,A_sputtRefDistIn,E_sputtRefDistIn,EDist_CDF_Y_regrid);
            //if(particles->test[indx] == 0.0)
            //{
            //    particles->test[indx] = 1.0;
            //    particles->test0[indx] = aInterpVal;
            //    particles->test1[indx] = eInterpVal;
            //    particles->test2[indx] = r8;
            //    particles->test3[indx] = r9;
            //}            
                //cout << " particle sputters with " << eInterpVal << aInterpVal <<  endl;
                  //newWeight=(Y0/sputtProb)*weight;
		  newWeight=weight*totalYR;
    #if FLUX_EA > 0
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist))
              {
                //cout << " particle sputters with " << EdistInd << AdistInd <<  endl;
            #if USE_CUDA > 0
                  atomicAdd1(&surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd],newWeight);
            #else      
                  surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  newWeight;
              #endif 
              }
	       #endif
                  if(sputtProb == 0.0) newWeight = 0.0;
                   //cout << " particle sputtered with newWeight " << newWeight << endl;
                  if(surface > 0)
                {

            #if USE_CUDA > 0
                    atomicAdd1(&surfaces->grossDeposition[surfaceHit],weight*(1.0-R0));
                    atomicAdd1(&surfaces->grossErosion[surfaceHit],newWeight);
                    atomicAdd1(&surfaces->aveSputtYld[surfaceHit],Y0);
                    if(weight > 0.0)
                    {
                        atomicAdd1(&surfaces->sputtYldCount[surfaceHit],1);
                    }
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight*(1.0-R0);
                    surfaces->grossErosion[surfaceHit] = surfaces->grossErosion[surfaceHit] + newWeight;
                    surfaces->aveSputtYld[surfaceHit] = surfaces->aveSputtYld[surfaceHit] + Y0;
                    surfaces->sputtYldCount[surfaceHit] = surfaces->sputtYldCount[surfaceHit] + 1;
            #endif
                }
            }
            //cout << "eInterpValYR " << eInterpVal << endl; 
            }
            else
            {       newWeight = 0.0;
                    particles->hitWall[indx] = 2.0;
                  if(surface > 0)
                {
            #if USE_CUDA > 0
                    atomicAdd1(&surfaces->grossDeposition[surfaceHit],weight);
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight;
            #endif
	        }
            //cout << "eInterpValYR_not " << eInterpVal << endl; 
            }
            //cout << "eInterpVal " << eInterpVal << endl; 
	    if(eInterpVal <= 0.0)
            {       newWeight = 0.0;
                    particles->hitWall[indx] = 2.0;
                  if(surface > 0)
                {
		    if(didReflect)
		    {
            #if USE_CUDA > 0
                    atomicAdd1(&surfaces->grossDeposition[surfaceHit],weight);
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight;
            #endif
	            }
		}
            }
            //if(particles->test[indx] == 1.0)
            //{
            //    particles->test3[indx] = eInterpVal;
            //    particles->test[indx] = 2.0;
            //}
                //deposit on surface
            if(surface)
            {
            #if USE_CUDA > 0
                atomicAdd1(&surfaces->sumWeightStrike[surfaceHit],weight);
                atomicAdd1(&surfaces->sumParticlesStrike[surfaceHit],1);
            #else
                surfaces->sumWeightStrike[surfaceHit] =surfaces->sumWeightStrike[surfaceHit] +weight;
                surfaces->sumParticlesStrike[surfaceHit] = surfaces->sumParticlesStrike[surfaceHit]+1;
              //boundaryVector[wallHit].impacts = boundaryVector[wallHit].impacts +  particles->weight[indx];
            #endif
            #if FLUX_EA > 0
                EdistInd = floor((E0-E0dist)/dEdist);
                AdistInd = floor((thetaImpact-A0dist)/dAdist);
              
	        if((EdistInd >= 0) && (EdistInd < nEdist) && 
                  (AdistInd >= 0) && (AdistInd < nAdist))
                {
#if USE_CUDA > 0
                    atomicAdd1(&surfaces->energyDistribution[surfaceHit*nEdist*nAdist + 
                                               EdistInd*nAdist + AdistInd], weight);
#else

                    surfaces->energyDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->energyDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  weight;
#endif
        }
#endif
      }
      //reflect with weight and new initial conditions
      //cout << "particle wall hit Z and nwweight " << boundaryVector[wallHit].Z << " " << newWeight << endl;
      if (boundaryVector[wallHit].Z > 0.0 && newWeight > 0.0)
      //if(newWeight > 0.0)
      {
        particles->weight[indx] = newWeight;
        particles->hitWall[indx] = 0.0;
        particles->charge[indx] = 0.0;
        double V0 = sqrt(2 * eInterpVal * 1.602e-19 / (particles->amu[indx] * 1.66e-27));
        particles->newVelocity[indx] = V0;
        vSampled[0] = V0 * sin(aInterpVal * 3.1415 / 180) * cos(2.0 * 3.1415 * r10);
        vSampled[1] = V0 * sin(aInterpVal * 3.1415 / 180) * sin(2.0 * 3.1415 * r10);
        vSampled[2] = V0 * cos(aInterpVal * 3.1415 / 180);
        boundaryVector[wallHit].transformToSurface(vSampled, particles->y[indx], particles->x[indx]);
        //double rr = sqrt(particles->x[indx]*particles->x[indx] + particles->y[indx]*particles->y[indx]);
        //if (particles->z[indx] < -4.1 && -signPartDotNormal*vSampled[0] > 0.0)
        //{
        //  cout << "particle index " << indx  << endl;
        //  cout << "aInterpVal" << aInterpVal  << endl;
        //  cout << "Surface Normal" << surfaceNormalVector[0] << " " << surfaceNormalVector[1] << " " << surfaceNormalVector[2] << endl;
        //  cout << "signPartDotNormal " << signPartDotNormal << endl;
        //  cout << "Particle hit wall with v " << vx << " " << vy << " " << vz<< endl;
        //  cout << "vSampled " << vSampled[0] << " " << vSampled[1] << " " << vSampled[2] << endl;
        //  cout << "Final transform" << -signPartDotNormal*vSampled[0] << " " << -signPartDotNormal*vSampled[1] << " " << -signPartDotNormal*vSampled[2] << endl;
        //  cout << "Position of particle0 " << particles->xprevious[indx] << " " << particles->yprevious[indx] << " " << particles->zprevious[indx] << endl;
        //  cout << "Position of particle " << particles->x[indx] << " " << particles->y[indx] << " " << particles->z[indx] << endl;
        //  }
        particles->vx[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[0] * vSampled[0];
        particles->vy[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[1] * vSampled[1];
        particles->vz[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[2] * vSampled[2];
        //        //if(particles->test[indx] == 0.0)
        //        //{
        //        //    particles->test[indx] = 1.0;
        //        //    particles->test0[indx] = aInterpVal;
        //        //    particles->test1[indx] = eInterpVal;
        //        //    particles->test2[indx] = V0;
        //        //    particles->test3[indx] = vSampled[2];
        //        //}

        particles->xprevious[indx] = particles->x[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[0] * 1e-4;
        particles->yprevious[indx] = particles->y[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[1] * 1e-4;
      particles->zprevious[indx] = particles->z[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[2] * 1e-4;
        //cout << "New vel " << particles->vx[indx] << " " << particles->vy[indx] << " " << particles->vz[indx] << endl;
        //cout << "New pos " << particles->xprevious[indx] << " " << particles->yprevious[indx] << " " << particles->zprevious[indx] << endl;
        //if(particles->test[indx] == 0.0)
        //{
        //    particles->test[indx] = 1.0;
        //    particles->test0[indx] = particles->x[indx];
        //    particles->test1[indx] = particles->y[indx];
        //    particles->test2[indx] = particles->z[indx];
        //    particles->test3[indx] = signPartDotNormal;
        //}
        //else
        //{
        //    particles->test[indx] = particles->test[indx] + 1.0;
        //}
      } else {
        particles->hitWall[indx] = 2.0;
      }
    }
  }
};
#endif

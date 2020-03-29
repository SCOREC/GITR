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
using namespace std;
#endif

#ifndef SURF_DEBUG_PRINT
#define SURF_DEBUG_PRINT 0
#endif
const int debug = 0;

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
   
particles->tt[indx] = particles->tt[indx]+1; 
   
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

int tstep = particles->tt[indx]-1;
int ptcl = particles->index[indx];
      if (wallHit < 0)
        wallHit = 0;
      if (surfaceHit < 0)
        surfaceHit = 0;
      double eInterpVal = 0.0;
      double aInterpVal = 0.0;
      double weight = particles->weight[indx];
      double vx = particles->vx[indx];
      double vy = particles->vy[indx];
      double vz = particles->vz[indx];
#if FLUX_EA > 0
      double dEdist = (Edist - E0dist) / nEdist; //cast ?
      double dAdist = (Adist - A0dist) / nAdist;
      int AdistInd = 0;
      int EdistInd = 0;
#endif
      particles->firstCollision[indx] = 1;
      particleTrackVector[0] = vx;
      particleTrackVector[1] = vy;
      particleTrackVector[2] = vz;
      norm_part = sqrt(particleTrackVector[0] * particleTrackVector[0] + particleTrackVector[1] * particleTrackVector[1] + particleTrackVector[2] * particleTrackVector[2]);
      E0 = 0.5 * particles->amu[indx] * 1.6737236e-27 * (norm_part * norm_part) / 1.60217662e-19;

      if(SURF_DEBUG_PRINT==1) {
        auto xx = particles->x[indx]; 
        auto yy = particles->y[indx];
        auto zz = particles->z[indx];
        auto amu = particles->amu[indx];
        printf("SURF1 ptcl %d timestep %d pos %.15e %.15e %.15e vel %g %g %g : wallHit %d " 
            "surfaceHit %d surface %d wallIndex %d    norm %.15e weight %.15e amu %g  E0 %.15e \n", 
            ptcl, tstep,  xx,yy,zz,vx, vy,vz, wallHit, surfaceHit, surface, 
            particles->wallIndex[indx], norm_part, weight, amu, E0);
      }

      if (E0 > 1000.0)
        E0 = 990.0;
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

      if(SURF_DEBUG_PRINT==1)
        printf("SURF4 ptcl %d timestep %d surfaceNormal %.15e %.15e %.15e ptclTrackV %.15e %.15e %.15e partDotNormal %.15e "
          " materialZ %g thetaImpact %.15e totalYR %.15e sputtProb %.15e Y0 %.15e R0 %.15e totalYR %.15e "
          " bdrxyz %g %g %g : %g %g %g : %g %g %g  bdry:abcd %g %g %g %g plane_norm %g \n",
          ptcl, tstep, surfaceNormalVector[0], surfaceNormalVector[1],surfaceNormalVector[2],
          particleTrackVector[0], particleTrackVector[1], particleTrackVector[2], partDotNormal,
          boundaryVector[wallHit].Z , thetaImpact,  totalYR, Y0/totalYR, Y0, R0, totalYR,
          boundaryVector[wallHit].x1, boundaryVector[wallHit].y1, 
          boundaryVector[wallHit].z1, boundaryVector[wallHit].x2, boundaryVector[wallHit].y2,
          boundaryVector[wallHit].z2, boundaryVector[wallHit].x3, boundaryVector[wallHit].y3,
          boundaryVector[wallHit].z3,   boundaryVector[wallHit].a,  boundaryVector[wallHit].b,
           boundaryVector[wallHit].c,  boundaryVector[wallHit].d,  boundaryVector[wallHit].plane_norm);
 
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
      int pindex = particles->index[indx];
      int beg = -1;
      if(dof_intermediate > 0 && particles->storeRnd[indx]) {
        auto pind = pindex;
        int rid = particles->storeRndSeqId[indx]; 
        pind = (rid >= 0) ? rid : pind;  
        beg = pind*nT*dof_intermediate + (nthStep-1)*dof_intermediate;
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
                  if(SURF_DEBUG_PRINT==1)
                    printf("SURF5 reflects ptcl %d  timestep %d weight %.15e newWeight %.15e rand8 %.15e "
                        " thetaImpact %.15e log10(E0) %.15e aInterpVal %.15e  eInterpVal %.15e \n",
                      ptcl, tstep, weight, newWeight, r8, thetaImpact,log10(E0), aInterpVal, eInterpVal);
    #if FLUX_EA > 0
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist))
              {
            #if USE_CUDA > 0
                  auto old = atomicAdd(&(surfaces->reflDistribution[
                        surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd]),newWeight);
                  if(debug)
                    printf("REFL  %g tot %g ind %d Ei %d Ai %d ptcl %d  t %d\n", newWeight, old+newWeight,
                      surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd, EdistInd, AdistInd, ptcl, tstep);
            #else      
                  surfaces->reflDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->reflDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  newWeight;
            #endif

                 if(SURF_DEBUG_PRINT==1)
                   printf("SURF6 reflDist ptcl %d  timestep %d reflIndx %d \n", 
                     ptcl, tstep, surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd);
              }
	       #endif
                  if(surface > 0)
                {

            #if USE_CUDA > 0
                    auto old = atomicAdd(&(surfaces->grossDeposition[surfaceHit]),weight*(1.0-R0));
            
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight*(1.0-R0);
            #endif

                 if(SURF_DEBUG_PRINT==1)
                   printf("SURF7 grossDep ptcl %d timestep %d GrossDep+ %.15e surfaceHit %d \n", 
                       ptcl, tstep, weight*(1.0-R0),  surfaceHit);
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

              if(SURF_DEBUG_PRINT==1)
                   printf("SURF8 sputtDist ptcl %d timestep %d  weight %.15e newWeight %.15e surface %d "
                    " aInterpVal %.15e eInterpVal %.15e \n", 
                       ptcl,  tstep, weight, newWeight, surface, aInterpVal, eInterpVal);
    #if FLUX_EA > 0
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist))
              {
                //cout << " particle sputters with " << EdistInd << AdistInd <<  endl;
            #if USE_CUDA > 0
                  auto old = atomicAdd(&(surfaces->sputtDistribution[
                        surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd]),newWeight);
                  if(debug)
                    printf("SPUTT  %g tot %g ind %d Ei %d Ai %d ptcl %d  t %d\n", newWeight, old+newWeight,
                      surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd, EdistInd, AdistInd, ptcl, tstep);
            #else      
                  surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  newWeight;
              #endif 
                  if(SURF_DEBUG_PRINT==1)
                    printf("SURF9 sputters FLUX_EA  ptcl %d timestep %d  sputtDist indx %d "
                      " tot sputtDist %.15e \n",ptcl, tstep, 
                      surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd, 
                      surfaces->sputtDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd]);
              }
	       #endif
                  if(sputtProb == 0.0) newWeight = 0.0;
                   //cout << " particle sputtered with newWeight " << newWeight << endl;
                  if(surface > 0)
                {

            #if USE_CUDA > 0
                    atomicAdd(&(surfaces->grossDeposition[surfaceHit]),weight*(1.0-R0));
                    atomicAdd(&(surfaces->grossErosion[surfaceHit]),newWeight);
                    atomicAdd(&(surfaces->aveSputtYld[surfaceHit]),Y0);
                    if(weight > 0.0)
                    {
                        atomicAdd(&(surfaces->sputtYldCount[surfaceHit]),1);
                    }
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight*(1.0-R0);
                    surfaces->grossErosion[surfaceHit] = surfaces->grossErosion[surfaceHit] + newWeight;
                    surfaces->aveSputtYld[surfaceHit] = surfaces->aveSputtYld[surfaceHit] + Y0;
                    surfaces->sputtYldCount[surfaceHit] = surfaces->sputtYldCount[surfaceHit] + 1;
            #endif

                 if(SURF_DEBUG_PRINT==1)
                   printf("SURF10 DepErosSput ptcl %d timestep %d surfaceHit %d newWeight  %.15e GrossDep+ %.15e "
                    " GrossEros+ %.15e AveSput+ %.15e SpYCount +1\n", 
                      ptcl, tstep, surfaceHit, newWeight, weight*(1.0-R0), newWeight, Y0);
                }
            }
            //cout << "eInterpValYR " << eInterpVal << endl; 
            }
            else // totalYR
            { 
              if(SURF_DEBUG_PRINT==1) printf("SURF10_ ptcl %d timestep %d weight %.15e  surface %d \n",
                ptcl, tstep, weight, surface);
                    newWeight = 0.0;
                    particles->hitWall[indx] = 2.0;
                  if(surface > 0)
                {
            #if USE_CUDA > 0
                    atomicAdd(&(surfaces->grossDeposition[surfaceHit]),weight);
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight;
            #endif

                  if(SURF_DEBUG_PRINT==1) printf("SURF11 ptcl %d timestep %d totalYR=0 newWeight => 0 "
                    "GrossDep+ %.15e grossDep %.15e\n", 
                    ptcl, tstep, weight, surfaces->grossDeposition[surfaceHit]);
                }
            //cout << "eInterpValYR_not " << eInterpVal << endl; 
            }
            //cout << "eInterpVal " << eInterpVal << endl; 
	    if(eInterpVal <= 0.0)
            {
                if(SURF_DEBUG_PRINT==1)
                  printf("SURF12 eInterpVal <= 0.0  newWeight = 0.0 surface %d didReflect %d\n",surface,didReflect);

                    newWeight = 0.0;
                    particles->hitWall[indx] = 2.0;

              if(surface > 0)
              {
        		    if(didReflect)
        		    {
            #if USE_CUDA > 0
                    atomicAdd(&(surfaces->grossDeposition[surfaceHit]),weight);
            #else
                    surfaces->grossDeposition[surfaceHit] = surfaces->grossDeposition[surfaceHit]+weight;
            #endif
	        
                    if(SURF_DEBUG_PRINT==1)
                      printf("SURF12b reflect ptcl %d timestep %d  surfaceHit %d GrossDep+ %.15e \n", 
                        ptcl, tstep, surfaceHit, weight);
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
                atomicAdd(&(surfaces->sumWeightStrike[surfaceHit]),weight);
                atomicAdd(&(surfaces->sumParticlesStrike[surfaceHit]),1);
            #else
                surfaces->sumWeightStrike[surfaceHit] =surfaces->sumWeightStrike[surfaceHit] +weight;
                surfaces->sumParticlesStrike[surfaceHit] = surfaces->sumParticlesStrike[surfaceHit]+1;
              //boundaryVector[wallHit].impacts = boundaryVector[wallHit].impacts +  particles->weight[indx];
            #endif
            #if FLUX_EA > 0
                EdistInd = floor((E0-E0dist)/dEdist);
                AdistInd = floor((thetaImpact-A0dist)/dAdist);

            if(SURF_DEBUG_PRINT==1)
                printf("SURF13 surface WeightStrike+ %.15e PtclStrike+1 n", weight );

	        if((EdistInd >= 0) && (EdistInd < nEdist) && 
                  (AdistInd >= 0) && (AdistInd < nAdist))
                {
#if USE_CUDA > 0
                    auto old = atomicAdd(&(surfaces->energyDistribution[surfaceHit*nEdist*nAdist + 
                                               EdistInd*nAdist + AdistInd]), weight);

                    if(debug)
                      printf("EDIST %g tot %g ind %d Ei %d Ai %d ptcl %d t %d\n", weight, old+weight, 
                          surfaceHit*nEdist*nAdist +EdistInd*nAdist + AdistInd, EdistInd, AdistInd, ptcl, tstep);
#else

                    surfaces->energyDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] = 
                    surfaces->energyDistribution[surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd] +  weight;
#endif

                    if(SURF_DEBUG_PRINT==1)
                      printf("SURF14 enDistr ptcl %d timestep %d indx %d weight %.15e surfaceHit %d nEdist %d nAdist %d \n",
                          ptcl, tstep, surfaceHit*nEdist*nAdist + EdistInd*nAdist + AdistInd, weight, 
                          surfaceHit, nEdist, nAdist);
                }
#endif
      }

      if(SURF_DEBUG_PRINT==1)
        printf("SURF16  ptcl %d timestep %d newweight %.15e bdryZ %g \n",
            ptcl, tstep, newWeight, boundaryVector[wallHit].Z);
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
        if(SURF_DEBUG_PRINT==1) {
          printf("SURF17 ptcl %d timestep %d V0 %.15e rand10 %g vsampled0 %.15e %.15e %.15e surfaceNormal %.15e %.15e %.15e \n", 
            ptcl, tstep, V0, r10, vSampled[0], vSampled[1], vSampled[2], surfaceNormalVector[0],
            surfaceNormalVector[1], surfaceNormalVector[2]);

        }
        boundaryVector[wallHit].transformToSurface(vSampled, particles->y[indx], particles->x[indx]);

        particles->vx[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[0] * vSampled[0];
        particles->vy[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[1] * vSampled[1];
        particles->vz[indx] = -static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[2] * vSampled[2];


        particles->xprevious[indx] = particles->x[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[0] * 1e-4;
        particles->yprevious[indx] = particles->y[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[1] * 1e-4;
      particles->zprevious[indx] = particles->z[indx] - static_cast<double>(boundaryVector[wallHit].inDir) * surfaceNormalVector[2] * 1e-4;

        if(SURF_DEBUG_PRINT==1) {
          auto xp =  particles->xprevious[indx];
          auto yp =  particles->yprevious[indx];
          auto zp =  particles->zprevious[indx];
          auto vx =  particles->vx[indx];
          auto vy = particles->vy[indx];
          auto vz = particles->vz[indx];
          printf("SURF18 ptcl %d timestep %d newpos %.15e %.15e %.15e newvel %.15e %.15e %.15e vsampled %.15e %.15e %.15e\n", 
              ptcl, tstep, xp, yp, zp, vx, vy, vz, vSampled[0], vSampled[1], vSampled[2]);
        }

      } else {
        particles->hitWall[indx] = 2.0;
      }
    }
  }
};
#endif

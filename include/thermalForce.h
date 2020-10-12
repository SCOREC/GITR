#ifndef _THERMAL_
#define _THERMAL_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
using namespace std;
#endif

#include "Particle.h"
#include <cmath>

struct thermalForce { 

    Particles *p;
    const double dt;
    double background_amu;
    int nR_gradT;
    int nZ_gradT;
    double* gradTGridr;
    double* gradTGridz;
    double* gradTiR;
    double* gradTiZ;
    double* gradTiT;
    double* gradTeR;
    double* gradTeZ;
    double* gradTeT;
            int nR_Bfield;
            int nZ_Bfield;
            double * BfieldGridRDevicePointer;
            double * BfieldGridZDevicePointer;
            double * BfieldRDevicePointer;
            double * BfieldZDevicePointer;
            double * BfieldTDevicePointer;
	    double dv_ITGx=0.0;
	    double dv_ITGy=0.0;
	    double dv_ITGz=0.0;
	    double dv_ETGx=0.0;
	    double dv_ETGy=0.0;
	    double dv_ETGz=0.0;
            
    thermalForce(Particles *_p,double _dt, double _background_amu,int _nR_gradT, int _nZ_gradT, double* _gradTGridr, double* _gradTGridz,
            double* _gradTiR, double* _gradTiZ, double* _gradTiT, double* _gradTeR, double* _gradTeZ,double* _gradTeT,
            int _nR_Bfield, int _nZ_Bfield,
            double * _BfieldGridRDevicePointer,
            double * _BfieldGridZDevicePointer,
            double * _BfieldRDevicePointer,
            double * _BfieldZDevicePointer,
            double * _BfieldTDevicePointer)
        
            : p(_p), dt(_dt), background_amu(_background_amu),nR_gradT(_nR_gradT),nZ_gradT(_nZ_gradT),
        gradTGridr(_gradTGridr), gradTGridz(_gradTGridz),
        gradTiR(_gradTiR), gradTiZ(_gradTiZ),gradTiT(_gradTiT), gradTeR(_gradTeR), gradTeZ(_gradTeZ),gradTeT(_gradTeT), 
             nR_Bfield(_nR_Bfield), nZ_Bfield(_nZ_Bfield), BfieldGridRDevicePointer(_BfieldGridRDevicePointer), BfieldGridZDevicePointer(_BfieldGridZDevicePointer),
    BfieldRDevicePointer(_BfieldRDevicePointer), BfieldZDevicePointer(_BfieldZDevicePointer), BfieldTDevicePointer(_BfieldTDevicePointer) {}

CUDA_CALLABLE_MEMBER    
void operator()(size_t indx)  { 
    if ((p->hitWall[indx] == 0.0) && (p->charge[indx] > 0.0)) {
      double MI = 1.6737236e-27;
      double alpha;
      double beta;
      double mu;
      double gradTe[3] = {0.0, 0.0, 0.0};
      double gradTi[3] = {0.0, 0.0, 0.0};
      double B[3] = {0.0, 0.0, 0.0};
      double B_unit[3] = {0.0, 0.0, 0.0};
      double Bmag = 0.0;
      double gradTiPar = 0.0;
      double dv_ITG[3] = {};
      double dv_ETG[3] = {};
      double vNorm = 0.0;
      double vNorm2 = 0.0;
      // std:cout << " grad Ti interp " << endl;
      interp2dVector(&gradTi[0], p->xprevious[indx], p->yprevious[indx], p->zprevious[indx], nR_gradT, nZ_gradT,
                     gradTGridr, gradTGridz, gradTiR, gradTiZ, gradTiT);
      //cout << "Position r z" << sqrt(p->xprevious*p->xprevious + p->yprevious*p->yprevious) << " " << p->zprevious << endl;
      //cout << "grad Ti " << copysign(1.0,gradTi[0])*sqrt(gradTi[0]*gradTi[0] + gradTi[1]*gradTi[1]) << " " << gradTi[2] << endl;
      interp2dVector(&gradTe[0], p->xprevious[indx], p->yprevious[indx], p->zprevious[indx], nR_gradT, nZ_gradT,
                     gradTGridr, gradTGridz, gradTeR, gradTeZ, gradTeT);
      mu = p->amu[indx] / (background_amu + p->amu[indx]);
      alpha = p->charge[indx] * p->charge[indx] * 0.71;
      beta = 3 * (mu + 5 * sqrt(2.0) * p->charge[indx] * p->charge[indx] * (1.1 * pow(mu, (5 / 2)) - 0.35 * pow(mu, (3 / 2))) - 1) / (2.6 - 2 * mu + 5.4 * pow(mu, 2));
       
       interp2dVector(&B[0],p->xprevious[indx],p->yprevious[indx],p->zprevious[indx],nR_Bfield,nZ_Bfield,
             BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,BfieldZDevicePointer,BfieldTDevicePointer);    
   #if DEBUG_PRINT > 0
       printf("Thermalforce: B %g %g %g GradTi %g %g %g GradTe %g %g %g\n",
           B[0],B[1],B[2], gradTi[0], gradTi[1], gradTi[2], gradTe[0], gradTe[1], gradTe[2]);
   #endif
        Bmag = sqrt(B[0]*B[0] + B[1]*B[1]+ B[2]*B[2]);
        B_unit[0] = B[0]/Bmag;
        B_unit[1] = B[1]/Bmag;
        B_unit[2] = B[2]/Bmag;

//	dv_ETG[0] = 1.602e-19*dt/(p->amu[indx]*MI)*(alpha*(gradTe[0]));
//	dv_ETG[1] = 1.602e-19*dt/(p->amu[indx]*MI)*(alpha*(gradTe[1]));
//	dv_ETG[2] = 1.602e-19*dt/(p->amu[indx]*MI)*(alpha*(gradTe[2]));

	dv_ITG[0] = 1.602e-19*dt/(p->amu[indx]*MI)*(beta*(gradTi[0]))*B_unit[0];
	dv_ITG[1] = 1.602e-19*dt/(p->amu[indx]*MI)*(beta*(gradTi[1]))*B_unit[1];
	dv_ITG[2] = 1.602e-19*dt/(p->amu[indx]*MI)*(beta*(gradTi[2]))*B_unit[2];
	dv_ITGx = dv_ITG[0];
	dv_ITGy = dv_ITG[1];
	dv_ITGz = dv_ITG[2];
    //cout << "mu " << mu << endl;
    //cout << "alpha beta " << alpha << " " << beta << endl;
    //cout << "ITG " << dv_ITG[0] << " " << dv_ITG[1] << " " << dv_ITG[2] << endl;
    //cout << "gradTi " << gradTi[0] << " " << gradTi[1] << " " << gradTi[2] << endl;
    //cout << "ETG " << dv_ETG[0] << " " << dv_ETG[1] << " " << dv_ETG[2] << endl;
    //cout << "v before thermal force " << p->vx[indx] << " " << p->vy[indx] << " " << p->vz[indx] << endl;
    /*
    double theta = atan2(p->yprevious,p->xprevious);
    double Ar = -1;
    double At = 0.0;
    double Az = 1;
    gradTi[0] = cos(theta)*Ar - sin(theta)*At;
    gradTi[1] = sin(theta)*Ar + cos(theta)*At;
    gradTi[2] = Az;
    */
    double vx = p->vx[indx];
    double vy = p->vy[indx];
    double vz = p->vz[indx];
 //       vNorm = sqrt(vx*vx + vy*vy + vz*vz);
    p->vD[indx] = dv_ITG[2];    
	//cout << "gradTi Parallel " << gradTiPar << endl;
        //cout << "gradTi Parallel " << gradTi[0]<<gradTi[1]<<gradTi[2] << endl;
        //p->vx[indx] = p->vx[indx] +dv_ITG[0];//alpha*(gradTe[0])   
	//p->vy[indx] = p->vy[indx] +dv_ITG[1];//alpha*(gradTe[1])
	//p->vz[indx] = p->vz[indx] +dv_ITG[2];//alpha*(gradTe[2])		
        //vNorm2 = sqrt(p->vx[indx]*p->vx[indx] + p->vy[indx]*p->vy[indx] + p->vz[indx]*p->vz[indx]);
		//SFT
    //    double k1 = dv_ITG[2] - dt*p->nu_s[indx]
    //                *(dv_ITG[2]);
        p->vx[indx] = vx + dv_ITG[0];///velocityCollisionsNorm;   	
		p->vy[indx] = vy + dv_ITG[1];///velocityCollisionsNorm;   	
		p->vz[indx] = vz + dv_ITG[2];// - dt*p->nu_s[indx]
   #if DEBUG_PRINT > 0
        printf("dv_ITG %.15f %.15f %.15f vel %.15f %.15f %.15f \n", dv_ITG[0], dv_ITG[1], dv_ITG[2], p->vx[indx],  p->vy[indx], p->vz[indx]);
   #endif
                         //*(0.5*k1);///velocityCollisionsNorm;   	
        //p.vx = p.vx + (dt/(p.amu*MI))*(  beta*(gradTi[0]));//alpha*(gradTe[0])
		//p.vy = p.vy + (dt/(p.amu*MI))*(  beta*(gradTi[1]));//alpha*(gradTe[1])
		//p.vz = p.vz + (dt/(p.amu*MI))*(  beta*(gradTi[2]));//alpha*(gradTe[2])		
     //   cout << "dv ion thermal x" << dt/(p.amu*MI)*(  beta*(gradTi[0])) << endl;	
     //  cout << "dv ion thermal y" << dt/(p.amu*MI)*(  beta*(gradTi[1])) << endl;	
     //  cout << "dv ion thermal z" << dt/(p.amu*MI)*(  beta*(gradTi[2])) << endl;	
        //cout << "theta " << theta << endl;
       //cout << "v after thermal force " << p.vx << " " << p.vy << " " << p.vz << endl;
        }
    	}
     
};

#endif

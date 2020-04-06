#ifndef _COULOMB_
#define _COULOMB_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include "Particles.h"
#include <cmath>
#ifdef __CUDACC__
#include <thrust/random.h>
#else
#include <random>
using namespace std;
#endif

#ifndef COULOMB_DEBUG_PRINT
#define COULOMB_DEBUG_PRINT 0
#endif

CUDA_CALLABLE_MEMBER
void getSlowDownFrequencies ( double& nu_friction, double& nu_deflection, double& nu_parallel,
			 	double& nu_energy, double x, double y,double z, double vx, double vy, double vz,double charge, double amu, 
                
    int nR_flowV,
    int nZ_flowV,
    double* flowVGridr,
    double* flowVGridz,
    double* flowVr,
    double* flowVz,
    double* flowVt,
    int nR_Dens,
    int nZ_Dens,
    double* DensGridr,
    double* DensGridz,
    double* ni,
    int nR_Temp,
    int nZ_Temp,
    double* TempGridr,
    double* TempGridz,
    double* ti, double* te,double background_Z, double background_amu,
                        int nR_Bfield, int nZ_Bfield,
                        double* BfieldGridR ,double* BfieldGridZ ,
                        double* BfieldR ,double* BfieldZ ,
                 double* BfieldT,double &T_background, int ptcl=0, int timestep=0 
                ) {
        double Q = 1.60217662e-19;
        double EPS0 = 8.854187e-12;
	double pi = 3.14159265;
        double MI = 1.6737236e-27;	
        double ME = 9.10938356e-31;
        double te_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,te);
        double ti_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,ti);
	T_background = ti_eV;
            double density = interp2dCombined(x,y,z,nR_Dens,nZ_Dens,DensGridr,DensGridz,ni);
            //cout << "ion t and n " << te_eV << "  " << density << endl;
    double flowVelocity[3]= {0.0};
	double relativeVelocity[3] = {0.0, 0.0, 0.0};
	double velocityNorm = 0.0;
	double lam_d;
	double lam;
	double gam_electron_background;
	double gam_ion_background;
	double a_electron = 0.0;
	double a_ion = 0.0;
	double xx;
	double psi_prime;
	double psi_psiprime;
	double psi;
	double xx_e;
	double psi_prime_e;
	double psi_psiprime_e;
	double psi_psiprime_psi2x = 0.0;
	double psi_psiprime_psi2x_e = 0.0;
	double psi_e;
	double nu_0_i;
	double nu_0_e;
    double nu_friction_i; 
    double nu_deflection_i;
    double nu_parallel_i; 
    double nu_energy_i;
    double nu_friction_e; 
    double nu_deflection_e;
    double nu_parallel_e; 
    double nu_energy_e;
                
#if FLOWV_INTERP == 3
        interp3dVector (&flowVelocity[0], x,y,z,nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(&flowVelocity[0],x,y,z,
                                 nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(&flowVelocity[0],x,y,z,
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
	relativeVelocity[0] = vx - flowVelocity[0];
	relativeVelocity[1] = vy - flowVelocity[1];
	relativeVelocity[2] = vz - flowVelocity[2];
	velocityNorm = sqrt( relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2]);                
	    //cout << "velocity norm " << velocityNorm << endl;	
    //for(int i=1; i < nSpecies; i++)
		//{
			lam_d = sqrt(EPS0*te_eV/(density*pow(background_Z,2)*Q));//only one q in order to convert to J
                	lam = 4.0*pi*density*pow(lam_d,3);
                	gam_electron_background = 0.238762895*pow(charge,2)*log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
                	gam_ion_background = 0.238762895*pow(charge,2)*pow(background_Z,2)*log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
                    //cout << "gam components " <<gam_electron_background << " " << pow(Q,4) << " " << " " << pow(background_Z,2) << " " << log(lam)<< endl; 
                if(gam_electron_background < 0.0) gam_electron_background=0.0;
                if(gam_ion_background < 0.0) gam_ion_background=0.0;
		       a_ion = background_amu*MI/(2*ti_eV*Q);// %q is just to convert units - no z needed
                	a_electron = ME/(2*te_eV*Q);// %q is just to convert units - no z needed
                
                	xx = pow(velocityNorm,2)*a_ion;
                	//psi_prime = 2*sqrtf(xx/pi)*exp(-xx);
                	//psi_psiprime = erf(1.0*sqrtf(xx));
                	//psi = psi_psiprime - psi_prime;
                        //if(psi < 0.0) psi = 0.0;
		        //psi_psiprime_psi2x = psi+psi_prime - psi/2.0/x;
		        //if(xx<1.0e-3)
		        //{
                            psi_prime = 1.128379*sqrt(xx);
                            psi = 0.75225278*pow(xx,1.5);
                            psi_psiprime = psi+psi_prime;
                            psi_psiprime_psi2x = 1.128379*sqrt(xx)*exp(-xx);
		        //}
                    //if(psi_prime/psi > 1.0e7) psi = psi_psiprime/1.0e7;
                    //if(psi_prime < 0.0) psi_prime = 0.0;
                    //if(psi_psiprime < 0.0) psi_psiprime = 0.0;
                	xx_e = pow(velocityNorm,2)*a_electron;
                	//psi_prime_e = 2*sqrtf(xx_e/pi)*exp(-xx_e);
                	//psi_psiprime_e = erf(1.0*sqrtf(xx_e));
                	//psi_e = psi_psiprime_e - psi_prime_e;
                        //if(psi_e < 0.0) psi_e = 0.0;
		        //psi_psiprime_psi2x_e = psi_e+psi_prime_e - psi_e/2.0/xx_e;
		        //if(xx_e<1.0e-3)
		        //{
                            psi_prime_e = 1.128379*sqrt(xx_e);
                            psi_e = 0.75225278*pow(xx_e,1.5);
                            psi_psiprime_e = psi_e+psi_prime_e;
                            psi_psiprime_psi2x_e = 1.128379*sqrt(xx_e)*exp(-xx_e);
		        //}
                    //if(psi_prime_e/psi_e > 1.0e7) psi_e = psi_psiprime_e/1.0e7;
                    //if(psi_prime_e < 0.0) psi_prime_e = 0.0;
                    //if(psi_psiprime_e < 0.0) psi_psiprime_e = 0.0;
                	nu_0_i = gam_electron_background*density/pow(velocityNorm,3);
                	nu_0_e = gam_ion_background*density/pow(velocityNorm,3);
                	nu_friction_i = (1+amu/background_amu)*psi*nu_0_i;
                	//nu_deflection_i = 2*(psi_psiprime - psi/(2*xx))*nu_0_i;
                	nu_deflection_i = 2*(psi_psiprime_psi2x)*nu_0_i;
                	nu_parallel_i = psi/xx*nu_0_i;
                	nu_energy_i = 2*(amu/background_amu*psi - psi_prime)*nu_0_i;
                	nu_friction_e = (1+amu/(ME/MI))*psi_e*nu_0_e;
                	//nu_deflection_e = 2*(psi_psiprime_e - psi_e/(2*xx_e))*nu_0_e;
                	nu_deflection_e = 2*(psi_psiprime_psi2x_e)*nu_0_e;
                	nu_parallel_e = psi_e/xx_e*nu_0_e;
                	nu_energy_e = 2*(amu/(ME/MI)*psi_e - psi_prime_e)*nu_0_e;
    nu_friction = nu_friction_i + nu_friction_e;
   //#if COULOMB_DEBUG_PRINT > 0
   //  printf("timestep %d ptcl %d Nufriction  %.15e  \n", timestep, ptcl, nu_friction);    
     nu_deflection = nu_deflection_i + nu_deflection_e;
     nu_parallel = nu_parallel_i + nu_parallel_e;
     nu_energy = nu_energy_i + nu_energy_e;
    if(te_eV <= 0.0 || ti_eV <= 0.0)
    {
        nu_friction = 0.0;
        nu_deflection = 0.0;
        nu_parallel = 0.0;
        nu_energy = 0.0;
       //cout << " ti_eV and nu_friction " << ti_eV<< " " << nu_friction << endl;
    }
    if(density <= 0.0)
    {
        nu_friction = 0.0;
        nu_deflection = 0.0;
        nu_parallel = 0.0;
        nu_energy = 0.0;
    }

    //if(abs(nu_energy) > 1.0e7) 
    //{
    //        cout << "velocity norm " << velocityNorm << endl;	
    //cout << "gams " << gam_electron_background << " " << gam_ion_background << endl;
    //cout << " a and xx " << a_electron <<" " <<  a_ion<< " " << xx << " " << xx_e << endl;
    //       cout << "psi_prime psi_psiprime psi" << psi_prime << " "<< psi_prime_e << " " << psi_psiprime<< " " << psi_psiprime_e << " " << psi<< " " << psi_e << endl;
    //cout << "nu_Ei and nuEe " << nu_energy_i << " " << nu_energy_e << endl;
    // cout << "nu0 "  << " " <<nu_0_i << " " << nu_0_e << " " << psi_psiprime_psi2x << " " << psi_psiprime_psi2x_e << endl;
    ////nu_energy = vx;
    ////nu_friction = vy;
    ////nu_deflection = vz;
    ////cout << "nu friction i e total " << nu_deflection_i << " " << nu_deflection_e << " " <<nu_deflection  << endl;
    //}
}

CUDA_CALLABLE_MEMBER
void getSlowDownDirections (double parallel_direction[], double perp_direction1[], double perp_direction2[],
        double xprevious, double yprevious, double zprevious,double vx, double vy, double vz,
    int nR_flowV,
    int nY_flowV,
    int nZ_flowV,
    double* flowVGridr,
    double* flowVGridy,
    double* flowVGridz,
    double* flowVr,
    double* flowVz,
    double* flowVt,
    
                        int nR_Bfield, int nZ_Bfield,
                        double* BfieldGridR ,double* BfieldGridZ ,
                        double* BfieldR ,double* BfieldZ ,
                 double* BfieldT 
    ) {
	        double flowVelocity[3]= {0.0};
                double relativeVelocity[3] = {0.0, 0.0, 0.0};
                double B[3] = {0.0};
                double Bmag = 0.0;
		double B_unit[3] = {0.0};
		double velocityRelativeNorm;
		double s1;
		double s2;
        interp2dVector(&B[0],xprevious,yprevious,zprevious,nR_Bfield,nZ_Bfield,
                                       BfieldGridR,BfieldGridZ,BfieldR,BfieldZ,BfieldT);
        Bmag = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
        B_unit[0] = B[0]/Bmag;
        B_unit[1] = B[1]/Bmag;
        B_unit[2] = B[2]/Bmag;
        if(Bmag ==0.0)
        {Bmag = 1.0;
            B_unit[0] = 1.0;
            B_unit[1] = 0.0;
            B_unit[2] = 0.0;
        }

#if FLOWV_INTERP == 3
        interp3dVector (&flowVelocity[0], xprevious,yprevious,zprevious,nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(flowVelocity,xprevious,yprevious,
                                 zprevious,nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(&flowVelocity[0],xprevious,yprevious,zprevious,
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
                relativeVelocity[0] = vx;// - flowVelocity[0];
                relativeVelocity[1] = vy;// - flowVelocity[1];
                relativeVelocity[2] = vz;// - flowVelocity[2];
                velocityRelativeNorm = sqrt( relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2]);

		parallel_direction[0] = relativeVelocity[0]/velocityRelativeNorm;
		parallel_direction[1] = relativeVelocity[1]/velocityRelativeNorm;
		parallel_direction[2] = relativeVelocity[2]/velocityRelativeNorm;

		s1 = parallel_direction[0]*B_unit[0]+parallel_direction[1]*B_unit[1]+parallel_direction[2]*B_unit[2];
            	s2 = sqrt(1.0-s1*s1);
		//cout << "s1 and s2 " << s1 << " " << s2 << endl;
           if(abs(s1) >=1.0) s2=0; 
            	perp_direction1[0] = 1.0/s2*(s1*parallel_direction[0] - B_unit[0]);
		perp_direction1[1] = 1.0/s2*(s1*parallel_direction[1] - B_unit[1]);
		perp_direction1[2] = 1.0/s2*(s1*parallel_direction[2] - B_unit[2]);
           
                perp_direction2[0] = 1.0/s2*(parallel_direction[1]*B_unit[2] - parallel_direction[2]*B_unit[1]);
                perp_direction2[1] = 1.0/s2*(parallel_direction[2]*B_unit[0] - parallel_direction[0]*B_unit[2]);
                perp_direction2[2] = 1.0/s2*(parallel_direction[0]*B_unit[1] - parallel_direction[1]*B_unit[0]);
        //cout << "SlowdonwDir par" << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << " " << endl;
        //cout << "SlowdonwDir perp" << perp_direction1[0] << " " <<perp_direction1[1] << " " << perp_direction1[2] << " " << endl;
        //cout << "SlowdonwDir perp" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << " " << endl;
            //perp_direction1[0] =  s1;
            //perp_direction1[1] =  s2;
            if (s2 == 0.0)
            {
                perp_direction1[0] =  s1;
                perp_direction1[1] =  s2;

                perp_direction2[0] = parallel_direction[2];
		perp_direction2[1] = parallel_direction[0];
		perp_direction2[2] = parallel_direction[1];

                s1 = parallel_direction[0]*perp_direction2[0]+parallel_direction[1]*perp_direction2[1]+parallel_direction[2]*perp_direction2[2];
                s2 = sqrt(1.0-s1*s1);
		perp_direction1[0] = -1.0/s2*(parallel_direction[1]*perp_direction2[2] - parallel_direction[2]*perp_direction2[1]);
                perp_direction1[1] = -1.0/s2*(parallel_direction[2]*perp_direction2[0] - parallel_direction[0]*perp_direction2[2]);
                perp_direction1[2] = -1.0/s2*(parallel_direction[0]*perp_direction2[1] - parallel_direction[1]*perp_direction2[0]);
            }
}

struct coulombCollisions { 
    Particles *particlesPointer;
    const double dt;
    int nR_flowV;
    int nY_flowV;
    int nZ_flowV;
    double* flowVGridr;
    double* flowVGridy;
    double* flowVGridz;
    double* flowVr;
    double* flowVz;
    double* flowVt;
    int nR_Dens;
    int nZ_Dens;
    double* DensGridr;
    double* DensGridz;
    double* ni;
    int nR_Temp;
    int nZ_Temp;
    double* TempGridr;
    double* TempGridz;
    double* ti;
    double* te;
    double background_Z;
    double background_amu;
    int nR_Bfield;
    int nZ_Bfield;
    double * BfieldGridR;
    double * BfieldGridZ;
    double * BfieldR;
    double * BfieldZ;
    double * BfieldT;
    double dv[3];

    int dof_intermediate = 0;
    int idof = -1;
    int nT = -1;
    double* intermediate;
    int select = 0;

#if __CUDACC__
            curandState *state;
#else
            mt19937 *state;
#endif

    coulombCollisions(Particles *_particlesPointer,double _dt, 
#if __CUDACC__
                            curandState *_state,
#else
                            mt19937 *_state,
#endif
            int _nR_flowV,int _nY_flowV, int _nZ_flowV,    double* _flowVGridr,double* _flowVGridy,
                double* _flowVGridz,double* _flowVr,
                        double* _flowVz,double* _flowVt,
                        int _nR_Dens,int _nZ_Dens,double* _DensGridr,
                            double* _DensGridz,double* _ni,int _nR_Temp, int _nZ_Temp,
                        double* _TempGridr, double* _TempGridz,double* _ti,double* _te,
                        double _background_Z, double _background_amu,
                        int _nR_Bfield, int _nZ_Bfield,
                        double * _BfieldGridR ,double * _BfieldGridZ ,
                        double * _BfieldR ,double * _BfieldZ ,
                 double * _BfieldT, double* intermediate=nullptr, int nT=0, int idof=0, 
                 int dof_intermediate=0, int select=0 )
      : particlesPointer(_particlesPointer),
        dt(_dt),
        nR_flowV(_nR_flowV),
        nY_flowV(_nY_flowV),
        nZ_flowV(_nZ_flowV),
        flowVGridr(_flowVGridr),
        flowVGridy(_flowVGridy),
        flowVGridz(_flowVGridz),
        flowVr(_flowVr),
        flowVz(_flowVz),
        flowVt(_flowVt),
        nR_Dens(_nR_Dens),
        nZ_Dens(_nZ_Dens),
        DensGridr(_DensGridr),
        DensGridz(_DensGridz),
        ni(_ni),
        nR_Temp(_nR_Temp),
        nZ_Temp(_nZ_Temp),
        TempGridr(_TempGridr),
        TempGridz(_TempGridz),
        ti(_ti),
        te(_te),
        background_Z(_background_Z),
        background_amu(_background_amu),
        nR_Bfield(_nR_Bfield),
        nZ_Bfield(_nZ_Bfield),
        BfieldGridR(_BfieldGridR),
        BfieldGridZ(_BfieldGridZ),
        BfieldR(_BfieldR),
        BfieldZ(_BfieldZ),
        BfieldT(_BfieldT),
        dv{0.0, 0.0, 0.0},
        state(_state),intermediate(intermediate),nT(nT),
        idof(idof), dof_intermediate(dof_intermediate), select(select){
  }
CUDA_CALLABLE_MEMBER_DEVICE    
void operator()(size_t indx)  { 
	    if(particlesPointer->hitWall[indx] == 0.0 && particlesPointer->charge[indx] != 0.0)
        {

         double Q = 1.60217662e-19;
         double EPS0 = 8.854187e-12;
         double pi = 3.14159265;
         double MI = 1.6737236e-27;
         double ME = 9.10938356e-31;
         double boltz = 1.380649e-23;
         double amu = 1.66053906e-27;
	double k_boltz = boltz*11604/amu;
	double T_background = 0.0;
		double nu_friction = 0.0;
		double nu_deflection = 0.0;
		double nu_parallel = 0.0;
		double nu_energy = 0.0;
		double flowVelocity[3]= {0.0};
		double vUpdate[3]= {0.0};
		double relativeVelocity[3] = {0.0};
		double velocityCollisions[3]= {0.0};	
		double velocityRelativeNorm;	
		double parallel_direction[3] = {0.0};
		double parallel_direction_lab[3] = {0.0};
		double perp_direction1[3] = {0.0};
		double perp_direction2[3] = {0.0};
		double parallel_contribution;
		double dv_perp1[3] = {0.0};
		double dv_perp2[3] = {0.0};
        double x = particlesPointer->xprevious[indx];
        double y = particlesPointer->yprevious[indx];
        double z = particlesPointer->zprevious[indx];
        double vx = particlesPointer->vx[indx];
        double vy = particlesPointer->vy[indx];
        double vz = particlesPointer->vz[indx];
        double vPartNorm = 0.0;
        double velx1 = vx;
        double vely1 = vy;
        double velz1 = vz;
        
        double xn = particlesPointer->x[indx];
        double yn = particlesPointer->y[indx];
        double zn = particlesPointer->z[indx];
      
        int selectThis = 1;
        if(select > 0)
          selectThis = particlesPointer->storeRnd[indx];      
       
      #if COULOMB_DEBUG_PRINT > 0
        if(selectThis > 0)
          printf("GITRCollision-In: ptcl %d timestep %d charge %.15e VelIn %.15e %.15e %.15e "
            " => Vel %.15e %.15e %.15e pos %.15e %.15e %.15e next_pos  %.15e %.15e %.15e \n", 
          particlesPointer->index[indx],  particlesPointer->tt[indx]-1,
          particlesPointer->charge[indx], velx1, vely1, velz1, vx, vy, vz, x,y, z, xn, yn, zn);
       #endif
#if FLOWV_INTERP == 3 
        interp3dVector (&flowVelocity[0], particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],nR_flowV,nY_flowV,nZ_flowV,
                flowVGridr,flowVGridy,flowVGridz,flowVr,flowVz,flowVt);
#elif FLOWV_INTERP < 3    
#if USEFIELDALIGNEDVALUES > 0
        interpFieldAlignedVector(&flowVelocity[0],
                                 particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],
                                 nR_flowV,nZ_flowV,
                                 flowVGridr,flowVGridz,flowVr,
                                 flowVz,flowVt,nR_Bfield,nZ_Bfield,
                                 BfieldGridR,BfieldGridZ,BfieldR,
                                 BfieldZ,BfieldT);
#else
               interp2dVector(flowVelocity,particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],
                        nR_flowV,nZ_flowV,
                        flowVGridr,flowVGridz,flowVr,flowVz,flowVt);
#endif
#endif
            vPartNorm = sqrt(vx*vx + vy*vy + vz*vz);

           //SFT 
	        relativeVelocity[0] = vx - flowVelocity[0];
        	relativeVelocity[1] = vy - flowVelocity[1];
        	relativeVelocity[2] = vz - flowVelocity[2];
		double vRel2 = relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2];
        	velocityRelativeNorm = vectorNorm(relativeVelocity);
#if PARTICLESEEDS > 0
#ifdef __CUDACC__
            double n1 = curand_normal(&state[indx]);
            double n2 = curand_normal(&state[indx]);
            double r1 = curand_uniform(&state[indx]);
            double r2 = curand_uniform(&state[indx]);
            double r3 = curand_uniform(&state[indx]);
            double xsi = curand_uniform(&state[indx]);
            //particlesPointer->test[indx] = xsi;
#else
            normal_distribution<double> distribution(0.0,1.0);
            uniform_real_distribution<double> dist(0.0, 1.0);
            double n1 = distribution(state[indx]);
            double n2 = distribution(state[indx]);
            double r1 = dist(state[indx]);
            double r2 = dist(state[indx]);
            double r3 = dist(state[indx]);
            double xsi = dist(state[indx]);
#endif
#else
#if __CUDACC__
            double n2 = curand_normal(&state[indx]);
            double xsi = curand_uniform(&state[indx]);
#else
            normal_distribution<double> distribution(0.0,1.0);
            uniform_real_distribution<double> dist(0.0, 1.0);
            double n2 = distribution(state[indx]);
            double xsi = dist(state[indx]);
#endif
#endif
      int nthStep = particlesPointer->tt[indx];
      auto pindex = particlesPointer->index[indx];
      int beg = -1;
      if(dof_intermediate > 0 && selectThis > 0) {
        auto pind = pindex;
        int rid = particlesPointer->storeRndSeqId[indx]; 
        pind = (rid >= 0) ? rid : pind;
        beg = pind*nT*dof_intermediate + (nthStep-1)*dof_intermediate;
        intermediate[beg+idof] = n1;
        intermediate[beg+idof+1] = n2;
        intermediate[beg+idof+2] = xsi;
       #if COULOMB_DEBUG_PRINT > 0 
         printf("Collision: beg %d @ %d n1 %.15e n2 %.15e xsi %.15e \n", beg, beg+idof, n1, n2, xsi );
       #endif
      }
      
      int ptcl =  particlesPointer->index[indx];
      getSlowDownFrequencies(nu_friction, nu_deflection, nu_parallel, nu_energy,
                             x, y, z,
                             vx, vy, vz,
                             particlesPointer->charge[indx], particlesPointer->amu[indx],
                             nR_flowV, nZ_flowV, flowVGridr,
                             flowVGridz, flowVr,
                             flowVz, flowVt,
                             nR_Dens, nZ_Dens, DensGridr,
                             DensGridz, ni, nR_Temp, nZ_Temp,
                             TempGridr, TempGridz, ti, te, background_Z, background_amu,
                             nR_Bfield,
                             nZ_Bfield,
                             BfieldGridR,
                             BfieldGridZ,
                             BfieldR,
                             BfieldZ,
                             BfieldT, T_background,ptcl, nthStep-1);

      getSlowDownDirections(parallel_direction, perp_direction1, perp_direction2,
                            x, y, z,
                            vx, vy, vz,
                            nR_flowV, nY_flowV, nZ_flowV, flowVGridr, flowVGridy,
                            flowVGridz, flowVr,
                            flowVz, flowVt,

                            nR_Bfield,
                            nZ_Bfield,
                            BfieldGridR,
                            BfieldGridZ,
                            BfieldR,
                            BfieldZ,
                            BfieldT);
      double ti_eV = interp2dCombined(x, y, z, nR_Temp, nZ_Temp, TempGridr, TempGridz, ti);
      double density = interp2dCombined(x, y, z, nR_Dens, nZ_Dens, DensGridr, DensGridz, ni);
      double tau_s = particlesPointer->amu[indx] * ti_eV * sqrt(ti_eV / background_amu) / (6.84e4 * (1.0 + background_amu / particlesPointer->amu[indx]) * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      double tau_par = particlesPointer->amu[indx] * ti_eV * sqrt(ti_eV / background_amu) / (6.84e4 * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      double tau_E = particlesPointer->amu[indx] * ti_eV * sqrt(ti_eV / background_amu) / (1.4e5 * density / 1.0e18 * particlesPointer->charge[indx] * particlesPointer->charge[indx] * 15);
      //cout << "tau_E " << tau_E << endl;
      //cout << "ti dens tau_s tau_par " << ti_eV << " " << density << " " << tau_s << " " << tau_par << endl;
      double vTherm = sqrt(ti_eV * Q / particlesPointer->amu[indx] /amu);
      double drag = -dt * nu_friction * velocityRelativeNorm / 1.2;
      double coeff_par = 1.4142 * n1 * sqrt(nu_parallel * dt);
      double cosXsi = cos(2.0 * pi * xsi);
      double sinXsi = sin(2.0 * pi * xsi);
      double coeff_perp1 = cosXsi * sqrt(nu_deflection * dt * 0.5);
      double coeff_perp2 = sinXsi * sqrt(nu_deflection * dt * 0.5);
//cout << "cosXsi and sinXsi " << cosXsi << " " << sinXsi << endl;
#if USEFRICTION == 0
      drag = 0.0;
#endif
#if USEANGLESCATTERING == 0
      coeff_perp1 = 0.0;
      coeff_perp2 = 0.0;
#endif
#if USEHEATING == 0
      coeff_par = 0.0;
#endif
      ////ALL COULOMB COLLISION OPERATORS///
      velocityCollisions[0] = (drag)*relativeVelocity[0] / velocityRelativeNorm;
      velocityCollisions[1] = (drag)*relativeVelocity[1] / velocityRelativeNorm;
      velocityCollisions[2] = (drag)*relativeVelocity[2] / velocityRelativeNorm;
      double velocityCollisionsNorm = vectorNorm(velocityCollisions);
      double nuEdt = nu_energy * dt;
      if (nuEdt < -1.0)
        nuEdt = -1.0;
      particlesPointer->vx[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[0] + abs(n2) * (coeff_perp1 * perp_direction1[0] + coeff_perp2 * perp_direction2[0])) + velocityCollisions[0];
      particlesPointer->vy[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[1] + abs(n2) * (coeff_perp1 * perp_direction1[1] + coeff_perp2 * perp_direction2[1])) + velocityCollisions[1];
      particlesPointer->vz[indx] = vPartNorm * (1.0 - 0.5 * nuEdt) * ((1 + coeff_par) * parallel_direction[2] + abs(n2) * (coeff_perp1 * perp_direction1[2] + coeff_perp2 * perp_direction2[2])) + velocityCollisions[2];
      vx = particlesPointer->vx[indx];
      vy = particlesPointer->vy[indx];
      vz = particlesPointer->vz[indx];
   
    #if COULOMB_DEBUG_PRINT > 0
      if(selectThis)
        printf("GITRCollision: ptcl %d timestep %d charge %.15e VelIn %.15e %.15e %.15e "
            " => Vel %.15e %.15e %.15e pos %.15e %.15e %.15e \n", 
         ptcl, nthStep-1, particlesPointer->charge[indx], velx1, vely1, velz1, vx, vy, vz, x,y, z);
     #endif
   
      this->dv[0] = velocityCollisions[0];
      this->dv[1] = velocityCollisions[1];
      this->dv[2] = velocityCollisions[2];
    }
  }
};

#endif

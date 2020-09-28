#ifndef _COULOMB_
#define _COULOMB_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Particles.h"
#include <cmath>
#include "interp2d.hpp"
#include "boris.h"
#include "array.h"

#ifdef __CUDACC__
#include <thrust/random.h>
#else
#include <random>
#endif
#include <fenv.h>

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
                 double* BfieldT,double &T_background, int tstep=-1 
                ) {
int feenableexcept(FE_INVALID | FE_OVERFLOW);			
        double Q = 1.60217662e-19;
        double EPS0 = 8.854187e-12;
	double pi = 3.14159265;
        double MI = 1.6737236e-27;	
        double ME = 9.10938356e-31;
        
	double te_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,te);
        double ti_eV = interp2dCombined(x,y,z,nR_Temp,nZ_Temp,TempGridr,TempGridz,ti);
	
	T_background = ti_eV;
        double density = interp2dCombined(x,y,z,nR_Dens,nZ_Dens,DensGridr,DensGridz,ni);
#if  DEBUG_PRINT > 0
         printf("tstep %d density %g te_eV %.15f ti_eV %.15f\n",
           tstep, density, te_eV, ti_eV);
#endif
        //cout << "ion t and n " << te_eV << "  " << density << endl;
	//printf ("te ti dens %f %f %f \n", te_eV, ti_eV, density);
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
	//printf ("flow Velocity %e %e %e \n", flowVelocity[0],flowVelocity[1],flowVelocity[2]);
	relativeVelocity[0] = vx - flowVelocity[0];
	relativeVelocity[1] = vy - flowVelocity[1];
	relativeVelocity[2] = vz - flowVelocity[2];
	velocityNorm = sqrt( relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2]);                
  #if DEBUG_PRINT > 0
     printf("relVel %g %g %g \n", relativeVelocity[0], relativeVelocity[1], relativeVelocity[2]);
  #endif
	lam_d = sqrt(EPS0*te_eV/(density*pow(background_Z,2)*Q));//only one q in order to convert to J
        lam = 12.0*pi*density*pow(lam_d,3)/charge;
        gam_electron_background = 0.238762895*pow(charge,2)*log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
        gam_ion_background = 0.238762895*pow(charge,2)*pow(background_Z,2)*log(lam)/(amu*amu);//constant = Q^4/(MI^2*4*pi*EPS0^2)
        if(gam_electron_background < 0.0) gam_electron_background=0.0;
	if(gam_ion_background < 0.0) gam_ion_background=0.0;
	a_ion = background_amu*MI/(2*ti_eV*Q);// %q is just to convert units - no z needed
	a_electron = ME/(2*te_eV*Q);// %q is just to convert units - no z needed
#if DEBUG_PRINT > 0
        printf("lam_d %g lam %g gam_el_bg %g gam_ion_bg %g \n", lam_d, lam, gam_electron_background, gam_ion_background);
#endif
    xx = pow(velocityNorm,2)*a_ion;
    psi_prime = 2.0*sqrt(xx/pi)*exp(-xx);
    psi_psiprime = erf(sqrt(xx));
    psi = psi_psiprime - psi_prime;
    xx_e = pow(velocityNorm,2)*a_electron;
#if DEBUG_PRINT > 0
    printf("xx_i %g xx_e %g psi_prime_i %g psi_psiprime_i %g \n", xx, xx_e,
       psi_prime, psi_psiprime);
#endif
    psi_prime_e = 1.128379*sqrt(xx_e);
    psi_e = 0.75225278*pow(xx_e,1.5);
    psi_psiprime_e = psi_e+psi_prime_e;
    psi_psiprime_psi2x_e = 1.128379*sqrt(xx_e)*expf(-xx_e);
    nu_0_i = gam_electron_background*density/pow(velocityNorm,3);
    nu_0_e = gam_ion_background*density/pow(velocityNorm,3);
    //printf ("nu i e %e %e \n", nu_0_i, nu_0_e);

    nu_friction_i = (1+amu/background_amu)*psi*nu_0_i;
    nu_deflection_i = 2*(psi_psiprime - psi/(2*xx))*nu_0_i;
    //nu_deflection_i = 2*(psi_psiprime_psi2x)*nu_0_i;
    nu_parallel_i = psi/xx*nu_0_i;
    nu_energy_i = 2*(amu/background_amu*psi - psi_prime)*nu_0_i;
    nu_friction_e = (1+amu/(ME/MI))*psi_e*nu_0_e;
    //nu_deflection_e = 2*(psi_psiprime_e - psi_e/(2*xx_e))*nu_0_e;
    nu_deflection_e = 2*(psi_psiprime_psi2x_e)*nu_0_e;
    nu_parallel_e = psi_e/xx_e*nu_0_e;
    nu_energy_e = 2*(amu/(ME/MI)*psi_e - psi_prime_e)*nu_0_e;
    //printf ("nu s d par e %e %e %e %e \n", nu_friction_i, nu_deflection_i, nu_parallel_i,nu_energy_i);
                    
    nu_friction = nu_friction_i ;//+ nu_friction_e;
    nu_deflection = nu_deflection_i ;//+ nu_deflection_e;
    nu_parallel = nu_parallel_i;// + nu_parallel_e;
    nu_energy = nu_energy_i;// + nu_energy_e;
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
#if  DEBUG_PRINT > 0
      printf("tstep %d NU_friction %.15f NU_deflection %.15f\n",
       tstep, nu_friction, nu_deflection);
      printf("tstep %d NU_parallel %.15f NU_energy %.15f\n",
       tstep, nu_parallel, nu_energy);
      printf("tstep %d Ion-temp %.15f el-temp %.15f ion-density %g \n",
        tstep, ti_eV, te_eV, density);
#endif
}
CUDA_CALLABLE_MEMBER
void getSlowDownDirections2 (double parallel_direction[], double perp_direction1[], double perp_direction2[],
        double vx, double vy, double vz)
{
	double v = sqrt(vx*vx + vy*vy + vz*vz);
	if(v == 0.0)
	{
		v = 1.0;
		vz = 1.0;
		vx = 0.0;
		vy = 0.0;
	}
        double ez1 = vx/v;
        double ez2 = vy/v;
        double ez3 = vz/v;
   #if DEBUG_PRINT > 0
     printf(" relvel %g %g %g norm %g \n", vx, vy,vz, v);
  #endif 
    // Get perpendicular velocity unit vectors
    // this comes from a cross product of
    // (ez1,ez2,ez3)x(0,0,1)
    double ex1 = ez2;
    double ex2 = -ez1;
    double ex3 = 0.0;
    
    // The above cross product will be zero for particles
    // with a pure z-directed (ez3) velocity
    // here we find those particles and get the perpendicular 
    // unit vectors by taking the cross product
    // (ez1,ez2,ez3)x(0,1,0) instead
    double exnorm = sqrt(ex1*ex1 + ex2*ex2);
    if(abs(exnorm) < 1.0e-12){
    ex1 = -ez3;
    ex2 = 0.0;
    ex3 = ez1;
    }
    // Ensure all the perpendicular direction vectors
    // ex are unit
    exnorm = sqrt(ex1*ex1+ex2*ex2 + ex3*ex3);
    ex1 = ex1/exnorm;
    ex2 = ex2/exnorm;
    ex3 = ex3/exnorm;
    
    if(isnan(ex1) || isnan(ex2) || isnan(ex3)){
       printf("ex nan %f %f %f v %f", ez1, ez2, ez3,v);
    }
    // Find the second perpendicular direction 
    // by taking the cross product
    // (ez1,ez2,ez3)x(ex1,ex2,ex3)
    double ey1 = ez2*ex3 - ez3*ex2;
    double ey2 = ez3*ex1 - ez1*ex3;
    double ey3 = ez1*ex2 - ez2*ex1;
    parallel_direction[0] = ez1; 
    parallel_direction[1] = ez2;
    parallel_direction[2] = ez3;
    
    perp_direction1[0] = ex1; 
    perp_direction1[1] = ex2;
    perp_direction1[2] = ex3;
    
    perp_direction2[0] = ey1; 
    perp_direction2[1] = ey2;
    perp_direction2[2] = ey3;
#if  DEBUG_PRINT > 0
    printf("parallel_dir %.15f %.15f %.15f\n",
       parallel_direction[0], parallel_direction[1], parallel_direction[2]);
     printf("perp1_dir %.15f %.15f %.15f\n",
       perp_direction1[0], perp_direction1[1], perp_direction1[2]);
     printf("perp2_dir %.15f %.15f %.15f\n",
       perp_direction2[0], perp_direction2[1], perp_direction2[2]);
#endif
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
                relativeVelocity[0] = vx - flowVelocity[0];
                relativeVelocity[1] = vy - flowVelocity[1];
                relativeVelocity[2] = vz - flowVelocity[2];
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
	    //if(parallel_direction[0]*parallel_direction[0] + parallel_direction[1]*parallel_direction[1] + parallel_direction[2]*parallel_direction[2] - 1.0 > 1e-6) cout << " parallel direction not one " << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << endl;
	    //if(perp_direction1[0]*perp_direction1[0] + perp_direction1[1]*perp_direction1[1] + perp_direction1[2]*perp_direction1[2] - 1.0 > 1e-6) cout << " perp direction1 not one" << perp_direction1[0] << " " << perp_direction1[1] << " " << perp_direction1[2] << endl;
	    //if(perp_direction2[0]*perp_direction2[0] + perp_direction2[1]*perp_direction2[1] + perp_direction2[2]*perp_direction2[2] - 1.0 > 1e-6) cout << " perp direction2 not one" << perp_direction2[0] << " " << perp_direction2[1] << " " << perp_direction2[2] << endl;
	    //if(vectorDotProduct(parallel_direction,perp_direction1)  > 1e-6)
	    //{cout << "par dot perp1 " << parallel_direction[0] << " " << parallel_direction[1] << " " << parallel_direction[2] << endl;
	    //cout << "par dot perp1 " << perp_direction1[0] << " " << perp_direction1[1] << " " << perp_direction1[2] << endl;
	    //}
	    //if(vectorDotProduct(parallel_direction,perp_direction2)  > 2e-6)
	    //{cout << "par dot perp2 " << parallel_direction[0] << " " << parallel_direction[2] << " " << parallel_direction[2] << endl;
	    //cout << "par dot perp2 " << perp_direction2[0] << " " << perp_direction2[2] << " " << perp_direction2[2] << endl;
	    //}
	    //if(vectorDotProduct(perp_direction1,perp_direction2)  > 2e-6)
	    //{cout << "perp1 dot perp2 " << perp_direction1[0] << " " << perp_direction1[2] << " " << perp_direction1[2] << endl;
	    //cout << "par dot perp2 " << perp_direction2[0] << " " << perp_direction2[2] << " " << perp_direction2[2] << endl;
	    //}
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

    int dof_randStore = 0;
    int idof = -1;
    int nT = -1;
    double* randStore = nullptr;
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
                 double * _BfieldT, double* randStore=nullptr, int nT=0, int idof=0, int dof_randStore=0 )
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
        state(_state), randStore(randStore), nT(nT),idof(idof), dof_randStore(dof_randStore) {
  }
CUDA_CALLABLE_MEMBER_DEVICE    
void operator()(size_t indx)  { 

	    if(particlesPointer->hitWall[indx] == 0.0 && particlesPointer->charge[indx] != 0.0)
        { 
        double pi = 3.14159265;   
	double k_boltz = 1.38e-23*11604/1.66e-27;
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
        int ptcl = indx;
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

	        relativeVelocity[0] = vx - flowVelocity[0];
        	relativeVelocity[1] = vy - flowVelocity[1];
        	relativeVelocity[2] = vz - flowVelocity[2];
		double vRel2 = relativeVelocity[0]*relativeVelocity[0] + relativeVelocity[1]*relativeVelocity[1] + relativeVelocity[2]*relativeVelocity[2];
        	velocityRelativeNorm = vectorNorm(relativeVelocity);

#if PARTICLESEEDS > 0
#ifdef __CUDACC__
        //int plus_minus1 = 1;//floor(curand_uniform(&particlesPointer->streams_collision1[indx]) + 0.5)*2 -1;
		//int plus_minus2 = 1;//floor(curand_uniform(&particlesPointer->streams_collision2[indx]) + 0.5)*2 -1;
		//int plus_minus3 = 1;//floor(curand_uniform(&particlesPointer->streams_collision3[indx]) + 0.5)*2 -1;
            double n1 = curand_normal(&state[indx]);
            double n2 = curand_normal(&state[indx]);
            //double r1 = curand_uniform(&state[indx]);
            //double r2 = curand_uniform(&state[indx]);
            //double r3 = curand_uniform(&state[indx]);
            double xsi = curand_uniform(&state[indx]);
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
            //double plus_minus1 = floor(curand_uniform(&state[3]) + 0.5)*2-1;
            //double plus_minus2 = floor(curand_uniform(&state[4]) + 0.5)*2-1;
            //double plus_minus3 = floor(curand_uniform(&state[5]) + 0.5)*2-1;
            double n1 = curand_normal(&state[indx]);
            double n2 = curand_normal(&state[indx]);
            double xsi = curand_uniform(&state[indx]);
#else
            normal_distribution<double> distribution(0.0,1.0);
            uniform_real_distribution<double> dist(0.0, 1.0);
            double n1 = distribution(state[indx]);
            double n2 = distribution(state[indx]);
            double xsi = dist(state[indx]);
#endif
#endif
      int tstep = (int)(particlesPointer->tt[indx] ) -1;
      int pindex = ptcl;//particlesPointer->index[indx];
      if(dof_randStore > 0) {
        auto pind = pindex;
        int beg = pind*nT*dof_randStore + tstep*dof_randStore;
        randStore[beg+idof] = n1;
        randStore[beg+idof+1] = n2;
        randStore[beg+idof+2] = xsi;
       #if  DEBUG_PRINT > 0 
         printf("Collision: beg %d @ %d n1 %.15f n2 %.15f xsi %.15f \n", beg, beg+idof, n1, n2, xsi );
       #endif
      }
  #if DEBUG_PRINT > 0
      printf("Collision: relVel %g %g %g flowVel %g %g %g\n", relativeVelocity[0] , relativeVelocity[1] ,
          relativeVelocity[2], flowVelocity[0], flowVelocity[1], flowVelocity[2]);
  #endif
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
                             BfieldT, T_background, tstep);

      //getSlowDownDirections(parallel_direction, perp_direction1, perp_direction2,
      //                      x, y, z,
      //                      vx, vy, vz,
      //                      nR_flowV, nY_flowV, nZ_flowV, flowVGridr, flowVGridy,
      //                      flowVGridz, flowVr,
      //                      flowVz, flowVt,

      //                      nR_Bfield,
      //                      nZ_Bfield,
      //                      BfieldGridR,
      //                      BfieldGridZ,
      //                      BfieldR,
      //                      BfieldZ,
      //                      BfieldT);
      
      getSlowDownDirections2(parallel_direction, perp_direction1, perp_direction2,
                            relativeVelocity[0] , relativeVelocity[1] , relativeVelocity[2] );
      
      double ti_eV = interp2dCombined(x, y, z, nR_Temp, nZ_Temp, TempGridr, TempGridz, ti);
      double density = interp2dCombined(x, y, z, nR_Dens, nZ_Dens, DensGridr, DensGridz, ni);
      
      if(nu_parallel <=0.0) nu_parallel = 0.0;
      double coeff_par = n1 * sqrt(2.0*nu_parallel * dt);
      double cosXsi = cos(2.0 * pi * xsi) - 0.0028;
      if(cosXsi > 1.0) cosXsi = 1.0;
      double sinXsi = sin(2.0 * pi * xsi);
      if(nu_deflection <=0.0) nu_deflection = 0.0;
      double coeff_perp1 = cosXsi * sqrt(nu_deflection * dt*0.5);
      double coeff_perp2 = sinXsi * sqrt(nu_deflection * dt*0.5);
#if USEFRICTION == 0
      //drag = 0.0;
#endif
#if USEANGLESCATTERING == 0
      coeff_perp1 = 0.0;
      coeff_perp2 = 0.0;
#endif
#if USEHEATING == 0
      coeff_par = 0.0;
      nu_energy = 0.0;
#endif
      
      double nuEdt = nu_energy * dt;
      if (nuEdt < -1.0) nuEdt = -1.0;
      
      double vx_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[0] + abs(n2)*(coeff_perp1 * perp_direction1[0] + coeff_perp2 * perp_direction2[0])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[0];
      double vy_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[1] + abs(n2)*(coeff_perp1 * perp_direction1[1] + coeff_perp2 * perp_direction2[1])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[1];
      double vz_relative = velocityRelativeNorm*(1.0-0.5*nuEdt)*((1.0 + coeff_par) * parallel_direction[2] + abs(n2)*(coeff_perp1 * perp_direction1[2] + coeff_perp2 * perp_direction2[2])) - velocityRelativeNorm*dt*nu_friction*parallel_direction[2];

      particlesPointer->vx[indx] = vx_relative + flowVelocity[0]; 
      particlesPointer->vy[indx] = vy_relative + flowVelocity[1]; 
      particlesPointer->vz[indx] = vz_relative + flowVelocity[2];

      vx = particlesPointer->vx[indx];
      vy = particlesPointer->vy[indx];
      vz = particlesPointer->vz[indx];
      //printf("xsi n2 vxr vx %f %f %f %f \n",xsi,n2,vx_relative,vx);
  #if  DEBUG_PRINT > 0
         printf("Collision: ptcl %d timestep %d n1 %.15f n2 %.15f xsi %.15f\n", ptcl, tstep, n1, n2, xsi);
         printf("Collision ptcl %d timestep %d cosXsi %.15f sinXsi %.15f n2 %.15f relVNorm %.15f charge %f\n",
           ptcl, tstep, cosXsi, sinXsi, n2, velocityRelativeNorm, particlesPointer->charge[indx]);
         printf("Collision ptcl %d timestep %d nuEdt %.15f coeff_par %.15f cf_perp1,2 %.15f %.15f \n",
           ptcl, tstep, nuEdt, coeff_par, coeff_perp1, coeff_perp2);

         printf("Collision: ptcl %d timestep %d par-dir %.15f %.15f %.15f \n", ptcl, tstep,
           parallel_direction[0], parallel_direction[1], parallel_direction[2]);
         printf("Collision: ptcl %d timestep %d perpdir1 %.15f %.15f %.15f perpdir2 %.15f %.15f %.15f \n",
           ptcl, tstep, perp_direction1[0], perp_direction1[1], perp_direction1[2],
           perp_direction2[0], perp_direction2[1], perp_direction2[2]);

        printf("Collision: ptcl %d timestep %d charge %f vel %.15f %.15f %.15f => %.15f %.15f %.15f \n",
          ptcl, tstep, particlesPointer->charge[indx], velx1, vely1, velz1, vx, vy, vz);
  #endif
      this->dv[0] = velocityCollisions[0];
      this->dv[1] = velocityCollisions[1];
      this->dv[2] = velocityCollisions[2];
    }
  }
};

#endif

#ifndef _BOUNDARYINIT_
#define _BOUNDARYINIT_


#include "Particle.h"
#include "Boundary.h"
#ifdef __CUDACC__
#include <thrust/random.h>
#endif

#ifdef __GNUC__ 
#include <random>
using namespace std;
#endif
#include <cmath>

struct boundary_init {
    double background_Z;
    double background_amu;
    int nR_Temp;
    int nZ_Temp;
    double* TempGridr;
    double* TempGridz;
    double* ti;
    double* te;
    int nx;
    int nz;
    double* densityGridx;
    double* densityGridz;
    double* density;
    double* ne;
    int nxB;
    int nzB;
    double* bfieldGridr;
    double* bfieldGridz;
    double* bfieldR;
    double* bfieldZ;
    double* bfieldT;
    double potential;
    
    boundary_init(double _background_Z, double _background_amu,int _nx, int _nz,
          double* _densityGridx, double* _densityGridz,double* _density,double* _ne,int _nxB,
          int _nzB, double* _bfieldGridr, double* _bfieldGridz,double* _bfieldR,
          double* _bfieldZ,double* _bfieldT,int _nR_Temp, int _nZ_Temp,
          double* _TempGridr, double* _TempGridz, double* _ti, double* _te, double _potential)

     : background_Z(_background_Z),
        background_amu(_background_amu),
        nR_Temp(_nR_Temp),
        nZ_Temp(_nZ_Temp),
        TempGridr(_TempGridr),
        TempGridz(_TempGridz),
        ti(_ti),
        te(_te),
        nx(_nx),
        nz(_nz),
        densityGridx(_densityGridx),
        densityGridz(_densityGridz),
        density(_density),
        ne(_ne),
        nxB(_nxB),
        nzB(_nzB),
        bfieldGridr(_bfieldGridr),
        bfieldGridz(_bfieldGridz),
        bfieldR(_bfieldR),
        bfieldZ(_bfieldZ),
        bfieldT(_bfieldT),
        potential(_potential) {}

    void operator()(Boundary &b) const {
#if USE3DTETGEOM
        double midpointx = b.x1 + 0.666666667*(b.x2 + 0.5*(b.x3-b.x2)-b.x1);
        double midpointy = b.y1 + 0.666666667*(b.y2 + 0.5*(b.y3-b.y2)-b.y1);
        double midpointz = b.z1 + 0.666666667*(b.z2 + 0.5*(b.z3-b.z2)-b.z1);
        b.midx = midpointx;
        b.midy = midpointy;
        b.midz = midpointz;
#else

        double midpointx = 0.5*(b.x2 - b.x1)+ b.x1;
        double midpointy = 0.0;
        double midpointz = 0.5*(b.z2 - b.z1) + b.z1;
#endif
        b.density = interp2dCombined(midpointx,midpointy,midpointz,nx,nz,densityGridx,densityGridz,density);
        b.ne = interp2dCombined(midpointx,midpointy,midpointz,nx,nz,densityGridx,densityGridz,ne);
        b.ti = interp2dCombined(midpointx,midpointy,midpointz,nR_Temp,nZ_Temp,TempGridr,TempGridz,ti);
        b.te = interp2dCombined(midpointx,midpointy,midpointz,nR_Temp,nZ_Temp,TempGridr,TempGridz,te);
        double B[3] = {0.0,0.0,0.0};
interp2dVector(&B[0],midpointx,midpointy,midpointz,nxB,nzB,bfieldGridr,
                 bfieldGridz,bfieldR,bfieldZ,bfieldT);
        double norm_B = vectorNorm(B);
#if USE3DTETGEOM
        double surfNorm[3] = {0.0,0.0,0.0};
        b.getSurfaceNormal(surfNorm,0.0,0.0);
        double theta = acos(vectorDotProduct(B,surfNorm)/(vectorNorm(B)*vectorNorm(surfNorm)));
        if (theta > 3.14159265359*0.5)
        {
          theta = abs(theta - (3.14159265359));
        }
#else
        double br = B[0];
        double bt = B[1];
        double bz = B[2];
        double theta = acos((-br*b.slope_dzdx + bz)/(sqrt(br*br+bz*bz+bt*bt)*sqrt(b.slope_dzdx*b.slope_dzdx + 1.0)));
 
        if (theta > 3.14159265359*0.5)
        {
            theta = acos((br*b.slope_dzdx - bz)/(sqrt(br*br+bz*bz+bt*bt)*sqrt(b.slope_dzdx*b.slope_dzdx + 1.0)));
        }
#endif        
        b.angle = theta*180.0/3.14159265359;
        b.debyeLength = sqrt(8.854187e-12*b.te/(b.ne*pow(background_Z,2)*1.60217662e-19));
	if(b.ne == 0.0) b.debyeLength = 1e12;
        b.larmorRadius = 1.44e-4*sqrt(background_amu*b.ti/2)/(background_Z*norm_B);
        b.flux = 0.25*b.density*sqrt(8.0*b.ti*1.602e-19/(3.1415*background_amu));
        b.impacts = 0.0;
#if BIASED_SURFACE
        b.potential = potential;
        //double cs = sqrt(2*b.ti*1.602e-19/(1.66e-27*background_amu));
        //double jsat_ion = 1.602e-19*b.density*cs;
        //b.ChildLangmuirDist = 2.0/3.0*pow(2*1.602e-19/(background_amu*1.66e-27),0.25)
        //*pow(potential,0.75)/(2.0*sqrt(3.1415*jsat_ion))*1.055e-5;
        if(b.te > 0.0)
        {
          b.ChildLangmuirDist = b.debyeLength*pow(abs(b.potential)/b.te,0.75);
        }
        else
        { b.ChildLangmuirDist = 1e12;
        }
#else
        b.potential = 3.0*b.te;
        //cout << "Surface number " << b.surfaceNumber << " has te and potential " << b.te << " " << b.potential << endl; 
#endif        
        //if(b.Z > 0.0)
        //{
        //cout << "Boundary ti density potensial and CLdist " <<b.ti << " " << 
        //    b.density << " " << b.potential << " " << b.ChildLangmuirDist << endl;   
        //}     
    }	
};

#endif

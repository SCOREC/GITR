#ifndef _FIELDTRACE_
#define _FIELDTRACE_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
using namespace std;
#endif
#include "Particles.h"
#include "Boundary.h"
#include <cmath>

struct field_line_trace {
    double BfieldFactor; 
    Particles *particles;
    double dr;
    Boundary *boundaries;
    int nLines;
    int nR_Lc;
    int nZ_Lc;
    double* gridRLc;
    double* gridZLc;
    double* Lc;
            int nR_Bfield;
            int nZ_Bfield;
            double * BfieldGridR;
            double * BfieldGridZ;
            double * BfieldR;
            double * BfieldZ;
            double * BfieldT;
            
    field_line_trace(double _BfieldFactor,Particles* _particles,double _dr,Boundary* _boundaries,int _nLines, int _nR_Lc, int _nZ_Lc, 
            double* _gridRLc, double* _gridZLc, double* _Lc,
            int _nR_Bfield, int _nZ_Bfield,
            double * _BfieldGridR,
            double * _BfieldGridZ,
            double * _BfieldR,
            double * _BfieldZ,
            double * _BfieldT)
        
            : BfieldFactor(_BfieldFactor),particles(_particles),dr(_dr),boundaries(_boundaries),nLines(_nLines),
        nR_Lc(_nR_Lc),nZ_Lc(_nZ_Lc),
        gridRLc(_gridRLc), gridZLc(_gridZLc),Lc(_Lc),
             nR_Bfield(_nR_Bfield), nZ_Bfield(_nZ_Bfield), BfieldGridR(_BfieldGridR), BfieldGridZ(_BfieldGridZ),
    BfieldR(_BfieldR), BfieldZ(_BfieldZ), BfieldT(_BfieldT) {}

CUDA_CALLABLE_MEMBER    
void operator()(size_t indx) const { 
    double B[3] = {0.0,0.0,0.0};
    double Bnorm[3] = {0.0,0.0,0.0};
    double Bmag = 0.0;
    double particleDistance = 0.0;
    double k1[3] = {0.0,0.0,0.0};
    double k2[3] = {0.0,0.0,0.0};
    double k3[3] = {0.0,0.0,0.0};
    double k4[3] = {0.0,0.0,0.0};
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double dr_fac = BfieldFactor*dr;

if(particles->hitWall[indx] == 0.0)
{
    x0 = particles->x[indx];
    y0 = particles->y[indx];
    z0 = particles->z[indx];

    interp2dVector(&B[0],x0, y0,z0,
            nR_Bfield,nZ_Bfield,BfieldGridR,BfieldGridZ,
            BfieldR,BfieldZ,BfieldT);
    //cout << "Bfield interp " << B[0] << " " << B[1] << " " << B[2] << endl;
    vectorNormalize(B,B);
    //Bmag = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
    //Bnorm[0] = B[0]/Bmag;
    //Bnorm[1] = B[1]/Bmag;
    //Bnorm[2] = B[2]/Bmag;
    //cout << "Bfield interp " << B[0] << " " << B[1] << " " << B[2] << endl;
    vectorScalarMult(dr_fac,B,k1);

    interp2dVector(&B[0],x0+0.5*k1[0],y0+0.5*k1[1],z0+0.5*k1[2],
            nR_Bfield,nZ_Bfield,BfieldGridR,BfieldGridZ,
            BfieldR,BfieldZ,BfieldT);
    vectorNormalize(B,B);

    vectorScalarMult(dr_fac,B,k2);
    interp2dVector(&B[0],x0+0.5*k2[0],y0+0.5*k2[1],z0+0.5*k2[2],
            nR_Bfield,nZ_Bfield,BfieldGridR,BfieldGridZ,
            BfieldR,BfieldZ,BfieldT);
    vectorNormalize(B,B);

    vectorScalarMult(dr_fac,B,k3);

    interp2dVector(&B[0],x0+k3[0],y0+k3[1],z0+k3[2],
            nR_Bfield,nZ_Bfield,BfieldGridR,BfieldGridZ,
            BfieldR,BfieldZ,BfieldT);
    vectorNormalize(B,B);

    vectorScalarMult(dr_fac,B,k4);
    x = x0+k1[0]/6.0+k2[0]/3.0+k3[0]/3.0+k4[0]/6.0;
    y = y0+k1[1]/6.0+k2[1]/3.0+k3[1]/3.0+k4[1]/6.0;
    z = z0+k1[2]/6.0+k2[2]/3.0+k3[2]/3.0+k4[2]/6.0;
    particles->x[indx] = x; 
    particles->y[indx] = y;
    particles->z[indx] = z;
    //particles->x[indx] = particles->xprevious[indx] + BfieldFactor*dr*Bnorm[0];
    //particles->y[indx] = particles->yprevious[indx] + BfieldFactor*dr*Bnorm[1];
    //particles->z[indx] = particles->zprevious[indx] + BfieldFactor*dr*Bnorm[2];
    particles->distanceTraveled[indx] = particles->distanceTraveled[indx] + dr; 
}
else if(particles->hitWall[indx] == 1.0)
{
    particleDistance = sqrt((particles->x[indx] - particles->xprevious[indx])*
                             (particles->x[indx] - particles->xprevious[indx]) 
                           + (particles->y[indx] - particles->yprevious[indx])*
                             (particles->y[indx] - particles->yprevious[indx])
                           + (particles->z[indx] - particles->zprevious[indx])*
                             (particles->z[indx] - particles->zprevious[indx]));  
            
    particles->distanceTraveled[indx] = particles->distanceTraveled[indx]
                                        + particleDistance; 
    particles->hitWall[indx] = particles->hitWall[indx]+1.0;        
}
  
       }
     
};

#endif

#ifndef _INTERPRATECOEFF2D_
#define _INTERPRATECOEFF2D_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include <thrust/device_vector.h>
#include <vector>
#include <cmath>

CUDA_CALLABLE_MEMBER


float rateCoeffInterp(int charge, float te, float ne,int nT, int nD, float* rateGrid_Tempp,float* rateGrid_Densp,float* Ratesp){

/*    vector<float>& rateGrid_Temp = *rateGrid_Tempp;
    vector<float>& rateGrid_Dens = *rateGrid_Densp;
    vector<float>& Rates = *Ratesp;
  */  
    int indT = 0;
    int indN = 0;
    float logT = log10(te);
    float logn = log10(ne);
    //cout << "Rategrid_temp in rateCoeffInterp " << rateGrid_Temp[1] << endl;
    float d_T = rateGrid_Tempp[1] - rateGrid_Tempp[0];
    float d_n = rateGrid_Densp[1] - rateGrid_Densp[0];
   // if (logT >= rateGrid_Tempp[0] && logT <= rateGrid_Tempp[nT-2])
   // {
        indT = floor((logT - rateGrid_Tempp[0])/d_T );//addition of 0.5 finds nearest gridpoint
    //}
    //if (logn >= rateGrid_Densp[0] && logn <= rateGrid_Densp[nD-2])
    //{
        indN = floor((logn - rateGrid_Densp[0])/d_n );
    //}
    //cout << "Indices density, temp " << indN << " " <<indT<<endl;
    //cout << "charge " << charge << endl;
    //cout << "Lower value " << Ratesp[charge*nT*nD + indT*nD + indN] << endl;

if(indT < 0 || indT > nT-2)
{indT = 0;}
if(indN < 0 || indN > nD-2)
{indN = 0;}
if(charge > 74-1)
{charge = 0;}
        float aT = pow(10.0f,rateGrid_Tempp[indT+1]) - te;
    float bT = te - pow(10.0f,rateGrid_Tempp[indT]);
    float abT = aT+bT;

    float aN = pow(10.0f,rateGrid_Densp[indN+1]) - ne;
    float bN = ne - pow(10.0f, rateGrid_Densp[indN]);
    float abN = aN + bN;

    //float interp_value = Rates[charge*rateGrid_Temp.size()*rateGrid_Dens.size()            + indT*rateGrid_Dens.size() + indN];

    float fx_z1 = (aN*pow(10.0f,Ratesp[charge*nT*nD + indT*nD + indN]) 
            + bN*pow(10.0f,Ratesp[charge*nT*nD            + indT*nD + indN + 1]))/abN;
    
    float fx_z2 = (aN*pow(10.0f,Ratesp[charge*nT*nD            + (indT+1)*nD + indN]) 
            + bN*pow(10.0f,Ratesp[charge*nT*nD            + (indT+1)*nD + indN+1]))/abN;
    float fxz = (aT*fx_z1+bT*fx_z2)/abT;
    //cout << "fxz1 and 2 " << fx_z1 << " " << fx_z2<< " "<< fxz << endl;
if(false)
  printf("rateCoeffInterp:logT %g logn %g d_T %g d_n %g indT %d indN %d  aT %g bT %g abT %g aN %g bN %g abN %g fx_z1 %g fx_z2 %g fxz %g\n", 
    logT, logn, d_T, d_n, indT, indN, aT, bT, abT, aN, bN, abN, fx_z1, fx_z2, fxz);
    return fxz;    
}

CUDA_CALLABLE_MEMBER
float interpRateCoeff2d ( int charge, float x, float y, float z,int nx, int nz, float* tempGridxp,
       float* tempGridzp, float* Tempp,
       float* densGridxp,float* densGridzp,float* Densp,int nT_Rates, int nD_Rates,
       float* rateGrid_Temp,float* rateGrid_Dens,float* Rates ) {
//    cout << "rate test " << Tempp[0] << endl;
    /*vector<float>& Tdata = *Tempp;
    vector<float>& Tgridx = *tempGridxp;
    vector<float>& Tgridz = *tempGridzp;
    vector<float>& DensityData = *Densp;
    vector<float>& DensGridx = *densGridxp;
    vector<float>& DensGridz = *densGridzp;
*/
    //cout << "at tlocal interp routine " <<x << y << z<< " " << nx << nz<< endl;
    //cout << "Interpolating local temp at "<<x << " " << y << " " << z << endl;
    float tlocal = interp2dCombined(x,y,z,nx,nz,tempGridxp,tempGridzp,Tempp);
    //cout << "Interpolating local dens " << endl;
    float nlocal = interp2dCombined(x,y,z,nx,nz,densGridxp,densGridzp,Densp);
    //cout << "tlocal" << tlocal << endl;
    //cout << "nlocal" << nlocal << endl;
    //cout << "Interpolating RC " << endl;
    float RClocal = rateCoeffInterp(charge,tlocal,nlocal,nT_Rates,nD_Rates,rateGrid_Temp, rateGrid_Dens, Rates);
    float tion = 1/(RClocal*nlocal);
    if(tlocal == 0.0 || nlocal == 0.0) tion=1.0e12;
   if(false && COMPARE_GITR_PRINT==1)
      printf("interpRateCoeff2d:tlocal %g nlocal %g RClocal %g charge %d \n", tlocal, nlocal, RClocal, charge);
    //cout << "Returning " << endl;
    return tion;
}

#endif


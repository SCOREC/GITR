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

#ifndef RATE_DEBUG_PRINT
#define RATE_DEBUG_PRINT 0
#endif

CUDA_CALLABLE_MEMBER


double rateCoeffInterp(int charge, double te, double ne,int nT, int nD, double* rateGrid_Tempp,double* rateGrid_Densp,double* Ratesp){

/*    vector<double>& rateGrid_Temp = *rateGrid_Tempp;
    vector<double>& rateGrid_Dens = *rateGrid_Densp;
    vector<double>& Rates = *Ratesp;
  */  
    int indT = 0;
    int indN = 0;
    double logT = log10(te);
    double logn = log10(ne);
    //cout << "Rategrid_temp in rateCoeffInterp " << rateGrid_Temp[1] << endl;
    double d_T = rateGrid_Tempp[1] - rateGrid_Tempp[0];
    double d_n = rateGrid_Densp[1] - rateGrid_Densp[0];
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
        double aT = pow(10.0,rateGrid_Tempp[indT+1]) - te;
    double bT = te - pow(10.0,rateGrid_Tempp[indT]);
    double abT = aT+bT;

    double aN = pow(10.0,rateGrid_Densp[indN+1]) - ne;
    double bN = ne - pow(10.0, rateGrid_Densp[indN]);
    double abN = aN + bN;

    //double interp_value = Rates[charge*rateGrid_Temp.size()*rateGrid_Dens.size()            + indT*rateGrid_Dens.size() + indN];

    double fx_z1 = (aN*pow(10.0,Ratesp[charge*nT*nD + indT*nD + indN]) 
            + bN*pow(10.0,Ratesp[charge*nT*nD            + indT*nD + indN + 1]))/abN;
    
    double fx_z2 = (aN*pow(10.0,Ratesp[charge*nT*nD            + (indT+1)*nD + indN]) 
            + bN*pow(10.0,Ratesp[charge*nT*nD            + (indT+1)*nD + indN+1]))/abN;
    double fxz = (aT*fx_z1+bT*fx_z2)/abT;
    //cout << "fxz1 and 2 " << fx_z1 << " " << fx_z2<< " "<< fxz << endl;
if(false)
  printf("rateCoeffInterp:logT %g logn %g d_T %g d_n %g indT %d indN %d  aT %g bT %g abT %g aN %g bN %g abN %g fx_z1 %g fx_z2 %g fxz %g\n", 
    logT, logn, d_T, d_n, indT, indN, aT, bT, abT, aN, bN, abN, fx_z1, fx_z2, fxz);
    return fxz;    
}

CUDA_CALLABLE_MEMBER
double interpRateCoeff2d ( int charge, double x, double y, double z,int nx, int nz, double* tempGridxp,
       double* tempGridzp, double* Tempp,
       double* densGridxp,double* densGridzp,double* Densp,int nT_Rates, int nD_Rates,
       double* rateGrid_Temp,double* rateGrid_Dens,double* Rates, double* t_at = nullptr, double* n_at = nullptr ) {
//    cout << "rate test " << Tempp[0] << endl;
    /*vector<double>& Tdata = *Tempp;
    vector<double>& Tgridx = *tempGridxp;
    vector<double>& Tgridz = *tempGridzp;
    vector<double>& DensityData = *Densp;
    vector<double>& DensGridx = *densGridxp;
    vector<double>& DensGridz = *densGridzp;
*/
    //cout << "at tlocal interp routine " <<x << y << z<< " " << nx << nz<< endl;
    //cout << "Interpolating local temp at "<<x << " " << y << " " << z << endl;
    double tlocal = interp2dCombined(x,y,z,nx,nz,tempGridxp,tempGridzp,Tempp);
    //cout << "Interpolating local dens " << endl;
    double nlocal = interp2dCombined(x,y,z,nx,nz,densGridxp,densGridzp,Densp);
    //cout << "tlocal" << tlocal << endl;
    //cout << "nlocal" << nlocal << endl;
    //cout << "Interpolating RC " << endl;
    double RClocal = rateCoeffInterp(charge,tlocal,nlocal,nT_Rates,nD_Rates,rateGrid_Temp, rateGrid_Dens, Rates);
    double tion = 1/(RClocal*nlocal);
    if(tlocal == 0.0 || nlocal == 0.0) tion=1.0e12;
   if(RATE_DEBUG_PRINT==1) {
      printf("interpRateCoeff2d: x %.15f y %.15f z %.15f nx %d nz %d\n",x,y,z,nx,nz);
      printf("interpRateCoeff2d:tlocal %g nlocal %g RClocal %g charge %d \n", tlocal, nlocal, RClocal, charge);
   }
    //cout << "Returning " << endl;
    if(t_at !=nullptr && n_at!=nullptr) { *t_at = tlocal; *n_at=nlocal;}

    return tion;
}

#endif


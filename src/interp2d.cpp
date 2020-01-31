#include "interp2d.hpp"
#include <iostream>
#ifdef __CUDACC__
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include <cmath>
#ifndef COMPARE_GITR
#define COMPARE_GITR 0
#endif

CUDA_CALLABLE_MEMBER

double interp2d ( double x, double z,int nx, int nz,
    double* gridx,double* gridz,double* data ) {
    
    double fxz = 0.0;
    double fx_z1 = 0.0;
    double fx_z2 = 0.0; 
    if(nx*nz == 1)
    {
        fxz = data[0];
    }
    else{
    double dim1 = x;
    double d_dim1 = gridx[1] - gridx[0];
    double dz = gridz[1] - gridz[0];
    int i = std::floor((dim1 - gridx[0])/d_dim1);//addition of 0.5 finds nearest gridpoint
    int j = std::floor((z - gridz[0])/dz);
    
    //double interp_value = data[i + j*nx];
    if (i < 0) i =0;
    if (j< 0 ) j=0;
    if (i >=nx-1 && j>=nz-1)
    {
        fxz = data[nx-1+(nz-1)*nx];
    }
    else if (i >=nx-1)
    {
        fx_z1 = data[nx-1+j*nx];
        fx_z2 = data[nx-1+(j+1)*nx];
        fxz = ((gridz[j+1]-z)*fx_z1+(z - gridz[j])*fx_z2)/dz;
    }
    else if (j >=nz-1)
    {
        fx_z1 = data[i+(nz-1)*nx];
        fx_z2 = data[i+(nz-1)*nx];
        fxz = ((gridx[i+1]-dim1)*fx_z1+(dim1 - gridx[i])*fx_z2)/d_dim1;
        
    }
    else
    {
      fx_z1 = ((gridx[i+1]-dim1)*data[i+j*nx] + (dim1 - gridx[i])*data[i+1+j*nx])/d_dim1;
      fx_z2 = ((gridx[i+1]-dim1)*data[i+(j+1)*nx] + (dim1 - gridx[i])*data[i+1+(j+1)*nx])/d_dim1; 
      fxz = ((gridz[j+1]-z)*fx_z1+(z - gridz[j])*fx_z2)/dz;
      //std::cout << "fxz1,2,fxz" << fx_z1 << fx_z2 << fxz <<std::endl;
      //std::cout << "gridz0,1 j dz" << gridz[0] <<gridz[1] << j << dz <<std::endl;
    }
    }

    return fxz;
}
double interp2dCombined ( double x, double y, double z,int nx, int nz,
    double* gridx,double* gridz,double* data ) {
    
    double fxz = 0.0;
    double fx_z1 = 0.0;
    double fx_z2 = 0.0; 
    if(nx*nz == 1)
    {
        fxz = data[0];
    }
    else{
#if USECYLSYMM > 0
    double dim1 = std::sqrt(x*x + y*y);
#else
    double dim1 = x;
#endif    
    double d_dim1 = gridx[1] - gridx[0];
    double dz = gridz[1] - gridz[0];
    int i = std::floor((dim1 - gridx[0])/d_dim1);//addition of 0.5 finds nearest gridpoint
    int j = std::floor((z - gridz[0])/dz);
    //double interp_value = data[i + j*nx];
    if (i < 0) i=0;
    if (j < 0) j=0;
    if (i >=nx-1 && j>=nz-1)
    {
        fxz = data[nx-1+(nz-1)*nx];
    }
    else if (i >=nx-1)
    {
        fx_z1 = data[nx-1+j*nx];
        fx_z2 = data[nx-1+(j+1)*nx];
        fxz = ((gridz[j+1]-z)*fx_z1+(z - gridz[j])*fx_z2)/dz;
    }
    else if (j >=nz-1)
    {
        fx_z1 = data[i+(nz-1)*nx];
        fx_z2 = data[i+(nz-1)*nx];
        fxz = ((gridx[i+1]-dim1)*fx_z1+(dim1 - gridx[i])*fx_z2)/d_dim1;
        
    }
    else
    {
      fx_z1 = ((gridx[i+1]-dim1)*data[i+j*nx] + (dim1 - gridx[i])*data[i+1+j*nx])/d_dim1;
      fx_z2 = ((gridx[i+1]-dim1)*data[i+(j+1)*nx] + (dim1 - gridx[i])*data[i+1+(j+1)*nx])/d_dim1; 
      fxz = ((gridz[j+1]-z)*fx_z1+(z - gridz[j])*fx_z2)/dz;
    }
  if(COMPARE_GITR)
    printf(" interp2dCombined: x %g y %g z %g dim1 %g nx %d, nz %d grid0 %g gridz0 %g i %d j %d d_dim1 %g dz %g inter2d-fxz %g \n", 
      x,y,z, dim1, nx, nz, gridx[0], gridz[0], i, j, d_dim1, dz, fxz);    
    }
    return fxz;
}

CUDA_CALLABLE_MEMBER

double interp3d ( double x, double y, double z,int nx,int ny, int nz,
    double* gridx,double* gridy, double* gridz,double* data ) {
    //std::cout << "xyz " << x << " "<<y << " " << z<< std::endl;
    //std::cout << "nxyz " << nx << " "<<ny << " " << nz<< std::endl;
    
    double fxyz = 0.0;

    double dx = gridx[1] - gridx[0];
    double dy = gridy[1] - gridy[0];
    double dz = gridz[1] - gridz[0];
    
    int i = std::floor((x - gridx[0])/dx);//addition of 0.5 finds nearest gridpoint
    int j = std::floor((y - gridy[0])/dy);
    int k = std::floor((z - gridz[0])/dz);
    //std::cout << "dxyz ijk " << dx << " "<<dy << " " << dz<< " " << i
      //  << " " << j << " " << k << std::endl;
    if(i <0 ) i=0;
    else if(i >=nx-1) i=nx-2;
    if(j <0 ) j=0;
    else if(j >=ny-1) j=ny-2;
    if(k <0 ) k=0;
    else if(k >=nz-1) k=nz-2;
    if(ny <=1) j=0;
    if(nz <=1) k=0;
    //std::cout << "dxyz ijk " << dx << " "<<dy << " " << dz<< " " << i
      //  << " " << j << " " << k << std::endl;
    //if(j <0 || j>ny-1) j=0;
    //if(k <0 || k>nz-1) k=0;
    double fx_z0 = (data[i + j*nx + k*nx*ny]*(gridx[i+1]-x) + data[i +1 + j*nx + k*nx*ny]*(x-gridx[i]))/dx;
    double fx_z1 = (data[i + j*nx + (k+1)*nx*ny]*(gridx[i+1]-x) + data[i +1 + j*nx + (k+1)*nx*ny]*(x-gridx[i]))/dx;
    //std::cout << "dataInd 1 2 3 4 " << i + j*nx + k*nx*ny << " "<<i+1 + j*nx + k*nx*ny << " " << i + j*nx + (k+1)*nx*ny<< " " << i +1 + j*nx + (k+1)*nx*ny
    //    << std::endl;

    //std::cout << "data 1 2 3 4 " << data[i + j*nx + k*nx*ny] << " "<<data[i+1 + j*nx + k*nx*ny] << " " << data[i + j*nx + (k+1)*nx*ny]<< " " << data[i +1 + j*nx + (k+1)*nx*ny]
    //    << std::endl;
    
    //std::cout << "fxz0 fxz1 " << fx_z0 << " "<<fx_z1 << std::endl;
    double fxy_z0 = (data[i + (j+1)*nx + k*nx*ny]*(gridx[i+1]-x) + data[i +1 + (j+1)*nx + k*nx*ny]*(x-gridx[i]))/dx;
    double fxy_z1 = (data[i + (j+1)*nx + (k+1)*nx*ny]*(gridx[i+1]-x) + data[i +1 + (j+1)*nx + (k+1)*nx*ny]*(x-gridx[i]))/dx;
    //std::cout << "fxyz0 fxyz1 " << fxy_z0 << " "<<fxy_z1 << std::endl;

    double fxz0 = (fx_z0*(gridz[k+1] - z) + fx_z1*(z-gridz[k]))/dz;
    double fxz1 = (fxy_z0*(gridz[k+1] - z) + fxy_z1*(z-gridz[k]))/dz;
    //std::cout << "fxz0 fxz1 " << fxz0 << " "<<fxz1 << std::endl;

    fxyz = (fxz0*(gridy[j+1] - y) + fxz1*(y-gridy[j]))/dy;
    if(ny <=1) fxyz=fxz0;
    if(nz <=1) fxyz=fx_z0;
    //std::cout <<"fxyz " << fxyz << std::endl;
    return fxyz;
}

CUDA_CALLABLE_MEMBER
void interp3dVector (double* field, double x, double y, double z,int nx,int ny, int nz,
        double* gridx,double* gridy,double* gridz,double* datar, double* dataz, double* datat ) {

    field[0] =  interp3d (x,y,z,nx,ny,nz,gridx, gridy,gridz,datar );
    field[1] =  interp3d (x,y,z,nx,ny,nz,gridx, gridy,gridz,datat );
    field[2] =  interp3d (x,y,z,nx,ny,nz,gridx, gridy,gridz,dataz );
}
CUDA_CALLABLE_MEMBER
void interp2dVector (double* field, double x, double y, double z,int nx, int nz,
double* gridx,double* gridz,double* datar, double* dataz, double* datat ) {

   double Ar = interp2dCombined(x,y,z,nx,nz,gridx,gridz, datar);
   double At = interp2dCombined(x,y,z,nx,nz,gridx,gridz, datat);
   field[2] = interp2dCombined(x,y,z,nx,nz,gridx,gridz, dataz);
#if USECYLSYMM > 0
            double theta = std::atan2(y,x);   
            field[0] = std::cos(theta)*Ar - std::sin(theta)*At;
            field[1] = std::sin(theta)*Ar + std::cos(theta)*At;
#else
            field[0] = Ar;
            field[1] = At;
#endif

}
CUDA_CALLABLE_MEMBER
void interpFieldAlignedVector (double* field, double x, double y, double z,int nx, int nz,
double* gridx,double* gridz,double* datar, double* dataz, double* datat,
int nxB, int nzB, double* gridxB,double* gridzB,double* datarB,double* datazB, double* datatB) {

   double Ar = interp2dCombined(x,y,z,nx,nz,gridx,gridz, datar);
   double B[3] = {0.0};
   double B_unit[3] = {0.0};
   double Bmag = 0.0;
   interp2dVector (&B[0],x,y,z,nxB,nzB,
                   gridxB,gridzB,datarB,datazB,datatB);
   Bmag = std::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
   B_unit[0] = B[0]/Bmag;
   B_unit[1] = B[1]/Bmag;
   B_unit[2] = B[2]/Bmag;
   //std::cout << " Ar and Bunit " << Ar << " " << B_unit[0] << " " <<
   //             " " << B_unit[1] << " " << B_unit[2] << std::endl; 
   field[0] = Ar*B_unit[0];
   field[1] = Ar*B_unit[1];
   field[2] = Ar*B_unit[2];

}
CUDA_CALLABLE_MEMBER
double interp1dUnstructured(double samplePoint,int nx, double max_x, double* data,int &lowInd)
{
    int done = 0;
    int low_index = 0;
    double interpolated_value = 0.0;

    for(int i=0;i<nx;i++)
    {
        if(done == 0)
        {
            if(samplePoint < data[i])
            {
                done = 1;
                low_index = i-1;
            }   
        }
    }

    //std::cout << " smple point nx max_x " << samplePoint << " " << nx << " " << max_x << std::endl;
    //std::cout << " lowInd " << low_index << " " << data[low_index] << " " << data[low_index+1] <<
      //  std::endl;
    interpolated_value =
        ((data[low_index+1] - samplePoint)*low_index*max_x/nx
        + (samplePoint - data[low_index])*(low_index+1)*max_x/nx)/(data[low_index+1]- data[low_index]);
      //(low_index+1)*max_x/nx
      //std::cout << "interpolated_value " << interpolated_value << std::endl;
    lowInd = low_index;
    if(low_index < 0)
    {
        //std::cout << "WARNING: interpolated value is outside range of CDF " << std::endl;
        lowInd = 0;
        //interpolated_value = samplePoint*data[0];
        if(samplePoint > 0.0)
        {
          interpolated_value = samplePoint;
        }
        else{
          interpolated_value = 0.0;
        }
    }
    if(low_index >= nx)
    {
        lowInd = nx-1;
        //interpolated_value = samplePoint*data[0];
          interpolated_value = max_x;
        
    }
    return interpolated_value;
}
CUDA_CALLABLE_MEMBER
double interp1dUnstructured2(double samplePoint,int nx, double *xdata, double* data)
{
    int done = 0;
    int low_index = 0;
    double interpolated_value = 0.0;

    for(int i=0;i<nx;i++)
    {
        if(done == 0)
        {
            if(samplePoint < data[i])
            {
                done = 1;
                low_index = i-1;
            }   
        }
    }
    //std::cout << " sample point low_index " << samplePoint<< " " << low_index << std::endl;
    interpolated_value =    xdata[low_index]  
        + (samplePoint - data[low_index])*(xdata[low_index + 1] - xdata[low_index])/(data[low_index+1]- data[low_index]);
    //std::cout << " xdatas " << xdata[low_index] << " " << xdata[low_index+1] << std::endl;
    return interpolated_value;
}
CUDA_CALLABLE_MEMBER
double interp2dUnstructured(double x,double y,int nx,int ny, double *xgrid,double *ygrid, double* data)
{
    int doneX = 0;
    int doneY = 0;
    int xInd = 0;
    double xDiffUp = 0.0;
    double xDiffDown = 0.0;
    int yInd = 0;
    double dx;
    double yLowValue; 
    double yHighValue;
    double yDiffUp;
    double yDiffDown; 
    double dy;
    double fxy=0.0;
    double factor = 1.0;

    if(x >= xgrid[0] && x<= xgrid[nx-1])
    {
      for(int i=0;i<nx;i++)
      {
          if(!doneX)
          {
             if(x<xgrid[i])
               {
                  doneX = 1;
                  xInd = i-1;
               }
          }
      }
    }
    else
    {
        factor = 0.0;
    }
    if(y >= ygrid[0] && y<= ygrid[ny-1])
    {
      for(int i=0;i<ny;i++)
      {
          if(!doneY)
          {
             if(y<ygrid[i])
               {
                  doneY = 1;
                  yInd = i-1;
               }
          }
      }
    }
    else
    {
        factor = 0.0;
    }
   
    //std::cout << "x vals " << xgrid[xInd] << " " << xgrid[xInd+1];
    //std::cout << "y vals " << ygrid[yInd] << " " << ygrid[yInd+1];
    xDiffUp = xgrid[xInd+1] - x;
    xDiffDown = x-xgrid[xInd];
    dx = xgrid[xInd+1]-xgrid[xInd];
    //std::cout << "dx, data vals " << dx << " " << data[xInd + yInd*nx] << " " <<
                 //data[xInd+1 + yInd*nx] << " " << data[xInd + (yInd+1)*nx] << " " << 
                 //data[xInd+1 + (yInd+1)*nx] << std::endl;
    yLowValue = (xDiffUp*data[xInd + yInd*nx] + xDiffDown*data[xInd+1 + yInd*nx])/dx;
    yHighValue = (xDiffUp*data[xInd + (yInd+1)*nx] + xDiffDown*data[xInd+1 + (yInd+1)*nx])/dx;
    yDiffUp = ygrid[yInd+1]-y;
    yDiffDown = y - ygrid[yInd];
    dy = ygrid[yInd+1] - ygrid[yInd];
    fxy = factor*(yDiffUp*yLowValue + yDiffDown*yHighValue)/dy;

    return fxy;

}

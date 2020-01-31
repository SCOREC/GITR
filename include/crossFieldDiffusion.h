#ifndef _CFDIFFUSION_
#define _CFDIFFUSION_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Particles.h"
#include <cmath>

struct crossFieldDiffusion { 
    Particles *particlesPointer;
    const double dt;
	const double diffusionCoefficient;
    int nR_Bfield;
    int nZ_Bfield;
    double * BfieldGridRDevicePointer;
    double * BfieldGridZDevicePointer;
    double * BfieldRDevicePointer;
    double * BfieldZDevicePointer;
    double * BfieldTDevicePointer;
#if __CUDACC__
        curandState *state;
#else
         mt19937 *state;
#endif

    int dof_intermediate = 0;
    int idof = -1;
    int nT = -1;
    double* intermediate;
    crossFieldDiffusion(Particles *_particlesPointer, double _dt,
#if __CUDACC__
                            curandState *_state,
#else
                                             mt19937 *_state,
#endif
            double _diffusionCoefficient,
            int _nR_Bfield, int _nZ_Bfield,
            double * _BfieldGridRDevicePointer,double * _BfieldGridZDevicePointer,
            double * _BfieldRDevicePointer,double * _BfieldZDevicePointer,
            double * _BfieldTDevicePointer,
            double* intermediate, int nT, int idof, int dof_intermediate  )
      : particlesPointer(_particlesPointer),
        dt(_dt),
        diffusionCoefficient(_diffusionCoefficient),
        nR_Bfield(_nR_Bfield),
        nZ_Bfield(_nZ_Bfield),
        BfieldGridRDevicePointer(_BfieldGridRDevicePointer),
        BfieldGridZDevicePointer(_BfieldGridZDevicePointer),
        BfieldRDevicePointer(_BfieldRDevicePointer),
        BfieldZDevicePointer(_BfieldZDevicePointer),
        BfieldTDevicePointer(_BfieldTDevicePointer),
        state(_state), intermediate(intermediate),nT(nT),
        idof(idof), dof_intermediate(dof_intermediate)  {
  }

CUDA_CALLABLE_MEMBER_DEVICE    
void operator()( size_t indx) const { 

	    if(particlesPointer->hitWall[indx] == 0.0)
        {
           if(particlesPointer->charge[indx] > 0.0)
           { 
       
	        double perpVector[3]= {0, 0, 0};
	        double B[3] = {0.0,0.0,0.0};
            double Bmag = 0.0;
		double B_unit[3] = {0.0, 0.0, 0.0};
		double phi_random;
		double norm;
		double step;
    double x0 = particlesPointer->xprevious[indx];
    double y0 = particlesPointer->yprevious[indx];
    double z0 = particlesPointer->zprevious[indx];
    // cout << "initial position " << x0 << " " << y0 << " " << z0 <<  endl;
        interp2dVector(&B[0],particlesPointer->xprevious[indx],particlesPointer->yprevious[indx],particlesPointer->zprevious[indx],nR_Bfield,nZ_Bfield,
                               BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,BfieldZDevicePointer,BfieldTDevicePointer);
        Bmag =  sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
        B_unit[0] = B[0]/Bmag;
        B_unit[1] = B[1]/Bmag;
        B_unit[2] = B[2]/Bmag;
    // cout << "B " << B[0] << " " <<  B[1]<< " " <<  B[2]<< " " << endl;
    // cout << "B_unit " << B_unit[0] << " " <<  B_unit[1]<< " " <<  B_unit[2]<< " " << endl;
#if PARTICLESEEDS > 0
#ifdef __CUDACC__
        	double r3 = curand_uniform(&state[indx]);
#else
        	 uniform_real_distribution<double> dist(0.0, 1.0);
        	double r3=dist(state[indx]);
        	double r4=dist(state[indx]);
#endif 
#else
#if __CUDACC__
            double r3 = curand_uniform(&state[2]);
#else
             uniform_real_distribution<double> dist(0.0, 1.0);
            double r3=dist(state[2]);
#endif
#endif
		step =  sqrt(6*diffusionCoefficient*dt);

      int nthStep = particlesPointer->tt[indx];
      auto pindex = particlesPointer->index[indx];
      int beg = -1;
      if(dof_intermediate > 0) { 
        beg = pindex*nT*dof_intermediate + (nthStep-1)*dof_intermediate;
        intermediate[beg+idof] = r3;
      }
#if USEPERPDIFFUSION > 1
      if(dof_intermediate > 0) { 
        intermediate[beg+idof+1] = r4;
      }
    double plus_minus1 = floor(r4 + 0.5)*2 - 1;
    double h = 0.001;
//    double k1x = B_unit[0]*h;
//    double k1y = B_unit[1]*h;
//    double k1z = B_unit[2]*h;
    double x_plus = x0+B_unit[0]*h;
    double y_plus = y0+B_unit[1]*h;
    double z_plus = z0+B_unit[2]*h;
//    double x_minus = x0-B_unit[0]*h;
//    double y_minus = y0-B_unit[1]*h;
//    double z_minus = z0-B_unit[2]*h;
//     x_plus = x0+k1x;
//     y_plus = y0+k1y;
//     z_plus = z0+k1z;
//     x_minus = x0-k1x;
//     y_minus = y0-k1y;
//     z_minus = z0-k1z;
    // cout << "pos plus " << x_plus << " " << y_plus << " " << z_plus <<  endl;
    // cout << "pos minus " << x_minus << " " << y_minus << " " << z_minus <<  endl;
    double B_plus[3] = {0.0f};
        interp2dVector(&B_plus[0],x_plus,y_plus,z_plus,nR_Bfield,nZ_Bfield,
                               BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,BfieldZDevicePointer,BfieldTDevicePointer);
        double Bmag_plus =  sqrt(B_plus[0]*B_plus[0] + B_plus[1]*B_plus[1] + B_plus[2]*B_plus[2]);
//    double k2x = B_plus[0]*h;
//    double k2y = B_plus[1]*h;
//    double k2z = B_plus[2]*h;
//   double xNew = x0+0.5*(k1x+k2x); 
//   double yNew = y0+0.5*(k1y+k2y); 
//   double zNew = z0+0.5*(k1z+k2z); 
//    cout <<"pps new plus " << xNew << " " << yNew << " " << zNew <<  endl;
//    double B_minus[3] = {0.0f};
//        interp2dVector(&B_minus[0],x_minus,y_minus,z_minus,nR_Bfield,nZ_Bfield,
//                               BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,BfieldZDevicePointer,BfieldTDevicePointer);
//        double Bmag_minus =  sqrt(B_minus[0]*B_minus[0] + B_minus[1]*B_minus[1] + B_minus[2]*B_minus[2]);
//    double k2x_minus = -B_minus[0]*h/Bmag_minus;
//    double k2y_minus = -B_minus[1]*h/Bmag_minus;
//    double k2z_minus = -B_minus[2]*h/Bmag_minus;
//   double xNew_minus = x0+0.5*(k1x+k2x); 
//   double yNew_minus = y0+0.5*(k1y+k2y); 
//   double zNew_minus = z0+0.5*(k1z+k2z); 
//    cout <<"pps new minus " << xNew_minus << " " << yNew_minus << " " << zNew_minus <<  endl;
    
    double B_deriv1[3] = {0.0f};
//    double B_deriv2[3] = {0.0f};
    // cout << "B_plus " << B_plus[0] << " " <<  B_plus[1]<< " " <<  B_plus[2]<< " " << endl;
    // cout << "B_minus " << B_minus[0] << " " <<  B_minus[1]<< " " <<  B_minus[2]<< " " << endl;
    B_deriv1[0] = (B_plus[0] - B[0])/(h);
    B_deriv1[1] = (B_plus[1] - B[1])/(h);
    B_deriv1[2] = (B_plus[2] - B[2])/(h);
    
   // B_deriv1[0] = (B_plus[0] - B_minus[0])/(2*h);
   // B_deriv1[1] = (B_plus[1] - B_minus[1])/(2*h);
   // B_deriv1[2] = (B_plus[2] - B_minus[2])/(2*h);
    // cout << "B_deriv1 " << B_deriv1[0] << " " <<  B_deriv1[1]<< " " <<  B_deriv1[2]<< " " << endl;
    // cout << "Bderiv2 " << B_deriv2[0] << " " <<  B_deriv2[1]<< " " <<  B_deriv2[2]<< " " << endl;
    //double pos_deriv1[3] = {0.0f};
    //double pos_deriv2[3] = {0.0f};
    //pos_deriv1[0] = (xNew-x0)/(h);
    //pos_deriv1[1] = (yNew-y0)/(h);
    //pos_deriv1[2] = (zNew-z0)/(h);
    //pos_deriv2[0] = (xNew - 2*x0 + xNew_minus)/(h*h);
    //pos_deriv2[1] = (yNew - 2*y0 + yNew_minus)/(h*h);
    //pos_deriv2[2] = (zNew - 2*z0 + zNew_minus)/(h*h);
    // cout << "pos_deriv1 " << pos_deriv1[0] << " " <<  pos_deriv1[1]<< " " <<  pos_deriv1[2]<< " " << endl;
    //double deriv_cross[3] = {0.0};
    //vectorCrossProduct(B_deriv1, B_deriv2, deriv_cross);
    double denom = vectorNorm(B_deriv1);
    //double norm_cross = vectorNorm(deriv_cross);
    // cout << "deriv_cross " << deriv_cross[0] << " " <<  deriv_cross[1]<< " " <<  deriv_cross[2]<< " " << endl;
    // cout << "denome and norm_cross " << denom << " " << norm_cross <<  endl;
    double R = 1.0e4;
    if(( abs(denom) > 1e-10) & ( abs(denom) < 1e10) )
    {
      R = Bmag/denom;
    }
    // cout << "Radius of curvature"<< R << endl;
    double initial_guess_theta = 3.14159265359*0.5;
    double eps = 0.01;
    double error = 2.0;
    double s = step;
    double drand = r3;
    double theta0 = initial_guess_theta;
    double theta1 = 0.0;
    double f = 0.0;
    double f_prime = 0.0;
    int nloops = 0;
    if(R > 1.0e-4)
    {
    while ((error > eps)&(nloops<10))
    {
        f = (2*R*theta0-s*sin(theta0))/(2*3.14159265359*R) - drand;
        f_prime = (2*R-s*cos(theta0))/(2*3.14159265359*R); 
        theta1 = theta0 - f/f_prime;
         error = abs(theta1-theta0);
         theta0=theta1;
         nloops++;
         // cout << " R rand and theta "<<R << " " <<  drand << " " << theta0 <<  endl;
    }
    if(nloops > 9)
    {
      theta0 = 2*3.14159265359*drand;

    }
    }
    else
    {R = 1.0e-4;
      theta0 = 2*3.14159265359*drand;
    }
         // cout << "out of newton"<<  endl;
      

    if(plus_minus1 < 0)
    {
      theta0 = 2*3.14159265359-theta0; 
    }
perpVector[0] = B_deriv1[0];
perpVector[1] = B_deriv1[1];
perpVector[2] = B_deriv1[2];
		norm =  sqrt(perpVector[0]*perpVector[0] + perpVector[1]*perpVector[1] + perpVector[2]*perpVector[2]);
		perpVector[0] = perpVector[0]/norm;
		perpVector[1] = perpVector[1]/norm;
		perpVector[2] = perpVector[2]/norm;
    double y_dir[3] = {0.0};
    vectorCrossProduct(B, B_deriv1, y_dir);
    double x_comp = s* cos(theta0);
    double y_comp = s* sin(theta0);
    double x_transform = x_comp*perpVector[0] + y_comp*y_dir[0];
    double y_transform = x_comp*perpVector[1] + y_comp*y_dir[1];
    double z_transform = x_comp*perpVector[2] + y_comp*y_dir[2];
    //double R = 1.2;
    //double r_in = R-step;
    //double r_out = R-step;
    //double A_in = R*R-r_in*r_in;
    //double A_out = r_out*r_out-R*R;
    //double theta[100] = {0.0f};
    //double f_theta[100] = {0.0f};
    //double cdf[100] = {0.0f};
    //for(int ii=0;ii<100;ii++)
    //{ theta[ii] = 3.1415/100.0*ii;
    //  f_theta[ii] = 
    //    //2*R*theta[ii] -step* sin(theta[ii]);}
    //    //A_in + ii/100*(A_out-A_in);}
    //    (2.0*R*step-step*step* cos(theta[ii]));}
    //cdf[0] = f_theta[0];
    //for(int ii=1;ii<100;ii++)
    //{cdf[ii] = cdf[ii-1]+f_theta[ii];}
    //for(int ii=0;ii<100;ii++)
    //{cdf[ii] = cdf[ii]+cdf[99];}
    //double minTheta = 0;
    //int found=0;
    //for(int ii=0;ii<100;ii++)
    //{if(r3>cdf[ii] && found < 1)
    //  {
    //    minTheta = theta[ii];
    //    found = 2;
    //  }
    //}

if( abs(denom) < 1.0e-8)
{
#endif
    perpVector[0] = 0.0;
    perpVector[1] = 0.0;
    perpVector[2] = 0.0;
		phi_random = 2*3.14159265*r3;
		perpVector[0] =  cos(phi_random);
		perpVector[1] =  sin(phi_random);
		perpVector[2] = (-perpVector[0]*B_unit[0] - perpVector[1]*B_unit[1])/B_unit[2];
                // cout << "perp Vector " << perpVector[0] << " " << perpVector[1] << " " << perpVector[2]<< endl;
		if (B_unit[2] == 0){
			perpVector[2] = perpVector[1];
			perpVector[1] = (-perpVector[0]*B_unit[0] - perpVector[2]*B_unit[2])/B_unit[1];
		}
               //  cout << "perp Vector " << perpVector[0] << " " << perpVector[1] << " " << perpVector[2]<< endl;
		
		if ((B_unit[0] == 1.0 && B_unit[1] ==0.0 && B_unit[2] ==0.0) || (B_unit[0] == -1.0 && B_unit[1] ==0.0 && B_unit[2] ==0.0))
		{
			perpVector[2] = perpVector[0];
			perpVector[0] = 0;
			perpVector[1] = sin(phi_random);
                // cout << "perp Vector " << perpVector[0] << " " << perpVector[1] << " " << perpVector[2]<< endl;
		}
		else if ((B_unit[0] == 0.0 && B_unit[1] ==1.0 && B_unit[2] ==0.0) || (B_unit[0] == 0.0 && B_unit[1] ==-1.0 && B_unit[2] ==0.0))
		{
			perpVector[1] = 0.0;
		}
		else if ((B_unit[0] == 0.0 && B_unit[1] ==0.0 && B_unit[2] ==1.0) || (B_unit[0] == 0.0 && B_unit[1] ==0.0 && B_unit[2] ==-1.0))
		{
			perpVector[2] = 0;
		}
		
		norm =  sqrt(perpVector[0]*perpVector[0] + perpVector[1]*perpVector[1] + perpVector[2]*perpVector[2]);
		perpVector[0] = perpVector[0]/norm;
		perpVector[1] = perpVector[1]/norm;
		perpVector[2] = perpVector[2]/norm;
//                // cout << "perp Vector " << perpVector[0] << " " << perpVector[1] << " " << perpVector[2]<< endl;
//		
//		step =  sqrt(6*diffusionCoefficient*dt);
    // cout << "y_dir " << y_dir[0] << " " <<  y_dir[1]<< " " <<  y_dir[2]<< " " << endl;
    // cout << "y_dir " << y_dir[0] << " " <<  y_dir[1]<< " " <<  y_dir[2]<< " " << endl;
    // cout << "transforms " << x_transform << " " << y_transform << " " << z_transform <<  endl; 
		particlesPointer->x[indx] = particlesPointer->xprevious[indx] + step*perpVector[0];
		particlesPointer->y[indx] = particlesPointer->yprevious[indx] + step*perpVector[1];
		particlesPointer->z[indx] = particlesPointer->zprevious[indx] + step*perpVector[2];
#if USEPERPDIFFUSION > 1    
}
else
{
    particlesPointer->x[indx] = x0 + x_transform;
		particlesPointer->y[indx] = y0 + y_transform;
		particlesPointer->z[indx] = z0 + z_transform;
}
#endif
    	}
    } }
};

#endif

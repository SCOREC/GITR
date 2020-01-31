#ifndef _HASHGEOMSHEATH_
#define _HASHGEOMSHEATH_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#define CUDA_CALLABLE_MEMBER_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
#define CUDA_CALLABLE_MEMBER_HOST
#endif

#include "Particles.h"
#include "Boundary.h"
#ifdef __CUDACC__
#include <thrust/random.h>
#include <curand_kernel.h>
#endif

#ifdef __GNUC__ 
#include <random>
using namespace std;
#endif

#include "interpRateCoeff.hpp"
#include <cmath>

struct hashGeom_sheath {
   //int k;
   int nLines; 
   Boundary* boundary;
   double* x;
   double* y;
   double* z;
   int n_closeGeomElements;
   //double* minDist;
   int* closeGeom;
   int nR;
   int nY;
   int nZ;


   hashGeom_sheath(int _nLines,
                Boundary* _boundary,
                double* _x,
                double* _y, 
                double* _z, 
                int _n_closeGeomElements, //double *_minDist, 
                int *_closeGeom,
                int _nR, int _nY, int _nZ)
               : nLines(_nLines),boundary(_boundary), x(_x), y(_y), z(_z), 
               n_closeGeomElements(_n_closeGeomElements), 
               //minDist(_minDist), 
               closeGeom(_closeGeom), nR(_nR), nY(_nY), nZ(_nZ) {}
    
    CUDA_CALLABLE_MEMBER_DEVICE 
    void operator()(size_t indx) const { 
      #if USE3DTETGEOM > 0
       double kk = indx/(nR*nY);
       int k = floor(kk);
       int jjj = indx - k*nR*nY;
       double jj = 1.0*jjj/nR;
       int j = floor(jj);
       int i = indx - j*nR - k*(nR*nY);
       int xyzIndx = indx;
       double x0 = x[i];
       double y0 = y[j];
       double z0 = z[k];
      #else
       double kk = indx/(nR);
       int k = floor(kk);
       int i = indx - k*(nR);
       double x0 = x[i];
       double y0 = 0.0;
       double z0 = z[k];
       int xyzIndx = indx;
      #endif
       double A[3] = {0.0,0.0,0.0};
            double B[3] = {0.0,0.0,0.0};
            double C[3] = {0.0,0.0,0.0};
            double AB[3] = {0.0,0.0,0.0};
            double AC[3] = {0.0,0.0,0.0};
            double BC[3] = {0.0,0.0,0.0};
            double CA[3] = {0.0,0.0,0.0};
            double p[3] = {0.0,0.0,0.0};
            double Ap[3] = {0.0,0.0,0.0};
            double Bp[3] = {0.0,0.0,0.0};
            double Cp[3] = {0.0,0.0,0.0};
            double normalVector[3] = {0.0,0.0,0.0};
            double crossABAp[3] = {0.0,0.0,0.0};
            double crossBCBp[3] = {0.0,0.0,0.0};
            double crossCACp[3] = {0.0,0.0,0.0};
            double signDot0 = 0.0;
            double signDot1 = 0.0;
            double signDot2 = 0.0;
            double totalSigns = 0.0;
#if USE_CUDA
           double *minDist  = new double[n_closeGeomElements];
           for(int i1=0;i1<n_closeGeomElements;i1++){ minDist[i1] = 1.0e6;}
           //double minDist[10] = {1.0e6,1.0e6,1.0e6,1.0e6,1.0e6,1.0e6,1.0e6,1.0e6,1.0e6,1.0e6};
#else
           sim::Array<double> minDist(n_closeGeomElements,1e6);      
#endif
                
    for(int l=0; l<nLines; l++)
    {
        if(boundary[l].Z > 0)
        {
                       double a = boundary[l].a;
                       double b = boundary[l].b;
                       double c = boundary[l].c;
                       double d = boundary[l].d;
    #if USE3DTETGEOM > 0
      double plane_norm = boundary[l].plane_norm;
      double t = -(a*x0 + b*y0 + c*z0 + d)/(a*a + b*b + c*c);
      p[0] = a*t + x0;
      p[1] = b*t + y0;
      p[2] = c*t + z0;
      double perpDist = sqrt((x0-p[0])*(x0-p[0]) + (y0-p[1])*(y0-p[1]) + (z0-p[2])*(z0-p[2]));
    #endif
      vectorAssign(boundary[l].x1, boundary[l].y1, 
          boundary[l].z1, A);    
      vectorAssign(boundary[l].x2, boundary[l].y2, 
          boundary[l].z2, B);    
    #if USE3DTETGEOM > 0
      vectorAssign(boundary[l].x3, boundary[l].y3, 
          boundary[l].z3, C); 
    #endif
      vectorSubtract(B,A,AB);
    #if USE3DTETGEOM > 0
      vectorSubtract(C,A,AC);
      vectorSubtract(C,B,BC);
      vectorSubtract(A,C,CA);

      vectorSubtract(p,A,Ap);
      vectorSubtract(p,B,Bp);
      vectorSubtract(p,C,Cp);

      vectorCrossProduct(AB,AC,normalVector);
      vectorCrossProduct(AB,Ap,crossABAp);
      vectorCrossProduct(BC,Bp,crossBCBp);
      vectorCrossProduct(CA,Cp,crossCACp);

        signDot0 = copysign(1.0,vectorDotProduct(crossABAp, normalVector));
        signDot1 = copysign(1.0,vectorDotProduct(crossBCBp, normalVector));
        signDot2 = copysign(1.0,vectorDotProduct(crossCACp, normalVector));
        totalSigns = abs(signDot0 + signDot1 + signDot2);
        if (totalSigns == 3.0) {
        } else
          perpDist = 1.0e6;
#endif
        p[0] = x0;
        p[1] = y0;
        p[2] = z0;
        double pA[3] = {0.0};
        double cEdge1[3] = {0.0};
        double dEdge1[3] = {0.0};
        vectorSubtract(A, p, pA);
        double cEdge1mag = vectorDotProduct(pA, AB) / vectorDotProduct(AB, AB);
        double distE1 = 1.0e6;
        if (cEdge1mag < 0.0 && cEdge1mag > -1.0) {
          vectorScalarMult(cEdge1mag, AB, cEdge1);
          vectorSubtract(pA, cEdge1, dEdge1);
          distE1 = sqrt(vectorDotProduct(dEdge1, dEdge1));
        }
#if USE3DTETGEOM > 0
        double pB[3] = {0.0};
        double cEdge2[3] = {0.0};
        double dEdge2[3] = {0.0};
        vectorSubtract(B, p, pB);
        double cEdge2mag = vectorDotProduct(pB, BC) / vectorDotProduct(BC, BC);
        double distE2 = 1.0e6;
        if (cEdge2mag < 0.0 && cEdge2mag > -1.0) {
          vectorScalarMult(cEdge2mag, BC, cEdge2);
          vectorSubtract(pB, cEdge2, dEdge2);
          distE2 = sqrt(vectorDotProduct(dEdge2, dEdge2));
        }
        double pC[3] = {0.0};
        double cEdge3[3] = {0.0};
        double dEdge3[3] = {0.0};
        vectorSubtract(C, p, pC);
        double cEdge3mag = vectorDotProduct(pC, CA) / vectorDotProduct(CA, CA);
        double distE3 = 1.0e6;
        if (cEdge3mag < 0.0 && cEdge3mag > -1.0) {
          vectorScalarMult(cEdge3mag, CA, cEdge3);
          vectorSubtract(pC, cEdge3, dEdge3);
          distE3 = sqrt(vectorDotProduct(dEdge3, dEdge3));
        }
        double minEdge = min(distE1, distE2);
        minEdge = min(distE3, minEdge);
#else
          //
          double minEdge = distE1;
#endif
        double d1 =sqrt((x0 - boundary[l].x1)*(x0 - boundary[l].x1)
                +  (y0 - boundary[l].y1)*(y0 - boundary[l].y1)
                +  (z0 - boundary[l].z1)*(z0 - boundary[l].z1));
        double d2 =sqrt((x0 - boundary[l].x2)*(x0 - boundary[l].x2)
                +  (y0 - boundary[l].y2)*(y0 - boundary[l].y2)
                +  (z0 - boundary[l].z2)*(z0 - boundary[l].z2));
          #if USE3DTETGEOM > 0
            double d3 =sqrt((x0 - boundary[l].x3)*(x0 - boundary[l].x3)
                    +  (y0 - boundary[l].y3)*(y0 - boundary[l].y3)
                    +  (z0 - boundary[l].z3)*(z0 - boundary[l].z3));
          #endif
          double minOf3 = min(d1,d2);
          minOf3 = min(minOf3,minEdge);
        //cout << "min of two " << minOf3 << endl;
          #if USE3DTETGEOM > 0
          minOf3 = min(minOf3,perpDist);
            minOf3 = min(minOf3,d3);
          #endif
          int minIndClose = n_closeGeomElements;
           for(int m=0; m< n_closeGeomElements; m++)
           {
              if(minDist[m] > minOf3)
              {
                  minIndClose = minIndClose-1;
              }
           }

           if((minIndClose < n_closeGeomElements) && (minIndClose > -1))
           {
               //%shift numbers down
               for(int n=n_closeGeomElements-1; n>minIndClose; n--)
               {
                    minDist[n] = 
                    minDist[n-1];  
               closeGeom[indx*n_closeGeomElements+ n] =    
               closeGeom[indx*n_closeGeomElements + n-1];
               }
               minDist[minIndClose] = minOf3;
              closeGeom[indx*n_closeGeomElements + minIndClose] = l;
     }
        }
    }
#if USE_CUDA
     delete[] minDist;
#endif
                
                }
};

#endif

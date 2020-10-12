#ifndef _BORIS_
#define _BORIS_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#include "thrust/extrema.h"
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include <algorithm>
#include "Particles.h"
#include "Boundary.h"
#include "interp2d.hpp"
#include <algorithm>

CUDA_CALLABLE_MEMBER
void vectorAdd(double A[], double B[],double C[])
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];
}

CUDA_CALLABLE_MEMBER
void vectorSubtract(double A[], double B[],double C[])
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    C[2] = A[2] - B[2];
}

CUDA_CALLABLE_MEMBER
void vectorScalarMult(double a, double B[],double C[])
{
    C[0] = a*B[0];
    C[1] = a*B[1];
    C[2] = a*B[2];
}

CUDA_CALLABLE_MEMBER
void vectorAssign(double a, double b,double c, double D[])
{
    D[0] = a;
    D[1] = b;
    D[2] = c;
}

CUDA_CALLABLE_MEMBER
double vectorNorm(double A[])
{
    double norm = 0.0;
    norm = sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);

        return norm;
}
CUDA_CALLABLE_MEMBER
void vectorNormalize(double A[],double B[])
{
    double norm = 0.0;
    norm = sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);
    B[0] = A[0]/norm;
    B[1] = A[1]/norm;
    B[2] = A[2]/norm;

}

CUDA_CALLABLE_MEMBER
double vectorDotProduct(double A[], double B[])
{
    double c = A[0]*B[0] +  A[1]*B[1] + A[2]*B[2];
    return c;
}

CUDA_CALLABLE_MEMBER
void vectorCrossProduct(double A[], double B[], double C[])
{
    double tmp[3] = {0.0,0.0,0.0};
    tmp[0] = A[1]*B[2] - A[2]*B[1];
    tmp[1] = A[2]*B[0] - A[0]*B[2];
    tmp[2] = A[0]*B[1] - A[1]*B[0];

    C[0] = tmp[0];
    C[1] = tmp[1];
    C[2] = tmp[2];
}

CUDA_CALLABLE_MEMBER
void closest_point_on_triangle(double* A, double* B, double*C, double* pt, double* ptq) {
  int debug = 0;
  int region = -1;
  // Check if P in vertex region outside A
  double ab[3], ac[3], ap[3], bp[3];
  vectorSubtract(B, A, ab); 
  vectorSubtract(C, A, ac); 
  vectorSubtract(pt, A, ap);
  double d1 = vectorDotProduct(ab, ap);
  double d2 = vectorDotProduct(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    // barycentric coordinates (1,0,0)
    for(int i=0; i<3; ++i)
      ptq[i] = A[i];
    region =0;
    return; 
  }
  // Check if P in vertex region outside B
  vectorSubtract(pt, B, bp);
  double d3 = vectorDotProduct(ab, bp);
  double d4 = vectorDotProduct(ac, bp);
  if(d3 >= 0 && d4 <= d3){ 
    // barycentric coordinates (0,1,0)
    for(int i=0; i<3; ++i)
      ptq[i] = B[i];
    region =1;
    return; 
  }
  // Check if P in edge region of AB, if so return projection of P onto AB
  double vc = d1*d4 - d3*d2;
  if(vc <= 0 && d1 >= 0 && d3 <= 0) {
    double v = d1 / (d1 - d3);
    // barycentric coordinates (1-v,v,0)
    vectorScalarMult(v, ab, ptq);
    vectorAdd(ptq, A, ptq); 
    region = 2; //FIX
    return;
  }

  // Check if P in vertex region outside C
  double cp[3];
  vectorSubtract(pt, C, cp);
  double d5 = vectorDotProduct(ab, cp);
  double d6 = vectorDotProduct(ac, cp);
  if(region <0 && d6 >= 0 && d5 <= d6) { 
    // barycentric coordinates (0,0,1)
    for(int i=0; i<3; ++i)
      ptq[i] = C[i]; 
    region =3;
    return;
  }

  // Check if P in edge region of AC, if so return projection of P onto AC
  double vb = d5*d2 - d1*d6;
  if(region <0 && vb <= 0 && d2 >= 0 && d6 <= 0) {
    double w = d2 / (d2 - d6);
    // barycentric coordinates (1-w,0,w)
    vectorScalarMult(w, ac, ptq); // w*vac;
    vectorAdd(ptq, A, ptq);
    region = 4;
    return;
  }

  // Check if P in edge region of BC, if so return projection of P onto BC
  double va = d3*d6 - d5*d4;
  if(region <0 && va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    // barycentric coordinates (0,1-w,w)
    double c_b[3], wc_b[3];
    vectorSubtract(C,B,c_b);
    vectorScalarMult(w, c_b, wc_b);
    vectorAdd(B, wc_b, ptq);
    //ptq =  ptb + w * (ptc - ptb); 
    region = 5;
    return;
  }

  // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
  if(region <0) {
    double inv = 1 / (va + vb + vc);
    double v = vb * inv;
    double w = vc * inv;
    // u*a + v*b + w*c, u = va * inv = 1 - v - w
    double wxac[3], vxab[3], plus[3];
    vectorScalarMult(w, ac, wxac);
    vectorScalarMult(v, ab, vxab);
    vectorAdd(vxab, wxac, plus);
    vectorAdd(A, plus, ptq);
    //ptq =  pta + v * vab+ w * vac;
    region = 6;
    return;
  }
  if(debug)
    printf("d's:: %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f \n", d1, d2, d3, d4, d5, d6);
}

CUDA_CALLABLE_MEMBER
void bccCoords(double* a, double* b, double* c, double * x, double*bcc) {
  double cross[3] ={0};
  double b_a[3] ={0};
  double c_a[3] = {0};
  vectorSubtract(b, a, b_a);
  vectorSubtract(c, a, c_a);
  vectorCrossProduct(b_a, c_a, cross);
  double crossX[3] = {0};
  vectorScalarMult(1/2.0, cross, crossX);
  double norm[3] = {0};
  vectorNormalize(crossX, norm);
  double area = vectorDotProduct(norm, cross);

  if(abs(area) < 1e-10) {
    printf("area is too small \n");
    return;
  }
  auto fac = 1/(area*2.0);
  double b_aXx_a[3] = {0};
  double c_bXx_b[3] = {0};
  double x_aXc_a[3] = {0};
  double x_a[3] = {0};
  double x_b[3] = {0};
  double c_b[3] = {0};
  vectorSubtract(c, b, c_b);
  vectorSubtract(x, a, x_a);
  vectorSubtract(x, b, x_b);
  vectorCrossProduct(b_a, x_a, b_aXx_a);
  vectorCrossProduct(c_b, x_b, c_bXx_b);
  vectorCrossProduct(x_a, c_a, x_aXc_a);
  bcc[0] = fac * vectorDotProduct(norm, b_aXx_a);
  bcc[1] = fac * vectorDotProduct(norm, c_bXx_b);
  bcc[2] = fac * vectorDotProduct(norm, x_aXc_a);
}

CUDA_CALLABLE_MEMBER

double getE ( double x0, double y, double z, double E[], Boundary *boundaryVector, int nLines,
       int nR_closeGeom, int nY_closeGeom,int nZ_closeGeom, int n_closeGeomElements, 
       double *closeGeomGridr,double *closeGeomGridy, double *closeGeomGridz, int *closeGeom, 
       int&  closestBoundaryIndex, int* csrHashPtrs=nullptr, int* csrHashes=nullptr, int ptcl=-1, 
       int tstep=-1, int* bdryMinInd=nullptr, int detail=0 ) {

   //int detail = (ptcl ==612 && tstep >4473 && tstep <4476) ? 1 : 0;

#if USE3DTETGEOM > 0
    double Emag = 0.0;
    double Er = 0.0;
    double Et = 0.0;
      double p0[3] = {x0,y,z};
    double angle = 0.0;
	double fd = 0.0;
	double pot = 0.0;
      double a = 0.0;
      double b = 0.0;
      double c = 0.0;
      double d = 0.0;
      double plane_norm = 0.0;
      double pointToPlaneDistance0 = 0.0;
      double pointToPlaneDistance1 = 0.0;
      double signPoint0 = 0.0;
      double signPoint1 = 0.0;
      double t = 0.0;
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
      double p0A[3] = {0.0,0.0,0.0};
      double p0B[3] = {0.0,0.0,0.0};
      double p0C[3] = {0.0,0.0,0.0};
      double p0AB[3] = {0.0,0.0,0.0};
      double p0BC[3] = {0.0,0.0,0.0};
      double p0CA[3] = {0.0,0.0,0.0};
      double p0Anorm = 0.0;
      double p0Bnorm = 0.0;
      double p0Cnorm = 0.0;
      double normalVector[3] = {0.0,0.0,0.0};
      double crossABAp[3] = {0.0,0.0,0.0};
      double crossBCBp[3] = {0.0,0.0,0.0};
      double crossCACp[3] = {0.0,0.0,0.0};
      double directionUnitVector[3] = {0.0,0.0,0.0};
      double dot0 = 0.0;
      double dot1 = 0.0;
      double dot2 = 0.0;

      double normAB = 0.0;
      double normBC = 0.0;
      double normCA = 0.0;
      double ABhat[3] = {0.0,0.0,0.0};
      double BChat[3] = {0.0,0.0,0.0};
      double CAhat[3] = {0.0,0.0,0.0};
      double tAB = 0.0;
      double tBC = 0.0;
      double tCA = 0.0;
      double projP0AB[3] = {0.0,0.0,0.0};
      double projP0BC[3] = {0.0,0.0,0.0};
      double projP0CA[3] = {0.0,0.0,0.0};
      double p0ABdist = 0.0;
      double p0BCdist = 0.0;
      double p0CAdist = 0.0;
      double perpDist = 0.0;
      double signDot0 = 0.0;
      double signDot1 = 0.0;
      double signDot2 = 0.0;
      double totalSigns = 0.0;
      double minDistance = 1e12;
      int nBoundariesCrossed = 0;
      int boundariesCrossed[6] = {0,0,0,0,0,0};
        int minIndex=0;
      double distances[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      double normals[21] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      double closestAll[21] =  {0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,0.0,0.0};

#if GEOM_HASH_SHEATH > 0
  double dr = closeGeomGridr[1] - closeGeomGridr[0];
  double dy = closeGeomGridy[1] - closeGeomGridy[0];
  double dz = closeGeomGridz[1] - closeGeomGridz[0];
  double shift = 0.5;
#if USE_CSR_SHEATH_HASH > 0
      shift = 0;
#endif
  int rInd = floor((x0 - closeGeomGridr[0])/dr + shift);
  int yInd = floor((y - closeGeomGridy[0])/dy + shift);
  int zInd = floor((z - closeGeomGridz[0])/dz + shift);
  int i;
  if(rInd < 0 || rInd >= nR_closeGeom)
    rInd =0;
  if(yInd < 0 || yInd >= nY_closeGeom)
    yInd =0;
  if(zInd < 0 || zInd >= nZ_closeGeom)
    zInd =0;

#if USE_CSR_SHEATH_HASH > 0
      int cell = zInd*nY_closeGeom*nR_closeGeom + yInd*nR_closeGeoms + rInd;
      int hBegin = csrHashPtrs[cell];
      int hEnd = csrHashPtrs[cell+1];
      for (int k = hBegin; k < hEnd; ++k) {
        i = csrHashes[k]; //return index of boundaryVector 
#else
  int in=-1;
  for (int k=0; k< n_closeGeomElements; k++) //n_closeGeomElements
    {
       i = closeGeom[zInd*nY_closeGeom*nR_closeGeom*n_closeGeomElements 
                   + yInd*nR_closeGeom*n_closeGeomElements
                   + rInd*n_closeGeomElements + k];
       //closestBoundaryIndex = i;
       //cout << "closest boundaries to check " << i << endl;
#endif
#else
      int in=-1;
      for (int i=0; i<nLines; i++)
      {
#endif
    //cout << "Z and index " << boundaryVector[i].Z << " " << i << endl;
    if (boundaryVector[i].Z != 0.0)
    {
      ++in;
    //cout << "Z and index " << boundaryVector[i].Z << " " << i << endl;
    a = boundaryVector[i].a;
    b = boundaryVector[i].b;
    c = boundaryVector[i].c;
    d = boundaryVector[i].d;
    plane_norm = boundaryVector[i].plane_norm;
    pointToPlaneDistance0 = (a * p0[0] + b * p0[1] + c * p0[2] + d) / plane_norm;
    //cout << "abcd plane_norm "<< a  << " " << b << " " << c << " " << d << " " << plane_norm << endl;
    //cout << i << endl;// " point to plane dist "  << pointToPlaneDistance0 << endl;
    //pointToPlaneDistance1 = (a*p1[0] + b*p1[1] + c*p1[2] + d)/plane_norm;
    //signPoint0 = copysign(1.0,pointToPlaneDistance0);
    //signPoint1 = copysign(1.0,pointToPlaneDistance1);
    vectorAssign(a / plane_norm, b / plane_norm, c / plane_norm, normalVector);
    //vectorNormalize(normalVector,normalVector);
    //cout << "normal " << normalVector[0] << " " << normalVector[1] << " " << normalVector[2] << endl;
    vectorAssign(p0[0] - pointToPlaneDistance0 * normalVector[0],
                 p0[1] - pointToPlaneDistance0 * normalVector[1],
                 p0[2] - pointToPlaneDistance0 * normalVector[2], p);
   // printf("i %d ind %d abcd %g %g %g %g norm %g pt2planeDist %g normalVector %g %g %g  p %g %g %g \n", 
   // i, in, a,b,c,d,plane_norm, pointToPlaneDistance0, normalVector[0], normalVector[1], normalVector[2], p[0],p[1],p[2]);

   vectorAssign(boundaryVector[i].x1, boundaryVector[i].y1,
                 boundaryVector[i].z1, A);
    vectorAssign(boundaryVector[i].x2, boundaryVector[i].y2,
                 boundaryVector[i].z2, B);
    vectorAssign(boundaryVector[i].x3, boundaryVector[i].y3,
                 boundaryVector[i].z3, C);

    vectorSubtract(B, A, AB);
    vectorSubtract(C, A, AC);
    vectorSubtract(C, B, BC);
    vectorSubtract(A, C, CA);

    vectorSubtract(p, A, Ap);
    vectorSubtract(p, B, Bp);
    vectorSubtract(p, C, Cp);
    vectorCrossProduct(AB, AC, normalVector);
    vectorCrossProduct(AB, Ap, crossABAp);
    vectorCrossProduct(BC, Bp, crossBCBp);
    vectorCrossProduct(CA, Cp, crossCACp);

    signDot0 = copysign(1.0,vectorDotProduct(crossABAp, normalVector));
    signDot1 = copysign(1.0,vectorDotProduct(crossBCBp, normalVector));
    signDot2 = copysign(1.0,vectorDotProduct(crossCACp, normalVector));

         totalSigns = abs(signDot0 + signDot1 + signDot2);

         vectorSubtract(A,p0,p0A);
         vectorSubtract(B,p0,p0B);
         vectorSubtract(C,p0,p0C);
         
         p0Anorm = vectorNorm(p0A);   
         p0Bnorm = vectorNorm(p0B);   
         p0Cnorm = vectorNorm(p0C);
         distances[1] = p0Anorm;   
         distances[2] = p0Bnorm;   
         distances[3] = p0Cnorm;
      closestAll[3] =A[0]; closestAll[4] =A[1]; closestAll[5] =A[2];
      closestAll[6] =B[0]; closestAll[7] =B[1]; closestAll[8] =B[2];
      closestAll[9] =C[0]; closestAll[10] =C[1]; closestAll[11] =C[2];
             normals[3] = p0A[0]/p0Anorm;
             normals[4] = p0A[1]/p0Anorm;
             normals[5] = p0A[2]/p0Anorm;
             normals[6] = p0B[0]/p0Bnorm;
             normals[7] = p0B[1]/p0Bnorm;
             normals[8] = p0B[2]/p0Bnorm;
             normals[9] = p0C[0]/p0Cnorm;
             normals[10] = p0C[1]/p0Cnorm;
             normals[11] = p0C[2]/p0Cnorm;
         //cout << "point to plane " << pointToPlaneDistance0 << endl;
         //cout << "point to ABC " << p0Anorm << " " << p0Bnorm << " " << p0Cnorm << endl;
         //cout << "total Signs " << totalSigns << endl;
         normAB = vectorNorm(AB);
         normBC = vectorNorm(BC);
         normCA = vectorNorm(CA);
         vectorAssign(AB[0]/normAB,AB[1]/normAB,AB[2]/normAB,ABhat);
         vectorAssign(BC[0]/normBC,BC[1]/normBC,BC[2]/normBC,BChat);
         vectorAssign(CA[0]/normCA,CA[1]/normCA,CA[2]/normCA,CAhat);
         
         tAB = vectorDotProduct(p0A,ABhat);
         tBC = vectorDotProduct(p0B,BChat);
         tCA = vectorDotProduct(p0C,CAhat);
         tAB = -1*tAB;
         tBC = -1*tBC;
         tCA = -1*tCA;

         if((tAB > 0.0) && (tAB < normAB))
         {
             vectorScalarMult(tAB,ABhat,projP0AB);
             vectorAdd(A,projP0AB,projP0AB);
             vectorSubtract(projP0AB,p0,p0AB);
             p0ABdist = vectorNorm(p0AB);
             distances[4] = p0ABdist;   
             normals[12] = p0AB[0]/p0ABdist;
             normals[13] = p0AB[1]/p0ABdist;
             normals[14] = p0AB[2]/p0ABdist;
           #if DEBUG_PRINT > 2
             printf("d2bdry: ptcl %d timestep %d i %d in %d (tAB > 0.0) && (tAB < normAB) distances[4] %g\n", 
               ptcl, tstep, i, in, distances[4]);
            #endif
         }
         else
         {
             p0ABdist = 1e12;
             distances[4] = p0ABdist;   
           #if DEBUG_PRINT > 2
             printf("d2bdry: ptcl %d timestep %d i %d in %d p0ABdist distances[4] %g\n",
                    ptcl, tstep, i, in, distances[4]);
           #endif
         } 
         
         
         if((tBC > 0.0) && (tBC < normBC))
         {
             vectorScalarMult(tBC,ABhat,projP0BC);
             vectorAdd(B,projP0BC,projP0BC);
             vectorSubtract(projP0BC,p0,p0BC);
             p0BCdist = vectorNorm(p0BC);
             distances[5] = p0BCdist;   
             normals[15] = p0BC[0]/p0BCdist;
             normals[16] = p0BC[1]/p0BCdist;
             normals[17] = p0BC[2]/p0BCdist;
             
           #if DEBUG_PRINT > 2
             printf("d2bdry: ptcl %d timestep %d  i %d in %d (tBC > 0.0) && (tBC < normBC) distances[5] %g\n", 
               ptcl, tstep, i, in, distances[5]);
             printf("d2bdry: ptcl %d tstep %d i %d in %d tBC %g ABhat %g %g %g projP0BC %g %g %g p0BC %g %g %g \n",
              ptcl, tstep, i, in, tBC, ABhat[0], ABhat[1], ABhat[2], projP0BC[0],projP0BC[1], projP0BC[2], p0BC[0], p0BC[1], p0BC[2]);
           #endif
         }
         else
         {
             p0BCdist = 1e12;
             distances[5] = p0BCdist;   
           #if DEBUG_PRINT > 2
            printf("d2bdry: ptcl %d timestep %d i %d in  %d p0BCdist distances[5] %g\n",
                 ptcl, tstep, i, in, distances[5]);
           #endif
         } 
         
         if((tCA > 0.0) && (tCA < normCA))
         {
             vectorScalarMult(tCA,CAhat,projP0CA);
             vectorAdd(C,projP0CA,projP0CA);
             //cout << "projP0CA " << projP0CA[0] << " " << projP0CA[1] << " " << projP0CA[2] << endl; 
             vectorSubtract(projP0CA,p0,p0CA);
             p0CAdist = vectorNorm(p0CA);
             distances[6] = p0CAdist;   
             normals[18] = p0CA[0]/p0CAdist;
             normals[19] = p0CA[1]/p0CAdist;
             normals[20] = p0CA[2]/p0CAdist;
             //cout << "p0CA " << p0CA[0] << " " << p0CA[1] << " " << p0CA[2] << endl; 
         
            #if  DEBUG_PRINT > 2
              printf("d2bdryDebug: ptcl %d timestep %d  i %d in %d (tCA > 0.0) && (tCA < normCA)  distances[6] %g\n", 
                 ptcl, tstep, i, in, distances[6]);
            #endif
         }
         else
         {
             p0CAdist = 1e12;
             distances[6] = p0CAdist;  

            #if  DEBUG_PRINT > 2
             printf("d2bdryDebug: ptcl %d timestep %d i %d in %d distances[6] %g\n", ptcl, tstep, i, in, distances[6]);
            #endif 
         } 

         if (totalSigns == 3.0)
         {
             //if (fabs(pointToPlaneDistance0) < minDistance)
             //{
                perpDist = abs(pointToPlaneDistance0); 
                //minDistance = fabs(pointToPlaneDistance0);
                //cout << "p " << p[0] << " " << p[1] << " " << p[2] << endl;
                //cout << "p0 " << p0[0] << " " << p0[1] << " " << p0[2] << endl;
                vectorSubtract(p,p0 ,normalVector);
                //cout << "unit vec " << directionUnitVector[0] << " " << directionUnitVector[1] << 
                //    " " << directionUnitVector[2] << endl;
                vectorNormalize(normalVector,normalVector);
                //cout << "unit vec " << directionUnitVector[0] << " " << directionUnitVector[1] << 
                //    " " << directionUnitVector[2] << endl;
                //cout << "perp distance " << endl;
             distances[0] = perpDist;   
             normals[0] = normalVector[0];
             normals[1] = normalVector[1];
             normals[2] = normalVector[2];
             //}

          #if  DEBUG_PRINT > 2
             printf("d2bdryDebug: ptcl %d timestep %d i %d in %d signs3  distances[0] %g\n", ptcl, tstep, i,in, distances[0]);
          #endif
         }
         else
         {
             perpDist = 1e12;
             distances[0] = perpDist;  

            #if  DEBUG_PRINT > 2
             printf("d2bdryDebug: ptcl %d timestep %d  i %d in %d reset distances[0] %g\n", ptcl, tstep, i,in, perpDist);
            #endif 

         }
         int index = 0;
         for(int j = 0; j < 7; j++)
         {
            if(distances[j] < distances[index])
            index = j;              
         }

         if (distances[index] < minDistance)
         {
                 minDistance = distances[index];
                 vectorAssign(normals[index*3], normals[index*3+1],normals[index*3+2], directionUnitVector);
                 //cout << "min dist " << minDistance << endl;
                 //cout << "min normal " << normals[index*3] << " " 
                 //   <<normals[index*3+1] << " " << normals[index*3+2] << endl;
               //closestBoundaryIndex = i;
          closestBoundaryIndex = i;
          minIndex = i;
         }

         #if  DEBUG_PRINT > 1
          printf("d2bdry: ptcl %d tstep %d minDistance %.15f i %d minIndex %d \n", 
               ptcl, tstep, i, minDistance, minIndex);
           double A1[3]={0}, B1[3]={0}, C1[3]={0};
           A1[0] = boundaryVector[i].x1; A1[1]=boundaryVector[i].y1;A1[2]= boundaryVector[i].z1;
           B1[0] = boundaryVector[i].x2; B1[1]=boundaryVector[i].y2;B1[2]= boundaryVector[i].z2;
           C1[0] = boundaryVector[i].x3; C1[1]=boundaryVector[i].y3;C1[2]= boundaryVector[i].z3;
           for(int j=0; j<3; ++j)
             printf("d2bdry: ptcl %d  face %g %g %g : %g %g %g : %g %g %g pt %g %g %g \n",
               ptcl, A1[0], A1[1], A1[2],  B1[0], B1[1], B1[2], C1[0] ,C1[1], C1[2]);
          #endif 
         //cout << "perp dist " << perpDist << endl;
         //cout << "point to AB BC CA " << p0ABdist << " " << p0BCdist << " " << p0CAdist << endl;
        } //materialZ
       } //nLines


       #if  DEBUG_PRINT > 0
         int ii = minIndex;
         A[0] = boundaryVector[ii].x1; A[1]=boundaryVector[ii].y1;A[2]= boundaryVector[ii].z1;
         B[0] = boundaryVector[ii].x2; B[1]=boundaryVector[ii].y2;B[2]= boundaryVector[ii].z2;
         C[0] = boundaryVector[ii].x3; C[1]=boundaryVector[ii].y3;C[2]= boundaryVector[ii].z3;
         double bcc[4];
         bccCoords(A, B, C, p, bcc);
         int allPositive = 0;
         if(bcc[0] >= 0 && bcc[1] >=0 && bcc[2] >= 0 && bcc[3] >= 0)
           allPositive = 1;
         double point[3] = {0};
         for(int j=0; j<3; ++j)
           point[j] = p0[j] - minDistance*directionUnitVector[j]; 
           //if(allPositive) 
         printf("\nd2bdrycalc: ptcl %d minDist %.15f  faceId %d p0 %.15f %.15f %.15f"
             "  point %.15f %.15f %.15f  contain %d dirunitVec %.15f %.15f %.15f\n"
             " face %g %g %g : %g %g %g : %g %g %g\n",
             ptcl, minDistance, minIndex, p0[0], p0[1], p0[2], point[0], point[1], point[2], allPositive,
           directionUnitVector[0], directionUnitVector[1], directionUnitVector[2],
            A[0], A[1], A[2],B[0], B[1], B[2], C[0] ,C[1], C[2]);
       #endif
      //vectorScalarMult(-1.0,directionUnitVector,directionUnitVector);
      //cout << "min dist " << minDistance << endl;
#else //2dGeom     
                
    double Emag = 0.0;
	double fd = 0.0;
	double pot = 0.0;
    int minIndex = 0;
    double minDistance = 1e12;
    int direction_type;
    double tol = 1e12;
    double point1_dist;
    double point2_dist;
    double perp_dist;
    double directionUnitVector[3] = {0.0,0.0,0.0};
    double vectorMagnitude;
    double max = 0.0;
    double min = 0.0;
    double angle = 0.0;
    double Er = 0.0;
    double Et = 0.0;
    double Bfabsfperp = 0.0;
    double distanceToParticle = 0.0;
    int pointLine=0;
//#if EFIELD_INTERP ==1
#if USECYLSYMM > 0
    double x = sqrt(x0*x0 + y*y);
#else
    double x = x0;
#endif 

#if GEOM_HASH_SHEATH > 0
  double dr = closeGeomGridr[1] - closeGeomGridr[0];
  double dz = closeGeomGridz[1] - closeGeomGridz[0];
  int rInd = floor((x - closeGeomGridr[0])/dr + 0.5);
  int zInd = floor((z - closeGeomGridz[0])/dz + 0.5);
  if(rInd >= nR_closeGeom) rInd = nR_closeGeom -1;
  if(zInd >= nZ_closeGeom) zInd = nZ_closeGeom -1;
  if(rInd < 0) rInd = 0;
  if(zInd < 0) zInd = 0;
  int j;
  for (int k=0; k< n_closeGeomElements; k++) //n_closeGeomElements
    {
       j = closeGeom[zInd*nR_closeGeom*n_closeGeomElements + rInd*n_closeGeomElements + k];

#else
    for (int j=0; j< nLines; j++)
    {  //cout << " surface check " << j << endl;
#endif
        //if(j > nLines)
        //{
        //    j = 0;
        //}
       double boundZhere = boundaryVector[j].Z;
       
        if (boundZhere != 0.0)
        {
            point1_dist = sqrt((x - boundaryVector[j].x1)*(x - boundaryVector[j].x1) + 
                    (z - boundaryVector[j].z1)*(z - boundaryVector[j].z1));
            point2_dist = sqrt((x - boundaryVector[j].x2)*(x - boundaryVector[j].x2) + 
                                        (z - boundaryVector[j].z2)*(z - boundaryVector[j].z2));
            perp_dist = (boundaryVector[j].slope_dzdx*x - z + boundaryVector[j].intercept_z)/
                sqrt(boundaryVector[j].slope_dzdx*boundaryVector[j].slope_dzdx + 1.0);   
	
	
          if (abs(boundaryVector[j].slope_dzdx) >= tol*0.75)
	  {
	   perp_dist = x0 - boundaryVector[j].x1;
	  }
	//cout << " x0 z " << x0 << " " << z << " slope " << boundaryVector[j].slope_dzdx << " intercept " << boundaryVector[j].intercept_z << endl;
        
	//cout << " surface check " << j << " point1dist " << point1_dist << " point2_dist " << point2_dist <<  
	   //           " perp_dist " << perp_dist << endl;
            if (point1_dist > point2_dist)
            {
                max = point1_dist;
                min = point2_dist;
            }
            else
            {
                max = point2_dist;
                min = point1_dist;
            }
    //        cout << "p1dist p2dist perpDist " << point1_dist << " " << point2_dist << " " << perp_dist << endl;
            if (boundaryVector[j].length*boundaryVector[j].length + perp_dist*perp_dist >=
                    max*max)
            {
                //boundaryVector[j].distanceToParticle =fabsf( perp_dist);
                distanceToParticle = abs(perp_dist);
                //boundaryVector[j].pointLine = 1;
                pointLine = 1;
            }
            else
            {
                //boundaryVector[j].distanceToParticle = min;
                distanceToParticle = min;
                if (boundaryVector[j].distanceToParticle == point1_dist)
                {
                    pointLine = 2;
                }
                else
                {
                    pointLine = 3;
                }
            }

            if (distanceToParticle < minDistance)
            {
                minDistance = distanceToParticle;
                minIndex = j;
                closestBoundaryIndex = j;
                direction_type = pointLine;
            }
        }
        else
        {
            distanceToParticle = tol;
        }

    }
    if (direction_type == 1)
    {
        if (boundaryVector[minIndex].slope_dzdx == 0)
        {
            directionUnitVector[0] = 0.0;
            directionUnitVector[1] = 0.0;
            directionUnitVector[2] = 1.0 * copysign(1.0,boundaryVector[minIndex].z1 - z);
        }
        else if (abs(boundaryVector[minIndex].slope_dzdx)>= 0.75*tol)
        {
            
            directionUnitVector[0] = boundaryVector[minIndex].x1 - x;
            directionUnitVector[1] = 0.0;
            directionUnitVector[2] = 0.0;
        }
        else
        {
            directionUnitVector[0] = 1.0 * copysign(1.0,(z - boundaryVector[minIndex].intercept_z)/(boundaryVector[minIndex].slope_dzdx) - x0);
            directionUnitVector[1] = 0.0;
            directionUnitVector[2] = 1.0 * copysign(1.0,perp_dist)/(boundaryVector[minIndex].slope_dzdx);
        }
    }
    else if (direction_type == 2)
    {
        directionUnitVector[0] = (boundaryVector[minIndex].x1 - x);
        directionUnitVector[1] = 0.0;
        directionUnitVector[2] = (boundaryVector[minIndex].z1 - z);
    }
    else
    {
        directionUnitVector[0] = (boundaryVector[minIndex].x2 - x);
        directionUnitVector[1] = 0.0;
        directionUnitVector[2] = (boundaryVector[minIndex].z2 - z);
    }

    vectorMagnitude = sqrt(directionUnitVector[0]*directionUnitVector[0] + directionUnitVector[1]*directionUnitVector[1]
                                + directionUnitVector[2]*directionUnitVector[2]);
    directionUnitVector[0] = directionUnitVector[0]/vectorMagnitude;
    directionUnitVector[1] = directionUnitVector[1]/vectorMagnitude;
    directionUnitVector[2] = directionUnitVector[2]/vectorMagnitude;
#endif   

#if BIASED_SURFACE > 0
    pot = boundaryVector[minIndex].potential;
    Emag = pot/(2.0*boundaryVector[minIndex].ChildLangmuirDist)*exp(-minDistance/(2.0*boundaryVector[minIndex].ChildLangmuirDist));
#else 
    angle = boundaryVector[minIndex].angle;    
    fd  =  0.98992 + 5.1220E-03 * angle  -
           7.0040E-04  * pow(angle,2.0) +
           3.3591E-05  * pow(angle,3.0) -
           8.2917E-07  * pow(angle,4.0) +
           9.5856E-09  * pow(angle,5.0) -
           4.2682E-11  * pow(angle,6.0);
    pot = boundaryVector[minIndex].potential;

        double debyeLength = boundaryVector[minIndex].debyeLength;
        double larmorRadius = boundaryVector[minIndex].larmorRadius;
        Emag = pot*(fd/(2.0 * boundaryVector[minIndex].debyeLength)*exp(-minDistance/(2.0 * boundaryVector[minIndex].debyeLength))+ (1.0 - fd)/(boundaryVector[minIndex].larmorRadius)*exp(-minDistance/boundaryVector[minIndex].larmorRadius) );
        double part1 = pot*(fd/(2.0 * boundaryVector[minIndex].debyeLength)*exp(-minDistance/(2.0 * boundaryVector[minIndex].debyeLength)));
        double part2 = pot*(1.0 - fd)/(boundaryVector[minIndex].larmorRadius)*exp(-minDistance/boundaryVector[minIndex].larmorRadius);

#endif
    if(minDistance == 0.0 || boundaryVector[minIndex].larmorRadius == 0.0)
    {
        Emag = 0.0;
        directionUnitVector[0] = 0.0;
        directionUnitVector[1] = 0.0;
        directionUnitVector[2] = 0.0;

    }
        Er = Emag*directionUnitVector[0];
        Et = Emag*directionUnitVector[1];
        E[2] = Emag*directionUnitVector[2];

#if USE3DTETGEOM > 0
            E[0] = Er;
            E[1] = Et;
#else
#if USECYLSYMM > 0
            //if cylindrical geometry
            double theta = atan2(y,x0);
  
            E[0] = cos(theta)*Er - sin(theta)*Et;
            E[1] = sin(theta)*Er + cos(theta)*Et;
#else
            E[0] = Er;
            E[1] = Et;
#endif
#endif

#if  DEBUG_PRINT > 0
  printf("ptcl %d tstep %d E %g %g %g \n", ptcl,tstep, E[0], E[1], E[2]);
  printf("ptcl %d, tstep %d minInd %d angle %g, pot %g, DL %g LR %g \n", ptcl,tstep, minIndex ,
    boundaryVector[minIndex].angle, boundaryVector[minIndex].potential,
    boundaryVector[minIndex].debyeLength,boundaryVector[minIndex].larmorRadius);
#endif
#if  DEBUG_PRINT > 2
   if(true || (detail > 0 && ptcl>=0)) {
     double pt[3]={0}, ptq[3]={0};
     pt[0]= x0; pt[1] = y; pt[2] = z;
     double A[3], B[3], C[3];
     int numBdr = nLines;
     int minI = -1;
     double minD = 1.0e+10;
     double minA[3]={0}, minB[3]={0}, minC[3]={0}, minq[3]={0};
     //numBdr=1;
     for(int i=0; i<numBdr; ++i){
       //int i = minIndex;
       if(boundaryVector[i].Z <= 0)
         continue; 
       A[0] = boundaryVector[i].x1; A[1]=boundaryVector[i].y1;A[2]= boundaryVector[i].z1;
       B[0] = boundaryVector[i].x2; B[1]=boundaryVector[i].y2;B[2]= boundaryVector[i].z2;
       C[0] = boundaryVector[i].x3; C[1]=boundaryVector[i].y3;C[2]= boundaryVector[i].z3;
       closest_point_on_triangle(A,B,C,pt, ptq);
       double mind_test[3];
       vectorSubtract(pt,ptq,mind_test);
       double mind = vectorNorm(mind_test);
       if(minD > mind){
         minD = mind;
         minI = i;
         minA[0]=A[0];minA[1]=A[1];minA[2]=A[2];
         minB[0]=B[0];minB[1]=B[1];minB[2]=B[2];
         minC[0]=C[0];minC[1]=C[1];minC[2]=C[2];
         minq[0] = ptq[0]; minq[1]=ptq[1];minq[2]=ptq[2];
       }
     }

     double pott = boundaryVector[minI].potential;
#if BIASED_SURFACE > 0
     double Efmag = pott/(2.0*boundaryVector[minI].ChildLangmuirDist)*
           exp(-minD/(2.0*boundaryVector[minI].ChildLangmuirDist));
#else
     double a = boundaryVector[minI].angle;    
     double f  =  0.98992 + 5.1220E-03 * a  -
             7.0040E-04  * pow(a,2.0) +
             3.3591E-05  * pow(a,3.0) -
             8.2917E-07  * pow(a,4.0) +
             9.5856E-09  * pow(a,5.0) -
             4.2682E-11  * pow(a,6.0);

      double dl = boundaryVector[minI].debyeLength;
      double lr = boundaryVector[minI].larmorRadius;
      double  Efmag = pott*(f/(2.0 * dl)*exp(-minD/(2.0 * dl))+ (1.0 - f)/(lr)*exp(-minD/lr) );
#endif
     double Ef[3]={0};
     double dirVec[3]={0}, diffV[3]={0}, vN[3]={0};
     vectorSubtract(pt, minq, diffV);
     vectorNormalize(diffV, vN);
     vectorScalarMult(Efmag, vN, Ef);
     printf("calcE: ptcl %d tstep %d pot %.15e CLD %.15e mindist %.15e minIndex %d Emag %.15e dirV %.15e %.15e %.15e \n",
        ptcl, tstep, pot, boundaryVector[minIndex].ChildLangmuirDist, minDistance, minIndex,Emag,
        directionUnitVector[0], directionUnitVector[1] , directionUnitVector[2]);
    printf("calcE: ptcl %d tstep %d pos %.15e %.15e %.15e closest_test %.15e %.15e %.15e\n",
       ptcl, tstep,pt[0], pt[1], pt[2], minq[0], minq[1], minq[2]);
    printf("calcE: ptcl %d tstep %d testFace: %g %g %g : %g %g %g : %g %g %g\n",
        ptcl, tstep, minA[0],minA[1],minA[2],minB[0],minB[1],minB[2], minC[0], minC[1],minC[2]);
    printf("calcE: ptcl %d tstep %d mindist_test %.15e minInd_test %d Efmag %.15e Ef: %.15e %.15e %.15e \n", 
        ptcl,tstep,minD, minI,Efmag, Ef[0], Ef[1], Ef[2]);
   } //debug print
#endif

    if(bdryMinInd)  
      *bdryMinInd = minIndex;
    return minDistance;
}

struct move_boris { 
    Particles *particlesPointer;
    //int& tt;
    Boundary *boundaryVector;
    int nR_Bfield;
    int nZ_Bfield;
    double * BfieldGridRDevicePointer;
    double * BfieldGridZDevicePointer;
    double * BfieldRDevicePointer;
    double * BfieldZDevicePointer;
    double * BfieldTDevicePointer;
    int nR_Efield;
    int nY_Efield;
    int nZ_Efield;
    double * EfieldGridRDevicePointer;
    double * EfieldGridYDevicePointer;
    double * EfieldGridZDevicePointer;
    double * EfieldRDevicePointer;
    double * EfieldZDevicePointer;
    double * EfieldTDevicePointer;
    int nR_closeGeom_sheath;
    int nY_closeGeom_sheath;
    int nZ_closeGeom_sheath;
    int n_closeGeomElements_sheath;
    double* closeGeomGridr_sheath;
    double* closeGeomGridy_sheath;
    double* closeGeomGridz_sheath;
    int* closeGeom_sheath; 
    const double span;
    const int nLines;
    double magneticForce[3];
    double electricForce[3];
    int select = 0;
    int* csrHashPtrs;
    int* csrHashes;
    
    move_boris(Particles *_particlesPointer, double _span, Boundary *_boundaryVector,int _nLines,
            int _nR_Bfield, int _nZ_Bfield,
            double * _BfieldGridRDevicePointer,
            double * _BfieldGridZDevicePointer,
            double * _BfieldRDevicePointer,
            double * _BfieldZDevicePointer,
            double * _BfieldTDevicePointer,
            int _nR_Efield,int _nY_Efield, int _nZ_Efield,
            double * _EfieldGridRDevicePointer,
            double * _EfieldGridYDevicePointer,
            double * _EfieldGridZDevicePointer,
            double * _EfieldRDevicePointer,
            double * _EfieldZDevicePointer,
            double * _EfieldTDevicePointer,
            int _nR_closeGeom, int _nY_closeGeom,int _nZ_closeGeom, int _n_closeGeomElements, 
            double *_closeGeomGridr,double *_closeGeomGridy, double *_closeGeomGridz, 
            int *_closeGeom, int select=0, int* csrHashPtrs=nullptr, int* csrHashes=nullptr)
: particlesPointer(_particlesPointer),
        boundaryVector(_boundaryVector),
        nR_Bfield(_nR_Bfield),
        nZ_Bfield(_nZ_Bfield),
        BfieldGridRDevicePointer(_BfieldGridRDevicePointer),
        BfieldGridZDevicePointer(_BfieldGridZDevicePointer),
        BfieldRDevicePointer(_BfieldRDevicePointer),
        BfieldZDevicePointer(_BfieldZDevicePointer),
        BfieldTDevicePointer(_BfieldTDevicePointer),
        nR_Efield(_nR_Efield),
        nY_Efield(_nY_Efield),
        nZ_Efield(_nZ_Efield),
        EfieldGridRDevicePointer(_EfieldGridRDevicePointer),
        EfieldGridYDevicePointer(_EfieldGridYDevicePointer),
        EfieldGridZDevicePointer(_EfieldGridZDevicePointer),
        EfieldRDevicePointer(_EfieldRDevicePointer),
        EfieldZDevicePointer(_EfieldZDevicePointer),
        EfieldTDevicePointer(_EfieldTDevicePointer),
        nR_closeGeom_sheath(_nR_closeGeom),
        nY_closeGeom_sheath(_nY_closeGeom),
        nZ_closeGeom_sheath(_nZ_closeGeom),
        n_closeGeomElements_sheath(_n_closeGeomElements),
        closeGeomGridr_sheath(_closeGeomGridr),
        closeGeomGridy_sheath(_closeGeomGridy),
        closeGeomGridz_sheath(_closeGeomGridz),
        closeGeom_sheath(_closeGeom),
        span(_span),
        nLines(_nLines),
        magneticForce{0.0, 0.0, 0.0},
        electricForce{0.0, 0.0, 0.0}, 
        select(select), csrHashPtrs(csrHashPtrs), csrHashes(csrHashes){}

CUDA_CALLABLE_MEMBER    
void operator()(size_t indx) { 
#ifdef __CUDACC__
#else
  double initTime = 0.0;
  double interpETime = 0.0;
  double interpBTime = 0.0;
  double operationsTime = 0.0;
#endif
  double v_minus[3]= {0.0, 0.0, 0.0};
  double v_prime[3]= {0.0, 0.0, 0.0};
  double position[3]= {0.0, 0.0, 0.0};
  double v[3]= {0.0, 0.0, 0.0};
  double E[3] = {0.0, 0.0, 0.0};
#if USEPRESHEATHEFIELD > 0
  double PSE[3] = {0.0, 0.0, 0.0};
#endif
  double B[3] = {0.0,0.0,0.0};
  double dt = span;
  double Bmag = 0.0;
  double q_prime = 0.0;
  double coeff = 0.0;
  int nSteps = floor( span / dt + 0.5);
#if USESHEATHEFIELD > 0
  double minDist = 0.0;
  int closestBoundaryIndex;
#endif
 

  int tstep = particlesPointer->tt[indx];
  int ptcl = particlesPointer->index[indx];
  int selectThis = 1;
  if(select)
    selectThis = particlesPointer->storeRnd[indx];

#if ODEINT ==	0 
  if(particlesPointer->hasLeaked[indx] == 0)
	{
	  if(particlesPointer->zprevious[indx] > particlesPointer->leakZ[indx])
	  {
	    particlesPointer->hasLeaked[indx] = 1;
	  }
	}
  double qpE[3] = {0.0,0.0,0.0};
  double vmxB[3] = {0.0,0.0,0.0};
  double vpxB[3] = {0.0,0.0,0.0};
  double qp_vmxB[3] = {0.0,0.0,0.0};
  double c_vpxB[3] = {0.0,0.0,0.0};
  vectorAssign(particlesPointer->xprevious[indx], particlesPointer->yprevious[indx], 
    particlesPointer->zprevious[indx],position);
    
  for ( int s=0; s<nSteps; s++ ) 
  {
    int bdryMinIndex = -1;

#if USESHEATHEFIELD > 0
    minDist = getE(particlesPointer->xprevious[indx], particlesPointer->yprevious[indx], 
      particlesPointer->zprevious[indx], E,boundaryVector,nLines,nR_closeGeom_sheath,  
      nY_closeGeom_sheath,nZ_closeGeom_sheath,  n_closeGeomElements_sheath,closeGeomGridr_sheath, 
      closeGeomGridy_sheath,  closeGeomGridz_sheath,closeGeom_sheath, closestBoundaryIndex,
      csrHashPtrs, csrHashes, ptcl, tstep,  &bdryMinIndex, selectThis);
#endif

#if USEPRESHEATHEFIELD > 0
#if LC_INTERP==3
              
     //double PSE2[3] = {0.0, 0.0, 0.0};
    interp3dVector(PSE,position[0], position[1], position[2],nR_Efield,nY_Efield,nZ_Efield,
             EfieldGridRDevicePointer,EfieldGridYDevicePointer,EfieldGridZDevicePointer,EfieldRDevicePointer,
             EfieldZDevicePointer,EfieldTDevicePointer);
    vectorAdd(E,PSE,E);

#else
    interp2dVector(&PSE[0],position[0], position[1], position[2],nR_Efield,nZ_Efield,
             EfieldGridRDevicePointer,EfieldGridZDevicePointer,EfieldRDevicePointer,
             EfieldZDevicePointer,EfieldTDevicePointer);
         
    vectorAdd(E,PSE,E);
#endif
#endif

   #if  DEBUG_PRINT > 1
   i// if(selectThis){
      int nthStep = particlesPointer->tt[indx];
      float qc = particlesPointer->charge[indx];
      auto minIndex = bdryMinIndex; 
      auto CLD = boundaryVector[minIndex].ChildLangmuirDist;
      auto te = boundaryVector[minIndex].te;
      auto ne = boundaryVector[minIndex].ne;
      printf("Boris1 ptcl %d timestep %d charge %f ne %.15e te %.15e \n",
        ptcl, nthStep-1, qc, ne, te);
    #endif            

    interp2dVector(&B[0],position[0], position[1], position[2],nR_Bfield,nZ_Bfield,
        BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,
        BfieldZDevicePointer,BfieldTDevicePointer); 
    Bmag = vectorNorm(B);
    q_prime = particlesPointer->charge[indx]*1.60217662e-19/(particlesPointer->amu[indx]*1.6737236e-27)*dt*0.5;
    coeff = 2.0*q_prime/(1.0+(q_prime*Bmag)*(q_prime*Bmag));
    vectorAssign(particlesPointer->vx[indx], particlesPointer->vy[indx], particlesPointer->vz[indx],v);
    double v0[] = {v[0],v[1],v[2]};
    vectorScalarMult(q_prime,E,qpE);
    vectorAdd(v,qpE,v_minus);
    this->electricForce[0] = 2.0*qpE[0];
    //cout << "e force " << q_prime << " " << PSE[0] << " " << PSE[1] << " " << PSE[2] << endl;
    this->electricForce[1] = 2.0*qpE[1];
    this->electricForce[2] = 2.0*qpE[2];
    //v_prime = v_minus + q_prime*(v_minus x B)
    vectorCrossProduct(v_minus,B,vmxB);
    vectorScalarMult(q_prime,vmxB,qp_vmxB);
    vectorAdd(v_minus,qp_vmxB,v_prime);       
    this->magneticForce[0] = qp_vmxB[0];
    this->magneticForce[1] = qp_vmxB[1];
    this->magneticForce[2] = qp_vmxB[2];

    //v = v_minus + coeff*(v_prime x B)
    vectorCrossProduct(v_prime, B, vpxB);
    vectorScalarMult(coeff,vpxB,c_vpxB);
    vectorAdd(v_minus, c_vpxB, v);
auto v1_ = v[0];
auto v2_ = v[1];
auto v3_ = v[2];
   //v = v + q_prime*E
    vectorAdd(v,qpE,v);

   #if  DEBUG_PRINT > 1
      printf("Boris2 ptcl %d timestep %d eField %.15e %.15e %.15e bField %.15e %.15e %.15e \n"
        "  ... qPrime %.15e coeff %.15e qpE %.15e %.15e %.15e vmxB %.15e %.15e %.15e " 
        "qp_vmxB %.15e %.15e %.15e  v_prime %.15e %.15e %.15e vpxB %.15e %.15e %.15e "
        " c_vpxB %.15e %.15e %.15e  v_ %.15e %.15e %.15e \n", 
        particlesPointer->index[indx],  particlesPointer->tt[indx]-1, E[0],E[1],E[2], B[0],B[1],B[2]   
        ,q_prime, coeff, qpE[0], qpE[1], qpE[2],vmxB[0], vmxB[1],vmxB[2], qp_vmxB[0], qp_vmxB[1],  qp_vmxB[2],
         v_prime[0], v_prime[1], v_prime[2] , vpxB[0], vpxB[1], vpxB[2], c_vpxB[0],c_vpxB[1],c_vpxB[2],
         v1_, v2_, v3_ );
    #endif


    if(particlesPointer->hitWall[indx] == 0.0)
    {
      //cout << "updating r and v " << endl;
      particlesPointer->x[indx] = position[0] + v[0] * dt;
      particlesPointer->y[indx] = position[1] + v[1] * dt;
      particlesPointer->z[indx] = position[2] + v[2] * dt;
      particlesPointer->vx[indx] = v[0];
      particlesPointer->vy[indx] = v[1];
      particlesPointer->vz[indx] = v[2];    
      
     #if  DEBUG_PRINT > 0
      if(selectThis > 0)
        printf("Boris3  ptcl %d pos %.15f %.15f %.15f => %.15f %.15f %.15f " 
            "vel %.15e %.15e %.15e => %.15e %.15e %.15e B %.15e %.15e %.15e\n", 
            particlesPointer->index[indx],position[0], position[1], position[2],
            particlesPointer->x[indx], particlesPointer->y[indx], 
            particlesPointer->z[indx], v0[0], v0[1], v0[2], v[0],v[1], v[2], B[0], B[1], B[2]);              
     #endif
    }
  }
#endif

#if ODEINT == 1
  double m = particlesPointer->amu[indx]*1.6737236e-27;
  double q_m = particlesPointer->charge[indx]*1.60217662e-19/m;
  double r[3]= {0.0, 0.0, 0.0};
  double r2[3]= {0.0, 0.0, 0.0};
  double r3[3]= {0.0, 0.0, 0.0};
  double r4[3]= {0.0, 0.0, 0.0};
  double v2[3]= {0.0, 0.0, 0.0};
  double v3[3]= {0.0, 0.0, 0.0};
  double v4[3]= {0.0, 0.0, 0.0};
  double k1r[3]= {0.0, 0.0, 0.0};
  double k2r[3]= {0.0, 0.0, 0.0};
  double k3r[3]= {0.0, 0.0, 0.0};
  double k4r[3]= {0.0, 0.0, 0.0};
  double k1v[3]= {0.0, 0.0, 0.0};
  double k2v[3]= {0.0, 0.0, 0.0};
  double k3v[3]= {0.0, 0.0, 0.0};
  double k4v[3]= {0.0, 0.0, 0.0};
  double dtqm = dt*q_m;
  double vxB[3] = {0.0,0.0,0.0};
  double EplusvxB[3] = {0.0,0.0,0.0};
  double halfKr[3] = {0.0,0.0,0.0};
  double halfKv[3] = {0.0,0.0,0.0};
  double half = 0.5;
  v[0] = particlesPointer->vx[indx];
  v[1] = particlesPointer->vy[indx];
  v[2] = particlesPointer->vz[indx];

  r[0] = particlesPointer->xprevious[indx];
  r[1] = particlesPointer->yprevious[indx];
  r[2] = particlesPointer->zprevious[indx];
#ifdef __CUDACC__
#else
#endif
  for ( int s=0; s<nSteps; s++ ) 
  {
#ifdef __CUDACC__
#else
#endif
#if USESHEATHEFIELD > 0
      minDist = getE(r[0],r[1],r[2],E,boundaryVector,nLines);
#endif
#if USEPRESHEATHEFIELD > 0
      interparticlesPointer->dVector(&particlesPointer->E[0],particlesPointer->xparticlesPointer->evious,particlesPointer->yparticlesPointer->evious,particlesPointer->zparticlesPointer->evious,nR_Efield,nZ_Efield,
          EfieldGridRDeviceparticlesPointer->inter,EfieldGridZDeviceparticlesPointer->inter,EfieldRDeviceparticlesPointer->inter,
          EfieldZDeviceparticlesPointer->inter,EfieldTDeviceparticlesPointer->inter);
                 
      vectorAdd(E,particlesPointer->E,E);
#endif              
#ifdef __CUDACC__
#else
#endif
      interp2dVector(&B[0],r[0],r[1],r[2],nR_Bfield,nZ_Bfield,
               BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,
               BfieldZDevicePointer,BfieldTDevicePointer);        
#ifdef __CUDACC__
#else
#endif
      //k1r = dt*v
      vectorScalarMult(dt,v,k1r);
      /*
      k1r[0] = v[0]*dt;
      k1r[1] = v[1]*dt;
      k1r[2] = v[2]*dt;
      */
      //k1v = dt*q_m * (E + (v x B))
      vectorCrossProduct(v,B,vxB);
      vectorAdd(E,vxB,EplusvxB);
      vectorScalarMult(dtqm,EplusvxB,k1v);
      /*
      k1v[0] = dt*q_m*(E[0] + (v[1]*B[2] - v[2]*B[1]));
      k1v[1] = dt*q_m*(E[1] + (v[2]*B[0] - v[0]*B[2]));
      k1v[2] = dt*q_m*(E[2] + (v[0]*B[1] - v[1]*B[0]));
      */
      //r2 = r + 0.5*k1r
      vectorScalarMult(half,k1r,halfKr);
      vectorAdd(r,k1r,r2);
      /*
      r2[0] = r[0] + k1r[0]*0.5;
      r2[1] = r[1] + k1r[1]*0.5;
      r2[2] = r[2] + k1r[2]*0.5;
      */

      //v2 = v + 0.5*k1v
      vectorScalarMult(half,k1v,halfKv);
      vectorAdd(v, halfKv,v2);
          /*
      v2[0] = v[0] + k1v[0]*0.5;
      v2[1] = v[1] + k1v[1]*0.5;
      v2[2] = v[2] + k1v[2]*0.5;
      */
#ifdef __CUDACC__
#else
#endif

#if USESHEATHEFIELD > 0	  
      minDist = getE(r2[0],r2[1],r2[2],E,boundaryVector,nLines);
#endif
#if USEPRESHEATHEFIELD > 0
      interparticlesPointer->dVector(&particlesPointer->E[0],particlesPointer->xparticlesPointer->evious,particlesPointer->yparticlesPointer->evious,particlesPointer->zparticlesPointer->evious,nR_Efield,nZ_Efield,
               EfieldGridRDeviceparticlesPointer->inter,EfieldGridZDeviceparticlesPointer->inter,EfieldRDeviceparticlesPointer->inter,
               EfieldZDeviceparticlesPointer->inter,EfieldTDeviceparticlesPointer->inter);
      vectorAdd(E,particlesPointer->E,E);
#endif              
#ifdef __CUDACC__
#else
#endif


      interp2dVector(&B[0],r2[0],r2[1],r2[2],nR_Bfield,nZ_Bfield,
             BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,
             BfieldZDevicePointer,BfieldTDevicePointer);        
#ifdef __CUDACC__
#else
#endif
      //k2r = dt*v2
      vectorScalarMult(dt,v2,k2r);
      /*
      k2r[0] = v2[0]*dt;
      k2r[1] = v2[1]*dt;
      k2r[2] = v2[2]*dt;
      */
      //k2v = dt*q_m*(E + (v x B))
      vectorCrossProduct(v2,B,vxB);
      vectorAdd(E,vxB,EplusvxB);
      vectorScalarMult(dtqm,EplusvxB,k2v);
      /*
      k2v[0] = dt*q_m*(E[0] + (v2[1]*B[2] - v2[2]*B[1]));
      k2v[1] = dt*q_m*(E[1] + (v2[2]*B[0] - v2[0]*B[2]));
      k2v[2] = dt*q_m*(E[2] + (v2[0]*B[1] - v2[1]*B[0]));
      */
      //r3 = r + 0.5*k2r
      vectorScalarMult(half,k2r,halfKr);
      vectorAdd(r,k2r,r3);
      /*
      r3[0] = r[0] + k2r[0]*0.5;
      r3[1] = r[1] + k2r[1]*0.5;
      r3[2] = r[2] + k2r[2]*0.5;
      */
      //v3 = v + 0.5*k2v
      vectorScalarMult(half,k2v,halfKv);
      vectorAdd(v, halfKv,v3);
      /*
      v3[0] = v[0] + k2v[0]*0.5;
      v3[1] = v[1] + k2v[1]*0.5;
      v3[2] = v[2] + k2v[2]*0.5;
      */
#ifdef __CUDACC__
#else
#endif

#if USESHEATHEFIELD > 0	  
      minDist = getE(r3[0],r3[1],r3[2],E,boundaryVector,nLines);
#endif
#if USEPRESHEATHEFIELD > 0
      interparticlesPointer->dVector(&particlesPointer->E[0],particlesPointer->xparticlesPointer->evious,particlesPointer->yparticlesPointer->evious,particlesPointer->zparticlesPointer->evious,nR_Efield,nZ_Efield,
               EfieldGridRDeviceparticlesPointer->inter,EfieldGridZDeviceparticlesPointer->inter,EfieldRDeviceparticlesPointer->inter,
               EfieldZDeviceparticlesPointer->inter,EfieldTDeviceparticlesPointer->inter);
      vectorAdd(E,particlesPointer->E,E);
#endif              

#ifdef __CUDACC__
#else
#endif
      interp2dVector(&B[0],r3[0],r3[1],r3[2],nR_Bfield,nZ_Bfield,
                 BfieldGridRDevicePointer,BfieldGridZDevicePointer,BfieldRDevicePointer,
                 BfieldZDevicePointer,BfieldTDevicePointer);        
                
#ifdef __CUDACC__
#else
#endif
      //k3r = dt*v3
      vectorScalarMult(dt,v3,k3r);
      /*
      k3r[0] = v3[0]*dt;
      k3r[1] = v3[1]*dt;
      k3r[2] = v3[2]*dt;
      */
      //k3v = dt*qm*(E + (v x B))
      vectorCrossProduct(v3,B,vxB);
      vectorAdd(E,vxB,EplusvxB);
      vectorScalarMult(dtqm,EplusvxB,k3v);
      /*
      k3v[0] = dt*q_m*(E[0] + (v3[1]*B[2] - v3[2]*B[1]));
      k3v[1] = dt*q_m*(E[1] + (v3[2]*B[0] - v3[0]*B[2]));
      k3v[2] = dt*q_m*(E[2] + (v3[0]*B[1] - v3[1]*B[0]));
      */
      //r4 = r + k3r
      vectorAdd(r, k3r,r4);
      /*
      r4[0] = r[0] + k3r[0];
      r4[1] = r[1] + k3r[1];
      r4[2] = r[2] + k3r[2];
      */
      //v4 = v + k3v
      vectorAdd(v, k3v, v4);
          /*
      v4[0] = v[0] + k3v[0];
      v4[1] = v[1] + k3v[1];
      v4[2] = v[2] + k3v[2];
      */
#ifdef __CUDACC__
#else
#endif

#if USESHEATHEFIELD > 0            
      minDist = getE(r4[0],r4[1],r4[2],E,boundaryVector,nLines);
#endif
#if USEPRESHEATHEFIELD > 0
      interp2dVector(&particlesPointer->E[0],particlesPointer->xparticlesPointer->evious,particlesPointer->yparticlesPointer->evious,particlesPointer->zparticlesPointer->evious,nR_Efield,nZ_Efield,
               EfieldGridRDeviceparticlesPointer->inter,EfieldGridZDeviceparticlesPointer->inter,EfieldRDeviceparticlesPointer->inter,
               EfieldZDeviceparticlesPointer->inter,EfieldTDeviceparticlesPointer->inter);
      vectorAdd(E,particlesPointer->E,E);
#endif              
#ifdef __CUDACC__
#else
#endif

      interp2dVector(&B[0],r4[0],r4[1],r4[2],nR_Bfield,nZ_Bfield,
                        BfieldGridRDevicePointer,BfieldGridZDevicePointer,
                        BfieldRDevicePointer,BfieldZDevicePointer,BfieldTDevicePointer);        
#ifdef __CUDACC__
#else
#endif

      //k4r = dt*v4
      vectorScalarMult(dt,v4,k4r);
      /*
      k4r[0] = v4[0]*dt;
      k4r[1] = v4[1]*dt;
      k4r[2] = v4[2]*dt;
      */
      //k4v = dt*q_m*(E + (v x B))
      vectorCrossProduct(v4,B,vxB);
      vectorAdd(E,vxB,EplusvxB);
      vectorScalarMult(dtqm,EplusvxB,k4v);
      /*
      k4v[0] = dt*q_m*(E[0] + (v4[1]*B[2] - v4[2]*B[1]));
      k4v[1] = dt*q_m*(E[1] + (v4[2]*B[0] - v4[0]*B[2]));
      k4v[2] = dt*q_m*(E[2] + (v4[0]*B[1] - v4[1]*B[0]));
      */
      particlesPointer->x[indx] = r[0] + (k1r[0] + 2*k2r[0] + 2*k3r[0] + k4r[0])/6;
      particlesPointer->y[indx] = r[1] + (k1r[1] + 2*k2r[1] + 2*k3r[1] + k4r[1])/6;
      particlesPointer->z[indx] = r[2] + (k1r[2] + 2*k2r[2] + 2*k3r[2] + k4r[2])/6;
      particlesPointer->vx[indx] = v[0] + (k1v[0] + 2*k2v[0] + 2*k3v[0] + k4v[0])/6;
      particlesPointer->vy[indx] = v[1] + (k1v[1] + 2*k2v[1] + 2*k3v[1] + k4v[1])/6;
      particlesPointer->vz[indx] = v[2] + (k1v[2] + 2*k2v[2] + 2*k3v[2] + k4v[2])/6;
#ifdef __CUDACC__
#else
#endif
    }
#endif
  } 
};

#endif

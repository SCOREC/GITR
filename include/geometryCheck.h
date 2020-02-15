#ifndef _GEOM_
#define _GEOM_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
using namespace std;
#endif

#include "Boundary.h"
#include "Particles.h"
#include "Surfaces.h"
#include "surfaceModel.h"
#include <cmath>

CUDA_CALLABLE_MEMBER_DEVICE
double findT(double x0, double x1, double y0, double y1, double intersectionx) {

  double a, b, c, a1, a2, t=0, discriminant, realPart, imaginaryPart;
  a = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
  b = 2.0 * x0 * (x1 - x0) + 2.0 * y0 * (y1 - y0);
  c = x0 * x0 + y0 * y0 - intersectionx * intersectionx;
  discriminant = b * b - 4 * a * c;

  if (discriminant > 0) {
    a1 = (-b + sqrt(discriminant)) / (2 * a);
    a2 = (-b - sqrt(discriminant)) / (2 * a);
    //cout << "Roots are real and different." << endl;
    //cout << "a1 = " << a1 << endl;
    //cout << "a2 = " << a2 << endl;
    t = min(abs(a1),abs(a2));
  }

  else if (discriminant == 0) {
    // cout << "Roots are real and same." << endl;
    a1 = (-b + sqrt(discriminant)) / (2 * a);
    // cout << "a1 = a2 =" << a1 << endl;
  }

  else {
    realPart = -b / (2 * a);
    imaginaryPart = sqrt(-discriminant) / (2 * a);
    // cout << "Roots are complex and different."  << endl;
    // cout << "a1 = " << realPart << "+" << imaginaryPart << "i" << endl;
    // cout << "a2 = " << realPart << "-" << imaginaryPart << "i" << endl;
  }

  return t;
}
struct geometry_check {
  Particles *particlesPointer;
  const int nLines;
  Boundary *boundaryVector;
  Surfaces *surfaces;
  double dt;
  // int& tt;
  int nHashes;
  int *nR_closeGeom;
  int *nY_closeGeom;
  int *nZ_closeGeom;
  int *n_closeGeomElements;
  double *closeGeomGridr;
  double *closeGeomGridy;
  double *closeGeomGridz;
  int *closeGeom;
  int nEdist;
  double E0dist;
  double Edist;
  int nAdist;
  double A0dist;
  double Adist;

  geometry_check(Particles *_particlesPointer, int _nLines,
                 Boundary *_boundaryVector, Surfaces *_surfaces, double _dt,
                 int _nHashes, int *_nR_closeGeom, int *_nY_closeGeom,
                 int *_nZ_closeGeom, int *_n_closeGeomElements,
                 double *_closeGeomGridr, double *_closeGeomGridy,
                 double *_closeGeomGridz, int *_closeGeom, int _nEdist,
                 double _E0dist, double _Edist, int _nAdist, double _A0dist,
                 double _Adist)
      :

        particlesPointer(_particlesPointer), nLines(_nLines),
        boundaryVector(_boundaryVector), surfaces(_surfaces), dt(_dt),
        nHashes(_nHashes), nR_closeGeom(_nR_closeGeom),
        nY_closeGeom(_nY_closeGeom), nZ_closeGeom(_nZ_closeGeom),
        n_closeGeomElements(_n_closeGeomElements),
        closeGeomGridr(_closeGeomGridr), closeGeomGridy(_closeGeomGridy),
        closeGeomGridz(_closeGeomGridz), closeGeom(_closeGeom), nEdist(_nEdist),
        E0dist(_E0dist), Edist(_Edist), nAdist(_nAdist), A0dist(_A0dist),
        Adist(_Adist) {}

  CUDA_CALLABLE_MEMBER_DEVICE
  void operator()(size_t indx) const {
    // cout << "geometry check particle x" << particlesPointer->x[indx] <<
    // particlesPointer->x[indx]previous <<endl; cout << "geometry
    // check particle y" << particlesPointer->y[indx] <<
    // particlesPointer->y[indx]previous <<endl; cout << "geometry
    // check particle z" << particlesPointer->z[indx] <<
    // particlesPointer->z[indx]previous <<endl; cout << "geometry
    // check particle hitwall" << p.hitWall <<endl;
    if (particlesPointer->hitWall[indx] == 0.0) {
      int hitSurface = 0;
      double x = particlesPointer->x[indx];
      double y = particlesPointer->y[indx];
      double z = particlesPointer->z[indx];
      double xprev = particlesPointer->xprevious[indx];
      double yprev = particlesPointer->yprevious[indx];
      double zprev = particlesPointer->zprevious[indx];
      double dpath =
          sqrt((x - xprev) * (x - xprev) + (y - yprev) * (y - yprev) +
                    (z - zprev) * (z - zprev));
#if FLUX_EA > 0
      double dEdist = (Edist - E0dist) / static_cast<double>(nEdist);
      double dAdist = (Adist - A0dist) / static_cast<double>(nAdist);
      int AdistInd = 0;
      int EdistInd = 0;
#endif
      double vxy[3] = {0.0};
      double vtheta[3] = {0.0};
#if USECYLSYMM > 0
      if (boundaryVector[nLines].periodic) // if periodic
      {
        double pi = 3.14159265;
        double theta =
            atan2(particlesPointer->y[indx], particlesPointer->x[indx]);
        double thetaPrev = atan2(particlesPointer->yprevious[indx],
                                 particlesPointer->xprevious[indx]);
        // double vtheta =
        // atan2(particlesPointer->vy[indx],particlesPointer->vx[indx]);
        double rprev = sqrt(particlesPointer->xprevious[indx] *
                               particlesPointer->xprevious[indx] +
                           particlesPointer->yprevious[indx] *
                               particlesPointer->yprevious[indx]);
        double r = sqrt(particlesPointer->x[indx] * particlesPointer->x[indx] +
                       particlesPointer->y[indx] * particlesPointer->y[indx]);
        double rHat[3] = {0.0};
        double vr[3] = {0.0};
        rHat[0] = particlesPointer->x[indx];
        rHat[1] = particlesPointer->y[indx];

        vectorNormalize(rHat, rHat);
        vxy[0] = particlesPointer->vx[indx];
        vxy[1] = particlesPointer->vy[indx];
        vectorScalarMult(vectorDotProduct(rHat, vxy), rHat, vr);
        double vrMag = vectorNorm(vr);
        vectorSubtract(vxy, vr, vtheta);
        double vthetaMag = vectorNorm(vtheta);
        double vx0 = 0.0;
        double vy0 = 0.0;
        if (theta <= boundaryVector[nLines].y1) {
          particlesPointer->xprevious[indx] =
              r * cos(boundaryVector[nLines].y2 + theta);
          particlesPointer->yprevious[indx] =
              r * sin(boundaryVector[nLines].y2 + theta);
          particlesPointer->x[indx] =
              rprev * cos(boundaryVector[nLines].y2 + theta);
          particlesPointer->y[indx] =
              rprev * sin(boundaryVector[nLines].y2 + theta);

          vx0 = vrMag * cos(boundaryVector[nLines].y2 + theta) -
                vthetaMag * sin(boundaryVector[nLines].y2 + theta);
          vy0 = vrMag * sin(boundaryVector[nLines].y2 + theta) +
                vthetaMag * cos(boundaryVector[nLines].y2 + theta);
          particlesPointer->vx[indx] = vx0;
          particlesPointer->vy[indx] = vy0;
        } else if (theta >= boundaryVector[nLines].y2) {
          particlesPointer->xprevious[indx] =
              rprev * cos(thetaPrev - boundaryVector[nLines].y2);
          particlesPointer->yprevious[indx] =
              rprev * sin(thetaPrev - boundaryVector[nLines].y2);
          particlesPointer->x[indx] =
              r * cos(theta - boundaryVector[nLines].y2);
          particlesPointer->y[indx] =
              r * sin(theta - boundaryVector[nLines].y2);

          vx0 = vrMag * cos(theta - boundaryVector[nLines].y2) -
                vthetaMag * sin(theta - boundaryVector[nLines].y2);
          vy0 = vrMag * sin(theta - boundaryVector[nLines].y2) +
                vthetaMag * cos(theta - boundaryVector[nLines].y2);
          particlesPointer->vx[indx] = vx0;
          particlesPointer->vy[indx] = vy0;
        }
      }
#else
      if (boundaryVector[nLines].periodic) {
        if (particlesPointer->y[indx] < boundaryVector[nLines].y1) {
          particlesPointer->y[indx] =
              boundaryVector[nLines].y2 -
              (boundaryVector[nLines].y1 - particlesPointer->y[indx]);
          particlesPointer->yprevious[indx] =
              boundaryVector[nLines].y2 -
              (boundaryVector[nLines].y1 - particlesPointer->y[indx]);

        } else if (particlesPointer->y[indx] > boundaryVector[nLines].y2) {
          particlesPointer->y[indx] =
              boundaryVector[nLines].y1 +
              (particlesPointer->y[indx] - boundaryVector[nLines].y2);
          particlesPointer->yprevious[indx] =
              boundaryVector[nLines].y1 +
              (particlesPointer->y[indx] - boundaryVector[nLines].y2);
        }
      }
#endif
#if USE3DTETGEOM > 0

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
      double A[3] = {0.0, 0.0, 0.0};
      double B[3] = {0.0, 0.0, 0.0};
      double C[3] = {0.0, 0.0, 0.0};
      double AB[3] = {0.0, 0.0, 0.0};
      double AC[3] = {0.0, 0.0, 0.0};
      double BC[3] = {0.0, 0.0, 0.0};
      double CA[3] = {0.0, 0.0, 0.0};
      double p[3] = {0.0, 0.0, 0.0};
      double Ap[3] = {0.0, 0.0, 0.0};
      double Bp[3] = {0.0, 0.0, 0.0};
      double Cp[3] = {0.0, 0.0, 0.0};
      double normalVector[3] = {0.0, 0.0, 0.0};
      double crossABAp[3] = {0.0, 0.0, 0.0};
      double crossBCBp[3] = {0.0, 0.0, 0.0};
      double crossCACp[3] = {0.0, 0.0, 0.0};
      double signDot0 = 0.0;
      double signDot1 = 0.0;
      double signDot2 = 0.0;
      double totalSigns = 0.0;
      int nBoundariesCrossed = 0;
      int boundariesCrossed[6] = {0, 0, 0, 0, 0, 0};
      /*
      if(particlesPointer->xprevious[indx] < 0 ||
      particlesPointer->yprevious[indx] < 0 ||
              particlesPointer->zprevious[indx]< 0)
      {
      cout << "pos " << particlesPointer->xprevious[indx] << " "
          << particlesPointer->yprevious[indx]
          << " " << particlesPointer->zprevious[indx]  << endl;
      }
       */

      double p0[3] = {particlesPointer->xprevious[indx],
                     particlesPointer->yprevious[indx],
                     particlesPointer->zprevious[indx]};
      double p1[3] = {particlesPointer->x[indx], particlesPointer->y[indx],
                     particlesPointer->z[indx]};
#if GEOM_HASH > 0
      // find which hash
      int nHash = 0;
      int rHashInd = 0;
      int yHashInd = 0;
      int zHashInd = 0;
      int rHashInd1 = 0;
      int yHashInd1 = 0;
      int zHashInd1 = 0;
      double r_position = particlesPointer->xprevious[indx];
      for (int i = 0; i < nHashes; i++) {
        rHashInd1 = nR_closeGeom[i] - 1;
        yHashInd1 = nY_closeGeom[i] - 1;
        zHashInd1 = nZ_closeGeom[i] - 1;
        if (i > 0)
          rHashInd = nR_closeGeom[i - 1];
        if (i > 0)
          yHashInd = nY_closeGeom[i - 1];
        if (i > 0)
          zHashInd = nZ_closeGeom[i - 1];
        if (i > 0)
          rHashInd1 = nR_closeGeom[i - 1] + nR_closeGeom[i] - 1;
        if (i > 0)
          yHashInd1 = nY_closeGeom[i - 1] + nY_closeGeom[i] - 1;
        if (i > 0)
          zHashInd1 = nZ_closeGeom[i - 1] + nZ_closeGeom[i] - 1;
        // cout << "rpos " <<rHashInd<< " " << rHashInd1 << " " <<
        // closeGeomGridr[rHashInd] << " "
        //          << closeGeomGridr[rHashInd1] << endl;
        // cout << "ypos " << closeGeomGridy[yHashInd] << " "
        //          << closeGeomGridy[yHashInd1] << endl;
        // cout << "zpos " << closeGeomGridz[zHashInd] << " "
        //         << closeGeomGridz[zHashInd1] << endl;
        if (r_position < closeGeomGridr[rHashInd1] &&
            r_position > closeGeomGridr[rHashInd] &&
            particlesPointer->yprevious[indx] < closeGeomGridy[yHashInd1] &&
            particlesPointer->yprevious[indx] > closeGeomGridy[yHashInd] &&
            particlesPointer->zprevious[indx] < closeGeomGridz[zHashInd1] &&
            particlesPointer->zprevious[indx] > closeGeomGridz[zHashInd]) {
          nHash = i;
        }
      }
      // cout << "nHash " << nHash << endl;
      rHashInd = 0;
      yHashInd = 0;
      zHashInd = 0;
      if (nHash > 0)
        rHashInd = nR_closeGeom[nHash - 1];
      if (nHash > 0)
        yHashInd = nY_closeGeom[nHash - 1];
      if (nHash > 0)
        zHashInd = nZ_closeGeom[nHash - 1];
      double dr = closeGeomGridr[rHashInd + 1] - closeGeomGridr[rHashInd];
      double dz = closeGeomGridz[zHashInd + 1] - closeGeomGridz[zHashInd];
      double dy = closeGeomGridy[yHashInd + 1] - closeGeomGridy[yHashInd];
      int rInd = floor((r_position - closeGeomGridr[rHashInd]) / dr + 0.5);
      int zInd = floor(
          (particlesPointer->zprevious[indx] - closeGeomGridz[zHashInd]) / dz +
          0.5);
      int i = 0;
      int yInd = floor(
          (particlesPointer->yprevious[indx] - closeGeomGridy[yHashInd]) / dy +
          0.5);
      // cout << "rHashInd " << rHashInd << " " << yHashInd << " " <<
      // zHashInd << endl; cout << "dr dy dz " << dr << " " << dy << "
      // " << dz << endl; cout << "rind y z " << rInd << " " << yInd <<
      // " " << zInd << endl;
      if (rInd < 0 || yInd < 0 || zInd < 0) {
        rInd = 0;
        yInd = 0;
        zInd = 0;
#if USE_CUDA
#else
        // cout << "WARNING: particle outside of geometry hash range (low)"
        // << endl;
#endif
      } else if (rInd > nR_closeGeom[nHash] - 1 ||
                 yInd > nY_closeGeom[nHash] - 1 ||
                 zInd > nZ_closeGeom[nHash] - 1) {
        rInd = 0;
        yInd = 0;
        zInd = 0;
      }
      int buffIndx = 0;
      if (nHash > 0)
        buffIndx = nR_closeGeom[nHash - 1] * nY_closeGeom[nHash - 1] *
                   nZ_closeGeom[nHash - 1] * n_closeGeomElements[nHash - 1];
      // cout << "buff Index " << buffIndx << endl;
      for (int j = 0; j < n_closeGeomElements[nHash]; j++) {
        i = closeGeom[buffIndx +
                      zInd * nY_closeGeom[nHash] * nR_closeGeom[nHash] *
                          n_closeGeomElements[nHash] +
                      yInd * nR_closeGeom[nHash] * n_closeGeomElements[nHash] +
                      rInd * n_closeGeomElements[nHash] + j];
        // cout << "i's " << i << endl;
#else
      for (int i = 0; i < nLines; i++) {
#endif
        a = boundaryVector[i].a;
        b = boundaryVector[i].b;
        c = boundaryVector[i].c;
        d = boundaryVector[i].d;
        plane_norm = boundaryVector[i].plane_norm;
        pointToPlaneDistance0 =
            (a * p0[0] + b * p0[1] + c * p0[2] + d) / plane_norm;
        pointToPlaneDistance1 =
            (a * p1[0] + b * p1[1] + c * p1[2] + d) / plane_norm;
        // cout << "plane coeffs "<< i << " " << a << " " << b << " " << c
        // << " " << d << " " << plane_norm << endl; cout << "point to
        // plane dists "<< i << " " << pointToPlaneDistance0 << " " <<
        // pointToPlaneDistance1 << endl;
        signPoint0 = copysign(1.0, pointToPlaneDistance0);
        signPoint1 = copysign(1.0, pointToPlaneDistance1);

        if (signPoint0 != signPoint1) {
          t = -(a * p0[0] + b * p0[1] + c * p0[2] + d) /
              (a * (p1[0] - p0[0]) + b * (p1[1] - p0[1]) + c * (p1[2] - p0[2]));
          vectorAssign(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]),
                       p0[2] + t * (p1[2] - p0[2]), p);
          // cout << " p " << p[0] << " " << p[1] << " " << p[2] <<
          // endl;
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
          // cout << "AB " << AB[0] << " " << AB[1] << " " << AB[2] <<
          // endl; cout << "Ap " << Ap[0] << " " << Ap[1] << " " <<
          // Ap[2] << endl; cout << "BC " << BC[0] << " " << BC[1] << "
          // " << BC[2] << endl; cout << "Bp " << Bp[0] << " " << Bp[1]
          // << " " << Bp[2] << endl; cout << "CA " << CA[0] << " " <<
          // CA[1] << " " << CA[2] << endl; cout << "Cp " << Cp[0] << "
          // " << Cp[1] << " " << Cp[2] << endl;
          signDot0 =
              copysign(1.0, vectorDotProduct(crossABAp, normalVector));
          signDot1 =
              copysign(1.0, vectorDotProduct(crossBCBp, normalVector));
          signDot2 =
              copysign(1.0, vectorDotProduct(crossCACp, normalVector));
          totalSigns = 1.0 * abs(signDot0 + signDot1 + signDot2);
          // cout << "signdots " << signDot0 << " " << signDot1 << " " <<
          // signDot2 << " " << totalSigns << " " << (totalSigns
          // == 3.0)<<endl;

          // cout << "before loop totalSigns hitSurface " << totalSigns <<
          // " " << hitSurface << endl;
          hitSurface = 0;
          if (totalSigns == 3.0) {
            // cout << "in loop totalSigns hitSurface " << totalSigns << "
            // " << hitSurface << endl;
            hitSurface = 1;
          }
          if (vectorNorm(crossABAp) == 0.0 || vectorNorm(crossBCBp) == 0.0 ||
              vectorNorm(crossCACp) == 0.0) {
            hitSurface = 1;
          }
          // cout << "totalSigns hitSurface " << totalSigns << " " <<
          // hitSurface << endl;
          if (hitSurface == 1) {
            boundariesCrossed[nBoundariesCrossed] = i;
            // cout << "boundary crossed " << i << endl;
            nBoundariesCrossed++;
            particlesPointer->hitWall[indx] = 1.0;
            particlesPointer->xprevious[indx] = p[0];
            particlesPointer->yprevious[indx] = p[1];
            particlesPointer->zprevious[indx] = p[2];
            particlesPointer->x[indx] = p[0];
            particlesPointer->y[indx] = p[1];
            particlesPointer->z[indx] = p[2];
            particlesPointer->wallHit[indx] = i;
            //#if USESURFACEMODEL == 0
            //  #if USE_CUDA > 0
            //    atomicAdd(&boundaryVector[i].impacts,
            //    particlesPointer->weight[indx]);
            //  #else
            //    boundaryVector[i].impacts = boundaryVector[i].impacts +
            //    particlesPointer->weight[indx];
            //  #endif
            //#endif
            double E0 =
                0.5 * particlesPointer->amu[indx] * 1.66e-27 *
                (particlesPointer->vx[indx] * particlesPointer->vx[indx] +
                 particlesPointer->vy[indx] * particlesPointer->vy[indx] +
                 particlesPointer->vz[indx] * particlesPointer->vz[indx]) /
                1.602e-19;
            // cout << "Energy of particle that hit surface " << E0 <<
            // endl;
#if USE_CUDA > 0
            // double surfNormal[3] = {0.0};
            // double partNormal[3] = {0.0};
            // double partDotNormal = 0.0;
            // partNormal[0] = particlesPointer->vx[indx];
            // partNormal[1] = particlesPointer->vy[indx];
            // partNormal[2] = particlesPointer->vz[indx];
            // getBoundaryNormal(boundaryVector,i,surfNormal,particlesPointer->x[indx],particlesPointer->y[indx]);
            // vectorNormalize(partNormal,partNormal);
            // partDotNormal = vectorDotProduct(partNormal,surfNormal);
            // double thetaImpact = acos(partDotNormal)*180.0/3.1415;
            // if(E0 < surfaces->E && E0> surfaces->E0)
            //{
            //    int tally_index = floor((E0-surfaces->E0)/surfaces->dE);
            //    if(thetaImpact > surfaces->A0 && thetaImpact < surfaces->A)
            //    {
            //        int aTally =
            //        floor((thetaImpact-surfaces->A0)/surfaces->dA);
            //        atomicAdd(&surfaces->energyDistribution[i*surfaces->nE*surfaces->nA+
            //        aTally*surfaces->nE + tally_index],
            //        particlesPointer->weight[indx]);
            //    }
            //}
#else
#endif
          }
        }
        if (nBoundariesCrossed == 0) {
          particlesPointer->xprevious[indx] = particlesPointer->x[indx];
          particlesPointer->yprevious[indx] = particlesPointer->y[indx];
          particlesPointer->zprevious[indx] = particlesPointer->z[indx];
        }
      }
#else // 2D geometry
#if USECYLSYMM > 0
      double pdim1 = sqrt(particlesPointer->x[indx] * particlesPointer->x[indx] +
                         particlesPointer->y[indx] * particlesPointer->y[indx]);
      double pdim1previous = sqrt(particlesPointer->xprevious[indx] *
                                     particlesPointer->xprevious[indx] +
                                 particlesPointer->yprevious[indx] *
                                     particlesPointer->yprevious[indx]);
      double theta0 = atan2(particlesPointer->yprevious[indx],
                            particlesPointer->xprevious[indx]);
      double theta1 =
          atan2(particlesPointer->y[indx], particlesPointer->x[indx]);
      double thetaNew = 0;
      double rNew = 0;
      double xNew = 0;
      double yNew = 0;
#else
      double pdim1 = particlesPointer->x[indx];
      double pdim1previous = particlesPointer->xprevious[indx];
#endif
      double particle_slope =
          (particlesPointer->z[indx] - particlesPointer->zprevious[indx]) /
          (pdim1 - pdim1previous);
      double particle_intercept =
          -particle_slope * pdim1 + particlesPointer->z[indx];
      double intersectionx[2] = {};
      // intersectionx = new double[nPoints];
      double intersectiony[2] = {};
      // intersectiony = new double[nPoints];
      double distances[2] = {};
      // distances = new double[nPoints];
      int intersectionIndices[2] = {};
      double tol_small = 1e-12;
      double tol = 1e12;
      int nIntersections = 0;
      double signPoint;
      double signPoint0;
      double signLine1;
      double signLine2;
      double minDist = 1e12;
      int minDistInd = 0;
 //cout << "particle slope " << particle_slope << " " << particle_intercept
 //<< endl; cout << "r " << boundaryVector[0].x1 << " " <<
 //boundaryVector[0].x1 << " " << boundaryVector[0].slope_dzdx << endl;
 //cout << "r0 " << particlesPointer->x[indx] << " " <<
 //particlesPointer->y[indx] << " " <<
 //particlesPointer->z[indx]<< endl;
#if GEOM_HASH > 0
#if USECYLSYMM > 0
      double r_position = sqrtf(particlesPointer->xprevious[indx] *
                                   particlesPointer->xprevious[indx] +
                               particlesPointer->yprevious[indx] *
                                   particlesPointer->yprevious[indx]);
#else
      double r_position = particlesPointer->xprevious[indx];
#endif
      double dr = closeGeomGridr[1] - closeGeomGridr[0];
      double dz = closeGeomGridz[1] - closeGeomGridz[0];
      int rInd = floor((r_position - closeGeomGridr[0]) / dr + 0.5);
      int zInd = floor(
          (particlesPointer->zprevious[indx] - closeGeomGridz[0]) / dz + 0.5);
      if (rInd < 0 || rInd >= nR_closeGeom[0])
        rInd = 0;
      if (zInd < 0 || zInd >= nZ_closeGeom[0])
        zInd = 0;
      int i = 0;
      int closeIndx = 0;
      for (int j = 0; j < n_closeGeomElements[0]; j++) {
        closeIndx = zInd * nR_closeGeom[0] * n_closeGeomElements[0] +
                    rInd * n_closeGeomElements[0] + j;
        // if(zInd*nR_closeGeom[0]*n_closeGeomElements[0] +
        // rInd*n_closeGeomElements[0] + j < 0)
        //{
        //        zInd=0;
        //        rInd=0;
        //        j=0;
        //    //cout << "index " <<
        //    zInd*nR_closeGeom[0]*n_closeGeomElements[0] +
        //    rInd*n_closeGeomElements[0] + j << endl;
        //}
        //    if(zInd*nR_closeGeom[0]*n_closeGeomElements[0] +
        //    rInd*n_closeGeomElements[0] + j > 1309440)
        //    {
        //        zInd=0;
        //        rInd=0;
        //        j=0;
        //        //cout << "index " <<
        //        zInd*nR_closeGeom[0]*n_closeGeomElements[0] +
        //        rInd*n_closeGeomElements[0] + j << endl;
        //    }
        i = closeGeom[closeIndx];
      
#else
      for (int i = 0; i < nLines; i++) {
#endif
        // cout << "vert geom " << i << "  " <<
        // fabs(boundaryVector[i].slope_dzdx) << " " << tol << endl;
        if (abs(boundaryVector[i].slope_dzdx) >= tol * 0.75) 
        {
          signPoint = copysign(1.0, pdim1 - boundaryVector[i].x1);
          signPoint0 = copysign(1.0, pdim1previous - boundaryVector[i].x1);
          // cout << "signpoint1 " << signPoint << " " << signPoint0 <<
          // endl;
        } 
        else 
        {
          signPoint =
              copysign(1.0, particlesPointer->z[indx] -
                                     pdim1 * boundaryVector[i].slope_dzdx -
                                     boundaryVector[i].intercept_z);
          signPoint0 = copysign(1.0, particlesPointer->zprevious[indx] -
                                              pdim1previous *
                                                  boundaryVector[i].slope_dzdx -
                                              boundaryVector[i].intercept_z);
           //cout << "signpoint2 " << signPoint << " " << signPoint0 <<
           //endl;
        }

        if (signPoint != signPoint0) 
        {
          if (abs(particle_slope) >= tol * 0.75) 
          {
            // cout << " isinf catch " << endl;
            particle_slope = tol;
          }
          if (abs(particle_slope) >= tol * 0.75) 
          {
            signLine1 = copysign(1.0, boundaryVector[i].x1 - pdim1);
            signLine2 = copysign(1.0, boundaryVector[i].x2 - pdim1);
            // cout << "signlines3 " << signLine1 << " " << signLine2 <<
            // endl;
          }
          else 
          {
            signLine1 =
                copysign(1.0, boundaryVector[i].z1 -
                                       boundaryVector[i].x1 * particle_slope -
                                       particle_intercept);
            signLine2 =
                copysign(1.0, boundaryVector[i].z2 -
                                       boundaryVector[i].x2 * particle_slope -
                                       particle_intercept);
            //cout << "signline 1 and 2 " << signLine1 << " " << signLine2 << endl;
          }

          ////if (signPoint != signPoint0) 
          ////{
          ////  if (abs(particle_slope) >= tol * 0.75) 
          ////  {
          ////    // cout << " isinf catch " << endl;
          ////    particle_slope = tol;
          ////  }
          ////  if (abs(particle_slope) >= tol * 0.75) 
          ////  {
          ////    signLine1 = copysign(1.0, boundaryVector[i].x1 - pdim1);
          ////    signLine2 = copysign(1.0, boundaryVector[i].x2 - pdim1);
          ////    // cout << "signlines3 " << signLine1 << " " << signLine2 <<
          ////    // endl;
          ////  } 
          ////  else 
          ////  {
          ////    signLine1 =
          ////        copysign(1.0, boundaryVector[i].z1 -
          ////                               boundaryVector[i].x1 * particle_slope -
          ////                               particle_intercept);
          ////    signLine2 =
          ////        copysign(1.0, boundaryVector[i].z2 -
          ////                               boundaryVector[i].x2 * particle_slope -
          ////                               particle_intercept);
          ////  }
            // cout << "signLines " << signLine1 << " " << signLine2 <<
            // endl; cout << "bound vec points " <<
            // boundaryVector[i].z1 << " " << boundaryVector[i].x1 <<
            // " " << boundaryVector[i].z2 << " " << boundaryVector[i].x2 <<
            // endl;
            if (signLine1 != signLine2) 
            {
              intersectionIndices[nIntersections] = i;
              nIntersections++;

              // cout << "nintersections " << nIntersections << endl;
              // cout << fabs(particlesPointer->x[indx] -
              // particlesPointer->xprevious[indx]) << tol_small << endl;
              if (abs(pdim1 - pdim1previous) < tol_small) 
              {
                //  cout << "vertical line" << cout;
                intersectionx[nIntersections - 1] = pdim1previous;
                intersectiony[nIntersections - 1] =
                    intersectionx[nIntersections - 1] *
                        boundaryVector[i].slope_dzdx +
                    boundaryVector[i].intercept_z;
              } 
              else 
              {
                // cout << "not vertical line" << endl;
                // cout << 0.0*7.0 << " " << i << " " << nParam << " " <<
                // lines[i*nParam+4] << "  " <<tol << endl; cout <<
                // "boundaryVector slope " << boundaryVector[i].slope_dzdx << " "
                // << tol*0.75 <<endl;
                if (abs(boundaryVector[i].slope_dzdx) >= tol * 0.75) 
                {
                  intersectionx[nIntersections - 1] = boundaryVector[i].x1;
                } 
                else 
                {
                  intersectionx[nIntersections - 1] =
                      (boundaryVector[i].intercept_z - particle_intercept) /
                      (particle_slope - boundaryVector[i].slope_dzdx);
                  //  cout << "in this else "<<
                  //  intersectionx[nIntersections -1] << endl;
                }
                intersectiony[nIntersections - 1] =
                    intersectionx[nIntersections - 1] * particle_slope +
                    particle_intercept;
                    //cout << "intersectionx and y"<<
                    //intersectionx[nIntersections -1] << " " << intersectiony[0] << endl;
              }
            }
          ////}
        }
      }
      //cout << " nIntersections " << nIntersections << endl;
        // if(particlesPointer->hitWall[indx] == 0.0)
        // {
        if (nIntersections == 0) 
        {
          particlesPointer->distTraveled[indx] =
              particlesPointer->distTraveled[indx] + dpath;
          particlesPointer->xprevious[indx] = particlesPointer->x[indx];
          particlesPointer->yprevious[indx] = particlesPointer->y[indx];
          particlesPointer->zprevious[indx] = particlesPointer->z[indx];
          // particlesPointer->test0[indx] = -50.0;

          // cout << "r " << particlesPointer->x[indx] << " " <<
          // particlesPointer->y[indx] << " " << particlesPointer->z[indx] <<
          // endl; cout << "r0 " << particlesPointer->xprevious[indx]
          // << " " << particlesPointer->yprevious[indx] << " " <<
          // particlesPointer->zprevious[indx] << endl;
        } 
        else if (nIntersections == 1) 
        {
          particlesPointer->hitWall[indx] = 1.0;
          particlesPointer->wallIndex[indx] = intersectionIndices[0];
          particlesPointer->wallHit[indx] = intersectionIndices[0];
          // particlesPointer->test0[indx] = -100.0;
          if (particle_slope >= tol * 0.75) 
          {
#if USECYLSYMM > 0
            double x0 = particlesPointer->xprevious[indx];
            double x1 = particlesPointer->x[indx];
            double y0 = particlesPointer->yprevious[indx];
            double y1 = particlesPointer->y[indx];
            double tt = findT(x0, x1, y0, y1, intersectionx[0]);
            xNew = x0 + (x1 - x0) * tt;
            yNew = y0 + (y1 - y0) * tt;
            rNew = sqrt(xNew * xNew + yNew * yNew);
            thetaNew = theta0 +
                       (intersectiony[0] - particlesPointer->zprevious[indx]) /
                           (particlesPointer->z[indx] -
                            particlesPointer->zprevious[indx]) *
                           (theta1 - theta0);
            particlesPointer->y[indx] = yNew;
            particlesPointer->yprevious[indx] = yNew;
#else
            // cout << "Particle index " << indx << " hit wall and is
            // calculating y point " << particlesPointer->y[indx] << endl;
            particlesPointer->y[indx] =
                particlesPointer->yprevious[indx] +
                (intersectiony[0] - particlesPointer->zprevious[indx]) /
                    (particlesPointer->z[indx] -
                     particlesPointer->zprevious[indx]) *
                    (particlesPointer->y[indx] -
                     particlesPointer->yprevious[indx]);
            // cout << "yprev,intersectiony,zprevious,z,y " <<
            // particlesPointer->yprevious[indx] << " " << intersectiony[0] << "
            // "<< particlesPointer->zprevious[indx] << " " <<
            // particlesPointer->z[indx] << " " << particlesPointer->y[indx] <<
            // endl;

#endif
          } 
          else 
          {
#if USECYLSYMM > 0
            double x0 = particlesPointer->xprevious[indx];
            double x1 = particlesPointer->x[indx];
            double y0 = particlesPointer->yprevious[indx];
            double y1 = particlesPointer->y[indx];
            double tt = findT(x0, x1, y0, y1, intersectionx[0]);
            xNew = x0 + (x1 - x0) * tt;
            yNew = y0 + (y1 - y0) * tt;
            rNew = sqrt(xNew * xNew + yNew * yNew);
            // particlesPointer->test0[indx] = -200.0;
            thetaNew = theta0 + (intersectionx[0] - pdim1previous) /
                                    (pdim1 - pdim1previous) * (theta1 - theta0);
            //cout << " tt xnew ynew " << tt << " " << xNew << " " << yNew << endl;
            particlesPointer->yprevious[indx] = yNew;
            particlesPointer->y[indx] = yNew;
            // double rrr  =
            // sqrt(particlesPointer->x[indx]*particlesPointer->x[indx] +
            // particlesPointer->y[indx]*particlesPointer->y[indx]);
            // if(particlesPointer->z[indx]< -4.1 & rrr > 5.5543)
            //{
            //  cout <<" positions of intersection 2" <<
            //  particlesPointer->x[indx] << " " << particlesPointer->y[indx]<<
            //  endl; cout <<" r " << rrr << " " <<
            //  boundaryVector[particlesPointer->wallHit[indx]].x1 << " " <<
            //  boundaryVector[particlesPointer->wallHit[indx]].x2<< endl;
            // cout << "x0 x1 y0 y1 rNew "  << " "<< x0 << " " << x1 << " "
            // << y0 << " " << y1 << " " << rNew << endl; cout << "xNew
            // yNew " << xNew << " " << yNew << endl; cout <<
            // "intersectionx " << intersectionx[0] << endl;
            //}
#else
            // cout << "Particle index " << indx << " hit wall and is
            // calculating y point " << particlesPointer->y[indx] << endl;
            particlesPointer->y[indx] =
                particlesPointer->yprevious[indx] +
                (intersectionx[0] - particlesPointer->xprevious[indx]) /
                    (particlesPointer->x[indx] -
                     particlesPointer->xprevious[indx]) *
                    (particlesPointer->y[indx] -
                     particlesPointer->yprevious[indx]);
            // cout << "yprev,intersectiony,zprevious,z,y " <<
            // particlesPointer->yprevious[indx] << " " << intersectiony[0] << "
            // "<< particlesPointer->zprevious[indx] << " " <<
            // particlesPointer->z[indx] << " " << particlesPointer->y[indx] <<
            // endl;
#endif
          }
#if USECYLSYMM > 0
          particlesPointer->xprevious[indx] = xNew;
          particlesPointer->x[indx] = particlesPointer->xprevious[indx];
#else
          particlesPointer->x[indx] = intersectionx[0];
          particlesPointer->xprevious[indx] = intersectionx[0];
#endif
          particlesPointer->zprevious[indx] = intersectiony[0];
          particlesPointer->z[indx] = intersectiony[0];
          // cout << "nInt = 1 position " << intersectionx[0] << " " <<
          // intersectiony[0]  << endl;
        } 
        else 
        {
          // cout << "nInts greater than 1 " << nIntersections <<
          // endl;
          for (int i = 0; i < nIntersections; i++) {
            distances[i] =
                (pdim1previous - intersectionx[i]) *
                    (pdim1previous - intersectionx[i]) +
                (particlesPointer->zprevious[indx] - intersectiony[i]) *
                    (particlesPointer->zprevious[indx] - intersectiony[i]);
            if (distances[i] < minDist) {
              minDist = distances[i];
              minDistInd = i;
            }
          }

          particlesPointer->wallIndex[indx] = intersectionIndices[minDistInd];
          particlesPointer->wallHit[indx] = intersectionIndices[minDistInd];
          particlesPointer->hitWall[indx] = 1.0;
#if USECYLSYMM > 0
          thetaNew = theta0 + (intersectionx[minDistInd] - pdim1previous) /
                                  (pdim1 - pdim1previous) * (theta1 - theta0);
          particlesPointer->yprevious[indx] =
              intersectionx[minDistInd] * sin(thetaNew);
          particlesPointer->y[indx] = particlesPointer->yprevious[indx];
          // particlesPointer->y[indx] =
          // intersectionx[minDistInd]*cosf(thetaNew);
          particlesPointer->x[indx] =
              intersectionx[minDistInd] * cos(thetaNew);
#else
          // cout << "Particle index " << indx << " hit wall and is
          // calculating y point " << particlesPointer->yprevious[indx] << " " <<
          // particlesPointer->y[indx] << endl;
          // particlesPointer->y[indx] = particlesPointer->yprevious[indx] +
          // (intersectionx[minDistInd] - pdim1previous)/(pdim1 -
          // pdim1previous)*(particlesPointer->y[indx] -
          // particlesPointer->yprevious[indx]); cout <<
          // "intersectionx,pdp,pd,y " << intersectionx[0] << " "<< pdim1previous
          // << " " << pdim1 << " " << particlesPointer->y[indx] << endl;
          particlesPointer->y[indx] =
              particlesPointer->yprevious[indx] +
              (intersectiony[0] - particlesPointer->zprevious[indx]) /
                  (particlesPointer->z[indx] -
                   particlesPointer->zprevious[indx]) *
                  (particlesPointer->y[indx] -
                   particlesPointer->yprevious[indx]);
          // cout << "yprev,intersectiony,zprevious,z,y " <<
          // particlesPointer->yprevious[indx] << " " << intersectiony[0] << "
          // "<< particlesPointer->zprevious[indx] << " " <<
          // particlesPointer->z[indx] << " " << particlesPointer->y[indx] <<
          // endl;
          particlesPointer->x[indx] = intersectionx[minDistInd];
#endif
          particlesPointer->z[indx] = intersectiony[minDistInd];
        }
      ////}
        // else
        //{
        //    if (particlesPointer->y[indx] < boundaryVector[nLines].y1)
        //    {
        //        particlesPointer->hitWall[indx] = 1.0;
        //    }
        //    else if (particlesPointer->y[indx] > boundaryVector[nLines].y2)
        //    {
        //        particlesPointer->hitWall[indx] = 1.0;
        //    }
        //}
#endif
      if (particlesPointer->hitWall[indx] == 1.0) {

#if (FLUX_EA > 0 && USESURFACEMODEL == 0)
        double E0 = 0.0;
        double thetaImpact = 0.0;
        double particleTrackVector[3] = {0.0};
        double surfaceNormalVector[3] = {0.0};
        double norm_part = 0.0;
        double partDotNormal = 0.0;
        particleTrackVector[0] = particlesPointer->vx[indx];
        particleTrackVector[1] = particlesPointer->vy[indx];
        particleTrackVector[2] = particlesPointer->vz[indx];
        norm_part = sqrt(particleTrackVector[0] * particleTrackVector[0] +
                         particleTrackVector[1] * particleTrackVector[1] +
                         particleTrackVector[2] * particleTrackVector[2]);
        E0 = 0.5 * particlesPointer->amu[indx] * 1.6737236e-27 *
             (norm_part * norm_part) / 1.60217662e-19;
        int wallHitP = particlesPointer->wallHit[indx];
        boundaryVector[particlesPointer->wallHit[indx]].getSurfaceNormal(
            surfaceNormalVector, particlesPointer->y[indx],
            particlesPointer->x[indx]);
        particleTrackVector[0] = particleTrackVector[0] / norm_part;
        particleTrackVector[1] = particleTrackVector[1] / norm_part;
        particleTrackVector[2] = particleTrackVector[2] / norm_part;

        partDotNormal =
            vectorDotProduct(particleTrackVector, surfaceNormalVector);
        thetaImpact = acos(partDotNormal);
        if (thetaImpact > 3.14159265359 * 0.5) {
          thetaImpact = abs(thetaImpact - (3.14159265359));
        }
        thetaImpact = thetaImpact * 180.0 / 3.14159265359;
        EdistInd = floor((E0 - E0dist) / dEdist);
        AdistInd = floor((thetaImpact - A0dist) / dAdist);
        int surfaceHit =
            boundaryVector[particlesPointer->wallHit[indx]].surfaceNumber;
        int surface = boundaryVector[particlesPointer->wallHit[indx]].surface;
        double weight = particlesPointer->weight[indx];
        // particlesPointer->test[indx] = norm_part;
        // particlesPointer->test0[indx] = partDotNormal;
        // particlesPointer->test1[indx] = particleTrackVector[0];
        // particlesPointer->test2[indx] = particleTrackVector[1];
        // particlesPointer->test3[indx] = particleTrackVector[2];
        // particlesPointer->test4[indx] = particles;
        // cout << "impact energy and angle " << E0 << " " << thetaImpact
        // << endl; cout << "surface EAinds " <<surface<< " " <<
        // EdistInd << " " << AdistInd << endl;
        if (surface) {
          if ((EdistInd >= 0) && (EdistInd < nEdist) && (AdistInd >= 0) &&
              (AdistInd < nAdist)) {
#if USE_CUDA > 0
            atomicAdd(
                &surfaces->energyDistribution[surfaceHit * nEdist * nAdist +
                                              EdistInd * nAdist + AdistInd],
                particlesPointer->weight[indx]);
            atomicAdd(&surfaces->grossDeposition[surfaceHit],
                      particlesPointer->weight[indx]);
            atomicAdd(&surfaces->sumWeightStrike[surfaceHit],
                      particlesPointer->weight[indx]);
            atomicAdd(&surfaces->sumParticlesStrike[surfaceHit], 1);
#else

            surfaces->energyDistribution[surfaceHit * nEdist * nAdist +
                                         EdistInd * nAdist + AdistInd] =
                surfaces->energyDistribution[surfaceHit * nEdist * nAdist +
                                             EdistInd * nAdist + AdistInd] +
                weight;
            surfaces->sumWeightStrike[surfaceHit] =
                surfaces->sumWeightStrike[surfaceHit] + weight;
            surfaces->sumParticlesStrike[surfaceHit] =
                surfaces->sumParticlesStrike[surfaceHit] + 1;
            surfaces->grossDeposition[surfaceHit] =
                surfaces->grossDeposition[surfaceHit] + weight;
#endif
          }
        }
#elif (FLUX_EA == 0 && USESURFACEMODEL == 0)
          particlesPointer->weight[indx] = 0.0;
#endif
        // particlesPointer->transitTime[indx] = tt*dt;
      }
    }

    // cout << "2geometry check particle x" << particlesPointer->x[indx] <<
    // particlesPointer->x[indx]previous <<endl; cout << "2geometry
    // check particle y" << particlesPointer->y[indx] <<
    // particlesPointer->y[indx]previous <<endl; cout << "2geometry
    // check particle z" << particlesPointer->z[indx] <<
    // particlesPointer->z[indx]previous <<endl;
  }
};

#endif

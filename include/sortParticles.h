#ifndef _SORT_
#define _SORT_

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#define CUDA_CALLABLE_MEMBER_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER_DEVICE
#define CUDA_CALLABLE_MEMBER_HOST
using namespace std;
#endif

#include "Particles.h"
#ifdef __CUDACC__
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#endif

#ifdef __GNUC__ 
#include <stdlib.h>
#endif
#ifdef USE_CUDA
#else
struct ordering {
    bool operator ()(thrust::pair<int, double> const& a,thrust::pair<int, double> const& b) {
            return (b.second) < (a.second);
	        }
};

struct sortParticles { 
    Particles *particles;
    int nP;
    double weightThreshold;
    int& tt;
    int nDtPerApply;
    int* startIndx;
    int* nActiveParticles;
    int rank;
#if __CUDACC__
    curandState *state;
#else
    mt19937 *state;
#endif

    sortParticles(Particles *_particles,int _nP, double _weightThreshold,int& _tt,int _nDtPerApply, int* _startIndx,int* _nActiveParticles,int _rank,
#if __CUDACC__
                curandState *_state)
#else
                mt19937 *_state)
#endif
    : particles(_particles),nP(_nP), weightThreshold(_weightThreshold),tt(_tt),nDtPerApply(_nDtPerApply), startIndx(_startIndx), nActiveParticles(_nActiveParticles),rank(_rank),state(_state) {}
    
    CUDA_CALLABLE_MEMBER_DEVICE 
    void operator()(size_t indx) const 
    { 
    if(tt%nDtPerApply==0 && tt>0)
    {
       int start = startIndx[rank];
       int nPonRank = nActiveParticles[rank];
       int end = start+nPonRank-1;
       sim::Array<double> weight(nPonRank,0.0); 
       sim::Array<thrust::pair<int,double>> pairs(nPonRank);
       //cout << "rank tt n start " << rank<<" "<< tt<<" "<< start<< endl;
//#ifdef __CUDACC__
	//#else
	//  uniform_real_distribution<double> dist(0.0, 1.0);
        //#endif
       for(int i=0;i<nPonRank;i++)
       {
	//#ifdef __CUDACC__
        //weight[i] =curand_uniform(&state[i]);
	//#else
        //weight[i] =dist(state[start+i]);
	//#endif
	//particles->weight[i] = weight[i];
        weight[i] =particles->weight[start+i];
	pairs[i].first = i;
        pairs[i].second = weight[i];
       //cout << "pair "  << " " << pairs[i].first << " " << pairs[i].second << endl;
       }
       //sim::Array<double> weight0(weight); 
       thrust::sort(thrust::device,pairs.begin(),pairs.end(),ordering());
       for(int i=0;i<nPonRank;i++)
       {
       //cout << "pair "  << i<<" " << pairs[i].first << " " << pairs[i].second << endl;
       weight[i] = pairs[i].second; 
       }
       sim::Array<double> weightThresholdA(1,weightThreshold);
       sim::Array<int> lowerBoundIndex(1,0);
       //cout << "weights " << " " << weightThresholdA[0] << endl;
       thrust::upper_bound(weight.begin(), weight.end(),
                           weightThresholdA.begin(),weightThresholdA.end() , 
        		   lowerBoundIndex.begin(),thrust::greater<double>());
       //cout << " min index " << lowerBoundIndex[0] << " " << weight[lowerBoundIndex[0]] << endl;
       int nUnderThresh=nPonRank-lowerBoundIndex[0];
       cout << " nPartivles under thresh " << nUnderThresh << endl;
       int nSwapsNeeded=0;
       int goodParticles = nPonRank-nUnderThresh;
       //cout << " n good particles " << goodParticles << endl;
       for(int i=0;i<nUnderThresh;i++)
       {
        if(pairs[nPonRank-nUnderThresh+i].first < goodParticles)
        {
          nSwapsNeeded=nSwapsNeeded+1;
        }
       }
       //cout << " nSwapsNeeded " << nSwapsNeeded << endl;
       sim::Array<int> swapBad(nSwapsNeeded),swapGood(nSwapsNeeded);
       int ind=0;
       for(int i=0;i<nUnderThresh;i++)
       {
        if(pairs[nPonRank-nUnderThresh+i].first < goodParticles)
        {
	  swapBad[ind] = pairs[nPonRank-nUnderThresh+i].first;
       //cout << " swapBad " << ind <<" " << swapBad[ind]<<" " << "weight0[swapBad[ind]]" << endl;
          ind=ind+1;
        }
       }
       ind=0;
       //cout << "swap good going from 0 to " << goodParticles << endl;
       for(int i=0;i<goodParticles;i++)
       {
       //cout << " swapGood1 " << i <<" "<<ind<< " "  << endl;
       //cout << " pairs[i].first " << pairs[i].first <<" "<< " "  << endl;
        if(pairs[i].first > goodParticles-1)
        {
          swapGood[ind]=pairs[i].first;
       //cout << " swapGood " << ind <<" " << swapGood[ind]<<" " << "weight0[swapGood[ind]]" << endl;
          ind=ind+1;
        }
       }
       for(int i=0;i<nSwapsNeeded;i++)
       { particles->swapP(swapBad[i]+start,swapGood[i]+start);
       }
       for(int i=0;i<nPonRank;i++)
       {
         //cout << " weight0 " << i << " " << particles->weight[start+i]<< " " << particles->index[i] << endl;
       }
       nActiveParticles[rank] = goodParticles;
    }
    } 
};
#endif
#endif

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bvani
"""
import numpy as np
from sys import stdout

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100,randomseed=0,periodicity=0):
    '''Regular space clustering.
    Args:
        data: ndarray containing (n,d)-shaped float data
        max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
        min_dist: the minimal distances between cluster centers
    '''
    num_observations, d = z.shape
    p = np.hstack((0,np.random.RandomState(seed=randomseed).permutation(num_observations-1)+1))
    data = z[p]
    center_list = data[0, :].copy().reshape(d,1)
    centerids=[p[0]+1]
    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        differences=np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1)
        #differences.shape
        differences=np.max(np.stack((differences,periodicity-differences)),axis=0)
        #differences.shape
        distances = np.sqrt((np.square(differences)).sum(axis=-1))
        indice = tuple(np.nonzero(np.all(distances > min_dist, axis=-1))[0])
        if len(indice) > 0:
            # the first element will be used
            #print(center_list.shape,x_active.shape,x_active[indice[0]].shape)
            center_list = np.hstack((center_list, x_active[indice[0]].reshape(d,1)))
            centerids.append(p[i+indice[0]]+1)
            i += indice[0]
        else:
            i += batch_size
        if len(centerids) >= max_centers:
            print("%i centers: Exceeded the maximum number of cluster centers!\n"%len(centerids))
            print("Please increase dmin!\n")
            raise ValueError
    print("Found %i centers!"%len(centerids))
    return center_list,centerids

def make_biased_plumed(plumedfile,weights,colvar,height,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2,temperature):
    f_unb=open(plumedfile)
    f=open('plumed_biased.dat','w')
    lines=f_unb.readlines()
    p=lines.pop(-2)
    w0=",".join([str(weights[0][i]) for i in range (len(weights[0]))])
    w1=",".join([str(weights[1][i]) for i in range (len(weights[1]))])
    lines.insert(-1,"\nsigma1: COMBINE ARG=%s COEFFICIENTS=%s PERIODIC=NO"%(colvar,w0))
    lines.insert(-1,"\nsigma2: COMBINE ARG=%s COEFFICIENTS=%s PERIODIC=NO"%(colvar,w1))

    lines.insert(-1,"\nMETAD ...\n \
      LABEL=metad\n \
      ARG=sigma1,sigma2\n \
      PACE=500 HEIGHT=%f TEMP=%i\n \
      BIASFACTOR=%i\n \
      SIGMA=%f,%f\n \
      FILE=HILLS GRID_MIN=%f,%f GRID_MAX=%f,%f GRID_BIN=200,200\n \
      CALC_RCT RCT_USTRIDE=500\n \
      ... METAD\n"%(height,temperature,biasfactor,width1,width2,gridmin1,gridmin2,gridmax1,gridmax2))
  
    f.writelines(lines)
    f.write("\n PRINT ARG=%s,sigma1,sigma2,metad.rbias STRIDE=500 FILE=COLVAR_biased.dat"%colvar)

    f.close()
    
#Functions for CSP demo only

def triginvert(x,sinx,cosx):
  if cosx<0:
    if sinx>0:
      x=np.pi-x
    elif sinx<0:
      x=-np.pi-x

  return x

def getTrp8(CVs):
  sinx=CVs[:,12]
  cosx=CVs[:,13]
  x=np.arcsin(sinx)
  chi1=  chi2=[triginvert(a,b,c) for (a,b,c) in zip(x,sinx,cosx)]

  sinx=CVs[:,116]
  cosx=CVs[:,117]
  x=np.arcsin(sinx)
  chi2=[triginvert(a,b,c) for (a,b,c) in zip(x,sinx,cosx)]

  return np.asarray(chi1),np.asarray(chi2)

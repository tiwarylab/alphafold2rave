#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bvani
"""
import numpy as np
from sys import stdout

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100,randomseed=0,periodicity=np.pi):
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
        differences=np.abs(np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1))
        #differences.shape
        #differences=np.max(np.stack(differences,2*np.pi-differences),axis=1)
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

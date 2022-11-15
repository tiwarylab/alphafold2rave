#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bvani
"""
import numpy as np

def RegSpaceClustering(z, min_dist, max_centers=200, batch_size=100):
    '''Regular space clustering.
    Args:
        data: ndarray containing (n,d)-shaped float data
        max_centers: the maximum number of cluster centers to be determined, integer greater than 0 required
        min_dist: the minimal distances between cluster centers
    '''
    num_observations, d = z.shape
    p = np.hstack((0,np.random.permutation(num_observations-1)+1))
    data = z[p]
    center_list = data[0, :].copy().reshape(d,1)
    centerids=[p[0]]
    i = 1
    while i < num_observations:
        x_active = data[i:i+batch_size, :]
        distances = np.sqrt((np.square(np.expand_dims(center_list.T,0) - np.expand_dims(x_active,1))).sum(axis=-1))
        indice = tuple(np.nonzero(np.all(distances > min_dist, axis=-1))[0])
        if len(indice) > 0:
            # the first element will be used
            #print(center_list.shape,x_active.shape,x_active[indice[0]].shape)
            center_list = np.hstack((center_list, x_active[indice[0]].reshape(d,1)))
            centerids.append(p[i+indice[0]])
            i += indice[0]
        else:
            i += batch_size
        if len(center_list) >= max_centers:
            print("Exceed the maximum number of cluster centers!\n")
            print("Please increase dmin!\n")
            raise ValueError
    return center_list,centerids

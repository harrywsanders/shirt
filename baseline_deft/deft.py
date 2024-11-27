#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:52:48 2024

@author: eochoa
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def deft(data, target, seed, n_target=10, n_neighbors_search=500):
    np.random.seed(seed)
    x_source = np.concatenate([np.array(data[bench]['Embs']) for bench in data.keys()
                               if bench != target])
    bench_source = np.concatenate([np.array([bench]*len(data[bench]['Embs'])) for bench in data.keys()
                               if bench != target])
    
    x_target = np.array(data[target]['Embs'])
    n = x_target.shape[0]
    idx = np.random.choice(n, size=n_target, replace=False)
    x_target = x_target[idx, :]
    
    neigh = NearestNeighbors(n_neighbors=n_neighbors_search)
    neigh.fit(x_source)

    index = neigh.kneighbors(x_target, return_distance=False)
    
    index = list(set(index.flatten()))
    
    qs_source = np.concatenate([np.array(data[bench]['Qs']) for bench in data.keys()
                                if bench != target])
    
    as_source = np.concatenate([np.array(data[bench]['As']) for bench in data.keys()
                                if bench != target])
    
    return (qs_source[index], as_source[index], bench_source[index])

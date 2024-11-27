#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:59:54 2024

@author: eochoa
"""
import json
import numpy as np
from deft import deft

file_path = 'data_QA_processed.json'
with open(file_path, "r") as datafile:
    data = json.load(datafile)
    
    
target = 'leaderboard_bbh_boolean_expressions'
seed = 0

qs_source, as_source, bench_source=deft(data,
                            target=target, 
                            seed=seed, 
                            n_target=10, 
                            n_neighbors_search=100)


print(qs_source)
print(as_source)
print(bench_source)
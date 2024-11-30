import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from irt import IRT
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from tqdm.auto import tqdm
from definitions import *

with open('../data/data_QA.json', "r") as datafile:
    data_QA = json.load(datafile)
with open('../data/data_QA_processed.json', "r") as datafile:
    data_QA_processed = json.load(datafile)
Y = np.load('../data/data_Y.pickle', allow_pickle=True)

def shirt(idx_sample_questions, target_benchmark, n=500):
 
    irt_model = IRT(device='cpu')
    irt_model.load(f"../irt_models/{target_benchmark}")
    
    M_aux = [Y[k]['models'] for k in benchmarks if k!=target_benchmark]
    M_aux = np.sort(list(reduce(set.intersection, map(set, M_aux)))).tolist()
    M = [Y[k]['models'] for k in benchmarks]
    M = np.sort(list(reduce(set.intersection, map(set, M)))).tolist()
    
    idx_aux = [int(np.argmax(np.array(M_aux)==m)) for m in M]
    idx_target = [int(np.argmax(np.array(Y[target_benchmark]['models'])==m)) for m in M]
    y = Y[target_benchmark]['correctness'][idx_target]
    
    assert irt_model.Theta.shape[0]==y.shape[0]
    Alpha_target = irt_model.fit_alpha_beta_kappa(y[:,idx_sample_questions], list(range(y.shape[0])), verbose=False)['new_Alpha']
    alpha_star = np.abs(np.linalg.eigh(Alpha_target.T@Alpha_target)[1][:,-1])
    projs = (irt_model.Alpha@alpha_star[:,None]).squeeze()
    index = np.sort(np.argsort(-projs)[:n]).tolist()
 
    return index

def deft(idx_sample_questions, target_benchmark, n=500):

    x_source = np.concatenate([np.array(data_QA_processed[bench]['Embs']) for bench in benchmarks
                               if bench != target_benchmark])
    bench_source = np.concatenate([np.array([bench]*len(data_QA_processed[bench]['Embs'])) for bench in benchmarks
                               if bench != target_benchmark])
    
    x_target = np.array(data_QA_processed[target_benchmark]['Embs'])
    x_target = x_target[idx_sample_questions, :]
    
    neigh = NearestNeighbors(n_neighbors=x_source.shape[0])
    neigh.fit(x_source)
    dist = neigh.kneighbors(x_target, return_distance=False)

    index = []
    for j in range(dist.shape[1]):
        if len(index) == n: break
        for i in range(dist.shape[0]):
            if len(index) == n: break
            if dist[i,j] not in index:
                index.append(int(dist[i,j]))
    index = np.sort(index).tolist()
    
    return index


def random_sampling(target_benchmark, n=500):

    seed=0
    np.random.seed(seed)
    
    x_source = np.concatenate([np.array(data_QA_processed[bench]['Embs']) for bench in benchmarks
                               if bench != target_benchmark])

    index = np.sort(np.random.choice(x_source.shape[0], size=n, replace=False)).tolist()
    
    return index
    
def save_ft_data(n_sample_qs, target_benchmark, n_aux):

    qs_source = np.concatenate([np.array(data_QA[bench]['Qs']) for bench in benchmarks
                                if bench != target_benchmark])
    
    as_source = np.concatenate([np.array(data_QA[bench]['As']) for bench in benchmarks
                                if bench != target_benchmark])

    bench_source = np.concatenate([np.array([bench]*len(data_QA[bench]['As'])) for bench in benchmarks
                               if bench != target_benchmark])
    def get_qas(index):
        return {'ft_data':{'Qs':qs_source[index].tolist(), 'As':as_source[index].tolist(), 'bench':bench_source[index].tolist()}}
    
    ###
    E = np.array(data_QA_processed[target_benchmark]['Embs'])
    kmeans = KMeans(n_clusters=n_sample_qs, random_state=0, n_init="auto").fit(E)
    idx_sample_qs = np.unique(pairwise_distances(kmeans.cluster_centers_, E).argmin(1))

    ###
    index_shirt = shirt(idx_sample_questions=idx_sample_qs,
                       target_benchmark=target_benchmark,
                       n=n_aux)
    dict_shirt = get_qas(index_shirt)

    index_deft = deft(idx_sample_questions=idx_sample_qs,
                     target_benchmark=target_benchmark,
                     n=n_aux)
    dict_deft = get_qas(index_deft)

    index_random = random_sampling(target_benchmark=target_benchmark, n=n_aux)
    dict_random = get_qas(index_random)

    ###
    with open(f"../ft_data/shirt_n-target={n_sample_qs}_n-aux={n_aux}_{target_benchmark}.json", "w") as outfile: 
        json.dump(dict_shirt, outfile)
    with open(f"../ft_data/deft_n-target={n_sample_qs}_n-aux={n_aux}_{target_benchmark}.json", "w") as outfile: 
        json.dump(dict_deft, outfile)
    with open(f"../ft_data/random_n-target={n_sample_qs}_n-aux={n_aux}_{target_benchmark}.json", "w") as outfile: 
        json.dump(dict_random, outfile)

if __name__=="__main__":
    for n in tqdm(ns_target):
        for bench in tqdm(benchmarks):
            save_ft_data(n_sample_qs=n, target_benchmark=bench, n_aux=n_aux)
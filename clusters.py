# import hdbscan
from sklearn.cluster import DBSCAN, SpectralClustering
import community
import networkx as nx
import numpy as np


def community_structure(dpos, algorithm, sigma=10.):

    data = dpos.loc[:, ['x', 'y', 'z']].values
    partition = None
    comm = None
    if algorithm == 'louvain':
        comm, partition, _ = louvain_community(data, sigma=sigma)
    # elif algorithm == 'hdbscan':
    #     clusterer = hdbscan.HDBSCAN().fit(data)
    #     comm = clusterer.labels_
    elif algorithm == 'dbscan':
        clusterer = DBSCAN().fit(data)
        comm = clusterer.labels_
    elif algorithm == 'spectral':
        clusterer = SpectralClustering().fit(data)
        comm = clusterer.labels_

    if partition is None and comm is not None:
        partition = com_to_partition(comm)

    return comm, partition


def louvain_community(data, sigma=10.):

    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import normalize
    import time
    t = time.time()
    A_Sc = rbf_kernel(data, gamma=1./sigma)
    print(time.time() - t)
    A_Sc[A_Sc < 1e-3] = 0
    save = False
    A_Sc_norm = normalize(A_Sc, norm='l1', axis=1)

    G_Sc = nx.from_numpy_array(A_Sc_norm)
    t = time.time()
    partition = community.best_partition(G_Sc)
    print(time.time() - t)
    return np.asarray([val for (key, val) in partition.items()]), partition, A_Sc


def com_to_partition(comm):

    partition = dict([])
    for key, c in enumerate(comm):
        partition[key] = c
    return partition


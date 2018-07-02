#Prepared by Arpan Mukherjee
from apt_importers import *
import argparse
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='Input File Name')
parser.add_argument('--rangefile', required=True, help='Range File')
parser.add_argument('--visu_type', type=int, default=0, help='Type of Visualization')
flags = parser.parse_args()
viz = flags.visu_type

epos = read_file(flags.filename)
ions, rrngs = read_rrng(flags.rangefile)

data = epos.loc[:, ['x', 'y', 'z']].values


lpos = label_ions(epos, rrngs)
dpos = deconvolve(lpos)

try:
    from pyobb.obb import OBB
    obb = OBB.build_from_points(dpos.loc[:, ['x', 'y', 'z']].values)
except ValueError:
    print 'Bounding Box could not be computed. Need package pyobb'
    obb = None


if viz == 0:
    volvis(dpos, obb=obb)
elif viz == 1:
    import numpy as np
    from sklearn.preprocessing import scale, normalize
    from pyflann import FLANN
    import scipy.stats as st
    from sklearn.metrics.pairwise import rbf_kernel
    from pyobb.obb import OBB
    import community
    import networkx as nx

    fl = FLANN()
    obb = OBB.build_from_points(dpos.loc[:, ['x', 'y', 'z']].values)
    Sc_pos = dpos.loc[dpos.element == 'Sc', :]
    data_Sc = Sc_pos.loc[:, ['x', 'y', 'z']].values

    results, dists = fl.nn(scale(data_Sc), scale(data_Sc), 10)  # calculating the distance to 10 nearest neighbors
    cov_dists = np.asarray([np.std(d[1:]) for d in dists])

    alpha = 0.01
    t_cr = st.t.interval(1 - alpha, len(cov_dists) - 1, loc=np.mean(cov_dists), scale=st.sem(cov_dists))[1]
    ind = np.where(cov_dists < t_cr)[0]

    pos1 = Sc_pos.iloc[ind]
    data_Sc = pos1.loc[:, ['x', 'y', 'z']].values

    A_Sc = rbf_kernel(data_Sc, gamma=1. / 10.)
    A_Sc[A_Sc < 1e-3] = 0
    A_Sc = normalize(A_Sc, norm='l1', axis=1)
    G_Sc = nx.from_numpy_array(A_Sc)
    partition = community.best_partition(G_Sc)
    comm_Sc = np.asarray([val for (key, val) in partition.iteritems()])
    pos2 = dpos.loc[dpos.element != 'Sc']
    convex_hull_clusters(pos1, comm_Sc, pos2, viz=True, obb=obb, same_color=True)


hull = ConvexHull(data)
print 'Volume of the specimen: ', hull.volume

import os
from apt_importers import *
import numpy as np
from pyflann import FLANN
from pyobb.obb import OBB
from scipy.spatial import ConvexHull
from clusters import community_structure

fl = FLANN()

#import tensorflow as tf
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
dpos= get_dpos('R12_Al-Sc.epos', 'ranges.rrng')
obb = OBB.build_from_points(dpos.loc[:, ['x', 'y', 'z']].values)

Sc_pos = dpos.loc[dpos.Element == 'Sc', :]

import time

t = time.time()
pos1, pos2, _ = singleton_removal(Sc_pos, k=10, alpha=0.01)
el_time = time.time() - t

viz = False
if viz is True:
    cm = plt.get_cmap('gist_rainbow')
    cmap = np.asarray([cm(1. * i / 2) for i in range(2)])
    colors = [np.tile(cmap[0, :], (len(pos1), 1)), np.tile(cmap[1, :], (len(pos2), 1))]

    canvas = multiple_volvis([pos1, pos2], colors=colors, obb=obb)

t = time.time()
comm_str, _ = community_structure(pos1, algorithm='louvain')
el_time = time.time() - t

big_cluster = True
if big_cluster is True:
    cpos = np.c_[dpos.loc[:, ['x', 'y', 'z']].values, np.ones((len(dpos),1))]
    n_clusters = len(np.unique(comm_str))
    a, b = np.histogram(comm_str, bins=n_clusters)
    sort_a = np.argsort(a)[::-1]
    ind = np.where(comm_str == sort_a[0])[0]
    pos2_ind = np.zeros(len(dpos), dtype=bool)
    vertices = pos1.iloc[ind].loc[:, ['x', 'y', 'z']].values
    hull = ConvexHull(vertices)
    print (hull.volume)
    eq = hull.equations
    for point_i, point in enumerate(cpos):
        if (np.dot(point, eq.T) < 0).all():
            pos2_ind[point_i] = True
    new_dpos = dpos[pos2_ind]
    volvis(new_dpos, size=5)
    print (ConvexHull(new_dpos.loc[:, ['x', 'y', 'z']].values).volume)

pos2 = dpos.loc[dpos.element != 'Sc']
convex_hull_clusters(pos=pos1, comm_str=comm_str, pos2=pos2, viz=True, obb=obb, same_color=True)

dpos_org = pos1
N = len(dpos)
ratio = (np.arange(0.1, 1, 0.02)[::-1] * N).astype(np.int)

hull = ConvexHull(dpos.loc[:, ['x', 'y', 'z']].values)
density = ratio/hull.volume


import scipy.stats as st
from sklearn import mixture
from sklearn.preprocessing import scale, normalize
t_test = []
t_mean =[]
com_test = []
com_mean = []
gmm_list = []
for n in ratio:
    vol = []
    com_num = []
    for i in range(50):
        ind = np.random.randint(N, size=n)
        dpos = dpos_org.iloc[ind]

        #Al_pos = dpos.loc[dpos.element == 'Al', :]
        Sc_pos = dpos.loc[dpos.element == 'Sc', :]
        data_Sc = Sc_pos.loc[:, ['x', 'y', 'z']].values

        #ind_al = np.random.randint(5000, size=len(Al_pos))
        #Al_pos =Al_pos.iloc[ind_al]

        ## Remove single atoms.
        results, dists = fl.nn(scale(data_Sc), scale(data_Sc), 8)   # calculating the distance to 10 nearest neighbors
        cov_dists = np.asarray([np.std(d[1:]) for d in dists])        # Calculating the covariance to the distances

        viz = False
        if viz is True:
            fig = plt.figure()
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            ax = fig.add_subplot(111)
            ax.hist(cov_dists)
            ax.set_xlabel('Covariance of Nearest Neighbor Distance')
            ax.set_ylabel('Frequency')
            ax.xaxis.label.set_size(26)
            ax.yaxis.label.set_size(26)
            for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
                            xtick.label.set_fontsize(20)
                            ytick.label.set_fontsize(20)
            plt.setp(ax.spines.values(), linewidth=3)
            ax.xaxis.set_tick_params(width=3)
            ax.yaxis.set_tick_params(width=3)
            plt.savefig('hist.png', bbox_inches='tight', dpi = 300)


        alpha = 0.01
        t_cr = st.t.interval(1-alpha, len(cov_dists)-1, loc=np.mean(cov_dists), scale=st.sem(cov_dists))[1]
        ind = np.where(cov_dists < t_cr)[0]

        # view multiple
        pos1 = Sc_pos.iloc[ind]
        pos2 = Sc_pos.drop(Sc_pos.index[ind])

        if viz is True:
            cm = plt.get_cmap('gist_rainbow')
            cmap = np.asarray([cm(1. * i / 2) for i in range(2)])
            colors = [np.tile(cmap[0, :], (len(pos1), 1)), np.tile(cmap[1, :], (len(pos2), 1))]

            multiple_volvis([pos1, pos2], colors=colors, obb=obb)

        Sc_pos = pos1
        data_Sc = Sc_pos.loc[:, ['x', 'y', 'z']].values
        dim = len(Sc_pos)

        # Measuring Gaussian Distance
        from sklearn.metrics.pairwise import rbf_kernel
        A_Sc = rbf_kernel(data_Sc, gamma=1./10.)
        A_Sc[A_Sc < 1e-3] = 0
        save = False
        if save is True:
            plt.imshow(A_Sc)
            plt.savefig("/home/larpanmuk/Documents/Papers_n_Thesis/APT_ML/fig/distance.png", bbox_inches='tight')
        A_Sc = normalize(A_Sc, norm='l1', axis=1)


        # Louvain Modularity Optimization

        G_Sc = nx.from_numpy_array(A_Sc)
        partition = community.best_partition(G_Sc)
        comm_Sc = np.asarray([val for (key, val) in partition.iteritems()])
        vol.append(convex_hull_clusters(Sc_pos, comm_Sc))
        com_num.append(len(np.unique(comm_Sc)))

        gmm = mixture.GaussianMixture(14, covariance_type='diag')
        gmm.fit(data_Sc)
        gmm_list.append(gmm)

    vol = np.asarray(vol)
    com_num = np.asarray(com_num)
    t_test.append(st.t.interval(1-alpha, len(vol)-1, loc=np.mean(vol), scale=st.sem(vol)))
    t_mean.append(np.mean(vol))
    com_mean.append(np.mean(com_num))
    com_test.append(st.t.interval(1-alpha, len(com_num)-1, loc=np.mean(com_num), scale=st.sem(com_num)))

import cPickle as pickle

gmm_list = pickle.load(open('gmm_list.p', 'rb'))
com_mean, com_test = pickle.load(open('com_list.p', 'rb'))
t_mean, t_test = pickle.load(open('t_list.p', 'rb'))

ft = 30
figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
ax = figure.add_subplot(111)
ax.plot(density, t_mean, linewidth=3, label='mean')
ax.plot(density, t_test, linewidth=3, label='3-sigma limit')
leg = ax.legend(prop={'size':ft})
ax.set_xlabel('Point Cloud Density')
ax.set_ylabel('Cluster Volume in cubic nm')
ax.xaxis.label.set_size(ft)
ax.yaxis.label.set_size(ft)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
       xtick.label.set_fontsize(ft)
       ytick.label.set_fontsize(ft)
plt.setp(ax.spines.values(), linewidth=3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.savefig('/home/larpanmuk/Documents/Papers_n_Thesis/APT_ML/fig/plot_vol.png', bbox_inches='tight')

figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
ax = figure.add_subplot(111)
ax.plot(density, com_mean, linewidth=3, label='mean')
ax.plot(density, com_test, linewidth=3, label='3-sigma limit')
leg = ax.legend(prop={'size':ft})
ax.set_xlabel('Point Cloud Density')
ax.set_ylabel('Cluster Number')
ax.xaxis.label.set_size(ft)
ax.yaxis.label.set_size(ft)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
       xtick.label.set_fontsize(ft)
       ytick.label.set_fontsize(ft)
plt.setp(ax.spines.values(), linewidth=3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.savefig('/home/larpanmuk/Documents/Papers_n_Thesis/APT_ML/fig/plot_com.png', bbox_inches='tight')

'''
unique_comm = np.unique(comm_Sc)
print 'Number of clusters are: ', len(unique_comm)
n_clusters = len(unique_comm)

clust_matrix = np.asarray([comm_Sc == k for k in np.unique(comm_Sc)]).T

overlapping = False
if not overlapping:
      clust_matrix = clust_matrix.astype(np.uint8)

if overlapping:
    D_cl = np.sum(A_Sc, axis=1)
    clust_matrix_overlap = np.zeros((dim, n_clusters))

    for i in range(dim):
        for j in np.unique(comm_Sc):
            clust_matrix_overlap[i, j] = np.sum(A_Sc[i, comm_Sc == j]) / D_cl[i]

    overlap_tol = 1e-3
    clust_matrix_overlap[clust_matrix_overlap < overlap_tol] = 0


cm = plt.get_cmap('gist_rainbow')
cmap = np.asarray([cm(1.*i/n_clusters) for i in range(n_clusters)])

cmap = np.random.rand(n_clusters, 3)
colors = []
for com in comm_Sc:
    ind = np.where(unique_comm == com)[0][0]
    colors.append(cmap[ind, :])

#colors = np.asarray(colors)

#colors = overlap_viz(clust_matrix_overlap)


'''
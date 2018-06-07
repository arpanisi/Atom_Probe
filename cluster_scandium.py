import os
from apt_importers import *
import numpy as np
from sklearn.preprocessing import scale, normalize


#import tensorflow as tf
import matplotlib.pyplot as plt

# Overlapping Visualization
def overlap_viz(cluster_matrix):

    dim = cluster_matrix.shape[0]
    n_clusters = cluster_matrix.shape[1]
    #cm = plt.get_cmap('gist_rainbow')
    #cmap = np.asarray([cm(1. * i / n_clusters) for i in range(n_clusters)])
    cmap = np.random.rand(n_clusters, 3)

    colors = []
    for cl in cluster_matrix:
        colors.append(np.average(cmap, axis=0, weights=cl))

    return np.asarray(colors)

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'Data', 'Scan')

# Reading the main file
example_file = os.path.join(DATA_DIR, '110-2.txt')
epos = read_file(example_file)

# Reading the range file
range_file = os.path.join(DATA_DIR, 'ranges.rrng')   # Loading the range file
ions, rrngs = read_rrng(range_file)

epos_label = label_ions(epos, rrngs)
dpos = deconvolve(epos_label)

volvis(dpos)
dpos_org = dpos
N = len(dpos)
ratio = (np.arange(0.1, 1, 0.1)[::-1] * N).astype(np.int)
from scipy.spatial import ConvexHull
hull = ConvexHull(dpos.loc[:, ['x', 'y', 'z']].values)

from pyflann import FLANN
fl = FLANN()

density = ratio/hull.volume
t_test = []
t_mean =[]
com_test = []
com_mean = []
for n in ratio:
    vol = []
    com_num = []
    for i in range(10):
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
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)
            for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
                            xtick.label.set_fontsize(20)
                            ytick.label.set_fontsize(20)
            plt.setp(ax.spines.values(), linewidth=3)
            ax.xaxis.set_tick_params(width=3)
            ax.yaxis.set_tick_params(width=3)
            plt.savefig('hist.png', bbox_inches='tight', dpi = 300)



        import scipy.stats as st
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

            multiple_volvis([pos1, pos2], colors=colors)



        Sc_pos = pos1
        data_Sc = Sc_pos.loc[:, ['x', 'y', 'z']].values
        dim = len(Sc_pos)



        # Measuring Gaussian Distance
        from sklearn.metrics.pairwise import rbf_kernel
        A_Sc = rbf_kernel(data_Sc, gamma=1./10.)
        A_Sc[A_Sc < 1e-3] = 0
        save=False
        if save is True:
            plt.imshow(A_Sc)
            plt.savefig("test.png", bbox_inches='tight')
        A_Sc = normalize(A_Sc, norm='l1', axis=1)


        # Louvain Modularity Optimization
        import community
        import networkx as nx
        G_Sc = nx.from_numpy_array(A_Sc)
        partition = community.best_partition(G_Sc)
        comm_Sc = np.asarray([val for (key, val) in partition.iteritems()])
        vol.append(convex_hull_clusters(Sc_pos, comm_Sc))
        com_num.append(len(np.unique(comm_Sc)))

    vol = np.asarray(vol)
    com_num = np.asarray(com_num)
    t_test.append(st.t.interval(1-alpha, len(vol)-1, loc=np.mean(vol), scale=st.sem(vol)))
    t_mean.append(np.mean(vol))
    com_mean.append(np.mean(com_num))
    com_test.append(st.t.interval(1-alpha, len(com_num)-1, loc=np.mean(com_num), scale=st.sem(com_num)))

figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
ax = figure.add_subplot(111)
ax.plot(density, t_mean, linewidth=2, label='mean')
ax.plot(density, t_test, linewidth=2, label='3-sigma limit')
leg = ax.legend(prop={'size':20})
ax.set_xlabel('Point Cloud Density')
ax.set_ylabel('Cluster Volume in cubic nm')
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
       xtick.label.set_fontsize(20)
       ytick.label.set_fontsize(20)
plt.setp(ax.spines.values(), linewidth=3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.savefig('plot_vol.png', bbox_inches='tight')

figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
ax = figure.add_subplot(111)
ax.plot(density, com_mean, linewidth=2, label='mean')
ax.plot(density, com_test, linewidth=2, label='3-sigma limit')
leg = ax.legend(prop={'size':20})
ax.set_xlabel('Point Cloud Density')
ax.set_ylabel('Cluster Number')
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
       xtick.label.set_fontsize(20)
       ytick.label.set_fontsize(20)
plt.setp(ax.spines.values(), linewidth=3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.savefig('plot_com.png', bbox_inches='tight')

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
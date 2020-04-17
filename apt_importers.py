#Code Imported from apt-tools https://github.com/oscarbranson/apt-tools
# Modified by Arpan Mukherjee
import pandas as pd
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from pyobb.obb import OBB
from vispy.scene import visuals
from vispy import app, scene, gloo
from open3d import *


def get_dpos(f, range_fille, corrected=True):

    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'Data', 'pos_files')
    example_file = os.path.join(DATA_DIR, f)
    epos = read_file(example_file)
    epos.insert(0, 'Index', epos.index)

    if corrected is True and f.endswith('epos'):
        epos = correct_ips(epos)

    # Reading the range file
    range_file = os.path.join(ROOT_DIR, 'Data', 'Range_Files', range_fille)  # Loading the range file
    ions, rrngs = read_rrng(range_file)

    epos_label = label_ions(epos, rrngs)

    #ind = epos_label.Comp == ''
    epos = deconvolve(epos_label)
    epos = epos.sort_values(by=['Index'])

    return epos

def correct_ips(epos):
    # Changing the ipp values
    unique_ipp = np.unique(epos.ipp)
    ipps = epos.ipp.values
    for ipp in unique_ipp:
        if ipp < 2:
            continue
        ind = np.where(epos.ipp == ipp)[0]
        for i in range(ipp - 1):
            ipps[ind + i + 1] = ipp

    ipps = ipps.reshape(ipps.size, 1)
    epos.ipp = ipps

    return epos

def read_file(f):
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'Data', 'pos_files')
    example_file = os.path.join(DATA_DIR, f)
    _, ext = os.path.splitext(f)
    if ext == '.epos':
        return read_epos(example_file)
    elif ext == '.txt':
        return read_txt(example_file)
    elif ext == '.pos':
        return read_pos(example_file)


def read_pos(f):
    """ Loads an APT .pos file as a pandas dataframe.
    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: mass/charge ratio of ion"""
    # read in the data
    n = len(open(f, 'rb').read())/4
    n = int(n)
    d = struct.unpack('>'+'f'*n, open(f, 'rb').read(4*n))
                    # '>' denotes 'big-endian' byte order
    # unpack data
    pos = pd.DataFrame({'x': d[0::4],
                        'y': d[1::4],
                        'z': d[2::4],
                        'Da': d[3::4]})
    return pos


def read_txt(f):

    fopen = open(f, 'rb')
    labels = fopen.readline().replace('m/n', 'Da')
    labels = labels.split()
    data = np.loadtxt(f, skiprows=1)

    return pd.DataFrame(data, columns=labels)


def read_epos(f):
    """Loads an APT .epos file as a pandas dataframe.
    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: Mass/charge ratio of ion
        ns: Ion Time Of Flight
        DC_kV: Potential
        pulse_kV: Size of voltage pulse (voltage pulsing mode only)
        det_x: Detector x position
        det_y: Detector y position
        pslep: Pulses since last event pulse (i.e. ionisation rate)
        ipp: Ions per pulse (multihits)
     [x,y,z,Da,ns,DC_kV,pulse_kV,det_x,det_y,pslep,ipp].
        pslep = pulses since last event pulse
        ipp = ions per pulse
    When more than one ion is recorded for a given pulse, only the
    first event will have an entry in the "Pulses since last evenT
    pulse" column. Each subsequent event for that pulse will have
    an entry of zero because no additional pulser firings occurred
    before that event was recorded. Likewise, the "Ions Per Pulse"
    column will contain the total number of recorded ion events for
    a given pulse. This is normally one, but for a sequence of records
    a pulse with multiply recorded ions, the first ion record will
    have the total number of ions measured in that pulse, while the
    remaining records for that pulse will have 0 for the Ions Per
    Pulse value.
        ~ Appendix A of 'Atom Probe tomography: A Users Guide',
          notes on ePOS format."""
    # read in the data
    n = len(open(f, 'rb').read())/4   # Number of variables
    n = int(n)
    rs = int(n / 11)                             # Number of atoms in the dataset
    fmt = '>'+'fffffffffII'*rs
    if struct.calcsize(fmt) != 4*n:
        return False
    # d = struct.unpack(fmt, open(f,'rb').read(4*n))
    d = struct.unpack('>' + 'f' * n, open(f, 'rb').read(4 * n))
                    # '>' denotes 'big-endian' byte order
    # unpack data
    pos = pd.DataFrame({'x': d[0::11],
                        'y': d[1::11],
                        'z': d[2::11],
                        'Da': d[3::11],
                        'ns': d[4::11],
                        'DC_kV': d[5::11],
                        'pulse_kV': d[6::11],
                        'det_x': d[7::11],
                        'det_y': d[8::11],
                        'pslep': d[9::11], # pulses since last event pulse
                        'ipp': d[10::11]}) # ions per pulse
    return pos

def read_epos_2(f):

    lines = open(f, 'rb').read()
    fmt = '>' + 'fffffffffII'
    pos = pd.DataFrame(columns=['x', 'y', 'z', 'Da', 'ns', 'DC_kV','pulse_kV', 'det_x', 'det_y', 'pslep', 'ipp'])
    line_list = []
    for i in range(0, len(lines), 44):
        line_list.append(list(struct.unpack(fmt, lines[i:i+44])))

def read_rrng(f):
    """Loads a .rrng file produced by IVAS. Returns two dataframes of 'ions'
    and 'ranges'."""
    import re

    rf = open(f,'r').readlines()

    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')

    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])

    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','Comp','Color'])
    rrngs.set_index('number',inplace=True)

    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['Comp','Color']] = rrngs[['Comp','Color']].astype(str)

    return ions, rrngs


def label_ions(pos, rrngs):
    """labels ions in a .pos or .epos dataframe (anything with a 'Da' column)
    with composition and colour, based on an imported .rrng file."""

    pos['Comp'] = ''
    pos['Color'] = '#FFFFFF'

    for n, r in rrngs.iterrows():
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper), ['Comp', 'Color']] = [r['Comp'],'#' + r['Color']]

    return pos


def deconvolve(lpos):
    """Takes a composition-labelled pos file, and deconvolves
    the complex ions. Produces a dataframe of the same input format
    with the extra columns:
       'element': element name
       'n': stoichiometry
    For complex ions, the location of the different components is not
    altered - i.e. xyz position will be the same for several elements."""

    import re

    out = []
    pattern = re.compile(r'([A-Za-z]+):([0-9]+)')
    lpos = lpos.loc[lpos.Comp != '']
    for g, d in lpos.groupby('Comp'):
         for i in range(len(g.split(' '))):
              tmp = d.copy()
              cn = pattern.search(g.split(' ')[i]).groups()
              tmp['Element'] = cn[0]
              tmp['n'] = cn[1]
              out.append(tmp.copy())
    return pd.concat(out)


def volvis_open3d(pos, obb=None):

    import matplotlib.colors as cols
    pcd = PointCloud()
    cpos = pos.loc[:, ['x', 'y', 'z']].values

    pcd.points = Vector3dVector(cpos)
    pcd.colors = Vector3dVector(np.asarray(list(pos.Color.apply(cols.hex2color))))

    if obb is None:
        obb = OBB.build_from_points(cpos)

    points = np.asarray(obb.points)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [5, 6], [6, 3],
                        [1, 4], [4, 7], [7, 2], [7, 6], [6, 5], [4, 5]]

    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector([[1, 0, 0] for i in range(len(lines))])

    draw_geometries([pcd, line_set])

def volvis(pos, size=3, alpha=1, colors=None, save=False, obb=None):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    import matplotlib.colors as cols
    import re
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    if isinstance(pos, pd.DataFrame):
        cpos = pos.loc[:, ['x', 'y', 'z']].values
    elif isinstance(pos, np.ndarray):
        cpos = pos

    if obb is None:
        obb = OBB.build_from_points(cpos)

    points = np.asarray(obb.points)
    faces = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [5, 6], [6, 3],
                        [1, 4], [4, 7], [7, 2], [7, 6], [6, 5], [4, 5]])

    if 'Color' in pos.columns:
        colours = np.asarray(list(pos.Color.apply(cols.hex2color)))

    if alpha is not 1:
        np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colours, edge_width=1, size=size) #, edge_width=0, size=size)

    view.add(p1)

    for f in faces:
        p2 = scene.visuals.Line()
        pois = np.asarray([points[f[0]], points[f[1]]])
        p2.set_data(pois, color='k')
        view.add(p2)

    face_colors = np.ones((faces.shape[0], 3))

    #mesh = scene.visuals.Mesh(vertices=points, faces=faces, face_colors=face_colors)
    #view.add(mesh)
    # make legend
    if 'Color' in pos.columns:
        ions = []
        cs = []
        for g, d in pos.groupby('Color'):
            ions.append(re.sub(r':1?|\s?', '', d.Comp.iloc[0]))
            cs.append(cols.hex2color(g))
        ions = np.array(ions)
        cs = np.asarray(cs)

        pts = np.array([[20] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T
        tpts = np.array([[30] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T

        legb = scene.widgets.ViewBox(parent=view, border_color='k', bgcolor='w')
        legb.pos = 200, 20
        legb.size = 100, 20 * len(ions) + 20

        leg = scene.visuals.Markers()
        leg.set_data(pts, face_color=cs)
        legb.add(leg)

        legt = scene.visuals.Text(text=ions, pos=tpts, color='k', anchor_x='left', anchor_y='center', font_size=20)

        legb.add(legt)

    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)
    # show viewer
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

    if save is True:
        from vispy import io
        img = canvas.render()
        io.write_png('original.png', img)

    return canvas


# Module to display multiple poses together in a single view
def multiple_volvis(poses, size=2, alpha=1, colors=None, save=False, obb=None):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    import matplotlib.colors as cols
    import re
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    if colors is None:
        colors = [None]*len(poses)
    for pos, color in zip(poses, colors):
        cpos = pos.loc[:, ['x', 'y', 'z']].values

        if colors is None:
            if 'colour' in pos.columns:
                colours2 = np.asarray(list(pos.colour.apply(cols.hex2color)))
            else:
                Dapc = pos.Da.values / pos.Da.max()
                colours2 = np.array(zip(Dapc, Dapc, Dapc))

            colours = colours2
        else:
            colours=color
            if alpha is not 1:
                np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

        p1 = scene.visuals.Markers()
        p1.set_data(cpos, face_color=colours, edge_width=1, size=5) #, edge_width=0, size=size)

        view.add(p1)

        if obb is not None:
            points = np.asarray(obb.points)
            faces = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [5, 6], [6, 3],
                                [1, 4], [4, 7], [7, 2], [7, 6], [6, 5], [4, 5]])
            for f in faces:
                p2 = scene.visuals.Line()
                pois = np.asarray([points[f[0]], points[f[1]]])
                p2.set_data(pois, color='k')
                view.add(p2)

        view.camera = 'turntable'
        axis = visuals.XYZAxis(parent=view.scene)

        ions = []
        cs = []
        for g, d in pos.groupby('Color'):
            ions.append(re.sub(r':1?|\s?', '', d.Comp.iloc[0]))
            cs.append(cols.hex2color(g))
        ions = np.array(ions)
        cs = np.asarray(cs)

        pts = np.array([[20] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T
        tpts = np.array([[30] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T


    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

    if save is True:
        from vispy import io
        img = canvas.render()
        io.write_png('multiple.png', img)

    return canvas


def convex_hull_clusters(pos, comm_str, pos2=None, save=False,
                         viz=False, obb=None, same_color=False, text=False):

    n_clusters = len(np.unique(comm_str))
    a, b = np.histogram(comm_str, bins=n_clusters)
    sort_a = np.argsort(a)[::-1]

    vertices = pos.loc[:, ['x', 'y', 'z']].values
    from scipy.spatial import ConvexHull
        #p = []
    volume = 0
    cm = plt.get_cmap('gist_rainbow')

    p = n_clusters
    if same_color is True:
        cmap = np.tile(cm(0), [p, 1])
    else:
        cmap = np.asarray([cm(1. * i / p) for i in range(p)])

    hulls = []
    if pos2 is not None:
        cpos_pos2 = np.c_[pos2.loc[:, ['x', 'y', 'z']].values, np.ones((len(pos2), 1))]
        pos2_ind = np.zeros(len(pos2), dtype=bool)
    for i in np.unique(comm_str):
        ind = np.where(comm_str == i)[0]
        if len(ind) < 10:
            continue
        hull = ConvexHull(vertices[ind, :])
        volume += hull.volume
        hulls.append(hull)

    if viz is True:
        from vispy import scene, app
        from vispy.scene import visuals

        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = canvas.central_widget.add_view()

        if obb is not None:
            points = np.asarray(obb.points)
            faces = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [5, 6], [6, 3],
                                [1, 4], [4, 7], [7, 2], [7, 6], [6, 5], [4, 5]])
            for f in faces:
                p2 = scene.visuals.Line()
                pois = np.asarray([points[f[0]], points[f[1]]])
                p2.set_data(pois, color='k')
                view.add(p2)

        for k, hull in enumerate(hulls):
            ''' 
            eq = hull.equations
            for point_i, point in enumerate(cpos_pos2):
                if (np.dot(point, eq.T) < 0).all():
                    pos2_ind[point_i] = True
            '''
            mesh = scene.visuals.Mesh(vertices=hull.points, faces=hull.simplices, color=cmap[k])
            view.add(mesh)


            #points = scene.visuals.Markers()
            #points.set_data(hull.points, size=5, face_color='k')
            #view.add(points)
            if text is True:
                t1 = scene.visuals.Text(str(k+1), pos=np.mean(hull.points[hull.vertices, :], axis=0), color='black')
                t1.font_size = 24
                view.add(t1)



        if pos2 is not None:
            import matplotlib.colors as cols
            cpos = pos2.loc[:, ['x', 'y', 'z']].values
            colours2 = np.asarray(list(pos2.Color.apply(cols.hex2color)))
            p1 = scene.visuals.Markers()
            p1.set_data(cpos, size=1, face_color=colours2)
            view.add(p1)

            #cpos_hull = cpos[pos2_ind, :]
            #p1 = scene.visuals.Markers()
            #p1.set_data(cpos_hull, size=5, face_color='r')
            #view.add(p1)

        view.camera = 'turntable'
        axis = visuals.XYZAxis(parent=view.scene)
        canvas.show()
        import sys
        if sys.flags.interactive == 0:
            app.run()

        if save is True:
            from vispy import io
            img = canvas.render()
            io.write_png('clusters.png', img)

    print('Volume of Cluster: ', volume)
    if viz is True:
        return canvas, hulls

    return hulls


def volvis_radius(pos, colors=None):

    import re
    import matplotlib.colors as cols
    if isinstance(pos, pd.DataFrame):
        cpos = pos.loc[:, ['x', 'y', 'z']].values
    elif isinstance(pos, np.ndarray):
        cpos = pos
    radius = pos.radius.values.astype(np.int)
    uniq_rad = np.unique(radius)

    p = len(uniq_rad)
    cm = plt.get_cmap('gist_rainbow')
    cmap = np.asarray([cm(1. * i / p) for i in range(p)])

    size = radius*5./100.

    if colors is None:
        colors = np.asarray([cmap[uniq_rad == rad] for rad in radius]).reshape(len(pos), 4)

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colors, size=size)

    view.add(p1)

    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)

    ion_table = dict()
    ion_table[73] = 'O'
    ion_table[117] = 'Si'
    ion_table[143] = 'Al'
    ion_table[216] = 'AlO'
    ion_table[260] = 'AlSi'
    ion_table[190] = 'SiO'
    ion_table[263] = 'SiO2'
    ions = uniq_rad
    cs = cmap

    pts = np.array([[20] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T
    tpts = np.array([[30] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T

    legb = scene.widgets.ViewBox(parent=view, border_color='k', bgcolor='w')
    legb.pos = 200, 20
    legb.size = 100, 20 * len(ions) + 20

    leg = scene.visuals.Markers()
    leg.set_data(pts, face_color=cs)
    legb.add(leg)

    legt = scene.visuals.Text(text=[ion_table[i] for i in uniq_rad], pos=tpts, color='k', anchor_x='left', anchor_y='center', font_size=16)

    legb.add(legt)

    canvas.show()
    import sys
    #if sys.flags.interactive == 0:
     #   app.run()

    return canvas


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


def singleton_removal(pos, k=10, alpha=0.01, algorithm=None, pos2=None):

    from pyflann import FLANN
    #from sklearn.preprocessing import scale
    from scipy.stats import chisquare, chi2

    fl = FLANN()
    data = pos.loc[:, ['x', 'y', 'z']].values
    if pos2 is None:
        data2 = data
    else:
        data2 = pos2.loc[:, ['x', 'y', 'z']].values
    if algorithm is None:
        _, dists = fl.nn(data2, data, k + 1, algorithm='kmeans',branching=32, iterations=7, checks=16)
    else:
        _, dists = fl.nn(data2, data, k + 1, algorithm=algorithm, branching=32, iterations=7, checks=16)
    dists = np.delete(dists, 0, axis=1)

    mu = np.average(dists, axis=0)
    cov_dists = np.cov(dists.T)
    chi_sq = np.asarray([chi2.cdf(np.dot(np.dot(x - mu, np.linalg.inv(cov_dists)), (x - mu).T), k) for x in dists])

    ind = chi_sq < alpha
    pos1 = pos.iloc[ind]
    pos2 = pos.drop(pos.index[ind])


    return pos1, pos2, ind
    #table = chisquare(dists.T)[1]


def animate(pos, batch=1000, size=3, alpha=1, colors=None, save=False, obb=None):

    import matplotlib.colors as cols
    data = pos.loc[:, ['x', 'y', 'z']].values
    det_data = np.c_[pos.loc[:, ['det_x', 'det_y']].values, -10. * np.ones((len(pos), 1))]

    if colors is None:
        colors = np.asarray(list(pos.colour.apply(cols.hex2color)))

    points = np.arange(batch, len(pos), batch)
    points = np.append(points, [len(pos)])
    n_times = len(points)

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    # 3D axis
    axis = scene.visuals.XYZAxis(parent=view.scene)
    view.camera = 'turntable'

    if obb is not None:
        points_obb = np.asarray(obb.points)
        faces = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [5, 6], [6, 3],
                            [1, 4], [4, 7], [7, 2], [7, 6], [6, 5], [4, 5]])
        for f in faces:
            p2 = scene.visuals.Line()
            pois = np.asarray([points_obb[f[0]], points_obb[f[1]]])
            p2.set_data(pois, color='k')
            view.add(p2)

    scatter = scene.visuals.Markers(pos=data[:1], face_color=colors[:1], parent=view.scene, size=size)
    detector = scene.visuals.Markers(pos=det_data, face_color=colors, parent=view.scene, size=size)


    def update(ev):
        global t, pos, color, line, k

        pos = data[:points[t]]
        print(len(pos))
        color = colors[:points[t]]
        scatter.set_data(pos, face_color=color, size=size)
        pos = det_data[points[t] + 1:]
        color = colors[points[t] + 1:]
        print(len(pos))
        detector.set_data(pos, face_color=color, size=5)

        t += 1

    t = 0
    print('Starting Animation')
    timer = app.Timer()
    timer.connect(update)
    timer.start(0.001, n_times)

    canvas.show()
    app.run()


def voxel_grid(dpos, n, obb=None):

    points = dpos.loc[:, ['x', 'y', 'z']].values

    xyzmin = points.min(0)
    xyzmax = points.max(0)

    if obb is None:
        margin = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
        xyzmin = xyzmin - margin / 2
        xyzmax = xyzmax + margin / 2



import pandas as pd
import struct
import numpy as np
import os

def read_pos(f):
    """ Loads an APT .pos file as a pandas dataframe.
    Columns:
        x: Reconstructed x position
        y: Reconstructed y position
        z: Reconstructed z position
        Da: mass/charge ratio of ion"""
    # read in the data
    n = len(file(f).read())/4
    d = struct.unpack('>'+'f'*n,file(f).read(4*n))
                    # '>' denotes 'big-endian' byte order
    # unpack data
    pos = pd.DataFrame({'x': d[0::4],
                        'y': d[1::4],
                        'z': d[2::4],
                        'Da': d[3::4]})
    return pos


def read_file(f):

    _, ext = os.path.splitext(f)
    if ext == '.epos':
        return read_epos(f)
    elif ext == '.txt':
        return read_txt(f)


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
    n = len(file(f, 'rb').read())/4
    rs = n / 11
    d = struct.unpack('>'+'fffffffffII'*rs,file(f,'rb').read(4*n))
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
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True)

    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)

    return ions, rrngs


def label_ions(pos, rrngs):
    """labels ions in a .pos or .epos dataframe (anything with a 'Da' column)
    with composition and colour, based on an imported .rrng file."""

    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'

    for n,r in rrngs.iterrows():
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp', 'colour']] = [r['comp'],'#' + r['colour']]

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

    for g,d in lpos.groupby('comp'):
        if g is not '':
            for i in range(len(g.split(' '))):
                tmp = d.copy()
                cn = pattern.search(g.split(' ')[i]).groups()
                tmp['element'] = cn[0]
                tmp['n'] = cn[1]
                out.append(tmp.copy())
    return pd.concat(out)

def volvis(pos, size=2, alpha=1, colors=None):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    from vispy import app, scene #, mpl_plot
    import numpy as np
    import sys
    import matplotlib.colors as cols
    import re
    from vispy.scene import visuals

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    cpos = pos.loc[:, ['x', 'y', 'z']].values

    if 'colour' in pos.columns:
        colours2 = np.asarray(list(pos.colour.apply(cols.hex2color)))
    else:
        Dapc = pos.Da.values / pos.Da.max()
        colours2 = np.array(zip(Dapc, Dapc, Dapc))

    if colors is not None:
        colours = colors
    else:
        colours = colours2
    if alpha is not 1:
        np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colours, edge_width=1, size=5) #, edge_width=0, size=size)

    view.add(p1)

    # make legend

    ions = []
    cs = []
    for g, d in pos.groupby('colour'):
        ions.append(re.sub(r':1?|\s?', '', d.comp.iloc[0]))
        cs.append(cols.hex2color(g))
    ions = np.array(ions)
    cs = np.asarray(cs)

    pts = np.array([[20] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T
    tpts = np.array([[30] * len(ions), np.linspace(20, 20 * len(ions), len(ions))]).T

    legb = scene.widgets.ViewBox(parent=view, border_color='red', bgcolor='k')
    legb.pos = 0, 0
    legb.size = 100, 20 * len(ions) + 20

    leg = scene.visuals.Markers()
    leg.set_data(pts, face_color=cs)
    legb.add(leg)

    legt = scene.visuals.Text(text=ions, pos=tpts, color='white', anchor_x='left', anchor_y='center', font_size=10)

    legb.add(legt)

    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)
    # show viewer
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

# Module to display multiple poses together in a single view
def multiple_volvis(poses, size=2, alpha=1, colors=None, save=False):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    from vispy import app, scene #, mpl_plot
    import numpy as np
    import sys
    import matplotlib.colors as cols
    import re
    from vispy.scene import visuals

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    for pos, color in zip(poses, colors):
        cpos = pos.loc[:, ['x', 'y', 'z']].values

        if 'colour' in pos.columns:
            colours2 = np.asarray(list(pos.colour.apply(cols.hex2color)))
        else:
            Dapc = pos.Da.values / pos.Da.max()
            colours2 = np.array(zip(Dapc, Dapc, Dapc))

        if colors is not None:
            colours = color
        else:
            colours = colours2
        if alpha is not 1:
            np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

        p1 = scene.visuals.Markers()
        p1.set_data(cpos, face_color=colours, edge_width=1, size=5) #, edge_width=0, size=size)

        view.add(p1)

        view.camera = 'turntable'
        axis = visuals.XYZAxis(parent=view.scene)

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

    if save is True:
        from vispy import io
        img = canvas.render()
        io.write_png('multiple.png', img)


def convex_hull_clusters(pos, comm_str, pos2=None, save=False, viz=False):

    n_clusters = len(np.unique(comm_str))
    a, b = np.histogram(comm_str, bins=n_clusters)
    sort_a = np.argsort(a)[::-1]

    vertices = pos.loc[:, ['x', 'y', 'z']].values


    if pos2 is not None:
        cpos = pos2.loc[:, ['x', 'y', 'z']].values
        import matplotlib.colors as cols
        colours2 = np.asarray(list(pos2.colour.apply(cols.hex2color)))

    if viz is True:
        cmap = np.random.rand(4, 3)
        from vispy import scene, app
        from vispy.scene import visuals

        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = canvas.central_widget.add_view()
        #p = []
    volume = []
    from scipy.spatial import ConvexHull
    for i in range(4):
        ind = np.where(comm_str == sort_a[i])[0]
        hull = ConvexHull(vertices[ind, :])

        if viz is True:
            faces = hull.simplices
            mesh = scene.visuals.Mesh(vertices=vertices[ind, :], faces=faces, color=cmap[i])

            #p1 = scene.visuals.Markers()
            #p1.set_data(cpos, size=1, face_color=colours2)

            view.add(mesh)
            #view.add(p1)

            view.camera = 'turntable'
            axis = visuals.XYZAxis(parent=view.scene)

        print 'Volume of Cluster: ', hull.volume
        volume.append(hull.volume)

    if viz is True:
        canvas.show()
        import sys
        if sys.flags.interactive == 0:
            app.run()

        if save is True:
            from vispy import io
            img = canvas.render()
            io.write_png('convex.png', img)

    print 'Volume of Cluster: ', hull.volume
    return hull.volume


def volvis_radius(pos, colors=None):
    from vispy import app, scene  # , mpl_plot
    import numpy as np
    import sys
    from vispy.scene import visuals

    cpos = pos.loc[:, ['x', 'y', 'z']].values
    radius = pos.loc[:,'radius'].values.astype(np.int)
    uniq_rad = np.unique(radius)
    cmap = np.random.rand(len(uniq_rad), 3)

    if colors is None:
        colors = np.asarray([cmap[uniq_rad == rad] for rad in radius]).reshape(len(pos), 3)

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    p1 = scene.visuals.Markers()
    p1.set_data(cpos, face_color=colors)

    view.add(p1)

    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)

    canvas.show()
    import sys
    if sys.flags.interactive == 0:
        app.run()





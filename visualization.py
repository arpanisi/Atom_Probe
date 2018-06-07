from apt_importers import *
import os
import numpy as np

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'Data', 'Scan')

filelist = []
for f in os.listdir(DATA_DIR):
    if f.endswith('.txt'):
        filelist.append(f)

df = pd.DataFrame()
for f in filelist:
    example_file = os.path.join(DATA_DIR, f)
    epos = read_file(example_file)
    df = df.append(epos)

example_file = os.path.join(DATA_DIR, 'R12_Al-Sc.epos')
range_file = os.path.join(DATA_DIR, 'ranges.rrng')

epos = read_file(example_file)
ions, rrngs = read_rrng(range_file)


lpos = label_ions(epos, rrngs)
dpos = deconvolve(lpos)

import matplotlib.pyplot as plt
figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
ax = figure.add_subplot(111)
dpos['radius'].value_counts().plot(ax=ax, kind='bar')
ax.set_xlabel('Element')
ax.set_ylabel('Frequency')
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
    xtick.label.set_fontsize(20)
data = dpos.loc[:, ['x', 'y', 'z']].values
plt.setp(ax.spines.values(), linewidth=3)
ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)
plt.savefig('bar.png', bbox_inches='tight')


'''
n = len(dpos)
N = 20000

ind = np.random.randint(n, size=N)
dpos = dpos.iloc[ind]
'''
Sc_pos = dpos.loc[dpos.element == 'Sc',:]

#print len(Sc_pos)
def volvis(pos, size=2, alpha=1):
    """Displays a 3D point cloud in an OpenGL viewer window.
    If points are not labelled with colours, point brightness
    is determined by Da values (higher = whiter)"""
    from vispy import app, scene #, mpl_plot
    from vispy.scene import visuals
    import numpy as np
    import sys
    import matplotlib.colors as cols
    import re
    from vispy import io

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    cpos = pos.loc[:, ['x', 'y', 'z']].values
    if 'colour' in pos.columns:
        colours = np.asarray(list(pos.colour.apply(cols.hex2color)))
    else:
        Dapc = lpos.Da.values / lpos.Da.max()
        colours = np.array(zip(Dapc, Dapc, Dapc))
    if alpha is not 1:
        np.hstack([colours, np.array([0.5] * len(colours))[..., None]])

    scatter = scene.visuals.Markers()
    scatter.set_data(cpos, face_color=colours, edge_width=1, size=5) #, edge_width=0, size=size)

    view.add(scatter)

    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)

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

    legb = scene.widgets.ViewBox(parent=view, border_color='k', bgcolor='w')
    legb.pos = 200, 20
    legb.size = 100, 20 * len(ions) + 20

    leg = scene.visuals.Markers()
    leg.set_data(pts, face_color=cs)
    legb.add(leg)

    legt = scene.visuals.Text(text=ions, pos=tpts, color='k', anchor_x='left', anchor_y='center', font_size=20)

    legb.add(legt)
    img = canvas.render()
    io.write_png('org_scan.png', img)
    # show viewer
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()


def convex_hull(pos):

    import sys
    from vispy import app, scene
    from scipy.spatial import ConvexHull
    datapos = pos.loc[:, ['x', 'y', 'z']].values

    hull = ConvexHull(datapos)
    simplices = hull.simplices

    indices = np.unique(simplices.reshape(simplices.size))

    vertices = datapos[indices, :]

    canvas = scene.SceneCanvas('APT Volume', keys='interactive')
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(up='z')

    p1 = scene.visuals.Markers()
    p1.set_data(datapos)  # , edge_width=0, size=size)

    mesh = scene.visuals.Mesh(vertices=datapos, faces=simplices, shading='smooth')
    view.add(mesh)
    canvas.show()

    print 'Area of the cloud: ', hull.area
    print 'Volume of the cloud: ', hull.volume

    if sys.flags.interactive == 0:
        app.run()

volvis(dpos)
#convex_hull(dpos)
#volvis(Sc_pos)
import matplotlib.colors as cols
from apt_importers import *
from clusters import community_structure
from scipy.spatial import ConvexHull
from PyQt5 import QtCore, QtWidgets, QtGui


class ObjectWidget(QtWidgets.QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_objet_changed = QtCore.pyqtSignal(name='objectChanged')

    def __init__(self, parent=None):
        super(ObjectWidget, self).__init__(parent)

        # Specifying Atom Size
        l_nbr_steps = QtWidgets.QLabel("Atom Size")
        self.nbr_steps = QtWidgets.QSpinBox()
        self.nbr_steps.setMinimum(1)
        self.nbr_steps.setMaximum(100)
        self.nbr_steps.setValue(6)
        self.nbr_steps.valueChanged.connect(self.update_param)

        # Camera Box
        l_cmap = QtWidgets.QLabel("Camera")
        self.camera_list = ['Turn Table', 'Arc Ball']
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItems(self.camera_list)
        self.combo.currentIndexChanged.connect(self.update_param)

        # Algorithm Box
        # self.pos_btn = QtWidgets.QPushButton('Load Pos File', self)
        # self.range_btn = QtWidgets.QPushButton('Load Range File', self)
        self.filter_btn = QtWidgets.QPushButton('Filtering', self)
        self.original_btn = QtWidgets.QPushButton('Reset', self)
        self.cluster_btn = QtWidgets.QPushButton('Cluster Analysis', self)

        #Slider
        self.k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.k_slider.setRange(5, 50)
        self.k_slider.setPageStep(5)

        gbox = QtWidgets.QGridLayout()
        # gbox.addWidget(self.pos_btn, 0, 0)
        # gbox.addWidget(self.range_btn, 0, 1)
        gbox.addWidget(l_cmap, 1, 0)
        gbox.addWidget(self.combo, 1, 1)
        gbox.addWidget(l_nbr_steps, 2, 0)
        gbox.addWidget(self.nbr_steps, 2, 1)
        gbox.addWidget(self.original_btn, 3, 0)
        gbox.addWidget(self.filter_btn, 3, 1)
        gbox.addWidget(self.k_slider)
        gbox.setColumnStretch(0, 1)
        gbox.addWidget(self.cluster_btn, 5, 0)
        self.k_slider.hide()
        self.cluster_btn.hide()

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(gbox)
        vbox.addStretch(2)

        self.setLayout(vbox)

    def update_param(self, option):
        self.signal_objet_changed.emit()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        # self.resize(200, 500)
        self.setWindowTitle('APT Demo')

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.props = ObjectWidget()
        splitter.addWidget(self.props)
        splitter.addWidget(self.canvas.native)

        self.setCentralWidget(splitter)
        self.props.signal_objet_changed.connect(self.update_view)
        self.props.filter_btn.clicked.connect(self.filter_view)
        self.props.original_btn.clicked.connect(self.original_view)
        self.update_view()

    def update_view(self):
        # banded, nbr_steps, cmap
        self.canvas.set_data(self.props.nbr_steps.value(),
                             self.props.combo.currentText())

    def filter_view(self):
        self.canvas.filter_data(self.props.nbr_steps.value(),
                                self.props.combo.currentText())
        self.props.k_slider.show()
        self.props.cluster_btn.show()
        self.props.k_slider.valueChanged.connect(self.update_slider)
        self.props.cluster_btn.clicked.connect(self.cluster_analysis)

    def original_view(self):
        self.canvas.original_view(self.props.nbr_steps.value(),
                                  self.props.combo.currentText())
        self.props.k_slider.hide()

    def update_slider(self, value):
        self.canvas.singleton(value, n_levels=self.props.nbr_steps.value(),
                              camera_name=self.props.combo.currentText())

    def cluster_analysis(self, camera_name):
        self.canvas.cluster_anomalies(camera_name)


class Canvas(scene.SceneCanvas):

    def __init__(self):
        scene.SceneCanvas.__init__(self, bgcolor='white', keys='interactive')
        self.size = 1200, 1200
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.radius = 2.0
        self.view.camera = scene.TurntableCamera(up='z')

        self.dpos = pd.read_csv('Data/dpos.csv')
        # self.cpos = np.zeros((1, 3)) #self.dpos.loc[:, ['x', 'y', 'z']].values
        # self.colours = np.ones((1, 3)) #np.asarray(list(self.dpos.Color.apply(cols.hex2color)))
        self.p1 = scene.visuals.Markers(parent=self.view.scene)


        self.filtered_cpos = None
        self.filtered_colours = None
        self.filtered_pos = None
        self.obb = None

        self.curr_cpos = self.dpos.loc[:, ['x', 'y', 'z']].values
        self.curr_cols = np.asarray(list(self.dpos.Color.apply(cols.hex2color)))
        self.p1.set_data(self.curr_cpos, face_color=self.curr_cols, edge_width=1, edge_color=(1, 1, 1, 1), size=6)
        self.pos1 = None

        self.comm_str = None
        self.meshes = []

        # self.freeze()

        # Add a 3D axis to keep us oriented
        # scene.visuals.XYZAxis(parent=self.view.scene)

    def set_data(self, n_levels, camera_name):

        self.p1.set_data(self.curr_cpos, face_color=self.curr_cols, edge_width=1, size=n_levels)
        if camera_name == 'Turn Table':
            self.view.camera = scene.TurntableCamera(up='z')
        elif camera_name == 'Arc Ball':
            self.view.camera = scene.ArcballCamera(up='z')

    def filter_data(self, n_levels, camera_name):

        if self.filtered_pos is None:
            self.filtered_pos = self.dpos.loc[self.dpos.Element == 'Sc', :]
            self.filtered_cpos = self.filtered_pos.loc[:, ['x', 'y', 'z']].values
            self.filtered_colours = np.asarray(list(self.filtered_pos.Color.apply(cols.hex2color)))
        self.pos1 = self.filtered_pos
        self.curr_cpos = self.filtered_cpos
        self.curr_cols = self.filtered_colours
        self.set_data(n_levels, camera_name)

    def original_view(self, n_levels, camera_name):

        self.curr_cols = np.asarray(list(self.dpos.Color.apply(cols.hex2color)))
        self.curr_cpos = self.dpos.loc[:, ['x', 'y', 'z']].values
        self.set_data(n_levels, camera_name)

    def singleton(self, k, n_levels, camera_name):

        self.pos1, _, _ = singleton_removal(self.filtered_pos, k=k, alpha=0.01)
        self.curr_cpos = self.pos1.loc[:, ['x', 'y', 'z']].values
        self.curr_cols = np.asarray(list(self.pos1.Color.apply(cols.hex2color)))
        self.set_data(n_levels, camera_name)

    def cluster_anomalies(self, camera_name):

        if self.comm_str is None:
            comm_str, _ = community_structure(self.pos1, algorithm='louvain')
            self.comm_str = comm_str
        vertices = self.pos1.loc[:, ['x', 'y', 'z']].values
        cm = plt.get_cmap('gist_rainbow')
        cmap = cm(0)
        # self.curr_cpos = self.cpos
        # self.curr_cols = self.colours
        # self.set_data(1, camera_name)
        for i in np.unique(self.comm_str):
            ind = np.where(self.comm_str == i)[0]
            if len(ind) < 10:
                continue
            hull = ConvexHull(vertices[ind, :])
            mesh = scene.visuals.Mesh(vertices=hull.points, faces=hull.simplices,
                                      color=cmap, parent=self.view.scene)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()

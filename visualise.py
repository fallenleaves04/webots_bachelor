"""
plik z funkcjami pomocniczymi wizualizacyjnymi
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import cv2 as cv
import camera_calibration as cc
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from park_algo import C
# Vehicle parameters
TRACK_FRONT = 1.628
TRACK_REAR = 1.628
WHEELBASE = 2.995
MAX_WHEEL_ANGLE = 0.5  # rad
CAR_WIDTH = 1.95
CAR_LENGTH = 4.85
s=1
car = cv.imread("pngegg.png")
car = cv.cvtColor(car,cv.COLOR_BGR2RGB)

class SpeedView(pg.GraphicsLayoutWidget):   
    def __init__(self,cont):
        super().__init__(title="Wykres prędkości")
        self.setBackground((235,235,250))
        
        
        self.running = True
        self.view_speed = self.addPlot(lockAspect=True)
        self.view_speed.setRange(xRange=[-30, 30], yRange=[-30, 30])
        self.view_speed.setMouseEnabled(x=True, y=True)
        self.view_speed.showGrid(x=True, y=True, alpha=0.3)
        self.view_speed.addLegend()

        # Bufory
        self.t1t, self.t1v1 = [], []
        self.t2v2 = []
        # Krzywe
        self.pl1 = self.view_speed.plot([], [], pen='r', name="Prędkość odometria [km/h]")
        self.pl2 = self.view_speed.plot([], [], pen='b', name="Prędkość supervisor [km/h]")
        #
        self.view_speed.setClipToView(True)
        self.view_speed.setDownsampling(mode='peak')

        cont.speedUpdated.connect(self.update_speed)
        cont.parkingToggled.connect(self.on_parking_change)

    @QtCore.pyqtSlot(object)
    def update_speed(self, speed_data):
        t = speed_data[0]
        v1 = speed_data[1] * 3.6  
        v2 = speed_data[2] * 3.6  
        # dopisz do buforów
        self.t1t.append(t); self.t1v1.append(v1); self.t2v2.append(v2)

        # zaktualizuj krzywe
        self.pl1.setData(self.t1t, self.t1v1)
        self.pl2.setData(self.t1t, self.t2v2)

    @QtCore.pyqtSlot(bool)
    def on_parking_change(self, is_parking):
        self.running = is_parking
        if is_parking:
            self.show()
        else:
            del self.t1t[:]
            del self.t1v1[:]
            del self.t2v2[:]
            self.hide()


       
class AngleView(pg.GraphicsLayoutWidget):   
    def __init__(self,cont):
        super().__init__(title="Wykres kąta skrętu i odchylenia")
        self.setBackground((235,235,250))

        
        self.running = True
        self.view_angle = self.addPlot(lockAspect=True)
        self.view_angle.setRange(xRange=[-30, 30], yRange=[-30, 30])
        self.view_angle.setMouseEnabled(x=True, y=True)
        self.view_angle.showGrid(x=True, y=True, alpha=0.3)
        self.view_angle.addLegend()

        # Bufory
        self.t1t, self.t1a = [], []
        self.t2y, self.t3y = [],[]
        # Krzywe
        self.pl1 = self.view_angle.plot([], [], pen='r', name="Kąt skrętu [rad]")
        self.pl2 = self.view_angle.plot([], [], pen='b', name="Kąt odchylenia odometria [rad]")
        self.pl3 = self.view_angle.plot([], [], pen='g', name="Kąt odchylenia z Webots [rad]")
        #
        self.view_angle.setClipToView(True)
        self.view_angle.setDownsampling(mode='peak')

        cont.angleUpdated.connect(self.update_angle)
        cont.parkingToggled.connect(self.on_parking_change)

    @QtCore.pyqtSlot(object)
    def update_angle(self, angle_data):
        t = angle_data[0]
        a = angle_data[1]  
        y = angle_data[2]
        y1 = angle_data[3]
        # dopisz do buforów
        self.t1t.append(t); self.t1a.append(a); self.t2y.append(y); self.t3y.append(y1)
        
        # zaktualizuj krzywe
        self.pl1.setData(self.t1t, self.t1a)
        self.pl2.setData(self.t1t, self.t2y)
        self.pl3.setData(self.t1t, self.t3y)

    @QtCore.pyqtSlot(bool)
    def on_parking_change(self, is_parking):
        self.running = is_parking
        if is_parking:
            self.show()
        else:
            del self.t1t[:]
            del self.t1a[:]
            del self.t2y[:]
            del self.t3y[:]
            self.hide()

class YawKappaView(pg.GraphicsLayoutWidget):   
    def __init__(self,cont):
        super().__init__(title="Wykresy skrętu kół, krzywizny i odchylenia w funkcji długości ścieżki")
        #self.setBackground((235,235,250))

        
        self.stack = QtWidgets.QStackedWidget()
        self.btns = QtWidgets.QHBoxLayout()

        self.btns_layout()

        self.kappa_plot = self.make_plots("Krzywizna κ(s)", "s [m]", "κ [1/m]", "Krzywizna śledzona", "Krzywizna rzeczywista")
        self.error_plot = self.make_plots("Błąd poprzeczny e(s)", "s [m]", "e [m]", None, None)
        self.delta_plot = self.make_plots("Skręt kół δ(s)", "s [m]", "δ [rad]", "δ_ref", "δ_exec")
        
        
        self.stack.addWidget(self.kappa_plot)
        self.stack.addWidget(self.error_plot)
        self.stack.addWidget(self.delta_plot)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.btns)
        layout.addWidget(self.stack)

        # dane
        self.s = []
        self.kappa_exec = []
        self.er_exec = []
        self.delta_exec = []

        self.kappa_ref = []
        self.delta_ref = []

        self.exec_data = {}   # key: ind, value: (er, delta, kappa)
        
        cont.funcCarUpdated.connect(self.update_data)
        cont.pathUpdated.connect(self.set_path)
        cont.parkingToggled.connect(self.on_parking_change)


    def btns_layout(self):
        for i, name in enumerate(["Krzywizna", "Błąd poprzeczny", "Skręt"]):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _, i=i: self.stack.setCurrentIndex(i))
            self.btns.addWidget(b)

    def make_plots(self,title, xlabel, ylabel, name_ref, name_exec):
        w = pg.PlotWidget(title=title)
        w.showGrid(True, True)
        w.setLabel('bottom', xlabel)
        w.setLabel('left', ylabel)
        w.addLegend()
        if name_ref is not None:
            w.ref = w.plot(pen=pg.mkPen('g', width=2), name=name_ref)
        w.exec = w.plot(pen=pg.mkPen('r', width=2), name=name_exec)
        return w
    
    def set_path(self, path):
        self.exec_data.clear()
        self.s = path.s[:]
        self.kappa_ref = path.curvs[:]
        self.delta_ref = [np.arctan(C.WHEELBASE * k) for k in self.kappa_ref] # tg(delta) = L/R; tg(delta) = L*kappa

        self.kappa_plot.ref.setData(self.s, self.kappa_ref)
        self.delta_plot.ref.setData(self.s, self.delta_ref)
        #self.error_plot.ref.setData(self.s, np.zeros(len(self.s)))

    def update_data(self, data):
        if not self.s:
            return 

        er, delta, kappa, ind = data
        if ind < 0 or ind >= len(self.s):
            return

        self.exec_data[ind] = (er, delta, kappa)

        inds = sorted(self.exec_data.keys())
        s_vals = [self.s[i] for i in inds]

        er_vals = [self.exec_data[i][0] for i in inds]
        delta_vals = [self.exec_data[i][1] for i in inds]
        kappa_vals = [self.exec_data[i][2] for i in inds]

        self.error_plot.exec.setData(s_vals, er_vals)
        self.delta_plot.exec.setData(s_vals, delta_vals)
        self.kappa_plot.exec.setData(s_vals, kappa_vals)

    def on_parking_change(self, is_parking):
        self.running = is_parking

        self.exec_data.clear()
        self.s = []
        self.kappa_ref = []
        self.delta_ref = []
        self.kappa_plot.exec.setData([], [])
        self.error_plot.exec.setData([], [])
        self.delta_plot.exec.setData([], [])

        if hasattr(self.kappa_plot, "ref"):
            self.kappa_plot.ref.setData([], [])
        if hasattr(self.delta_plot, "ref"):
            self.delta_plot.ref.setData([], [])

        if is_parking:
            self.show()
        else:
            self.hide()

class TrajView(pg.GraphicsLayoutWidget):
    def __init__(self,cont):
        super().__init__(title="Trajectory Display")
        self.setBackground((235,235,250))

        self.running = True
        self.controller = cont
        self.path_car_inserted = False
        # siatka 2, trajektoria
        #self.nextColumn()  # obok, w tej samej linii
        self.view_trajectory = self.addPlot(lockAspect=True)
        self.view_trajectory.setAspectLocked(lock=True, ratio=1)
        self.view_trajectory.setRange(xRange=[-30, 30], yRange=[-30, 30])
        self.view_trajectory.setMouseEnabled(x=True, y=True)
        self.view_trajectory.showGrid(x=True, y=True, alpha=0.3)
        self.view_trajectory.addLegend()
        
        # Bufory
        self.t1x, self.t1y = [], []
        self.t2x, self.t2y = [], []
        self.t3x, self.t3y = [], []
        self.t4x, self.t4y = [], []
        self.t5x, self.t5y = [], []
        self.t6x, self.t6y = [], []
        self.tx_yolo,self.ty_yolo = [],[]
        # Krzywe
        self.traj_curve1 = self.view_trajectory.plot([], [], pen='r', name="Odometria")
        self.traj_curve2 = self.view_trajectory.plot([], [], pen='b', name="Webots")
        self.traj_curve3 = self.view_trajectory.plot([], [], pen=None, symbol='s',symbolSize=4,symbolBrush='k',symbolPen=None, name="Przeszkody")
        #self.traj_curve_yolo = self.view_trajectory.plot([], [], pen=None, symbol='t',symbolSize=2,symbolBrush='k',symbolPen=None, name="YOLO")
        self.traj_curve4 = self.view_trajectory.plot([], [], pen=None, symbol='o',symbolSize=5,symbolBrush='k',symbolPen=None, name="Miejsca pojazdu")
        self.traj_curve5 = self.view_trajectory.plot([], [], pen='g', name="Przykładowa ścieżka")
        self.traj_curve6 = self.view_trajectory.plot([], [], pen=pg.mkPen(color=(0, 100, 100), width=1.5), name="Ścieżka Hybrid")
        #
        self.view_trajectory.setClipToView(True)
        self.view_trajectory.setDownsampling(mode='peak')
        #
        self.path_drawn = False
        self.car_rect = self.make_car()
        pen = pg.mkPen(None)
        self.car_rect.setPen(pen)
        self.car_rect.setBrush(QtGui.QBrush(QtGui.QColor(60, 10, 60, 150)))
        self.view_trajectory.addItem(self.car_rect)
        #
        self.obstacle_items = []
        self.spots_items = []
        self.text_items = []
        
        self.view_trajectory.scene().sigMouseClicked.connect(self.on_mouse_click)

        cont.parkingToggled.connect(self.on_parking_change)
        cont.trajUpdated.connect(self.update_trajectory)
        cont.pathUpdated.connect(self.draw_path)
        cont.expansionUpdated.connect(self.draw_expansion_cars)
        cont.hmapUpdated.connect(self.draw_hmap)
        cont.pathCarUpdated.connect(self.draw_path_car)
        
    def make_car(self,l_c=C.CAR_LENGTH,w_c=C.CAR_WIDTH):
        pts = [
            QtCore.QPointF(-1, -w_c/2),
            QtCore.QPointF(l_c-1, -w_c/2),
            QtCore.QPointF(l_c-1, w_c/2),
            QtCore.QPointF(-1, w_c/2)
        ]
        polygon = QtGui.QPolygonF(pts)
        car = QtWidgets.QGraphicsPolygonItem(polygon)
        t = QtGui.QTransform()
        return car 
    def make_rect(self,x_center,y_center,l_c,w_c,yaw):
        pts = [
            QtCore.QPointF(-l_c/2, -w_c/2),
            QtCore.QPointF(l_c/2, -w_c/2),
            QtCore.QPointF(l_c/2, w_c/2),
            QtCore.QPointF(-l_c/2, w_c/2)
        ]
        polygon = QtGui.QPolygonF(pts)
        item = QtWidgets.QGraphicsPolygonItem(polygon)
        t = QtGui.QTransform()
        t.translate(x_center, y_center)
        t.rotate(np.degrees(yaw))   
        item.setTransform(t)
        pen = pg.mkPen(color=(0, 0, 0), width=1)
        item.setPen(pen)
        item.setBrush(QtGui.QBrush(QtGui.QColor(80,80,80,120)))
        
        return item 
    
    def on_mouse_click(self, event):
        """Obsługa kliknięć myszy"""
        # jak się kliknie podwójnie na miejsce, to ono się podświetli i wyśle się wybrane miejsce (np. w controller)
        if event.double() and event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if not self.view_trajectory.sceneBoundingRect().contains(pos):
                return
            mouse_point = self.view_trajectory.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            clicked_items = self.view_trajectory.scene().items(pos)
            
                
            for item in clicked_items:
                if hasattr(item, 'spot_data'):
                    print("[TrajView] Kliknięto w miejsce:", item.spot_data['center'])
                    self.controller.goal_pose = (item.spot_data['target_rear_axle'][0],item.spot_data['target_rear_axle'][1],item.spot_data['orientation'])
                    self.controller.state = "waiting_for_confirm_start"
                    print("[TrajView] Miejsce wybrane. Wciśnij przycisk E, aby rozpocząć planowanie")
                    item.setPen(pg.mkPen(color=(0, 255, 0), width=3))
                    item.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 0, 100)))
                    break # Znaleziono, przerywamy pętlę
        
                    

    @QtCore.pyqtSlot(object)
    def update_trajectory(self, data):
        x1, y1 = data[0][0], data[0][1]
        x2, y2 = data[1][0], data[1][1]
        # dopisz douforów
        self.t1x.append(x1); self.t1y.append(y1)
        self.t2x.append(x2); self.t2y.append(y2)
        if not all(v is None for v in data[2]):
            self.t3x = data[2][0]; self.t3y = data[2][1]
            self.traj_curve3.setData(self.t3x, self.t3y)
        # if not all(v is None for v in data[5]):
        #     self.tx_yolo = data[5][0]; self.ty_yolo = data[5][1]
        #     self.traj_curve_yolo.setData(self.tx_yolo, self.ty_yolo)
        # zaktualizuj krzywe
        self.traj_curve1.setData(self.t1x, self.t1y)
        self.traj_curve2.setData(self.t2x, self.t2y)
        
        
        
        for item in self.obstacle_items:
            self.view_trajectory.removeItem(item)
        for item in self.spots_items:
            self.view_trajectory.removeItem(item)  
        for item in self.text_items:
            self.view_trajectory.removeItem(item)  
        self.obstacle_items.clear()
        self.spots_items.clear()
        self.text_items.clear()
        for obstacle in data[3]:
            center_x,center_y = obstacle['center'][0],obstacle['center'][1]
            leng,wid,ang = obstacle['length'],obstacle['width'],obstacle['angle']
            obs = self.make_rect(center_x,center_y,leng,wid,ang)
            self.obstacle_items.append(obs)

            #text = pg.TextItem(f"{obstacle['type']}", color='r', anchor=(0.5, 0.5))
            #text.setPos(center_x, center_y)  # Ustaw pozycję w układzie współrzędnych wykresu
            #self.text_items.append(text)
        for obs in self.obstacle_items:
            self.view_trajectory.addItem(obs)
        
        for spotss in data[4]:
            x_r,y_r = spotss['target_rear_axle'][0],spotss['target_rear_axle'][1]
            self.t4x.append(x_r); self.t4y.append(y_r)
            yaw = spotss['orientation']
            
            x_center = spotss['center'][0]
            y_center = spotss['center'][1]
            spot = self.make_rect(x_center,y_center,C.CAR_LENGTH,C.CAR_WIDTH,yaw)

            spot.spot_data = spotss
            self.spots_items.append(spot)
            self.traj_curve4.setData(self.t4x,self.t4y)
        for spot in self.spots_items:
            self.view_trajectory.addItem(spot)
        
        # for p_path in data[5]:
        #     for pp_path in p_path:
        #         p_x,p_y,p_yaw = pp_path
        #         self.t5x.append(p_x);self.t5y.append(p_y)
        # if len(data) > 6 and data[6] is not None:
        #     for j in range(len(data[6])):
        #         self.t6x.append(data[6][j])
        #         self.t6y.append(data[6][j])
        
        self.traj_curve5.setData(self.t5x,self.t5y)
        self.t4x.clear();self.t4y.clear()
        self.t5x.clear();self.t5y.clear()
        car_yaw = data[0][2]
        self.transform_car_item(self.car_rect,(x1,y1,car_yaw))

    @QtCore.pyqtSlot(object)
    def draw_path_car(self,pose):
        if not self.path_car_inserted:
            self.path_car_item = self.make_car()
            pen = pg.mkPen(None)
            self.path_car_item.setPen(pen)
            self.path_car_item.setBrush(QtGui.QBrush(QtGui.QColor(110, 110, 110, 50)))
            self.path_car_item.setZValue(-10) 
            self.view_trajectory.addItem(self.path_car_item)
            self.path_car_inserted = True
        self.transform_car_item(self.path_car_item,pose)
    @QtCore.pyqtSlot(object)
    def draw_path(self, path):
        if not self.controller.parking or self.controller.state == "inactive":
            return
        self.t6x = path.xs[:] 
        self.t6y = path.ys[:]
        self.traj_curve6.setData(self.t6x,self.t6y)
        
        print(f"[TrajView] Ścieżka narysowana: {len(self.t6x)} punktów")

    def transform_car_item(self,item,pose):
        x,y,yaw = pose
        t = QtGui.QTransform()
        t.translate(x, y)
        t.rotate(np.degrees(yaw))  
        item.setTransform(t)

    @QtCore.pyqtSlot(object)
    def draw_expansion_cars(self,car_pos):
        car_item = self.make_car()
        pen = pg.mkPen(None)
        car_item.setPen(pen)
        car_item.setBrush(QtGui.QBrush(QtGui.QColor(220, 220, 220, 10)))
        car_item.setZValue(-10) 
        self.transform_car_item(car_item,car_pos)
        self.view_trajectory.addItem(car_item)

    @QtCore.pyqtSlot(object)
    def draw_hmap(self,hmap_output):
        hmap, minx, miny, xw, yw = hmap_output
        # zaczynając od górnego lewego
        pts = [
            QtCore.QPointF(minx - xw, miny - yw),
            QtCore.QPointF(minx,miny - yw),
            QtCore.QPointF(minx,miny),
            QtCore.QPointF(minx - xw,miny)
        ]
        polygon = QtGui.QPolygonF(pts)
        map = QtWidgets.QGraphicsPolygonItem(polygon)
        pen = pg.mkPen(color=(0, 0, 0), width=1)
        map.setPen(pen)
        map.setBrush(QtGui.QBrush(QtGui.QColor("transparent")))
        self.view_trajectory.addItem(map)
        hmap_copy = hmap.copy()
        max_val = np.max(hmap_copy[~np.isinf(hmap_copy)]) if np.any(~np.isinf(hmap_copy)) else 10
        hmap_copy[np.isinf(hmap_copy)] = max_val * 1.5  # Ściany jako bardzo jasne/ciemne

        # Tworzenie ImageItem
        self.hmap_item = pg.ImageItem(hmap_copy)
        
        colormap = pg.colormap.get('viridis') 
        self.hmap_item.setLookupTable(colormap.getLookupTable())

        self.hmap_item.setOpacity(0.6)
        tr = QtGui.QTransform()
        
        origin_x = minx * C.XY_RESOLUTION
        origin_y = miny * C.XY_RESOLUTION
        
        tr.translate(origin_x, origin_y)
        tr.scale(C.XY_RESOLUTION, C.XY_RESOLUTION)
        
        self.hmap_item.setTransform(tr)
        self.hmap_item.setZValue(-10)
        self.view_trajectory.addItem(self.hmap_item)
        print("[TrajView] Mapa heurystyki narysowana.")

    @QtCore.pyqtSlot(bool)
    def on_parking_change(self, is_parking):
        self.running = is_parking
        if is_parking:
            self.show()
        else:
            self.t1x.clear(); self.t1y.clear()
            self.t2x.clear(); self.t2y.clear()
            self.t6x.clear(); self.t6y.clear()
            self.traj_curve1.setData([], [])
            self.traj_curve2.setData([], [])
            self.traj_curve6.setData([], [])   
            self.traj_curve3.setData([], [])
            self.traj_curve4.setData([], [])
            #self.traj_curve_yolo.setData([], [])
            self.hide()

class SensorView(pg.GraphicsLayoutWidget):
    def __init__(self,cont):
        super().__init__(title="Parking-Sensor Display")
        self.view = self.addViewBox(lockAspect=True)
        self.view.setRange(xRange=[-8,10], yRange=[-10,10])

        self.setBackground((235,235,250))

        self.view.setMouseEnabled(x=True, y=True)
        self.running = True

        self.n_strips = 8
        self.acceptable_dist = 5.0
        self.cx_b = 0
        self.cy_b_front = 3
        self.cy_b_rear = 0
        self.ang_start = -15 #-15
        self.ang_span = -37.5 #-37.5
        self.delta_r = 0.3
        self.dist_span = 3
        self.margin_car = CAR_WIDTH/2+0.1
        self.a_out_b = self.dist_span
        self.a_in_b = self.margin_car-self.delta_r

        self.div = (self.a_out_b - self.a_in_b)/self.n_strips
        self.margin_color_dist = 0.2
        # wizualizacja składa się z kapsuły, której sektory składają się z zapalanych w zależności od odległości pasków
        # wokół samochodu są tworzone ograniczone PathItem, których jest 12,

        # nazwy czujników - ZMIENIĆ JEŻELI SIĘ ZMIENIA W MAINIE
        self.front_sensor_names = [
            "distance sensor front right",
            "distance sensor front righter",
            "distance sensor front lefter",
            "distance sensor front left",
        ]
        self.rear_sensor_names = [
            "distance sensor right",
            "distance sensor righter",
            "distance sensor lefter",
            "distance sensor left",
        ]
        self.left_side_sensor_names = [
            "distance sensor left front side",
            "distance sensor left side",
        ]
        self.right_side_sensor_names = [
            "distance sensor right front side",
            "distance sensor right side",
        ]

        # iterowanie segmentów przednich i tylnych
        self._idx_front = {name:i for i,name in enumerate(self.front_sensor_names)}
        self._idx_rear  = {name:i for i,name in enumerate(self.rear_sensor_names)}
        self._idx_left_side = {name:i for i,name in enumerate(self.left_side_sensor_names)}
        self._idx_right_side = {name:i for i,name in enumerate(self.right_side_sensor_names)}

        # cache segmentów
        self._seg_front = [[None]*self.n_strips for _ in range(4)]
        self._seg_rear  = [[None]*self.n_strips for _ in range(4)]
        self._seg_left_side  = [[None]*self.n_strips for _ in range(2)]
        self._seg_right_side = [[None]*self.n_strips for _ in range(2)]

        for i_seg in range(4): # # cztery kliny sensory z przodu
            for k in range(self.n_strips):
                _ = self._build_front_segments(k, i_seg)
                _.setVisible(False)  # na starcie wyłączone

        for i_seg in range(4): # # cztery kliny sensory z tyłu
            for k in range(self.n_strips):
                _ = self._build_rear_segments(k, i_seg)
                _.setVisible(False)

        for i_seg in range(2): # # cztery kliny sensory z prawej
            name = self.right_side_sensor_names[i_seg]
            for k in range(self.n_strips):
                _ = self._build_right_side_segments(name,k, i_seg)
                _.setVisible(False)  # na starcie wyłączone

        for i_seg in range(2):
            name = self.left_side_sensor_names[i_seg]
            for k in range(self.n_strips): # # cztery kliny sensory z lewej
                _ = self._build_left_side_segments(name,k,i_seg)
                _.setVisible(False)


        self.colors = [
            QtGui.QColor(255,   0,   0, 127),   # czerwony
            QtGui.QColor(255, 128,   0, 127),   # pomarańczowy
            QtGui.QColor(255, 255,  51, 127),   # żółty
            QtGui.QColor(102, 255, 102, 127)    # zielony
        ]
        self._insert_car()
        self._create_base_front_sector()
        self._create_base_rear_sector()
        self._create_base_side_left()
        self._create_base_side_right()

        cont.parkingToggled.connect(self.on_parking_change)
        cont.sensorUpdated.connect(self.update_sensors)
        #cont.locUpdated.connect(self.update_location)

    def color_for_dist(self,strip_idx):
        if strip_idx is None:
            return QtGui.QColor(255, 255, 255, 0)  

        n = self.n_strips
        n_colors = len(self.colors)
        zone_size = n / n_colors
        color_idx = int(strip_idx / zone_size)

        color_idx = min(color_idx, n_colors - 1)
        return self.colors[color_idx]

    def segment_for_dist(self,val,minim):
        n = self.n_strips
        maxd = self.acceptable_dist
        if val > maxd - 1e-2:
            return None  

        width = (maxd - minim) / n
        idx = int((val - minim) / width)

        return min(max(idx, 0), n - 1)

    def _insert_car(self):

        # dodaj samochod na srodku
        pix = QtGui.QPixmap(r"D:\\User Files\BACHELOR DIPLOMA\\Pliki symulacyjne\\controllers\\parking_parallel_new\\pngegg.png")
        assert not pix.isNull()
        item = QtWidgets.QGraphicsPixmapItem(pix)

        item.setTransformationMode(QtCore.Qt.FastTransformation)
        pw, ph = pix.width(), pix.height()
        t = QtGui.QTransform()
        t.translate(CAR_WIDTH/2, CAR_LENGTH/2 + (CAR_LENGTH/2 - 1)) # żeby środek w (0,0)
        t.scale(CAR_WIDTH / pw, CAR_LENGTH / ph) # (Y) szerokość, (X) długość
        t.rotate(180)
        item.setTransform(t)
        self.view.addItem(item)

    def _create_base_front_sector(self):
        #zbudować przednie sektory
        # od 15 stopni do 165 (180-15) co 37.5

        cx_b = 0
        cy_b = 3
        ang_start = -15 #-15
        ang_span = -37.5 #-37.5
        delta_r = 0.3
        dist_span = 3
        margin_car = CAR_WIDTH/2+0.1
        a_out_b = dist_span
        a_in_b = margin_car-delta_r

        pen = QtGui.QPen(QtGui.QColor(30,30,30,60))
        pen.setCosmetic(True)
        brush = QtGui.QBrush(QtGui.QColor(120,190,255,60))

        for i in range(4):
            arc = QtGui.QPainterPath()
            global_span = i*ang_span
            ang_start_i = ang_start + global_span

            a_out = a_out_b
            a_in = a_in_b
            div = (a_out_b - a_in_b)/self.n_strips
            cx = cx_b
            cy = cy_b

            rect_out = QtCore.QRectF(cx-a_out,cy - a_out,a_out*2,a_out*2)
            rect_in = QtCore.QRectF(cx-a_in,cy - a_in,a_in*2,a_in*2)
            arc.moveTo(cx+a_out*np.cos(np.deg2rad(ang_start_i)),cy-a_out*np.sin(np.deg2rad(ang_start_i)))
            arc.arcTo(rect_out,ang_start_i,ang_span)
            arc.arcTo(rect_in,ang_start_i+ang_span,-ang_span)
            arc.closeSubpath()

            item = QtWidgets.QGraphicsPathItem(arc)
            item.setPen(pen)
            t = QtGui.QTransform()
            t.translate(delta_r*np.cos(np.deg2rad(ang_start_i + ang_span/2)),-delta_r*np.sin(np.deg2rad(ang_start_i + ang_span/2)))
            item.setTransform(t)
            self.view.addItem(item)


    def _create_base_rear_sector(self):

        cx_b = 0
        cy_b = 0
        ang_start = 15 #-15
        ang_span = 37.5 #-37.5
        delta_r = 0.3
        dist_span = 3
        margin_car = CAR_WIDTH/2+0.1
        a_out_b = dist_span
        a_in_b = margin_car-delta_r

        pen = QtGui.QPen(QtGui.QColor(30,30,30,60))
        pen.setCosmetic(True)


        for i in range(4):

            arc = QtGui.QPainterPath()
            global_span = i*ang_span
            ang_start_i = ang_start + global_span

            a_out = a_out_b
            a_in = a_in_b
            cx = cx_b
            cy = cy_b
            rect_out = QtCore.QRectF(cx-a_out,cy - a_out,a_out*2,a_out*2)
            rect_in = QtCore.QRectF(cx-a_in,cy - a_in,a_in*2,a_in*2)
            arc.moveTo(cx+a_out*np.cos(np.deg2rad(ang_start_i)),cy-a_out*np.sin(np.deg2rad(ang_start_i)))
            arc.arcTo(rect_out,ang_start_i,ang_span)
            arc.arcTo(rect_in,ang_start_i+ang_span,-ang_span)
            arc.closeSubpath()

            item = QtWidgets.QGraphicsPathItem(arc)
            item.setPen(pen)
            #item.setBrush(brush)
            t = QtGui.QTransform()
            t.translate(delta_r*np.cos(np.deg2rad(ang_start_i + ang_span/2)),-delta_r*np.sin(np.deg2rad(ang_start_i + ang_span/2)))
            item.setTransform(t)
            self.view.addItem(item)


    def _create_base_side_right(self):
        cx_b = 0
        cy_b = 3

        ang_start = -15 #-15
        ang_span = -150 #-37.5
        delta_r = 0.3

        dist_span = 3
        margin_car = CAR_WIDTH/2+0.1
        a_out_b = dist_span
        a_in_b = margin_car-delta_r

        pen = QtGui.QPen(QtGui.QColor(30,30,30,60))
        pen.setCosmetic(True)


        side_upper = QtGui.QPainterPath()
        side_upper.moveTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_upper.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_upper.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
        side_upper.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
        side_upper.closeSubpath()

        cx_b = 0
        cy_b = 0
        ang_start = 15 #-15
        ang_span = 150 #-37.5
        delta_r = 0.3
        a_out_b = dist_span
        a_in_b = margin_car-delta_r
        side_lower = QtGui.QPainterPath()
        side_lower.moveTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_lower.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_lower.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
        side_lower.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
        side_lower.closeSubpath()

        t = QtGui.QTransform()
        t.translate(delta_r,0)

        item_upper = QtWidgets.QGraphicsPathItem(side_upper)
        item_upper.setTransform(t)
        item_upper.setPen(pen)
        self.view.addItem(item_upper)

        item_lower = QtWidgets.QGraphicsPathItem(side_lower)
        item_lower.setTransform(t)
        item_lower.setPen(pen)
        self.view.addItem(item_lower)
        ########################

    def _create_base_side_left(self):
        delta_r = 0.3
        dist_span = 3
        margin_car = CAR_WIDTH/2+0.1
        a_out_b = dist_span
        a_in_b = margin_car-delta_r


        ang_start = 15 #-15
        ang_span = 150 #-37.5
        cx_b = 0
        cy_b = 3

        pen = QtGui.QPen(QtGui.QColor(30,30,30,60))
        pen.setCosmetic(True)

        side_upper = QtGui.QPainterPath()
        side_upper.moveTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))

        side_upper.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_upper.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
        side_upper.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
        side_upper.closeSubpath()
        ang_start = -ang_start #-15
        ang_span = -ang_span #-37.5
        cx_b = 0
        cy_b = 0
        side_lower = QtGui.QPainterPath()
        side_lower.moveTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_lower.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
        side_lower.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
        side_lower.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
        side_lower.closeSubpath()

        t = QtGui.QTransform()
        t.translate(-delta_r,0)

        item = QtWidgets.QGraphicsPathItem(side_upper)
        item.setTransform(t)
        item.setPen(pen)
        self.view.addItem(item)

        item = QtWidgets.QGraphicsPathItem(side_lower)
        item.setTransform(t)
        item.setPen(pen)
        self.view.addItem(item)

        ########################


    def _build_front_segments(self,n,i_seg,pen=QtGui.QPen(QtGui.QColor(30,30,30,60))):
        # z przodu buduje

        item = self._seg_front[i_seg][n]
        if item is not None:
            return item
        pen.setCosmetic(True)
        #brush = QtGui.QBrush(color)
        ang_span = self.ang_span
        ang_start = self.ang_start
        cx = self.cx_b
        cy = self.cy_b_front
        delta_r = self.delta_r

        arc = QtGui.QPainterPath()
        ang_start_i = ang_start + i_seg*ang_span

        a_out = self.a_in_b + (n+1)*self.div
        a_in = self.a_in_b + n*self.div

        rect_out = QtCore.QRectF(cx-a_out,cy - a_out,a_out*2,a_out*2)
        rect_in = QtCore.QRectF(cx-a_in,cy - a_in,a_in*2,a_in*2)
        rad = np.deg2rad(ang_start_i)
        sx = cx + a_out*np.cos(rad); sy = cy - a_out*np.sin(rad)
        arc.moveTo(sx, sy)
        arc.arcTo(rect_out, ang_start_i, ang_span)
        arc.arcTo(rect_in,  ang_start_i + ang_span, -ang_span)
        arc.closeSubpath()

        item = QtWidgets.QGraphicsPathItem(arc)
        item.setPen(pen)

        t = QtGui.QTransform()
        t.translate(delta_r*np.cos(np.deg2rad(ang_start_i + ang_span/2)),-delta_r*np.sin(np.deg2rad(ang_start_i + ang_span/2)))
        item.setTransform(t)
        self.view.addItem(item)
        self._seg_front[i_seg][n] = item
        return item

    def _build_rear_segments(self,n,i_seg,pen=QtGui.QPen(QtGui.QColor(30,30,30,60))):
        # z tyłu buduje
        item = self._seg_rear[i_seg][n]
        if item is not None:
            return item

        pen.setCosmetic(True)
        ang_start = -self.ang_start
        ang_span = -self.ang_span
        cx = self.cx_b
        cy = self.cy_b_rear
        delta_r = self.delta_r


        arc = QtGui.QPainterPath()

        ang_start_i = ang_start + i_seg*ang_span



        a_out = self.a_in_b + (n+1)*self.div
        a_in = self.a_in_b + n*self.div

        rect_out = QtCore.QRectF(cx-a_out,cy - a_out,a_out*2,a_out*2)
        rect_in = QtCore.QRectF(cx-a_in,cy - a_in,a_in*2,a_in*2)
        arc.moveTo(cx+a_out*np.cos(np.deg2rad(ang_start_i)),cy-a_out*np.sin(np.deg2rad(ang_start_i)))
        arc.arcTo(rect_out,ang_start_i,ang_span)
        arc.arcTo(rect_in,ang_start_i+ang_span,-ang_span)
        arc.closeSubpath()

        item = QtWidgets.QGraphicsPathItem(arc)
        item.setPen(pen)
        t = QtGui.QTransform()
        t.translate(delta_r*np.cos(np.deg2rad(ang_start_i + ang_span/2)),-delta_r*np.sin(np.deg2rad(ang_start_i + ang_span/2)))
        item.setTransform(t)
        self.view.addItem(item)
        self._seg_rear[i_seg][n] = item
        return item


    def _build_left_side_segments(self,name,n,i_seg,pen=QtGui.QPen(QtGui.QColor(30,30,30,60))):
        # z lewej buduje
        item = self._seg_left_side[i_seg][n]
        if item is not None:
            return item
        pen.setCosmetic(True)
        delta_r = self.delta_r

        ang_start = -self.ang_start #15
        ang_span = -self.ang_span*len(self._idx_front) #150
        cx_b = self.cx_b
        cy_b = self.cy_b_front

        a_out_b = self.a_in_b + (n+1)*self.div
        a_in_b = self.a_in_b + n*self.div


        t = QtGui.QTransform()
        t.translate(-delta_r,0)

        if name == "distance sensor left front side":
            cy_b = self.cy_b_front

            side_upper = QtGui.QPainterPath()
            side_upper.moveTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
            #print(side_upper.currentPosition())
            side_upper.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_upper.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
            side_upper.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
            side_upper.closeSubpath()

            item_upper = QtWidgets.QGraphicsPathItem(side_upper)
            item_upper.setTransform(t)
            item_upper.setPen(pen)
            self.view.addItem(item_upper)
            self._seg_left_side[i_seg][n] = item_upper
            return item_upper

        if name == "distance sensor left side":
            ang_start = -ang_start #-15
            ang_span = -ang_span #-37.5

            cy_b = self.cy_b_rear
            side_lower = QtGui.QPainterPath()
            side_lower.moveTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_lower.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b+a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_lower.lineTo(cx_b+a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
            side_lower.lineTo(cx_b+a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
            side_lower.closeSubpath()


            item_lower = QtWidgets.QGraphicsPathItem(side_lower)
            item_lower.setTransform(t)
            item_lower.setPen(pen)
            self.view.addItem(item_lower)
            self._seg_left_side[i_seg][n] = item_lower
            return item_lower


        # z prawej buduje
    def _build_right_side_segments(self,name,n,i_seg,pen=QtGui.QPen(QtGui.QColor(30,30,30,60))):
        item = self._seg_right_side[i_seg][n]
        if item is not None:
            return item

        pen.setCosmetic(True)
        cx_b = self.cx_b

        ang_start = self.ang_start #-15
        ang_span = self.ang_span*len(self._idx_front) #-37.5
        delta_r = self.delta_r

        a_out_b = self.a_in_b + (n+1)*self.div
        a_in_b = self.a_in_b + n*self.div

        t = QtGui.QTransform()
        t.translate(delta_r,0)

        if name == "distance sensor right front side":
            cy_b = self.cy_b_front
            side_upper = QtGui.QPainterPath()
            side_upper.moveTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_upper.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_upper.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
            side_upper.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1+delta_r/2)
            side_upper.closeSubpath()

            item_upper = QtWidgets.QGraphicsPathItem(side_upper)
            item_upper.setTransform(t)
            item_upper.setPen(pen)
            self.view.addItem(item_upper)
            self._seg_right_side[i_seg][n] = item_upper
            return item_upper

        if name == "distance sensor right side":
            cy_b = self.cy_b_rear
            ang_start = -ang_start
            ang_span = -ang_span

            side_lower = QtGui.QPainterPath()
            side_lower.moveTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_out_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_lower.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),cy_b-a_in_b*np.sin(np.deg2rad(ang_start+ang_span)))
            side_lower.lineTo(cx_b-a_in_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
            side_lower.lineTo(cx_b-a_out_b*np.cos(np.deg2rad(ang_start+ang_span)),CAR_LENGTH/2-1-delta_r/4)
            side_lower.closeSubpath()

            item_lower = QtWidgets.QGraphicsPathItem(side_lower)
            item_lower.setTransform(t)
            item_lower.setPen(pen)
            self.view.addItem(item_lower)
            self._seg_right_side[i_seg][n] = item_lower
            return item_lower

    @QtCore.pyqtSlot(object)
    def update_sensors(self, names_dists):

        for seg_list in [self._seg_front, self._seg_rear, self._seg_left_side, self._seg_right_side]:
            for row in seg_list:
                for item in row:
                    if item:
                        item.setVisible(False)
        max_min_dict = names_dists[4]

        for name, dist in names_dists[0].items():
            i_seg = self._idx_front[name]
            minim = max_min_dict[name][0]
            n = self.segment_for_dist(dist, minim)
            color = self.color_for_dist(n)

            if n is not None and n >= 0:
                item = self._build_front_segments(n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)


        for name, dist in names_dists[1].items():
            i_seg = self._idx_rear[name]
            minim = max_min_dict[name][0]
            n = self.segment_for_dist(dist, minim)
            color = self.color_for_dist(n)

            if n is not None and n >= 0:
                item = self._build_rear_segments(n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)

        for name,dist in names_dists[2].items():
            i_seg = self._idx_left_side[name]
            min = max_min_dict[name][0]
            n = self.segment_for_dist(dist,min)
            color = self.color_for_dist(n)
            if n is not None and n >= 0:
                item = self._build_left_side_segments(name,n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)


        for name,dist in names_dists[3].items():
            i_seg = self._idx_right_side[name]
            min = max_min_dict[name][0]
            n = self.segment_for_dist(dist,min)
            color = self.color_for_dist(n)
            if n is not None and n >= 0:
                item = self._build_right_side_segments(name,n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)

    # @QtCore.pyqtSlot(object)
    # def update_location(self,pose):
    #     x_odo = pose.get("x_odo")
    #     y_odo = pose.get("y_odo")
    #     sp_odo = pose.get("sp_odo")
    #     encoders = pose.get("encoders")
    #     node_pos = pose.get("node_pos")
    #     acc = pose.get("acc")
    #     node_or = pose.get("node_or")
    #     node_vel = pose.get("node_vel")

    #     #<b>Kąty kół przednich:</b> FR = {encoders[0]:.3f}, FL = {encoders[1]:.3f}<br>
    #     #<b>Kąty kół tylnych:</b>   RR = {encoders[2]:.3f}, RL = {encoders[3]:.3f}<br>
    #     # <b>Kąty z IMU:</b>   roll={im[0]:+.4f} rad, pitch={im[1]:+.4f} rad, yaw={im[2]:+.4f} rad<br>
    #     # <b>Żyroskop:</b>     gx={gyr[0]:+.4f}, gy={gyr[1]:+.4f}, gz={gyr[2]:+.4f} rad/s<br>
    #     # <b>Akcelerometr:<\b> ax={acc[0]:+.4f}, ay={acc[1]:+.4f}, az={acc[2]:+.4f} m/s^2<br>
    #     html = f"""
    #     <div style="font-family: Consolas, 'Courier New', monospace; font-size:12pt; line-height:1.2;">
    #       <b>Wszystkie współrzędne w odniesieniu do początkowych<br>
    #       <b>Pozycja w Webots:</b>   x = {node_pos[0]:.4f}, y = {node_pos[1]:.4f}<br>
    #       <b>Pozycja odometrii:</b>  x = {x_odo:.4f},       y = {y_odo:.4f}<br>
    #       <b>Prędkość z odometrii:</b> {sp_odo:.2f} m/s ({sp_odo*3.6:.2f} km/h)<br>
    #       <b>Prędkość z Webots:</b> {node_vel[0]:.2f} m/s ({node_vel[0]*3.6:.2f}) km/h)<br>
    #     </div>
    #     """

    #     self.text.setHtml(html)

    @QtCore.pyqtSlot(bool)
    def on_parking_change(self, is_parking):
        self.running = is_parking
        if is_parking:
            self.show()



class ProgressArrow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0.0  # 0..1
        self.direction = +1 
        self.setMinimumWidth(40)
        
    def set_progress(self, value):
        self.progress = np.clip(value, 0.0, 1.0)
        self.update()
        
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        shaft_w = 10
        head_h = 14
        center_x = w // 2

        p.setPen(QtGui.QPen(QtGui.QColor("#4D4C4C"), 2))
        p.setBrush(QtGui.QColor("#4D4C4C"))

        
        p.drawRect(center_x - shaft_w//2, 10, shaft_w, h - 20 - head_h)

        # head = QtGui.QPolygon([
        #     QtCore.QPoint(center_x - 12, h - 20),
        #     QtCore.QPoint(center_x + 12, h - 20),
        #     QtCore.QPoint(center_x, h - 5)
        # ])
        # p.drawPolygon(head)

        fill_h = int((h - 20 - head_h) * self.progress)
        p.setBrush(QtGui.QColor("#66B4FC"))
        p.drawRect(
            center_x - shaft_w//2,
            10 + (h - 20 - head_h - fill_h),
            shaft_w,
            fill_h
        )

        

class MainWindow(QtWidgets.QWidget):
    """Główne okno aplikacji"""
    
    mapData = QtCore.pyqtSignal(object)   # obstacles, start, goal
    carData = QtCore.pyqtSignal(object)   # (x, y, yaw)

    def __init__(self,cont):
        super().__init__()
        self.setWindowTitle("Wizualizacja parkowania")
        self.resize(1200, 900)
        
        # Controller
        self.controller = cont
        self.state = self.controller.state
        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # HEADER
        layout.addWidget(self.build_header())
        self.btns = []
        self.btns.extend([self.btn_left,self.btn_right,self.btn_parallel,self.btn_perpendicular])
        # ŚRODEK
        center = QtWidgets.QHBoxLayout()

        self.progress_arrow = ProgressArrow()
        center.addWidget(self.progress_arrow)

        self.sensor_widget = SensorView(self.controller)
        center.addWidget(self.sensor_widget, stretch=1)

        layout.addLayout(center)

        # DÓŁ – przyciski akcji
        bottom = QtWidgets.QHBoxLayout()

        self.btn_plan = QtWidgets.QPushButton("Uruchom poszukiwanie ścieżki")
        self.btn_plan.clicked.connect(self.send_for_planning)

        self.btn_activate = QtWidgets.QPushButton("Przełącz parkowanie")
        self.btn_activate.clicked.connect(self.on_activate_clicked)

        self.btn_plan.setMinimumHeight(48)
        self.btn_activate.setMinimumHeight(48)

        bottom.addWidget(self.btn_plan)
        bottom.addWidget(self.btn_activate)

        layout.addLayout(bottom)

        # Połącz sygnały
        self.controller.stateUpdated.connect(self.on_state_changed)
        #self.controller.segmentProgressUpdated.connect(self.progress_arrow.set_progress)
        btn_plan_state = self.state == "waiting_for_confirm_start"
        self.btn_plan.setEnabled(btn_plan_state)

    def build_header(self):
        header = QtWidgets.QFrame()
        header.setFrameShape(QtWidgets.QFrame.StyledPanel)
        header.setStyleSheet("""
            QFrame {
                background: #F5F7FA;
                border-bottom: 1px solid #D0D4DA;
            }
        """)

        layout = QtWidgets.QHBoxLayout(header)
        
        
        self.btn_left = QtWidgets.QPushButton("<-")
        self.btn_right = QtWidgets.QPushButton("->")
        self.btn_left.setCheckable(True)
        self.btn_right.setCheckable(True)

        self.side_group = QtWidgets.QButtonGroup(self)
        self.side_group.setExclusive(True)
        self.side_group.addButton(self.btn_left)
        self.side_group.addButton(self.btn_right)

        self.btn_left.clicked.connect(self.send_left_side)
        self.btn_right.clicked.connect(self.send_right_side)
        for b in (self.btn_left, self.btn_right):
            b.setFixedSize(100, 40)
        layout.addWidget(self.btn_left)
        layout.addWidget(self.btn_right)

        self.btn_parallel = QtWidgets.QPushButton("Równoległe")
        self.btn_parallel.setCheckable(True)
        self.btn_parallel.clicked.connect(self.send_parallel)
        
        self.btn_perpendicular = QtWidgets.QPushButton("Prostopadłe")
        self.btn_perpendicular.setCheckable(True)
        self.btn_perpendicular.clicked.connect(self.send_perpendicular)
        for b in (self.btn_parallel, self.btn_perpendicular):
            b.setMinimumSize(140, 44)
        self.type_group = QtWidgets.QButtonGroup(self)
        self.type_group.setExclusive(True)
        self.type_group.addButton(self.btn_parallel)
        self.type_group.addButton(self.btn_perpendicular)

        layout.addWidget(self.btn_parallel)
        layout.addWidget(self.btn_perpendicular)
        layout.addSpacing(20)
        layout.addStretch()

        self.status_label = QtWidgets.QLabel("Parkowanie nieaktywne")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #1F2A44;
            }
        """)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        self.status_label.setWordWrap(True)
        self.status_label.setFixedHeight(50)
        layout.addWidget(self.status_label)


        return header

    def send_perpendicular(self):
        self.controller.find_type = "perpendicular"

    def send_parallel(self):
        self.controller.find_type = "parallel"

    def send_left_side(self):
        self.btn_left.setChecked(True)
        self.controller.side = "left"

    def send_right_side(self):
        self.btn_right.setChecked(True)
        self.controller.side = "right"

    def send_for_planning(self):
        if self.controller.parking and self.controller.state == "waiting_for_confirm_start" and self.controller.stopped:
            self.controller.timer = 0.0
            self.controller.state = "planning"

    # def set_buttons_state(self):
        
    #     if self.state in ["planning","executing",
    #                       "waiting_for_confirm_start", "parking_finished"
    #                       "drive_forward","drive_backward"]:
    #         for btn in self.btns:
    #             btn.setEnabled(False)
    #     else:
    #         if self.controller.side == "left":
    #             self.btn_left.setEnabled(False)
    #             self.btn_right.setEnabled(True)
    #         elif self.controller.side == "right":
    #             self.btn_right.setEnabled(False)
    #             self.btn_left.setEnabled(True)
    #         if self.controller.find_type == "parallel":
    #             self.btn_parallel.setEnabled(False)
    #             self.btn_perpendicular.setEnabled(True)
    #         elif self.controller.find_type == "perpendicular":
    #             self.btn_parallel.setEnabled(True)
    #             self.btn_perpendicular.setEnabled(False)

    def on_activate_clicked(self):
        self.controller.toggle_parking()

    def on_state_changed(self,state):
        self.state = state
        
        state_text = {
            "inactive_waiting": "Proszę wybrać typ miejsca.",
            "inactive": "Parkowanie nieaktywne.",
            "searching_left_parallel": "Szukam miejsca równoległego po lewej...",
            "searching_left_perpendicular": "Szukam miejsca prostopadłego po lewej...",
            "searching_right_parallel": "Szukam miejsca równoległego po prawej...",
            "searching_right_perpendicular": "Szukam miejsca prostopadłego po prawej...",
            "found_spot": "Znaleziono miejsce! Proszę się zatrzymać.",
            "return_searching": "Miejsce porzucone. Poszukiwanie dalsze...",
            "found_another_spot": "Znaleziono nowe miejsce! Proszę się zatrzymać.",
            "waiting_for_confirm_start": "Proszę wcisnąć przycisk aktywacji manewru i zaczekać.",
            "planning": "Trwa poszukiwanie najkrótszej ścieżki...",
            "executing": "Ścieżka znaleziona! Proszę się poruszać zgodnie z sygnałami...",
            "drive_forward": "Jedź do przodu.",
            "drive_backward": "Jedź do tyłu.",
            "parking_finished": "Parkowanie ukończone. Proszę wyłączyć asystent parkowania."
        }
        
        
        self.status_label.setText(state_text.get(self.state, "Stan: Nieznany"))
        btn_plan_state = self.state == "waiting_for_confirm_start"
        self.btn_plan.setEnabled(btn_plan_state)

        if self.controller.side == "left":
            self.btn_left.setChecked(True)
        elif self.controller.side == "right":
            self.btn_right.setChecked(True)

        if self.controller.find_type == "parallel":
            self.btn_parallel.setChecked(True)
        elif self.controller.find_type == "perpendicular":
            self.btn_perpendicular.setChecked(True)
        
        locked = state in [
            "planning", "executing",
            "drive_forward", "drive_backward",
            "parking_finished"
        ]
        self.btns
        for b in self.btns:
            b.setEnabled(not locked)
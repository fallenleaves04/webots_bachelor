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

class TrajView(pg.GraphicsLayoutWidget):
    def __init__(self,cont):
        super().__init__(title="Trajectory Display")
        self.setBackground((235,235,250))

        self.running = True
        self.controller = cont
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
        self.traj_curve_yolo = self.view_trajectory.plot([], [], pen=None, symbol='t',symbolSize=2,symbolBrush='k',symbolPen=None, name="YOLO")
        self.traj_curve4 = self.view_trajectory.plot([], [], pen=None, symbol='o',symbolSize=5,symbolBrush='k',symbolPen=None, name="Miejsca pojazdu")
        self.traj_curve5 = self.view_trajectory.plot([], [], pen='g', name="Przykładowa ścieżka")
        self.traj_curve6 = self.view_trajectory.plot([], [], pen=pg.mkPen(color=(0, 100, 100), width=1.5), name="Ścieżka Hybrid")
        #
        self.view_trajectory.setClipToView(True)
        self.view_trajectory.setDownsampling(mode='peak')
        #
        self.path_drawn = False
        self.car_rect = self.make_car()

        #
        self.obstacle_items = []
        self.spots_items = []
        self.text_items = []
        self.arrow = pg.ArrowItem(
            pos=(0.0, 0.0),
            angle=np.degrees(0.0),
            headLen=25,        # px
            tipAngle=30,       # stopnie (szerokość grotu)
            tailLen=60,        # px długość „trzonu”
            tailWidth=10,      # px grubość „trzonu”
            pen=pg.mkPen(0,0,0, width=2),
            brush=QtGui.QBrush(QtGui.QColor(20,20,20))
        )
        #self.view_trajectory.addItem(self.arrow)
        
        self.view_trajectory.scene().sigMouseClicked.connect(self.on_mouse_click)

        cont.parkingToggled.connect(self.on_parking_change)
        cont.trajUpdated.connect(self.update_trajectory)
        cont.pathUpdated.connect(self.draw_path)
        cont.expansionUpdated.connect(self.draw_expansion_cars)
        cont.hmapUpdated.connect(self.draw_hmap)
        
    def make_car(self,x_center=0.0,y_center=0.0,l_c=CAR_LENGTH,w_c=CAR_WIDTH,yaw=0.0):
        pts = [
            QtCore.QPointF(-1, -w_c/2),
            QtCore.QPointF(l_c-1, -w_c/2),
            QtCore.QPointF(l_c-1, w_c/2),
            QtCore.QPointF(-1, w_c/2)
        ]
        polygon = QtGui.QPolygonF(pts)
        car = QtWidgets.QGraphicsPolygonItem(polygon)
        t = QtGui.QTransform()
        car.setTransformOriginPoint(QtCore.QPointF(1,w_c/2))
        t.translate(x_center, y_center)
        t.rotate(np.degrees(yaw), QtCore.Qt.ZAxis)   
        car.setTransform(t)
        pen = pg.mkPen(color=(0, 0, 0), width=1)
        car.setPen(pen)
        car.setBrush(QtGui.QBrush(QtGui.QColor(80,80,80,150)))
        self.view_trajectory.addItem(car)
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
        item.setBrush(QtGui.QBrush(QtGui.QColor(80,80,80,150)))
        
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
        if not all(v is None for v in data[5]):
            self.tx_yolo = data[5][0]; self.ty_yolo = data[5][1]
            self.traj_curve_yolo.setData(self.tx_yolo, self.ty_yolo)
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
        t = QtGui.QTransform()
        t.translate(x1, y1)
        t.rotate(np.degrees(car_yaw))  
        self.car_rect.setTransform(t)

    @QtCore.pyqtSlot(object)
    def draw_path(self, path):
        
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
        x, y, yaw = car_pos
        car_item = self.make_car(x,y,yaw)
        #self.transform_car_item(car_item,car_pos)
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
            del self.t1x[:]
            del self.t1y[:]
            del self.t2x[:]
            del self.t2y[:]
            self.hide()

class SensorView(pg.GraphicsLayoutWidget):
    def __init__(self,cont):
        super().__init__(title="Parking-Sensor Display")
        self.view = self.addViewBox(lockAspect=True)
        self.view.setRange(xRange=[-8,10], yRange=[-10,10])

        self.setBackground((235,235,250))

        self.view.setMouseEnabled(x=True, y=True)
        self.running = True

        # textItem do pozycji
        self.text = pg.TextItem("", anchor=(0, 0))
        self.view.addItem(self.text)
        self.text.setPos(-15, 15)

        self.n_strips = 8
        self.acceptable_dist = 6.0
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

        for i_seg in range(4):  # cztery kliny sensory z przodu
            for k in range(self.n_strips):
                _ = self._build_front_segments(k, i_seg)
                _.setVisible(False)  # na starcie wyłączone

        for i_seg in range(4): # # cztery kliny sensory z tyłu
            for k in range(self.n_strips):
                _ = self._build_rear_segments(k, i_seg)
                _.setVisible(False)

        for i_seg in range(2):  # cztery kliny sensory z prawej
            name = self.right_side_sensor_names[i_seg]
            for k in range(self.n_strips):
                _ = self._build_right_side_segments(name,k, i_seg)
                _.setVisible(False)  # na starcie wyłączone

        for i_seg in range(2):
            name = self.left_side_sensor_names[i_seg]
            for k in range(self.n_strips): # # cztery kliny sensory z lewej
                _ = self._build_left_side_segments(name,k,i_seg)
                _.setVisible(False)

        self._insert_car()
        self._create_base_front_sector()
        self._create_base_rear_sector()
        self._create_base_side_left()
        self._create_base_side_right()

        cont.parkingToggled.connect(self.on_parking_change)
        cont.sensorUpdated.connect(self.update_sensors)
        cont.locUpdated.connect(self.update_location)
        
        


    def color_for_dist(self,d):
        """Mapa koloru wg odległości (bliżej = 'bardziej alarmowo')."""
        if d < 1.00:              return QtGui.QColor(255,   0,   0, 127)  # czerwony
        if d < 3.00:              return QtGui.QColor(255, 128,   0, 127)  # pomarańczowy
        if d < 4.50:              return QtGui.QColor(255, 255,  51, 127)  # żółty
        if d < self.acceptable_dist:              return QtGui.QColor(102, 255, 102, 127)  # zielony
        return QtGui.QColor(255,255,255,0)

    def segment_for_dist(self,val,minim):
        # definiuje numer segmentu dla pojedynczego sektora
        #min = max_min_dict[name][0]
        n = self.n_strips
        maxd = self.acceptable_dist
        if val < minim:
            return -1
        if n <= 0 or maxd <= minim:
            return -1

        width = (maxd - minim) / n
        # val == maxd trafia do ostatniego segmentu
        if val >= maxd:
            return n - 1

        idx = int((val - minim) / width)  # 0..n-1
        if idx >= n:
            idx = n - 1
        if idx < 0:
            idx = 0
        return idx

    def _insert_car(self):

        # dodaj samochod na srodku
        pix = QtGui.QPixmap(r"D:\User Files\BACHELOR DIPLOMA\Pliki symulacyjne\controllers\parking_parallel_new\pngegg.png")
        assert not pix.isNull()
        item = QtWidgets.QGraphicsPixmapItem(pix)

        item.setTransformationMode(QtCore.Qt.FastTransformation)
        pw, ph = pix.width(), pix.height()
        t = QtGui.QTransform()
        t.translate(CAR_WIDTH/2, CAR_LENGTH/2 + (CAR_LENGTH/2 - 1))              # żeby środek w (0,0)
        t.scale(CAR_WIDTH / pw, CAR_LENGTH / ph)          # (Y) szerokość, (X) długość
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
        dist_span = self.dist_span
        margin_car = self.margin_car

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
            color = self.color_for_dist(dist)

            if n >= 0:
                item = self._build_front_segments(n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)


        for name, dist in names_dists[1].items():
            i_seg = self._idx_rear[name]
            minim = max_min_dict[name][0]
            n = self.segment_for_dist(dist, minim)
            color = self.color_for_dist(dist)

            if n >= 0:
                item = self._build_rear_segments(n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)

        for name,dist in names_dists[2].items():
            i_seg = self._idx_left_side[name]
            min = max_min_dict[name][0]
            n = self.segment_for_dist(dist,min)
            color = self.color_for_dist(dist)
            if n >= 0:
                item = self._build_left_side_segments(name,n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)


        for name,dist in names_dists[3].items():
            i_seg = self._idx_right_side[name]
            min = max_min_dict[name][0]
            n = self.segment_for_dist(dist,min)
            color = self.color_for_dist(dist)
            if n >= 0:
                item = self._build_right_side_segments(name,n,i_seg)
                item.setBrush(QtGui.QBrush(color))
                item.setVisible(True)

    @QtCore.pyqtSlot(object)
    def update_location(self,pose):
        x_odo = pose.get("x_odo")
        y_odo = pose.get("y_odo")
        sp_odo = pose.get("sp_odo")
        encoders = pose.get("encoders")
        node_pos = pose.get("node_pos")
        acc = pose.get("acc")
        node_or = pose.get("node_or")
        node_vel = pose.get("node_vel")

        #<b>Kąty kół przednich:</b> FR = {encoders[0]:.3f}, FL = {encoders[1]:.3f}<br>
        #<b>Kąty kół tylnych:</b>   RR = {encoders[2]:.3f}, RL = {encoders[3]:.3f}<br>
        # <b>Kąty z IMU:</b>   roll={im[0]:+.4f} rad, pitch={im[1]:+.4f} rad, yaw={im[2]:+.4f} rad<br>
        # <b>Żyroskop:</b>     gx={gyr[0]:+.4f}, gy={gyr[1]:+.4f}, gz={gyr[2]:+.4f} rad/s<br>
        # <b>Akcelerometr:<\b> ax={acc[0]:+.4f}, ay={acc[1]:+.4f}, az={acc[2]:+.4f} m/s^2<br>
        html = f"""
        <div style="font-family: Consolas, 'Courier New', monospace; font-size:12pt; line-height:1.2;">
          <b>Wszystkie współrzędne w odniesieniu do początkowych<br>
          <b>Pozycja w Webots:</b>   x = {node_pos[0]:.4f}, y = {node_pos[1]:.4f}<br>
          <b>Pozycja odometrii:</b>  x = {x_odo:.4f},       y = {y_odo:.4f}<br>
          <b>Prędkość z odometrii:</b> {sp_odo:.2f} m/s ({sp_odo*3.6:.2f} km/h)<br>
          <b>Prędkość z Webots:</b> {node_vel[0]:.2f} m/s ({node_vel[0]*3.6:.2f}) km/h)<br>
        </div>
        """

        self.text.setHtml(html)

    @QtCore.pyqtSlot(bool)
    def on_parking_change(self, is_parking):
        self.running = is_parking
        if is_parking:
            self.show()
        else:
            self.hide()






def collect_homo(names_images,homographies,car,streams):
    """
    DRUGI SPOSÓB tworzenia widoku "z lotu ptaka". Sklejanie połówek
    """
    h, w = int(3600/s),int(3600/s)
    #pobieramy streamy do przetwarzania obrazów na GPU
    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []

    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    #dalej się skaluje homografie, jeżeli była zmieniona rozdzielczość, bo do 4K było
    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)

    right_to_front_H = np.array([[ 1.0103254e+00,  8.9369901e-03,  2.4237607e+03],
 [-4.8033521e-03,  1.0200684e+00 , 1.8881282e+03],
 [-1.9079014e-06,  2.8290551e-06,  1.0000000e+00]]).astype(np.float32)
    right_to_front_H = S @ right_to_front_H @ np.linalg.inv(S)

    left_to_front_H = np.array([[ 1.0053335e+00 ,-6.1842590e-04, -2.4580974e+03],
 [-5.9892777e-03,  1.0021796e+00,  1.8966011e+03],
 [-2.1722203e-06, -4.8802093e-07,  1.0000000e+00]]).astype(np.float32)
    left_to_front_H = S @ left_to_front_H @ np.linalg.inv(S)

    left_to_rear_H = np.array([[ 9.7381920e-01, -2.3735706e-03, -2.2782476e+03],
 [ 4.3283560e-04,  9.7134042e-01, -1.7275361e+03],
 [-3.2049848e-06,  5.5537777e-07,  1.0000000e+00]]).astype(np.float32)

    right_to_rear_H = np.array([[ 9.8992777e-01,  1.6606528e-02,  4.6191958e+03],
 [-2.1194941e-03 , 9.9684310e-01, -1.7071634e+01],
 [-7.0434027e-07,  3.0691269e-06 , 1.0000000e+00]]).astype(np.float32)



    left_to_rear_H = S @ left_to_rear_H @ np.linalg.inv(S)
    right_to_rear_H = S @ right_to_rear_H @ np.linalg.inv(S)

    rear_to_front_H = np.array([[ 1.1127317e+00,  7.0112962e-03 ,-8.0307449e+01],
 [ 5.2037694e-02,  1.0801761e+00,  4.4567339e+03],
 [ 1.0615680e-05,  6.2511299e-07,  1.0000000e+00]]).astype(np.float32)
    rear_to_front_H = S @ rear_to_front_H @ np.linalg.inv(S)

    #blendujemy na GPU, wejściem jest GpuMat, nazwy macierzy wskazują kierunek
    canvas_front = blend_warp_GPUONLY(front,right,right_to_front_H,stream7)
    canvas_front = blend_warp_GPUONLY(canvas_front,left,left_to_front_H,stream8)
    canvas_rear = blend_warp_GPUONLY(rear,left_fender,left_to_rear_H,stream9)
    canvas_rear = blend_warp_GPUONLY(canvas_rear,right_fender,right_to_rear_H,stream10)
    canvas = blend_warp_GPUONLY(canvas_front,canvas_rear,rear_to_front_H,stream11)

    canvas = canvas.download()

    #cv.namedWindow("bev",cv.WINDOW_NORMAL)
    #cv.imshow("bev",canvas)

    cropped = canvas
    h, w = cropped.shape[:2]
    canvas.clear()
    crop_top_px    = int(0.184 * h)
    crop_bottom_px = int(0.215 * h)
    crop_left_px = int(0.14 * w)
    crop_right_px = int(0.14 * w)
    #obcinamy o te wartości powyżej z góry, dołu i z boków
    y1 = crop_top_px
    y2 = h - crop_bottom_px
    x1 = crop_left_px
    x2 = w - crop_right_px
    cropped = cropped[y1:y2, x1:x2]

    #prób i błędów wstawiamy obrazek samochodu
    scalex = 0.18
    scaley = 0.45
    bev_h, bev_w = cropped.shape[:2]
    new_w = int(bev_w * scalex)
    new_h = int(bev_h * scaley)
    car_resized = cv.resize(car, (new_w, new_h), interpolation=cv.INTER_AREA)

    #przesuń
    x_offset = (bev_w - new_w) // 2 + 15
    y_offset = (bev_h - new_h) // 2 - 15
    #wstaw w tym obszarze
    cropped[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = car_resized

    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)
    #cv.imwrite("img_vis.png",cropped)

    return canvas

def test_homo(names_images,homographies,streams):
    """
    Była pomocnicza, żeby wykonać sprawozdanie, dla homografii
    """
    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)

    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    H,_ = cc.chess_homography(left,front,(7,5))
    if H is not None:
        vis = warp_and_blend_gpu(front,left,H)
        cv.imwrite("img_vis.jpg",cv.cvtColor(vis,cv.COLOR_BGR2RGB))


def alt_collect_homo(names_images,homographies,car,streams):
    """
    TRZECI SPOSOB na widok "z lotu ptaka". Nakładanie na współną kanwę
    """
    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)

    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    #CZĘŚĆ KODU DO SPORZĄDZENIA HOMOGRAFII RZUTOWANIA NA KANWĘ

    canvas_size = (6000,6000)
    canvas_cpu = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas = cv.cuda_GpuMat()
    canvas.upload(canvas_cpu,stream7)

    px = np.array([[0,0],[6000/s,0],[6000/s,6000/s],[0,6000/s]]).astype(np.float32)
    met = np.array([[10,10],[10,-10],[-10,-10],[-10,10]],dtype = np.float32) #

    H_px_to_m_bev,_ = cv.findHomography(met,px,cv.RANSAC,5.0)

    #znowu skalujemy - dodatkowo jeszcze korygowanie macierzy

    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)
    H_left_to_bev= np.array([[5.78196165e-01, 1.36768422e-03 ,5.75901552e+02],
 [2.08805960e-03, 5.78244815e-01, 1.31412439e+03],
 [9.24389963e-07 ,7.19948103e-07, 1.00000000e+00]],dtype=np.float32)
    H_left_to_bev = S @ H_left_to_bev @ np.linalg.inv(S)

    H_right_to_bev = np.array([[ 5.74277218e-01, -8.39567193e-05,  3.34911956e+03],
 [ 3.93911249e-06,  5.73836613e-01,  1.31777314e+03],
 [ 1.31049336e-07, -1.42342160e-07,  1.00000000e+00]],dtype=np.float32)
    H_right_to_bev = S @ H_right_to_bev @ np.linalg.inv(S)

    #korygowanie macierzy, bo można tak każdy obraz sobie poprawić, jeżeli
    #złe jest dopasowanie

    H_left_fender_to_bev = np.array([[ 5.60635651e-01,  3.13768463e-02,  6.29376393e+02],
 [-1.14353803e-02,  6.03217079e-01, 2.80548010e+03],
 [-4.06125598e-06,  1.18455527e-05,  1.00000000e+00]],dtype=np.float32)
    H_corr_lf = np.array(
    [[ 9.95078761e-01, -5.70508469e-04,  5.23265217e+00],
     [-3.64430112e-03,  9.96849037e-01,  7.87846839e+00],
     [-1.04463326e-06,-2.72402103e-07,  1.00000000e+00]]).astype(np.float32)
    H_left_fender_to_bev = H_corr_lf @ H_left_fender_to_bev
    H_left_fender_to_bev = S @ H_left_fender_to_bev @ np.linalg.inv(S)

    H_right_fender_to_bev = np.array([[ 5.68654217e-01,  3.69548635e-02,  3.30027369e+03],
 [-5.18751953e-03,  6.04990446e-01 , 2.80305923e+03],
 [-2.20235231e-06,  1.11229040e-05 , 1.00000000e+00]],dtype=np.float32)
    H_corr_rf = np.array(
    [[ 9.86315188e-01, -3.03481721e-04,  2.78378345e+01],
     [-5.49422481e-03,  9.93028602e-01,  2.23023238e+01],
     [-1.63834578e-06, -8.44384115e-08 , 1.00000000e+00]]).astype(np.float32)
    H_right_fender_to_bev = H_corr_rf @ H_right_fender_to_bev
    H_right_fender_to_bev = S @ H_right_fender_to_bev @ np.linalg.inv(S)

    H_rear_to_bev = np.array([[5.72595558e-01, 2.77600165e-02 ,1.96983903e+03],
 [2.49365825e-04, 6.06721803e-01, 3.77306058e+03],
 [6.15294083e-08, 9.20599720e-06, 1.00000000e+00]],dtype=np.float32)
    H_corr_rear = np.array([[ 9.86009607e-01, -5.12444388e-05,  4.22106295e+01],
 [ 6.09014276e-05,  9.87099849e-01 , 4.70971490e+01],
 [ 1.79688786e-08, -1.22849686e-08,  1.00000000e+00]]).astype(np.float32)
    H_rear_to_bev = H_corr_rear @ H_rear_to_bev
    H_rear_to_bev = S @ H_rear_to_bev @ np.linalg.inv(S)

    H_front_to_bev = np.array([[5.78772185e-01, 3.67819798e-03, 1.95954679e+03],
 [2.99076766e-04, 5.83709774e-01, 2.15103801e+02],
 [1.77376103e-07 ,1.22892995e-06, 1.00000000e+00]],dtype = np.float32)
    H_front_to_bev = S @ H_front_to_bev @ np.linalg.inv(S)

    #tutaj już zamiast dynamicznie dobierać rozmiar kanwy dajemy stały
    #po kolei na kanwę, gdzie już obrazy są, nakładamy kolejny
    # można w dowolnej kolejnosci!
    bev_left = blend_warp_GPUONLY(canvas,left,H_left_to_bev,stream7,canvas_size=(6000//s,6000//s))
    bev_right = blend_warp_GPUONLY(bev_left,right,H_right_to_bev,stream8,canvas_size=(6000//s,6000//s))
    bev_left_fender = blend_warp_GPUONLY(bev_right,left_fender,H_left_fender_to_bev,stream9,canvas_size=(6000//s,6000//s))
    bev_right_fender = blend_warp_GPUONLY(bev_left_fender,right_fender,H_right_fender_to_bev,stream10,canvas_size=(6000//s,6000//s))
    bev_rear = blend_warp_GPUONLY(bev_right_fender,rear,H_rear_to_bev,stream11,canvas_size=(6000//s,6000//s))
    bev_front = blend_warp_GPUONLY(bev_rear,front,H_front_to_bev,stream12,canvas_size=(6000//s,6000//s))

    bev = bev_front.download()

    # Granice w metrach, tutaj pierwsze - X samochodu, drugie - Y samochod, czyli wlewo-wprawo
    meters = np.array([
        [ 5.6,  6],
        [ 5.6, -6],
        [-5.95, -6],
        [-5.95,  6]
    ], dtype=np.float32)

    # Dodaj trzecią współrzędną (homogeniczne)
    meters_hom = np.hstack([meters, np.ones((meters.shape[0], 1))])

    # Przekształć na piksele
    pxs = (H_px_to_m_bev @ meters_hom.T).T

    # Rzutuj na int (piksele)
    pxs_int = pxs[:, :2].astype(int)

    # Ustal minimalne i maksymalne współrzędne pikseli (x_min, x_max, y_min, y_max)
    x_min = np.min(pxs_int[:, 0])
    x_max = np.max(pxs_int[:, 0])
    y_min = np.min(pxs_int[:, 1])
    y_max = np.max(pxs_int[:, 1])

    # sprawdzenie czy zakres jest w granicach obrazu - aby nie było błędów
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, canvas_cpu.shape[1])
    y_max = min(y_max, canvas_cpu.shape[0])

    # Obcinanie obrazu (ROI -> Region of Interest)
    cropped = bev[y_min:y_max, x_min:x_max]


    # dwa punkty homogeniczne w metrach dla osi X
    e_x = np.array([[0,0,1],
                [1,0,1]], dtype=np.float32)
    # dwa punkty homogeniczne w metrach dla osi Y
    e_y = np.array([[0,0,1],
                [0,1,1]], dtype=np.float32)

    # rzut tych punktów na piksele:
    p_x = (H_px_to_m_bev @ e_x.T).T
    p_x /= p_x[:, [2]]
    p_y = (H_px_to_m_bev @ e_y.T).T
    p_y /= p_y[:, [2]]

    # skaluj punkty
    scale_x = np.linalg.norm(p_x[1,:2] - p_x[0,:2])  # px na 1 m w osi X
    scale_y = np.linalg.norm(p_y[1,:2] - p_y[0,:2])  # px na 1 m w osi Y
    # 1) Policz w pikselach samochód
    # rozmiar samochodu w pikselach na kanwie
    W_real = 1.95  # szerokość samochodu w metrach
    L_real = 4.95  # długość samochodu w metrach
    w_car_px = int(round(W_real * scale_x))
    h_car_px = int(round(L_real * scale_y))

    # 2) Skaluj
    car_resized = cv.resize(car, (w_car_px, h_car_px), interpolation=cv.INTER_AREA)

    # 3) Wylicz środek ROI jako środek samego cropped
    Hc, Wc = cropped.shape[:2]
    cx_px = Wc // 2
    cy_px = Hc // 2

    # 4) Przesunąć jeżeli trzeba
    d_forward = 0.1 # np. 0.5 m do przodu
    pix_offset = int(round(d_forward * scale_y))
    # w obrazie „do przodu” = w górę, czyli y maleje:
    cy_px -= pix_offset

    # 5) Oblicz ROI w pikselach:
    x0 = cx_px - w_car_px//2
    y0 = cy_px - h_car_px//2
    x1 = x0 + w_car_px
    y1 = y0 + h_car_px

    # 6) Przytnij do granic i wklej:
    x0c, y0c = max(x0,0), max(y0,0)
    x1c, y1c = min(x1,Wc), min(y1,Hc)
    sx0, sy0 = x0c - x0, y0c - y0
    sx1, sy1 = sx0 + (x1c-x0c), sy0 + (y1c-y0c)

    cropped[y0c:y1c, x0c:x1c] = car_resized[sy0:sy1, sx0:sx1]

    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)
    return cropped



def chain_collect_homo(names_images,homographies,car,streams):
    """
    PIERWSZY SPOSÓB na widok "z lotu ptaka". Każdy obraz łańcuchowo łączy się z
    poprzednim w taki sposób, żeby wizualnie nie było zbytnio zniekształćeń
    """
    h, w = int(3600/s),int(3600/s)

    (stream1,stream2,stream3,stream4,
    stream5,stream6,stream7,stream8,
    stream9,stream10,stream11,stream12,stream13) = streams
    (front_H,right_H,right_fender_H,
    rear_H,left_fender_H,left_H)= homographies
    imgs = []
    # warp CUDA wszystkie kamery
    left = warp_with_cuda(names_images["camera_left_pillar"], left_H, "left homo", h, w,stream1)
    right = warp_with_cuda(names_images["camera_right_pillar"], right_H, "right homo", h, w,stream2)
    rear = warp_with_cuda(names_images["camera_rear"], rear_H, "rear homo", h, w,stream3,show=True)
    front = warp_with_cuda(names_images["camera_front_bumper_wide"], front_H, "front homo", h, w,stream4)
    # front_wind = warp_with_cuda(names_images["camera_front_top"], front_wind_H, "front wind homo", h, w)
    right_fender = warp_with_cuda(names_images["camera_right_fender"], right_fender_H, "right fender homo", h, w,stream5)
    left_fender = warp_with_cuda(names_images["camera_left_fender"], left_fender_H, "left fender homo", h, w,stream6)

    #cv.imwrite("lewa_kolumna_6.jpg",left)
    #cv.imwrite("lewy_blotnik_6.jpg",left_fender)

    #H1 - lewa do frontalnej,
    #H2 - H1 do prawej
    #H3 - H2 do prawego błotnika
    #H4 - tylna zarówno do prawego, jak i lewego błotnika (dołączono )
    #H5 - lewy błotnik ostatni,
    S = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]]).astype(np.float32)

    H1 = np.array([[      1.013,   0.0036699,       -2443],
 [   0.008471,     1.0221,      1880.2],
 [ 3.1404e-06,  5.2511e-06,           1]]).astype(np.float32)
    H1 = S @ H1 @ np.linalg.inv(S)
    H2 =np.array([[    0.99331,    0.024914 ,     4847.5],
 [ -0.0033566,      1.0098,        1888],
 [-1.2409e-06,  4.6596e-06,           1]]).astype(np.float32)
    H2 = S @ H2 @ np.linalg.inv(S)
    H3 = np.array([[    0.99215,    0.027157,      4769.2],
 [ -0.0035934 ,     1.0118 ,     4438.5],
 [-1.3227e-06 , 5.0418e-06,           1]]).astype(np.float32)
    H3 = S @ H3 @ np.linalg.inv(S)
    H4 = np.array([[     1.0145,    0.034837,      2429.5],
 [    0.01286,       1.038,      6119.2],
[  1.771e-06,  7.0018e-06 ,          1]]).astype(np.float32)
    H4 = S @ H4 @ np.linalg.inv(S)
    H5 = np.array([[1.0146343e+00, 1.3305261e-02, 6.5544266e+01],
 [1.1184645e-02, 1.0057985e+00, 4.4326597e+03],
 [1.5989363e-06 ,4.4307935e-06, 1.0000000e+00]]).astype(np.float32)
    H5 = S @ H5 @ np.linalg.inv(S)
    """
    H6 = np.array([[ 1.0074713e+00,  1.0635464e-03,  3.0394157e+01],
 [-2.9231559e-03,  1.0117992e+00 , 2.5832144e+03],
 [-7.9523011e-07,  3.8797717e-07,  1.0000000e+00]]).astype(np.float32)
    H6 = S @ H6 @ np.linalg.inv(S)
    """
    canvas = blend_warp_GPUONLY(front,left,H1,stream7)
    canvas = blend_warp_GPUONLY(canvas,right,H2,stream8)
    canvas = blend_warp_GPUONLY(canvas,right_fender,H3,stream9)
    canvas = blend_warp_GPUONLY(canvas,rear,H4,stream10)
    canvas = blend_warp_GPUONLY(canvas,left_fender,H5,stream11)

    canvas = canvas.download()

    #Przytnij obraz tak samo jak i w 2. sposobie collect_homo
    cropped = canvas
    h, w = cropped.shape[:2]

    crop_top_px    = int(0.1965 * h)
    crop_bottom_px = int(0.177 * h)
    crop_left_px = int(0.11 * w)
    crop_right_px = int(0.11 * w)

    y1 = crop_top_px
    y2 = h - crop_bottom_px
    x1 = crop_left_px
    x2 = w - crop_right_px
    cropped = cropped[y1:y2, x1:x2]

    scalex = 0.18
    scaley = 0.45
    bev_h, bev_w = cropped.shape[:2]
    new_w = int(bev_w * scalex)
    new_h = int(bev_h * scaley)
    car_resized = cv.resize(car, (new_w, new_h), interpolation=cv.INTER_AREA)

    # przesuń żeby był na środku około
    x_offset = (bev_w - new_w) // 2 + 5
    y_offset = (bev_h - new_h) // 2 - 19

    cropped[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = car_resized

    cv.namedWindow("bev",cv.WINDOW_NORMAL)
    cv.imshow("bev",cropped)
    return cropped


def warp_with_cuda(image, H, name,h,w,stream, gpu=True, show=False,first_time=True):
    """
    Funkcja pozwalająca na wyprostowanie obrazu za pomocą homografii H
    Można zarówno jak i wykorzystywać ją do gładkiego przetwarzania na GPU,
    jak i z pobieraniem z powrotem do CPU. Również show pozwala pokazać w
    oddzielnym oknie obraz.
    """


    if first_time:
        gpu_img = cv.cuda.GpuMat()
        gpu_img.upload(image,stream=stream)
    else:
        gpu_img = image
    H_corrected = np.array(np.zeros((3,3))).astype(np.float32)
    # przesuń obrazek, dla tej kamery ekskluzywnie
    if name == "left fender homo":
        #pierwsza w kolumnie translacji - x, druga liczba - y;
        translation = np.array([
        [1, 0, 24],
        [0, 1, 0],
        [0, 0, 1]
        ], dtype=np.float32)
        H_corrected = translation @ H
        #print(H_corrected)
    else:
        H_corrected = H

    warped_gpu = cv.cuda.warpPerspective(gpu_img, H_corrected, (w, h),stream=stream)
    warped = cv.cuda.cvtColor(warped_gpu,cv.COLOR_BGR2RGB,stream=stream)

    if not gpu:
        warped = warped.download()
    else:
        warped = warped
    if show:
        if gpu:
            warp = warped.download()
        else:
            warp=warped
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, warp)
    #stream.waitForCompletion()
    return warped

def warp_and_blend_gpu(img1, img2, H, canvas_size=None, alpha=0.8):

    """
    Szybkie blendowanie obrazó ze wzajemną homografią:
      1) policz rozmiar kanwy i przesunięcie
      2) wyprostuj na kanwę oba obrazy
      3) policz binarne maski na GPU
      4) wyważony alpha-blending na wspólnym obszarze

    img1, img2 : BGR uint8
    H          : homografia float32 z img2 na img1
    canvas_size: (w,h) aby mieć stały rozmiar (nie zaleca się, nie po to to robione)
    alpha      : waga mieszania
    """
    # załaduj
    g1 = cv.cuda_GpuMat(); g1.upload(img1)
    g2 = cv.cuda_GpuMat(); g2.upload(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # policz rozmiar kanwy i translację, wymagane zeby dynamicznie się doklejały obrazy
    if canvas_size is None:
        pts1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        pts2t = cv.perspectiveTransform(pts2, H)
        all_pts = np.vstack([pts1, pts2t])
        x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        trans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]],dtype=np.float32)
        out_w, out_h = x_max - x_min, y_max - y_min
    else:
        trans = np.eye(3, dtype=np.float32)
        out_w, out_h = canvas_size

    # warpuj oba na kanwę
    g1w = cv.cuda.warpPerspective(g1, trans,     (out_w, out_h))
    g2w = cv.cuda.warpPerspective(g2, trans @ H, (out_w, out_h))

    # progowanie na gpu
    gray1 = cv.cuda.cvtColor(g1w, cv.COLOR_BGR2GRAY)
    gray2 = cv.cuda.cvtColor(g2w, cv.COLOR_BGR2GRAY)
    _, m1w = cv.cuda.threshold(gray1, 1, 255, cv.THRESH_BINARY)
    _, m2w = cv.cuda.threshold(gray2, 1, 255, cv.THRESH_BINARY)

    # gdzie się pokrywają
    overlap = cv.cuda.bitwise_and(m1w, m2w)

    # blendowanie w obszarze, gdzie się pokrywają obrazy, zwykłe alfa
    blend = cv.cuda.addWeighted(g1w, 1-alpha, g2w, alpha, 0)

    # tam gdzie się nie pokrywają, zrób normalnie
    inv2 = cv.cuda.bitwise_not(m2w)
    inv1 = cv.cuda.bitwise_not(m1w)
    part1 = cv.cuda.bitwise_and(g1w, cv.cuda.cvtColor(inv2, cv.COLOR_GRAY2BGR))
    part2 = cv.cuda.bitwise_and(g2w, cv.cuda.cvtColor(inv1, cv.COLOR_GRAY2BGR))
    partB = cv.cuda.bitwise_and(blend, cv.cuda.cvtColor(overlap, cv.COLOR_GRAY2BGR))

    # dodaj do siebie te dwie części
    tmp = cv.cuda.add(part1, part2)
    out = cv.cuda.add(tmp, partB)
    # zwróc normalny obraz numpy
    return out.download()

def blend_warp_GPUONLY(g1, g2, H,stream, canvas_size=None, alpha=0.85):
    """
    Szybkie mieszanie i prostowanie dwóch obrazów na GPU.
    Wykorzystuje całkowicie GPU, wejściem są obrazy GpuMat,
    homografia g2->g1, ponadto jeszcze strumień dla GPU.
    Dla finalnej implementacji, kiedy nie trzeba robić już dopasowań.
    """
    # załaduj

    w1, h1 = g1.size()
    w2, h2 = g2.size()


    # policz translację i rozmiar kanwy
    if canvas_size is None:
        #liczy na podstawie obu rozmiarów obrazów wymagany do dopasowania
        pts1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        #perspective aby drugi nakładał się na pierwszy
        pts2t = cv.perspectiveTransform(pts2, H)
        #dalej przekształcenia, żeby policzyć maksymalne i minimalne granice nowego obszaru
        all_pts = np.vstack([pts1, pts2t])
        x_min, y_min = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        trans = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]],dtype=np.float32)
        out_w, out_h = x_max - x_min, y_max - y_min
    else:
        #tutaj jak dajemy rozmiar kanwy, to nie liczy i po prostu przesuwa na miejsce
        trans = np.eye(3, dtype=np.float32)
        out_w, out_h = canvas_size

    # przesuń z homografią
    g1w = cv.cuda.warpPerspective(g1, trans,     (out_w, out_h),stream=stream)
    g2w = cv.cuda.warpPerspective(g2, trans @ H, (out_w, out_h),stream=stream)

    # progowanie aby odnaleźć wspólny obrszar
    gray1 = cv.cuda.cvtColor(g1w, cv.COLOR_BGR2GRAY,stream=stream)
    gray2 = cv.cuda.cvtColor(g2w, cv.COLOR_BGR2GRAY,stream=stream)
    _, m1w = cv.cuda.threshold(gray1, 1, 255, cv.THRESH_BINARY,stream=stream)
    _, m2w = cv.cuda.threshold(gray2, 1, 255, cv.THRESH_BINARY,stream=stream)

    #mieszenie na wspólnym obszarze
    overlap = cv.cuda.bitwise_and(m1w, m2w,stream=stream)
    blend = cv.cuda.addWeighted(g1w, 1-alpha, g2w, alpha, 0,stream=stream)

    # dodaj do siebie tamte fragmenty i otrzymaj końcowy obraz
    inv2 = cv.cuda.bitwise_not(m2w,stream=stream)
    inv1 = cv.cuda.bitwise_not(m1w,stream=stream)
    g1w = cv.cuda.bitwise_and(g1w, cv.cuda.cvtColor(inv2, cv.COLOR_GRAY2BGR),stream=stream)
    g2w = cv.cuda.bitwise_and(g2w, cv.cuda.cvtColor(inv1, cv.COLOR_GRAY2BGR),stream=stream)
    blend = cv.cuda.bitwise_and(blend, cv.cuda.cvtColor(overlap, cv.COLOR_GRAY2BGR),stream=stream)


    out = cv.cuda.add(g1w,g2w,stream=stream)
    out = cv.cuda.add(out, blend,stream=stream)

    return out

#DALSZE FRAGMENTY KODU DO SKOPIOWANIA W ALT_COLLECT_HOMO - TO SĄ DO WYZNACZENIA
#POŁOŻENIA SZACHOWNIC I ODPOWIADAJĄCYCH PUNKTÓW NA KANWIE

"""
def ch_points_calc(pattern_size,square_size,centerpoint):
    half_width = square_size*pattern_size[0]/2
    half_height = square_size*pattern_size[1]/2
    points = np.array(
    [[centerpoint[0]+half_height,centerpoint[1]+half_width],
    [centerpoint[0]+half_height,centerpoint[1]-half_width],
    [centerpoint[0]-half_height,centerpoint[1]-half_width],
    [centerpoint[0]-half_height,centerpoint[1]+half_width]],
    dtype=np.float32)
    return points

left_cp = np.array([-0.425+2.6,3.23],dtype=np.float32)
right_cp = np.array([-0.425+2.2,-3.51],dtype=np.float32)
left_fender_cp = np.array([-0.425-2.85,3.58],dtype=np.float32)
right_fender_cp = np.array([-0.425-2.85,-3.58],dtype=np.float32)
front_cp = np.array([-0.425+4.66,0],dtype=np.float32)
rear_cp = np.array([-0.425-5.25,0],dtype=np.float32)

objp_left = ch_points_calc((6,8),0.6,left_cp)
objp_right = ch_points_calc((5,6),0.6,right_cp)
objp_left_fender = ch_points_calc((8,7),0.5,left_fender_cp)
objp_right_fender = ch_points_calc((8,7),0.5,right_fender_cp)
objp_front = ch_points_calc((10,4),0.4,front_cp)
objp_rear = ch_points_calc((8,5),0.6,rear_cp)

#cor_left = cc.solve_chess_size(left,"left",(7,9),None)
#cor_right = cc.solve_chess_size(right,"right",(6,7),None)
#cor_left_fender = cc.solve_chess_size(left_fender,"left1",(9,8),None)
#cor_right_fender = cc.solve_chess_size(right_fender,"right1",(9,8),None)
#cor_front = cc.solve_chess_size(front,"front",(11,5),None)
#cor_rear = cc.solve_chess_size(rear,"rear",(11,5),None)

def apply_homography_to_points(points, H):
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Dodanie współrzędnych jedności
    points_transformed = np.dot(H, points_homogeneous.T).T  # Mnożenie homografii
    points_transformed /= points_transformed[:, 2].reshape(-1, 1)  # Normalizacja przez Z (homogenizacja)
    return points_transformed[:, :2]  # Zwrócenie tylko współrzędnych x i y

# Zastosowanie homografii do punktów szachownic (lewa i prawa szachownica)
transformed_left_points = apply_homography_to_points(objp_left, H_px_to_m_bev)
transformed_right_points = apply_homography_to_points(objp_right, H_px_to_m_bev)

transformed_front_points = apply_homography_to_points(objp_front, H_px_to_m_bev)
transformed_rear_points = apply_homography_to_points(objp_rear, H_px_to_m_bev)

transformed_left_fender_points = apply_homography_to_points(objp_left_fender, H_px_to_m_bev)
transformed_right_fender_points = apply_homography_to_points(objp_right_fender, H_px_to_m_bev)
bev_left= np.eye(3,3).astype(np.float32)
bev_right = np.eye(3,3).astype(np.float32)
bev_left_fender = np.eye(3,3).astype(np.float32)
bev_right_fender = np.eye(3,3).astype(np.float32)




#rear = warp_and_blend_gpu(canvas,rear,H_rear_to_bev)
#cor_rear = cc.solve_chess_size(rear,"rear",(9,6),None)
"""
"""
if cor_left is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_left_to_bev,_ = cv.findHomography(cor_left,transformed_left_points,cv.RANSAC,2.0)


    if H_left_to_bev is not None:
        bev_left = warp_and_blend_gpu(canvas,left,H_left_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_left_points:
            cv.circle(bev_left, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_points:
            cv.circle(bev_left, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)
        print("h_left_to_bev")
        print(H_left_to_bev)
        print("------------------------------------")
        #cv.namedWindow("left_bev",cv.WINDOW_NORMAL)
        #cv.imshow("left_bev",bev_left)

if cor_right is not None and cor_left is not None:
    #H_right_met_to_px,_ = cv.findHomography(objp_right,cor_right,cv.RANSAC,3.0)
    #H_right_to_bev = H_px_to_m_bev @ H_right_met_to_px
    H_right_to_bev,_ = cv.findHomography(cor_right,transformed_right_points,cv.RANSAC,3.0)
    if H_right_to_bev is not None:
        bev_right = warp_and_blend_gpu(bev_left,right,H_right_to_bev)
        for point in transformed_left_points:
            cv.circle(bev_right, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_points:
            cv.circle(bev_right, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("H_right_to_bev")
        print(H_right_to_bev)
        print("------------------------------------")
        #print(bev_right.shape)
        #cv.namedWindow("right_bev",cv.WINDOW_NORMAL)
        #cv.imshow("right_bev",bev_right)
"""
"""
#
if cor_left_fender is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_left_fender_to_bev,_ = cv.findHomography(cor_left_fender,transformed_left_fender_points,cv.RANSAC,3.0)
    if H_left_fender_to_bev is not None:
        bev_left_fender = warp_and_blend_gpu(canvas,left_fender,H_left_fender_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_left_fender_points:
            cv.circle(bev_left_fender, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_fender_points:
            cv.circle(bev_left_fender, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        cv.namedWindow("left_bev",cv.WINDOW_NORMAL)
        cv.imshow("left_bev",bev_left_fender)

        print("H_left_fender_to_bev")
        print( H_left_fender_to_bev)
        print("------------------------------------")

#
if cor_right_fender is not None :
    #H_right_met_to_px,_ = cv.findHomography(objp_right,cor_right,cv.RANSAC,3.0)
    #H_right_to_bev = H_px_to_m_bev @ H_right_met_to_px
    H_right_fender_to_bev,_ = cv.findHomography(cor_right_fender,transformed_right_fender_points,cv.RANSAC,3.0)
    if H_right_fender_to_bev is not None:
        bev_right_fender = warp_and_blend_gpu(canvas,right_fender,H_right_fender_to_bev)
        for point in transformed_left_fender_points:
            cv.circle(bev_right_fender, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_right_fender_points:
            cv.circle(bev_right_fender, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        #print(bev_right.shape)
        cv.namedWindow("right_bev",cv.WINDOW_NORMAL)
        cv.imshow("right_bev",bev_right_fender)

        print("H_right_fender_to_bev")
        print(H_right_fender_to_bev)
        print("------------------------------------")
"""
"""
if cor_rear is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_rear_to_bev,_ = cv.findHomography(cor_rear,transformed_rear_points,cv.RANSAC,3.0)
    if H_rear_to_bev is not None:
        bev_rear = warp_and_blend_gpu(canvas,rear,H_rear_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_rear_points:
            cv.circle(bev_rear, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        for point in transformed_front_points:
            cv.circle(bev_rear, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("git")
        print(H_rear_to_bev)
        cv.namedWindow("rear_bev",cv.WINDOW_NORMAL)
        cv.imshow("rear_bev",bev_rear)
"""
"""
if cor_front is not None:
    #H_left_met_to_px,_ = cv.findHomography(objp_left,cor_left,cv.RANSAC,3.0)
    #H_left_to_bev = H_px_to_m_bev @ H_left_met_to_px
    H_front_to_bev,_ = cv.findHomography(cor_front,transformed_front_points,cv.RANSAC,3.0)
    if H_front_to_bev is not None:
        bev_front = warp_and_blend_gpu(bev_rear,front,H_front_to_bev)
        # Rysowanie punktów na obrazie (kanwie)
        for point in transformed_front_points:
            cv.circle(bev_front, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)  # Zielone punkty dla lewej szachownicy

        #for point in transformed_right_points:
            #cv.circle(bev_front, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        print("jest git")
        print(H_front_to_bev)
        cv.namedWindow("front_bev",cv.WINDOW_NORMAL)
        cv.imshow("front_bev",bev_front)
"""

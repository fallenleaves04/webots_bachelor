import math
import numpy as np
import time

import heapq
from heapdict import heapdict
from scipy.spatial import KDTree
import pyqtgraph as pg
from pyqtgraph import QtCore
from sklearn.decomposition import PCA
import reeds_shepp
import cv2 as cv
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from typing import List
import traceback

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

"""
Parametry samochodu
"""
# Vehicle parameters
# class C:
#     # parametry samochodu 
#     TRACK_FRONT = 1.628
#     TRACK_REAR = 1.628
#     WHEELBASE = 2.995
#     MAX_WHEEL_ANGLE = 0.5  # rad
#     CAR_WIDTH = 1.95
#     CAR_LENGTH = 4.85
#     MAX_SPEED = -3.0
#     MAX_RADIUS = WHEELBASE/np.tan(MAX_WHEEL_ANGLE)
#     MAX_CURVATURE = 1/MAX_RADIUS
#     # parametry dla A*
#     c_val = 1
#     FORWARD_COST = c_val*1.5
#     BACKWARD_COST = c_val*1.0
#     GEAR_CHANGE_COST = c_val*15.0
#     STEER_CHANGE_COST = c_val*2.5
#     STEER_ANGLE_COST = c_val*2.0
#     OBS_COST = c_val*5.0
#     H_COST = 5.0
#     # parametry dla próbkowania
#     XY_RESOLUTION = 0.1 # m
#     YAW_RESOLUTION = np.deg2rad(5)
#     # parametry dla Stanley
#     K_STANLEY = 0.5

# z pyt1.py, można zamienić później
class C:
    # parametry samochodu 
    TRACK_FRONT = 1.628
    TRACK_REAR = 1.628
    WHEELBASE = 2.995
    MAX_WHEEL_ANGLE = 0.5  # rad
    CAR_WIDTH = 1.95
    CAR_LENGTH = 4.85
    MAX_SPEED = 4.0
    MAX_RADIUS = WHEELBASE/np.tan(MAX_WHEEL_ANGLE) # tg(delta) = L/R 
    MAX_CURVATURE = 1/MAX_RADIUS
    # parametry dla A*
    c_val = 1.5
    FORWARD_COST = c_val*1.2
    BACKWARD_COST = c_val*1.2
    GEAR_CHANGE_COST = c_val*3.0
    STEER_CHANGE_COST = c_val*2.5
    STEER_ANGLE_COST = c_val*2.0
    OBS_COST = c_val*5.0
    H_COST = 1.0
    # parametry dla próbkowania
    XY_RESOLUTION = 0.1 # m
    YAW_RESOLUTION = np.deg2rad(5)
    INTERP_STEP_SIZE = 0.005
    # parametry dla Stanley
    K_STANLEY = 0.5


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def mod2pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def build_pose_matrix(pose:np.ndarray):
    x,y,yaw = pose
    #yaw_rad = np.deg2rad(yaw_deg)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    position = np.array([x,y,0.0]).reshape(1,3)
    T[:3, 3] = position
    return T

from sklearn.cluster import DBSCAN

class OccupancyGrid:
    """
    Siatka zajętości z KD-tree dla detekcji kolizji; aktualizowana z czujników ultradźwiękowych 
    """
    def __init__(self,controller):
        
        self.controller = controller
        self.car_width = C.CAR_WIDTH
        self.car_length = C.CAR_LENGTH
        self.wheel_base = C.WHEELBASE
        self.cg_to_rear_axle = np.array([1.2975, 0, 0.1]) # dane z Webots
        self.cg_x = self.cg_to_rear_axle[0]
        self.rear_bumper_to_axle = 1.0
        self.front_bumper_to_axle = C.CAR_LENGTH - self.rear_bumper_to_axle
        self.collision_radius = np.hypot(max(self.front_bumper_to_axle - self.cg_x,
                                             self.rear_bumper_to_axle - self.cg_x),
                                             0.5*self.car_width) + 1.0

        self.size = (100,50)
        self.xy_resolution = C.XY_RESOLUTION
        self.grid_width = int(round(self.size[0] / self.xy_resolution))
        self.grid_height =  int(round(self.size[1] / self.xy_resolution))
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=np.float32) # indeksy w zasadzie
        #self.x_min = -self.size[0] / 2
        #self.y_min = -self.size[1] / 2
        self.x_min = -20 
        self.y_min = -self.size[1] / 2
        # mapować z czterech czujników prostopadłych do osi samochodu 
        # z każdym odświeżeniem Webots musimy sprawdzić zajętość siatki zgodnie z nowymi pomiarami;
        # obszar mapowania definiowany zgodnie z zdyskretyzowanym stożkiem czujnika - koniec stożka (ostatnie klatki) są zawsze 1.0; poza stożkiem niezdefiniowane
        # wszystkie klatki są domyślnie nieoznaczone, dopóki na nich nie najedzie stożek; jak najedzie, to klatki sa na pewno 0 lub 1, czyli nie aktualizujemy
        # 1: Algorithm occupancy grid mapping(flt−1;ig; xt; zt):
        #     2: for all cells mi do
        #        3: if mi in perceptual field of zt then
        #           4: lt;i = lt−1;i + inverse sensor model(mi; xt; zt) − l0
        #        5: else
        #           6: lt;i = lt−1;i
        #        7: endif
        #     8: endfor
        #     9: return flt;i
        # prawdopodobieństwo p(m_i|z_1:t,x_1:t) tego, że stan klatki mapy 'i' polega na tym, że pomiar i stan samochodu są takie jakie są
        # lt;i = log ( p(mi|z1:t, x1:t) / ( 1 − p(mi|z1:t, x1:t) ) )
        # p(mi|z1:t; x1:t) = 1 − 1 / (1 + exp{l_t,i}) --- log odds
        self.l_0 = 0.0
        self.l_occ = 0.1 
        self.l_free = -0.8
        
        self.obstacles = []
        self.spots = []
        self.detected_spots = []
        self.yolo_points_buffer = []
        self.yolo_x_pts = []
        self.yolo_y_pts = []
        self.ox = None
        self.oy = None

        self.parking_mode = "SEARCH"     # SEARCH | LOCKED | PARKING
        self.locked_slot = None
        self.stable_counter = 0
        self.N_STABLE = 5


    def setup_sensors(self,front_sensor_poses:dict,rear_sensor_poses:dict,ultrasonic_sensors:list,ultrasonic_sensors_apertures:dict,max_min_dict:dict):

        self.front_sensors = ultrasonic_sensors[0]
        self.rear_sensors = ultrasonic_sensors[1]
        self.right_side_sensors = ultrasonic_sensors[2]
        self.left_side_sensors = ultrasonic_sensors[3]
        
        self.ultrasonic_sensors = self.front_sensors + self.rear_sensors + self.right_side_sensors + self.left_side_sensors

        self.max_min_dict = max_min_dict

        self.front_apertures = ultrasonic_sensors_apertures[0]
        self.rear_apertures = ultrasonic_sensors_apertures[1]
        self.right_side_apertures = ultrasonic_sensors_apertures[2]
        self.left_side_apertures = ultrasonic_sensors_apertures[3]

        self.front_sensor_poses = front_sensor_poses
        self.rear_sensor_poses = rear_sensor_poses
        self.sensor_pose_matrices = {**self.front_sensor_poses,**self.rear_sensor_poses}
        self.sensor_poses = {}
        for name in self.ultrasonic_sensors:
            pose_matrix = self.sensor_pose_matrices[name]
            x = pose_matrix[0,3]
            y = pose_matrix[1,3]
            theta = np.arctan2(pose_matrix[1,0],pose_matrix[0,0]) # r10,r00
            self.sensor_poses[name] = (x,y,theta)
        
        # params: słownik, gdzie dla każdego imienia sensora jest wpis, będący słownikiem parametrów
        self.params = {}
        self.front_params = {}
        self.rear_params = {}
        self.right_side_params = {}
        self.left_side_params = {}
        for name in self.front_sensors:
            self.front_params[name] = {"min_range":max_min_dict[name][0],"max_range":max_min_dict[name][1],"aperture":self.front_apertures[name],"pose":self.sensor_poses[name]}
        for name in self.rear_sensors:
            self.rear_params[name] = {"min_range":max_min_dict[name][0],"max_range":max_min_dict[name][1],"aperture":self.rear_apertures[name],"pose":self.sensor_poses[name]}
        for name in self.right_side_sensors:
            self.right_side_params[name] = {"min_range":max_min_dict[name][0],"max_range":max_min_dict[name][1],"aperture":self.right_side_apertures[name],"pose":self.sensor_poses[name]}
        for name in self.left_side_sensors:
            self.left_side_params[name] = {"min_range":max_min_dict[name][0],"max_range":max_min_dict[name][1],"aperture":self.left_side_apertures[name],"pose":self.sensor_poses[name]}
        self.params = {**self.front_params,**self.rear_params,**self.right_side_params,**self.left_side_params}
        if not self.controller.sensors_set:
            self.controller.sensorStats.emit(self.params,self.sensor_poses)

    # ta funkcja musi znaleźć indeks celi w układzie siatki, czyli tak na prawdę to ona 
    def make_cell(self,state) -> tuple:
        x,y = state
        i = int(np.floor((x - self.x_min) / self.xy_resolution))
        j = int(np.floor((y - self.y_min) / self.xy_resolution))

        if -self.grid_width <= i < self.grid_width and -self.grid_height <= j < self.grid_height:
            return (i, j)
        return None
    
    def setup_obstacles(self, obstacles: list):
        """
        Inicjalizacja na podstawie listy słowników przeszkód (z klastrowania).
        """
        self.obstacles = obstacles
        
        if len(obstacles) > 0:
            # Budujemy KD-Tree na środkach przeszkód (centers)
            centers = np.array([obs['center'] for obs in obstacles])
            self.kd_tree = KDTree(centers)
            
            # Obliczamy maksymalny promień przeszkody (połowa przekątnej)
            # Potrzebne do bezpiecznego wyszukiwania w KD-Tree
            self.max_obs_radius = max([np.hypot(o['length'], o['width'])/2 for o in obstacles])
            #print(f"[OccupancyGrid] Ustawiono z {len(obstacles)} przeszkodami (obiektami).")
        else:
            self.kd_tree = None
            self.max_obs_radius = 0
            #print("[OccupancyGrid] Brak przeszkód.")
        
    
    def get_rect_corners(self, center, length, width, angle):
        """Generuje 4 rogi obróconego prostokąta."""
        cx, cy = center
        # Wektory połówek boków
        # Oś wzdłużna (length) - cos, sin
        # Oś poprzeczna (width) - -sin, cos
        
        c, s = np.cos(angle), np.sin(angle)
        
        # Wektory od środka do krawędzi
        dx_l = (length / 2) * c
        dy_l = (length / 2) * s
        
        dx_w = (width / 2) * -s
        dy_w = (width / 2) * c
        
        # przedni lewy, przedni prawy, tylny prawy, tylny lewy
        p1 = np.array([cx + dx_l + dx_w, cy + dy_l + dy_w])
        p2 = np.array([cx + dx_l - dx_w, cy + dy_l - dy_w])
        p3 = np.array([cx - dx_l - dx_w, cy - dy_l - dy_w])
        p4 = np.array([cx - dx_l + dx_w, cy - dy_l + dy_w])
        
        return np.array([p1, p2, p3, p4])

    
    def check_sat_collision(self, rect1, rect2):
        edges1 = rect1 - np.roll(rect1, 1, axis=0)
        edges2 = rect2 - np.roll(rect2, 1, axis=0)
        normals = np.vstack([
            np.column_stack([-edges1[:, 1], edges1[:, 0]]),
            np.column_stack([-edges2[:, 1], edges2[:, 0]])
        ])

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= (norms + 1e-9)

        projections1 = np.matmul(normals, rect1.T)
        projections2 = np.matmul(normals, rect2.T)
        min1, max1 = projections1.min(axis=1), projections1.max(axis=1)
        min2, max2 = projections2.min(axis=1), projections2.max(axis=1)
        collision_free = (max1 < min2) | (max2 < min1)
        
        return not np.any(collision_free)

    def is_collision(self, x, y, yaw) -> bool:
        if self.kd_tree is None:
            return False
        assert len(self.obstacles) == self.kd_tree.data.shape[0]
        box_center_offset = (self.front_bumper_to_axle - self.rear_bumper_to_axle) / 2.0
        
        box_center_x = x + box_center_offset * np.cos(yaw)
        box_center_y = y + box_center_offset * np.sin(yaw)
        
        car_len = self.front_bumper_to_axle + self.rear_bumper_to_axle
        car_wid = self.car_width
        
        rect = self.get_rect_corners((box_center_x, box_center_y), car_len, car_wid, yaw)
        search_radius = self.collision_radius + self.max_obs_radius
        
        query_point = np.array([box_center_x, box_center_y])
        indices = self.kd_tree.query_ball_point(query_point, search_radius)

        if not indices:
            return False

        for idx in indices:
            obs = self.obstacles[idx]
            obs_rect = self.get_rect_corners(
                obs['center'], 
                obs['length'], 
                obs['width'], 
                obs['angle']
            )
            if self.check_sat_collision(rect, obs_rect):
                return True 

        return False  
    
    def sensor_in_wrld(self,car_pose,sensor_pose):
        x_c,y_c,yaw_c = car_pose
        x_s, y_s, yaw_s = sensor_pose

        x_rel = x_c + x_s*np.cos(yaw_c) - y_s*np.sin(yaw_c)
        y_rel = y_c + x_s*np.sin(yaw_c) + y_s*np.cos(yaw_c)
        yaw_rel = yaw_c + yaw_s
        return x_rel,y_rel,yaw_rel

    def extract_obstacles(self):
        indices_i, indices_j = np.where(self.grid > 0)

        # Jeśli pusto, zwróć puste tablice
        if len(indices_i) == 0:
            return np.array([]), np.array([])

        ox = self.x_min + indices_i * self.xy_resolution
        oy = self.y_min + indices_j * self.xy_resolution
        self.ox = ox
        self.oy = oy
        return ox, oy

    def update_grid(self, sensor_name, dist, car_pose):
        # pozycja sensora względem samochodu
        sensor_pose_rel = self.sensor_poses[sensor_name]
        # bieżąca pozycja sensora plus jego orientacja
        sensor_global_x, sensor_global_y, sensor_global_theta = self.sensor_in_wrld(car_pose, sensor_pose_rel)
        params = self.params[sensor_name]
        # maksymalna odległość na którą może sięgnąć czujnik
        max_range = params["max_range"]
        # kąt rozwarcia w granicach którego będzie zbadana zajętość klatek
        beta = params["aperture"]
        # minimalna odległość na którą może sięgnąć czujnik
        min_range = params["min_range"]
        # rozwiązanie problemu, kiedy czujnik zwracał wartość mniejszą o amplitudę szumu
        if dist >= max_range - 0.05:
            effective_dist = max_range - 0.05
            is_hit = False # nie ma wskazania, bo czujnik zwraca domyślnie tą wartość
        else:
            # jeżeli wykryto że odległość wskazana przez czujnik jest mniejsza niż maksymalna - szum, to daj zielone na przeszkodę
            effective_dist = dist
            is_hit = True
        
        # zrób wokół czujnika kwadrat o rozmiarze max_range
        x_min = sensor_global_x - max_range
        x_max = sensor_global_x + max_range
        y_min = sensor_global_y - max_range
        y_max = sensor_global_y + max_range
        i_min, j_min = self.make_cell((x_min, y_min)) or (0, 0) # daj na 0 jeśli None
        i_max, j_max = self.make_cell((x_max, y_max)) or (self.grid_width-1, self.grid_height-1)
        # ogranicz do rozmiarów siatki
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.grid_width - 1, i_max)
        j_max = min(self.grid_height - 1, j_max)
        # poza siatką
        if i_min > i_max or j_min > j_max:
            return
        # zakres indeksów siatki
        range_i = np.arange(i_min, i_max + 1)
        range_j = np.arange(j_min, j_max + 1)
        # przekształć do rozmiarów siatki i jej rozdzielczości
        world_x = self.x_min + range_i * self.xy_resolution
        world_y = self.y_min + range_j * self.xy_resolution
        # zrób zakres X,Y dla oznaczenia siatek we współrzędnych geometrycznych
        X, Y = np.meshgrid(world_x, world_y, indexing='ij')
        # przekształć z powrotem na lokalny układ czujnika
        DX = X - sensor_global_x
        DY = Y - sensor_global_y
        # przekształć na współrzędne biegunowe
        R_grid = np.hypot(DX, DY)
        PHI_grid = mod2pi(np.arctan2(DY, DX) - sensor_global_theta)
        #PHI_grid = (PHI_grid + np.pi) % (2 * np.pi) - np.pi
        # PHI_grid musi być mniejsze od kąta rozwarcia
        mask_angle = np.abs(PHI_grid) <= (beta / 2.0)
        mask_greater_than_min = R_grid >= min_range
        thickness = self.xy_resolution
        buffer = 0.1
        mask_free = mask_angle & (R_grid < (effective_dist - buffer)) & mask_greater_than_min
        mask_occ = mask_angle & (np.abs(R_grid - effective_dist) <= thickness/2) & is_hit & mask_greater_than_min

        sigma = beta / 4.0 
        
        angle_weights = np.exp(-0.5 * (PHI_grid / sigma)**2)
        
        dist_weights = 1.0 - (R_grid / max_range)**2 
        dist_weights[dist_weights < 0] = 0
        #weights = angle_weights * dist_weights
        weights = dist_weights
        
        grid_slice = self.grid[i_min:i_max+1, j_min:j_max+1]
        grid_slice[mask_free] += self.l_free * weights[mask_free]
        grid_slice[mask_occ]  += self.l_occ * weights[mask_occ]

        np.clip(grid_slice, -20.0, 20.0, out=grid_slice)

        self.grid[i_min:i_max+1, j_min:j_max+1] = grid_slice

        #self.controller.sensorConeDraw.emit(sensor_name,sensor_global_x, sensor_global_y, sensor_global_theta, dist, beta)

    # jaka jest najlepsza reprezentacja tej siatki? aby samochód zaczynał po środku mapy, to trzeba uwzględnić ujemne indeksy
    # ale jak liczyć indeksy, jeżeli poza pojazdu jest ciągła? 
    def interpret_readings(self,names_dists:dict,car_pose:tuple):
        name = "distance sensor left front side"
        dist = names_dists[name]
        self.update_grid(name,dist,car_pose)
        name = "distance sensor left side"
        dist = names_dists[name]
        self.update_grid(name,dist,car_pose)
        name = "distance sensor right front side"
        dist = names_dists[name]
        self.update_grid(name,dist,car_pose)
        name = "distance sensor right side"
        dist = names_dists[name]
        self.update_grid(name,dist,car_pose)

    def analyze_clusters(self,car_pose):
        if self.ox is None: return []
        points = np.column_stack((self.ox, self.oy))
        car_x,car_y,car_yaw = car_pose
        # 1. Klastrowanie
        clustering = DBSCAN(eps=0.4, min_samples=4).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        obstacles = []

        for label in unique_labels:
            if label == -1: continue
            cluster_points = points[labels == label]
            
            # Odrzucamy szum (za małe obiekty)
            if len(cluster_points) < 5: continue
            
            rect = cv.minAreaRect(cluster_points.astype(np.float32))
            (center_x, center_y), (w, h), ang_deg = rect
            
            if w > h:
                obs_len, obs_wid = w, h
                ang = np.radians(ang_deg)
            else:
                obs_len, obs_wid = h, w
                # zamiana boków
                ang = np.radians(ang_deg + 90)
            ang = mod2pi(ang)
            add = 1.5
            obs_wid1 = obs_wid + add
            obs_len1 = obs_len  
            vec1 = np.array([-np.sin(ang),np.cos(ang)])
            vec2 = np.array([center_x - car_x,center_y - car_y])

            dot = np.dot(vec2,vec1)
            sign = 1.0 if dot > 0 else -1.0

            shift = (add / 2.0) * sign
            center_x += vec1[0] * shift
            center_y += + vec1[1] * shift
            obstacles.append({
                'center': np.array([center_x,center_y]),
                'angle': ang,
                'length': obs_len1,
                'width': obs_wid1,
                'points': cluster_points
            })
        
        return obstacles

    def find_spots_scanning(self, car_pose, spot_type, side):
        if spot_type is None or side is None:
            return []
        
        car_x, car_y, car_yaw = car_pose
        
        if spot_type == 'parallel':
            required_len = 5.8
            min_spot_depth = 2.1
            orientation_offset = 0.0
        else: # perpendicular
            required_len = 2.7
            min_spot_depth = 5.3
            orientation_offset = np.pi/2 if side == 'right' else -np.pi/2

        side_threshold = 0.25  
        c, s = np.cos(car_yaw), np.sin(car_yaw)
        R = np.array([[c, -s], [s, c]])
        
        all_obs = []
        
        for obs in self.obstacles:
            l, w, ang = obs['length'], obs['width'], obs['angle']
            corners_glob = self.get_rect_corners(obs['center'], l, w, ang)
            corners_loc = (corners_glob - np.array([car_x, car_y])) @ R.T
            min_x, max_x = np.min(corners_loc[:,0]), np.max(corners_loc[:,0])
            min_y, max_y = np.min(corners_loc[:,1]), np.max(corners_loc[:,1])
            centroid_x = (min_x + max_x) / 2.0
            centroid_y = (min_y + max_y) / 2.0
            obs_data = {
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'avg_y': (min_y + max_y) / 2.0,
                'centroid': (float(centroid_x), float(centroid_y)),
                'is_car': bool(obs.get('is_car', False)),
                'original': obs
            }
            all_obs.append(obs_data)
        side_sign = -1.0 if side == 'right' else 1.0
        on_side = lambda o: side_sign * o['avg_y'] > side_threshold
        cars = [o for o in all_obs if o['is_car'] and on_side(o)]
        if len(cars) < 2:
            return []
        cars.sort(key=lambda c: c['min_x'])
        spots = []
        if len(cars) < 2:
            return []
        
        
        
        for i in range(len(cars) - 1):
            # iterując po liście samochodów odnajdujemy luki między dwoma sąsiednimi
            obs1 = cars[i]
            obs2 = cars[i+1]

            if 'ref_pose' not in obs1:
                obs1['ref_pose'] = car_pose

            car_x0, car_y0, car_yaw0 = obs1['ref_pose']
            c0, s0 = np.cos(car_yaw0), np.sin(car_yaw0)
            R0 = np.array([[c0, -s0], [s0, c0]])

            # tylko wystarczająco długie luki są akceptowane
            gap_start = obs1['max_x']
            gap_end = obs2['min_x']
            gap_size = gap_end - gap_start
            if gap_size < required_len:
                continue
            # w tej chwili głębokość jest równa najdalszemu samochodowi
            # jak udało się znaleźć taką przeszkodę, to możemy na podstawie tego odsunąć miesce parkingowe docelowe od niej
            # wyznaczamy linię referencyjną wzdłuż samochodu jako krawędź najbliższego samochodu (jeśli po prawej np. jest (-3.0) i (-3.5), to się wybierze -3.0, tak samo po lewej)
            ref_line_y =  min(abs(obs1['max_y']),abs(obs2['max_y'])) if side == "right" else \
                            min(abs(obs1['min_y']),abs(obs2['min_y'])) 
            prev_depth =  max(abs(obs1['max_y']),abs(obs2['max_y'])) if side == "right" else \
                            max(abs(obs1['min_y']),abs(obs2['min_y']))
            current_depth = prev_depth 
            curb_detected = False
            # wśród innych pozostałych przeszkód musimy odnaleźć taką co leży głębiej niż samochody i również pomiędzy nimi
            for obs in all_obs:
                # płot, słup, drzewo czy cokolwiek musi leżeć w okolicach max_x < gap_end + 0.2 i min_y > gap_start - 0.2
                if obs is obs1 or obs is obs2:
                    continue
                cx, cy = obs['centroid']
                if not (gap_start <= cx <= gap_end):
                    continue
                # musi być po tej samej stronie
                if not on_side(obs):
                    continue
                obs_depth = abs(obs['max_y']) if side == "right" else abs(obs['min_y'])
                if obs_depth > current_depth + 1e-6:
                    current_depth = obs_depth
                    curb_detected = True
            
            usable_depth = ref_line_y + min_spot_depth / 2.0     
            if curb_detected:
                if (current_depth - ref_line_y) < min_spot_depth - 1e-6:
                    # za mało głębokości od linii referencyjnej do krawężnika
                    continue
                usable_depth = usable_depth
            else:
                usable_depth = usable_depth
            # Obliczenia liczby miejsc
            num_spots = int(gap_size // required_len)
            if num_spots <= 0:
                continue
            spot_positions_x = []
            if num_spots == 1:
                spot_positions_x.append((gap_start + gap_end) / 2.0)
            elif num_spots > 1:
                step = gap_size / num_spots
                for j in range(num_spots):
                    spot_x = gap_start + step * (j + 0.5)
                    spot_positions_x.append(spot_x)
            
            for spot_x in spot_positions_x:
                
                spot_center_y = side_sign * usable_depth
                vec_local = np.array([spot_x, spot_center_y])
                #center_global = np.array([car_x, car_y]) + R.T @ vec_local
                center_global = np.array([car_x0, car_y0]) + R0.T @ vec_local
                final_angle = car_yaw + orientation_offset
                slot_rect = self.get_rect_corners(
                    center=center_global,
                    length=required_len,
                    width=min_spot_depth,
                    angle=final_angle
                )
                
                blocked = False
                for obs in all_obs:
                    if obs is obs1 or obs is obs2:
                        continue
                    # ignoruj samochody (chcemy sprawdzać płoty/słupy)
                    if obs['is_car']:
                        continue
                    obs_rect = self.get_rect_corners(
                        obs['original']['center'],
                        obs['original']['length'],
                        obs['original']['width'],
                        obs['original']['angle']
                    )
                    if self.check_sat_collision(slot_rect, obs_rect):
                        blocked = True
                        break

                if blocked:
                    continue

                
                offset = (self.front_bumper_to_axle-self.rear_bumper_to_axle)/2.00
                target_x = center_global[0] - offset * np.cos(final_angle)
                target_y = center_global[1] - offset * np.sin(final_angle)                
                spots.append({
                    'type': spot_type,
                    'side': side,
                    'center': center_global,
                    'target_rear_axle': np.array([target_x, target_y]),
                    'orientation': final_angle,
                    'length': C.CAR_LENGTH,
                    'width': C.CAR_WIDTH,      
                    'obs1': obs1['original'],
                    'obs2': obs2['original'],
                    # "id": int,                        
                    #         "confidence": float,              
                    #         "last_seen": timestamp,
                    #         "size_ok": bool,
                    #         "state": one_of([
                    #             "candidate",       # wykryte, ale niepewne
                    #             "stable",          # stabilne, można rozważać
                    #             "selected",        # WYBRANE do parkowania
                    #             "rejected",        # odrzucone (np. za małe)
                    #             "invalid"          # unieważnione (śmietnik itd.)
                    #         ])
                })

        return spots

    def choose_spot(self,car_pose):
        cx, cy, _ = car_pose

        if not self.spots:
            return None, None

        best = min(
            self.spots,
            key=lambda sp: np.hypot(
                sp['target_rear_axle'][0] - cx,
                sp['target_rear_axle'][1] - cy
            )
        )

        dist = np.hypot(
            best['target_rear_axle'][0] - cx,
            best['target_rear_axle'][1] - cy
        )
        return best,dist


    def match_semantics_with_sat(self):
        if len(self.yolo_points_buffer) < 5 or not self.obstacles:
            return

        yolo_pts = np.array(self.yolo_points_buffer)
        sem_clustering = DBSCAN(eps=1.0, min_samples=3).fit(yolo_pts)
        sem_labels = sem_clustering.labels_
        yolo_clusters_rects = []
        rects_to_send = []
        for label in set(sem_labels):
            if label == -1: continue
            pts = yolo_pts[sem_labels == label]
            if pts is None or len(pts) < 4:
                continue
            pts = pts.astype(np.float32)
            try:
                rect = cv.minAreaRect(pts)
                rect_info = [rect[0], rect[1][0], rect[1][1], np.radians(rect[2])]
                corners = self.get_rect_corners(rect[0], rect[1][0], rect[1][1], np.radians(rect[2]))
                rects_to_send.append(rect_info)
                yolo_clusters_rects.append(corners)
            except cv.error as e:
                continue
        for obs in self.obstacles:
            obs['is_car'] = False
            obs_rect = self.get_rect_corners(obs['center'], obs['length'], obs['width'], obs['angle'])
            for yolo_rect in yolo_clusters_rects:
                if self.check_sat_collision(obs_rect, yolo_rect):
                    obs['is_car'] = True
                    obs['type'] = 'car' 
                    break
        #self.controller.yoloRects.emit(rects_to_send)

class Path:
    """Reprezentuje ścieżkę (wynik planowania)"""
    def __init__(self, xs, ys, yaws, directions, curvs,costs):
        self.xs = xs       # Lista punktów
        self.ys = ys
        self.yaws = yaws
        self.costs = costs
        def ch(d):
            if d == "reverse":
                d = -1
            elif d == "forward":
                d = 1
            else: 
                d = d 
            return d
        self.directions = [ch(d) for d in directions]
        self.curvs = curvs 
        self.ind_old = 0

        self.int_v = 0.0
        self.kp = 2.0
        self.ki = 5.0
        self.v_cmd_prev = 0.0

        self.s = self._build_len_s()
        self.last_ind = 0

        self.K_theta = 1.5
        self.K_e = 0.7
        self.segments = []
        
        self.v_prev = 0.0
        self.pid_switch = False

        self.active_segment = 0
        self.segment_hold = True
        self.changed = False

        self.ind_la = 0
        start = 0
        for i in range(1, len(self.directions)):
            if self.directions[i] != self.directions[i-1]:
                self.segments.append((start, i-1))
                start = i

        self.segments.append((start, len(self.directions)-1))

    # Metody pomocnicze:
    def __len__(self):          # len(path)
        return len(self.xs)
    def get_point(self, idx):
        return (self.xs[idx], self.ys[idx], self.yaws[idx])
    def reset_runtime_state(self):
        self.active_segment = 0
        self.segment_hold = True
        self.changed = False
        self.last_ind = 0
    # po to żeby śledzić odcinki długości   
    def _build_len_s(self):
        if len(self.xs) == 0:
            return []
        s = [0.0]
        for i in range(1, len(self.xs)):
            ds = np.hypot(self.xs[i]-self.xs[i-1], self.ys[i]-self.ys[i-1])
            s.append(s[-1] + ds)
            
        self.goal = np.array([self.xs[-1], self.ys[-1], self.yaws[-1]])
        return s
    
    def calc_theta_e_and_er(self, x,y,yaw):

        ind = self.nearest_index(x, y)
        k = self.curvs[ind]
        path_yaw = self.yaws[ind]
        n = np.array([math.sin(path_yaw), -math.cos(path_yaw)])
        d = np.array([x - self.xs[ind], y - self.ys[ind]])
        er = float(np.dot(d, n))    
        theta_e = mod2pi(yaw-path_yaw)

        return theta_e, er, k, yaw, ind

    def nearest_index(self, xx, yy, window=10):
        start = self.last_ind
        end = min(start + window, len(self.xs))
        dx = np.array(self.xs[start:end]) - xx
        dy = np.array(self.ys[start:end]) - yy
        ind = start + int(np.argmin(np.hypot(dx, dy)))
        self.last_ind = ind

        return ind

    def get_longitudinal_error(self, x, y, seg_end_idx):
        dx = self.xs[seg_end_idx] - x
        dy = self.ys[seg_end_idx] - y

        yaw_end = self.yaws[seg_end_idx]
        e_long = dx * np.cos(yaw_end) + dy * np.sin(yaw_end)
        
        return e_long

    def speed_control(self,target_v,v_meas,dist_to_end,e_long,delta,dt,a_max = 4.0,a_dec = 4.0,a_lat_max = 1.7,kp_long = 10.0, e_long_thresh = 0.2):
        
        v_stop = np.sqrt(2 * a_dec * max(dist_to_end,0.0))
        v_ref = min(abs(target_v), v_stop)
        v_ref *= np.sign(target_v)
        kappa = np.tan(delta)/C.WHEELBASE
        if dist_to_end <= e_long_thresh and e_long < 0:
            # e_long < 0 znaczy, że przejechał za daleko
            # Chcemy dodać ujemną składową do prędkości do przodu (lub dodatnią do tyłu)
            correction = kp_long * e_long  # to będzie ujemne
            v_ref += correction
            
        
        v_ref = np.clip(v_ref, -C.MAX_SPEED, C.MAX_SPEED)
        kappa = np.tan(delta) / C.WHEELBASE
        v_curve = np.sqrt(a_lat_max / max(abs(kappa), 1e-3))
        v_ref = min(abs(v_ref), v_curve) * np.sign(v_ref)
        v_cmd = np.clip(
            v_ref,
            self.v_cmd_prev - a_dec * dt,
            self.v_cmd_prev + a_max * dt
        )

        self.v_cmd_prev = v_cmd
        self.v_prev = v_meas
        return v_cmd
    
    def rear_wheel_feedback_control(self,x,y,v,yaw): 
        self.changed = False
        theta_e, er, k, yaw, ind = self.calc_theta_e_and_er(x,y,yaw)
        
        seg_start, seg_end = self.get_segment_bounds()
        ind_nearest = np.clip(ind, seg_start, seg_end)

        if abs(theta_e) < 1e-3:
            sinc = 1.0
        else:
            sinc = math.sin(theta_e) / theta_e

        term_e = self.K_e * v * sinc * er
        omega = (
            v * k * math.cos(theta_e) / (1.0 - k * er)
            - self.K_theta * abs(v) * theta_e
            - term_e
        )

        if abs(v) < 1e-4:
            delta = 0.0
        else:
            delta = math.atan(C.WHEELBASE * omega / v)
            if v < 0:
                delta = -delta
        return delta, ind,self.changed

    def get_segment_bounds(self):
        return self.segments[self.active_segment]
    
    def advance_segment(self):
        if self.active_segment < len(self.segments) - 1:
            self.active_segment += 1
            self.segment_hold = True
            return True
        return False
    

    def pure_pursuit(self, x, y, yaw, v, ld=0.5):
        self.changed = False
        ind_nearest = self.nearest_index(x, y)
        seg_start, seg_end = self.get_segment_bounds()
        ind_nearest = np.clip(ind_nearest, seg_start, seg_end)

        s_now = self.s[ind_nearest]
        s_end = self.s[seg_end]

        s_target = min(s_now + ld, s_end)

        self.ind_la = ind_nearest
        for i in range(ind_nearest, seg_end + 1):
            if self.s[i] >= s_target:
                self.ind_la = i
                break
        
        # if (seg_end - ind_nearest) <= 2:
        #     self.ind_la = seg_end
        #     if self.segment_hold:
        #         self.segment_hold = False
        #         self.changed = self.advance_segment()

        
        tx = self.xs[self.ind_la]
        ty = self.ys[self.ind_la]

        dx = tx - x
        dy = ty - y
        alpha = mod2pi(np.arctan2(dy, dx) - yaw)

        delta = math.atan2(
            2.0 * C.WHEELBASE * math.sin(alpha),
            max(ld, 1e-3)
        )

        # if self.segment_transition:
        #     return 0.0,ind_nearest
        
        return delta, self.ind_la, self.changed
    

def savitzky_golay_filt(x, window=11, poly=3):
    try:

        # window musi być nieparzyste i <= len(x)
        window = min(window, len(x) if len(x)%2==1 else len(x)-1)
        window = max(window, poly+2 if (poly+2)%2==1 else poly+3)
        if window >= len(x):
            return x
        return savgol_filter(x, window_length=window, polyorder=poly, mode='interp')
    except Exception:
        return x

def resample_and_smooth_path(path:Path, ds=C.INTERP_STEP_SIZE):
    
    xs = np.asarray(path.xs, dtype=float)
    ys = np.asarray(path.ys, dtype=float)
    dirs = np.asarray(path.directions, dtype=int)

    if len(xs) < 3:
        return path  # nie ma co robić

    segs = path.segments

    out_x, out_y, out_dir, out_kappa, out_seglen = [], [], [], [], []

    for seg_i, (i0, i1) in enumerate(segs):
        # wycinek segmentu
        x_seg = xs[i0:i1+1]
        y_seg = ys[i0:i1+1]
        d_seg = dirs[i0:i1+1]
        if len(x_seg) < 2:
            continue

        # lokalne s dla segmentu
        ds_local = np.hypot(np.diff(x_seg), np.diff(y_seg))
        s_local = np.concatenate([[0.0], np.cumsum(ds_local)])
        s_end = s_local[-1]
        if s_end < 1e-6:
            continue

        s_dense = np.arange(0.0, s_end + 1e-9, ds)

        # interpolacja liniowa x(s), y(s) 
        x_dense = np.interp(s_dense, s_local, x_seg)
        y_dense = np.interp(s_dense, s_local, y_seg)

        x_dense = savitzky_golay_filt(x_dense, window=11, poly=3)
        y_dense = savitzky_golay_filt(y_dense, window=11, poly=3)
        
        # kierunek stały w segmencie
        dir_val = int(np.sign(d_seg[0])) if np.sign(d_seg[0]) != 0 else 1
        dir_dense = np.full_like(x_dense, dir_val, dtype=int)

        seglen_dense = np.full_like(x_dense, dir_val * s_end, dtype=float)

        kappa_dense = np.interp(s_dense, s_local, path.curvs[i0:i1+1])
        kappa_dense = savitzky_golay_filt(kappa_dense, window=21, poly=3)

        if seg_i > 0:
            # usuń pierwszy punkt żeby nie dublować granicy
            x_dense = x_dense[1:]
            y_dense = y_dense[1:]
            dir_dense = dir_dense[1:]
            seglen_dense = seglen_dense[1:]
            kappa_dense = kappa_dense[1:]

        out_x.append(x_dense)
        out_y.append(y_dense)
        out_kappa.append(kappa_dense)
        out_dir.append(dir_dense)
        out_seglen.append(seglen_dense)

    x_all = np.concatenate(out_x)
    y_all = np.concatenate(out_y)
    dir_all = np.concatenate(out_dir)
    kappa = np.concatenate(out_kappa)
    seglen_all = np.concatenate(out_seglen)
    dx = np.gradient(x_all, ds)
    dy = np.gradient(y_all, ds)
    yaw = np.unwrap(np.arctan2(dy, dx))
    
    new_path = Path(
        xs=list(x_all),
        ys=list(y_all),
        yaws=list(yaw),
        directions=list(dir_all),
        curvs=list(kappa),
        costs=path.costs
    )
    return new_path

class Node:
    def __init__(self, cell: tuple, state: tuple, delta, direction: str, g_cost, h_cost, parent=None):
        self.cell = cell # pozycja klatki na zdyskretyzowanej siatce
        self.state = state # stan rzeczywisty samochodu
        self.delta = delta
        self.direction = direction
        self.g_cost = g_cost # koszt globalny
        self.h_cost = h_cost # koszt heurystyki (znamy zgodnie z końcem i początkiem ścieżki)
        self.f_cost = g_cost + h_cost
        self.parent:Node = parent # poprzednia klatka

class PriorityQueue:
    def __init__(self):
        self.heap = heapdict()
        self.nodes = {} 

    def push(self, node:Node):
        cell = node.cell
        existing:Node = self.nodes.get(cell)
        if existing is None or node.g_cost < existing.g_cost - 1e-6:
            self.heap[cell] = node.f_cost
            self.nodes[cell] = node

    def pop(self) -> Node:
        cell, _ = self.heap.popitem()
        return self.nodes.pop(cell)

    def contains(self, cell):
        return cell in self.heap

    def get_node(self, cell) -> Node:
        return self.nodes.get(cell, None)
    
    def empty(self):
        return len(self.heap) == 0


class NewPlanner(QtCore.QObject):
    expansionData = QtCore.pyqtSignal(object)
    hmapData = QtCore.pyqtSignal(object)
    def __init__(self,controller):
        super().__init__()
        self.goal_tolerance = 0.5
        self.xy_resolution = C.XY_RESOLUTION
        self.yaw_resolution = C.YAW_RESOLUTION  
        self.expansion_counter = 0.0
        self.step_size = 0.5
        self.n_steers = 5
        self.actions = self.calc_actions()
        self.hmap = None
        self.dist_map = None
        self.controller = controller
        self.expansionData.connect(self.controller.expansionUpdated)
        self.hmapData.connect(self.controller.hmapUpdated)

    def calc_actions(self):
        motion_actions = []
        steers = np.linspace(-C.MAX_WHEEL_ANGLE, C.MAX_WHEEL_ANGLE, self.n_steers,endpoint=True)
        steers = np.unique(np.concatenate([steers, np.array([0.0])]))  
        for delta in steers:
            for direction in ["forward", "reverse"]:
                motion_actions.append((delta, direction))
        return motion_actions
    
    def discretize_state(self,cur_node_state,direction) -> tuple:
        x, y, theta = cur_node_state
        yaw_bins = int(round(2*np.pi / self.yaw_resolution))
        yaw_i = int(round((theta % (2*np.pi)) / self.yaw_resolution)) % yaw_bins

        return (int(round(x/self.xy_resolution)),
                int(round(y/self.xy_resolution)),
                yaw_i,
                direction)
    

    def calculate_unconstrained_heuristic(self, start_pose, goal_pose, grid: OccupancyGrid):
        
        obstacles = grid.obstacles 
        
        gx, gy, _ = goal_pose
        sx, sy, _ = start_pose

        all_x = [sx, gx]
        all_y = [sy, gy]

        collision_radius = 0.5
        obs_polygons = []
        for obs in obstacles:
            corners = grid.get_rect_corners(obs['center'],obs['length']+ 2 * collision_radius,obs['width']+ 2 * collision_radius,obs['angle'])
            obs_polygons.append(corners)
            for p in corners:
                all_x.append(p[0])
                all_y.append(p[1])

        
        pad = 2
        minx = int(np.floor((min(all_x) - pad )/ self.xy_resolution)) 
        maxx = int(np.ceil((max(all_x) + pad )/ self.xy_resolution)) 
        miny = int(np.floor((min(all_y) - pad )/ self.xy_resolution)) 
        maxy = int(np.ceil((max(all_y) + pad )/ self.xy_resolution)) 

        xw, yw = maxx - minx, maxy - miny
        hmap = np.full((xw, yw), np.inf)
        obs_map = np.zeros((xw, yw), dtype=bool)

        grid_x, grid_y = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy), indexing='ij')

        world_grid_x = grid_x * self.xy_resolution
        world_grid_y = grid_y * self.xy_resolution
        
        for obs in obstacles:
            # Parametry przeszkody powiększonej o margines
            l = obs['length'] + 2 * collision_radius
            w = obs['width'] + 2 * collision_radius
            cx, cy = obs['center']
            ang = obs['angle']

            dx = world_grid_x - cx
            dy = world_grid_y - cy

            c, s = np.cos(-ang), np.sin(-ang)
            local_x = dx * c - dy * s
            local_y = dx * s + dy * c
            mask = (np.abs(local_x) <= l/2) & (np.abs(local_y) <= w/2)

            obs_map[mask] = True
        
        goal_node_x = int(gx / self.xy_resolution) - minx
        goal_node_y = int(gy / self.xy_resolution) - miny
        
        if not (0 <= goal_node_x < xw and 0 <= goal_node_y < yw):
            print("[Heurystyka] Cel poza granicami mapy heurystycznej.")
             
        if obs_map[goal_node_x, goal_node_y]:
            print("[Heurystyka] Cel jest wewnątrz przeszkody (po rozszerzeniu).")
            

        hmap[goal_node_x, goal_node_y] = 0.0
        open_set = []
        heapq.heappush(open_set, (0.0, goal_node_x, goal_node_y))
        
        motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        
        while open_set:
            cost, cx, cy = heapq.heappop(open_set)
            
            if cost > hmap[cx, cy]:
                continue

            for dx, dy, move_cost in motions:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < xw and 0 <= ny < yw:
                    if not obs_map[nx, ny]:

                        new_cost = cost + move_cost
                        if new_cost < hmap[nx, ny]:
                            hmap[nx, ny] = new_cost
                            heapq.heappush(open_set, (new_cost, nx, ny))
        
        #self.current_potential_map = potential_map
        self.map_offset_x = minx
        self.map_offset_y = miny               
        return hmap, minx, miny, xw, yw
    
    def calculate_hybrid_heuristic(self,pose,goal_pose):
        h_rs = reeds_shepp.path_length(pose, goal_pose, C.MAX_RADIUS) 
        dist = np.hypot(goal_pose[0] - pose[0], goal_pose[1] - pose[1])
        h_a_star = dist 
        if self.hmap is not None:
            h_map, minx, miny, xw, yw = self.hmap
            pose_cell_x = int(round(pose[0]/self.xy_resolution)) - minx
            pose_cell_y = int(round(pose[1]/self.xy_resolution)) - miny

            if 0 <= pose_cell_x < xw and 0 <= pose_cell_y < yw:
                v = h_map[pose_cell_x, pose_cell_y] * self.xy_resolution
                if np.isfinite(v):
                    h_a_star = v
                else:
                    h_a_star = dist
        
        return max(h_a_star,h_rs) 
        
    def simulate_motion(self, state, delta, direction):
        x, y, theta = state
        d = 1.0 if direction == "forward" else -1.0
        theta_new = mod2pi(theta + np.tan(delta)/C.WHEELBASE * self.step_size)
        theta = (theta + theta_new)/2.0
        x_new = x + d * self.step_size * np.cos(theta)
        y_new = y + d * self.step_size * np.sin(theta)
        return (x_new, y_new, theta_new)
    
    def get_neighbours(self, node:Node, grid:OccupancyGrid):
        neighbours = []
        
        actions = self.actions
        for delta,direction in actions:
            next_state = self.simulate_motion(
                node.state,
                delta,
                direction 
            )
            if grid.is_collision(*next_state):
                continue

            cost = self.motion_cost(node, next_state, delta, direction)
            neighbour = Node(
                cell=self.discretize_state(next_state,direction),
                state=next_state,
                delta=delta,
                direction=direction,
                g_cost=node.g_cost + cost,
                h_cost=0.0,  
                parent=node
            )
            neighbours.append(neighbour)
        
        return neighbours   
    
    def motion_cost(self, node:Node, to_state, delta, direction):
        dist = self.step_size
        cost = dist
        # dyaw = to_state[2] - node.state[2]
        # cost = abs(C.MAX_RADIUS * dyaw) 
        
        cost *= C.FORWARD_COST if direction == "forward" else C.BACKWARD_COST
        cost *= C.STEER_CHANGE_COST * abs(node.delta - delta) if abs(node.delta - delta) > 1e-2 else 1.0
        cost *= C.STEER_ANGLE_COST * abs(delta) if delta != 0.0 else 1.0
        cost *= C.GEAR_CHANGE_COST if node.direction != direction else 1.0
        
        return cost

    def try_reeds_shepp(self,node:Node,goal_pose,grid:OccupancyGrid):
        rs_path = reeds_shepp.path_sample(node.state,goal_pose,C.MAX_RADIUS,self.xy_resolution)
        
        if not rs_path:
            return None,None
        for i in range(0,len(rs_path),1): 
            (rs_xs,rs_ys,rs_yaws,rs_curvs,rs_segment_lengths) = rs_path[i]
            if grid.is_collision(rs_xs,rs_ys,rs_yaws):
                return None,None
        
        path_cost = reeds_shepp.path_length(node.state,goal_pose,C.MAX_RADIUS)
        goal_node = Node(
            cell=self.discretize_state(goal_pose,node.direction),
            state=goal_pose,
            delta=0.0,
            direction=node.direction,
            g_cost=node.g_cost + path_cost,
            h_cost=0.0,
            parent=node
        )   
        return rs_path,goal_node
        
    def reconstruct_path(self,rs_path,goal_node:Node):
        path_xs = []
        path_ys = []
        path_yaws = []
        #deltas = []
        dirs = []
        costs = []
        curvs = []
        

        curr = goal_node.parent
        while curr is not None:
            path_xs.append(curr.state[0])
            path_ys.append(curr.state[1])
            path_yaws.append(curr.state[2])
            dirs.append(curr.direction)
            costs.append(curr.g_cost)
            #deltas.append(curr.delta)
            curvs.append(np.tan(curr.delta)/C.WHEELBASE)
            curr = curr.parent

        path_xs = path_xs[::-1]
        path_ys = path_ys[::-1]
        path_yaws = path_yaws[::-1]
        dirs = dirs[::-1]
        costs = costs[::-1]
        #deltas = deltas[::-1]

        last = np.array([path_xs[-1], path_ys[-1], path_yaws[-1]])
        first = np.array([rs_path[0][0], rs_path[0][1], rs_path[0][2]])
        if np.linalg.norm(last[:2] - first[:2]) < 1e-3:
            rs_iter = rs_path[1:]
        else:
            rs_iter = rs_path

        for pt in rs_iter:
            path_xs.append(pt[0])
            path_ys.append(pt[1])
            path_yaws.append(pt[2])
            dirs.append(np.sign(pt[4]))
            curvs.append(pt[3])
            costs.append(costs[-1] if costs else 0.0)
        
        return Path(path_xs, path_ys, path_yaws, dirs, curvs,costs)
    
    def should_try_rs(self, node, goal):
        # d = np.hypot(node.state[0]-goal[0], node.state[1]-goal[1])
        # if d < 5.0:   return (self.expansion_counter % 5) == 0
        # if d < 15.0:  return (self.expansion_counter % 10) == 0
        return (self.expansion_counter % 20) == 0
    def is_goal(self, pose, goal_pose):
        dx = pose[0] - goal_pose[0]
        dy = pose[1] - goal_pose[1]
        d = np.hypot(dx, dy)
        dyaw = abs(mod2pi(pose[2] - goal_pose[2]))
        return (d <= self.goal_tolerance) and (dyaw <= self.yaw_resolution)
    def hybrid_a_star_planning(self,start_pose,goal_pose,grid:OccupancyGrid):
        print("[Planner] Zaczęto planowanie")
        open_set = PriorityQueue()
        closed_set = {}

        self.hmap = self.calculate_unconstrained_heuristic(start_pose,goal_pose,grid)
        #self.hmapData.emit(self.hmap)
        init_dir = "forward" 

        start_node = Node(
            cell = self.discretize_state(start_pose,init_dir),
            state=start_pose,
            delta = 0.0,
            direction=init_dir,
            g_cost=0.0,
            h_cost = self.calculate_hybrid_heuristic(start_pose,goal_pose)
        )
        open_set.push(start_node)

        while not open_set.empty():
            if self.controller.state != "planning":
                break
            self.expansion_counter += 1
            current_node = open_set.pop() 
            current_cell = current_node.cell
            
            if current_cell in closed_set:
                if closed_set[current_cell] <= current_node.g_cost:
                    continue # już byliśmy tu
            closed_set[current_cell] = current_node.g_cost

            if self.should_try_rs(current_node,goal_pose):
                rs_path, possible_goal_node = self.try_reeds_shepp(current_node, goal_pose, grid)
                if rs_path is not None and possible_goal_node is not None:
                    print(f"[Planner] Znaleziono optymalną trasę! Koszt: {possible_goal_node.g_cost:.2f}")
                    return self.reconstruct_path(rs_path, possible_goal_node)
         
            neighbours = self.get_neighbours(current_node,grid)
            
            for neighbour in neighbours:
                neighbour_cell = neighbour.cell
                same_cell = (neighbour_cell == current_cell)
                if (neighbour_cell in closed_set) and (closed_set[neighbour_cell] <= neighbour.g_cost) and (not same_cell):
                    continue
                existing = open_set.get_node(neighbour_cell) if open_set.contains(neighbour_cell) else None
                better_path = (existing is None) or (neighbour.g_cost < existing.g_cost - 1e-6) or same_cell
                if not better_path:
                    continue
                neighbour.h_cost = self.calculate_hybrid_heuristic(neighbour.state,goal_pose)
                neighbour.f_cost = neighbour.g_cost + neighbour.h_cost
                if same_cell:
                    if neighbour.f_cost > current_node.f_cost + 1e-2: # tie breaker
                        continue
                    neighbour.parent = current_node.parent
                open_set.push(neighbour)
                
            
            #self.expansionData.emit(current_node.state)
            print(f"[Planner] h_cost:{current_node.h_cost}, g_cost:{current_node.g_cost}, f_cost:{current_node.f_cost}")
            
        return None
               
class PlanningWorker(QtCore.QObject):

    stateData = QtCore.pyqtSignal(str)
    pathData = QtCore.pyqtSignal(object)  
    finished   = QtCore.pyqtSignal(bool)

    def __init__(self,controller,grid:OccupancyGrid):
        super().__init__() 
        self.controller = controller
        self.grid = grid
        self.pathData.connect(self.controller.pathUpdated)
        
    @QtCore.pyqtSlot()
    def run(self):
        if self.controller.planning_active:
            
            print("[PlannerWorker] Zaczynam planowanie ścieżki. Proszę czekać...")
            grid = self.grid
            start = self.controller.start_pose
            goal = self.controller.goal_pose
            print("planner obstacles:", len(grid.obstacles))
            planner = NewPlanner(self.controller)
            path = None
            try:
                path = planner.hybrid_a_star_planning(start,goal,grid)
            except Exception: 
                print(f"[PlanningWorker] Błąd w planowaniu!")
                traceback.print_exc()
            if path is not None:
                path = resample_and_smooth_path(path)
                print(f"[PlannerWorker] Segmenty ścieżki: {path.segments}")
                self.controller.path = path
                self.pathData.emit(path)
                print("[PlannerWorker] Znaleziono ścieżkę.")
                self.finished.emit(True)
            else:
                print("[PlannerWorker] Nie znaleziono ścieżki.")
                self.finished.emit(False)
        else:
            self.finished.emit(True)

class Kalman():
    def __init__(self,wheelbase):
        self.states = 5

        self.Q = np.diag([1e-5,1e-5,1e-5,1e-12,1e-12]) # macierz kowariancji szumu procesowego
        self.R = np.diag([(1e-5)**2]) # kowariancji szumu pomiarowego
        self.H = np.array([[0,0,1,0,0]]) # macierz obserwacji
        self.E = np.eye(self.states) # macierz kowariancji błędu
        self.I = np.eye(self.states)
        self.wheelbase = wheelbase
        
        # szumy procesowe i pomiarowe
    
    def compute_F(self,x, dt):
        F = np.eye(self.states)
        v = x[3]
        psi = x[2]
        delta = x[4]
        # rząd 1 x
        # F[0,2] = -dt*np.sin(psi)*np.cos(delta)*v
        # F[0,3] = dt*np.cos(psi)*np.cos(delta)
        # F[0,4] = -dt*np.cos(psi)*np.sin(delta)*v
        F[0,2] = -dt*np.sin(psi)*v
        F[0,3] = dt*np.cos(psi)
        # rząd 2 y
        # F[1,2] = dt*np.cos(psi)*np.cos(delta)*v
        # F[1,3] = dt*np.sin(psi)*np.cos(delta)
        # F[1,4] = -dt*np.sin(psi)*np.sin(delta)*v
        F[1,2] = dt*np.cos(psi)*v
        F[1,3] = dt*np.sin(psi)
        # rząd 3 psi
        # F[2,3] = 1/self.wheelbase*np.sin(delta)*dt
        # F[2,4] = v/self.wheelbase*np.cos(delta)*dt
        F[2,3] = 1/self.wheelbase*np.tan(delta)*dt
        F[2,4] = v/self.wheelbase*(1/(np.cos(delta))**2)*dt
        # rząd 4 i 5 (v i delta) nie tykamy
        return F
    def f(self, x, dt):
        # z przednich kół
        x_model = x[0]
        y_model = x[1]
        psi_model = x[2]
        v_model = x[3]
        delta_model = x[4]
        #x_model = x_model + dt * v_model * np.cos(psi_model) * np.cos(delta_model)
        x_model = x_model + dt * v_model * np.cos(psi_model)
        #y_model = y_model + dt * v_model * np.sin(psi_model) * np.cos(delta_model)
        y_model = y_model + dt * v_model * np.sin(psi_model)
        #psi_model = wrap_angle(psi_model + dt * (v_model/self.wheelbase) * np.sin(delta_model))
        psi_model = mod2pi(psi_model + dt * (v_model/self.wheelbase) * np.tan(delta_model))
        return np.array([x_model, y_model, psi_model, v_model, delta_model])
    def predict(self,x,dt):
        x_pred = self.f(x,dt)
        F = self.compute_F(x,dt)
        self.E = F @ self.E @ F.T + self.Q
        return x_pred
    def update(self,x_hat,z):
        # z = [psi_meas, vF_meas, delta_meas]
        zhat = np.array([x_hat[2]])
        innov = z - zhat
        innov[0] = mod2pi(innov[0]) # obetnij yaw, żeby był maks. 2pi
        S = self.H @ self.E @ self.H.T + self.R     # 3x3
        K = self.E @ self.H.T @ np.linalg.inv(S)    # 5x3
        x_upd = x_hat + K @ innov
        x_upd[2] = mod2pi(x_upd[2])
        I_KH = self.I - K @ self.H
        self.E = I_KH @ self.E @ I_KH.T + K @ self.R @ K.T
        self.x = x_upd
        return x_upd

        
class TrajStateMachine():
    def __init__(self,driver,worker):
        # do symulacji
        self.worker = worker
        self.driver = driver
        # do pomiarów
        self.straight_phase_time = 5.0
        self.curve_phase_time = 5.0
        self.prev_time = 0.0
        self.target_steer = 0.0
        self.target_speed = 0.0
        self.max_angle = 0.5
        self.deriv = 0.0
        self.prev_heading = 0.0
        self.delta_rate = 0.6
        self.speed_rate = 7.0
        self.state_index = 0
        # do symulacji kąta
        self.input = 0.0
        self.u_old = 0.0
        self.u_old_old = 0.0
        self.y = 0.0
        self.y_old = 0.0
        self.y_old_old = 0.0
        self.steer_cmd = 0.0
        # do symulacji prędkości
        self.input_s = 0.0
        self.u_old_s = 0.0
        self.u_old_old_s = 0.0
        self.y_s = 0.0
        self.y_old_s = 0.0
        self.y_old_old_s = 0.0
        self.speed_cmd = 0.0
        self.sequence = [
            ("prosto",        6.0,  8.0,  0.0),
            ("hamowanie",      5.0,  0.0,  0.0),
            ("wstecz_rozp",    3.0, -4.0,  0.0),
            ("wstecz_lewo",    5.0, -4.0, -0.5),
            ("wstecz_prawo",   7.5, -4.0,  0.5),
            ("wstecz_prosto",  4.0, -4.0,  0.0),
            ("hamowanie_stop", 3.0,  0.0,  0.0),
        ]

    def steer(self,dt):
        
        self.input = self.target_steer
        
        ts = dt
        tc = 1.0
        e = 0.4 
        k = 1.0
        w = 2 * np.pi / tc 
        a=k*w*ts*ts/(4+4*e*w*ts+ts*ts*w*w)
        b=(ts*ts*w*w/2-2)/(1+e*w*ts+ts*ts/4*w*w)
        c=(1-e*w*ts+ts*ts/4*w*w)/(1+e*w*ts+ts*ts/4*w*w)
        g0 = 4 * a / (1 + b + c)
        if g0 != 0.0:
            a = a / g0
        self.y = a*self.input+ 2*a*self.u_old + a*self.u_old_old - b*self.y_old - c*self.y_old_old
        max_rate = self.delta_rate  # [rad/s] – dostosuj do swojego modelu
        dy = np.clip(self.y - self.steer_cmd, -max_rate * dt, max_rate * dt)
        self.steer_cmd += dy

        # fizyczny limit kąta
        self.steer_cmd = np.clip(self.steer_cmd, -self.max_angle, self.max_angle)

        # przesunięcie stanów
        self.u_old_old = self.u_old
        self.u_old = self.input
        self.y_old_old = self.y_old
        self.y_old = self.y
        
        self.driver.setSteeringAngle(self.steer_cmd)
    def cont_speed(self,dt):
        self.input_s = self.target_speed
        
        ts = dt
        tc = 0.1
        e = 0.9
        k = 1.0
        w = 2 * np.pi / tc 
        a=k*w*ts*ts/(4+4*e*w*ts+ts*ts*w*w)
        b=(ts*ts*w*w/2-2)/(1+e*w*ts+ts*ts/4*w*w)
        c=(1-e*w*ts+ts*ts/4*w*w)/(1+e*w*ts+ts*ts/4*w*w)
        g0 = 4 * a / (1 + b + c)
        if g0 != 0.0:
            a = a / g0
        self.y_s = a*self.input_s + 2*a*self.u_old_s + a*self.u_old_old_s - b*self.y_old_s - c*self.y_old_old_s
        max_rate = self.speed_rate 
        dy = np.clip(self.y_s - self.speed_cmd, -max_rate * dt, max_rate * dt)
        self.speed_cmd += dy

        # przesunięcie stanów
        self.u_old_old_s = self.u_old_s
        self.u_old_s = self.input_s
        self.y_old_old_s = self.y_old_s
        self.y_old_s = self.y_s
        if self.target_speed == 0.0 and abs(self.speed_cmd - self.target_speed) < 1e-3:
            self.driver.setCruisingSpeed(self.target_speed)
        else:
            self.driver.setCruisingSpeed(self.speed_cmd)

    def update(self,now,dt):
        """
        if self.state_index >= len(self.sequence):
            
            self.target_speed = 0.0
            self.target_steer = 0.0
            self.steer(dt)
            self.cont_speed(dt)
        
        """
        if self.worker.first_call_traj:
            self.prev_time = time.time()
            self.worker.first_call_traj = False
            
        state_name, duration, self.target_speed, self.target_steer = self.sequence[self.state_index]
            

        # przejście do następnego stanu po upływie czasu
        if now - self.prev_time >= duration:
            self.state_index += 1
            self.prev_time = now
            if self.state_index < len(self.sequence):
                next_state = self.sequence[self.state_index][0]
                print(f"{now:6.2f}s -> zmiana: {state_name} - {next_state}")
            else:
                self.state_index = 0

        # wykonanie dynamiki
        self.steer(dt)
        self.cont_speed(dt)



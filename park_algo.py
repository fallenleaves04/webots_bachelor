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
from typing import List
"""
Parametry samochodu
"""
# Vehicle parameters
class C:
    # parametry samochodu 
    TRACK_FRONT = 1.628
    TRACK_REAR = 1.628
    WHEELBASE = 2.995
    MAX_WHEEL_ANGLE = 0.5  # rad
    CAR_WIDTH = 1.95
    CAR_LENGTH = 4.85
    MAX_SPEED = -3.0
    MAX_RADIUS = WHEELBASE/np.tan(MAX_WHEEL_ANGLE)
    MAX_CURVATURE = 1/MAX_RADIUS
    # parametry dla A*
    c_val = 1
    FORWARD_COST = c_val*2.0
    GEAR_CHANGE_COST = c_val*20.0
    STEER_CHANGE_COST = c_val*4.0
    STEER_ANGLE_COST = c_val*2.0
    OBS_COST = c_val*6.0
    H_COST = 5.0
    # parametry dla próbkowania
    XY_RESOLUTION = 0.1 # m
    YAW_RESOLUTION = np.deg2rad(5)
    # parametry dla Stanley
    K_STANLEY = 0.5

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

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

"""
DLA STANLEY, SKOPIOWANE Z https://github.com/zhm-real/MotionPlanning I DOPASOWANE
"""
    
class Trajectory:
    def __init__(self, cx, cy, cyaw,ccurv):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ccurv = ccurv
        self.ind_old = 0

        self.int_v = 0.0
        self.kp = 0.1
        self.ki = 0.6
        self.kd = 0.001

        self.len = len(self.cx)
        self.s0 = 1

    def calc_theta_e_and_er(self, x,y,yaw):
        """
        calc theta_e and er.
        theta_e = theta_car - theta_path
        er = lateral distance in frenet frame

        :param node: current information of vehicle
        :return: theta_e and er
        """

        ind = self.nearest_index(x,y)

        k = self.ccurv[ind]
        yaw = self.cyaw[ind]

        rear_axle_vec_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                         [math.sin(yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[x - self.cx[ind]],
                                      [y - self.cy[ind]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        theta_e = wrap_angle(yaw- self.cyaw[ind])

        return theta_e, er, k, yaw, ind

    def nearest_index(self, xx,yy):
        """
        find the index of the nearest point to current position.
        :param node: current information
        :return: nearest index
        """

        dx = [xx - x for x in self.cx]
        dy = [yy - y for y in self.cy]
        dist = np.hypot(dx, dy)
        self.s0 += np.argmin(dist[self.s0:self.len])

        return self.s0
    
    def pid_control(self,target_v, v, dist,dt):
        """
        PID controller and design speed profile.
        :param target_v: target speed
        :param v: current speed
        :param dist: distance to end point
        :return: acceleration
        """
        
        
        if dist < 10.0:
            if v > 3.0:
                a = -2.5
            elif v < -2.0:
                a = -1.0
        else:
            err = target_v - v
            diff_v = err/dt
            self.int_v += err*dt 
            a = self.kp*err + self.ki*self.int_v + self.kd*diff_v
            
        return a

def rear_wheel_feedback_control(x,y,v,yaw, ref_path):
    """
    rear wheel feedback controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :return: optimal steering angle
    """

    theta_e, er, k, yaw, ind = ref_path.calc_theta_e_and_er(x,y,yaw)
    

    omega = v * k * math.cos(theta_e) / (1.0 - k * er) - \
            C.K_theta * abs(v) * theta_e - C.K_e * v * math.sin(theta_e) * er / theta_e

    delta = math.atan2(C.WB * omega, v)

    return delta, ind




from sklearn.cluster import DBSCAN

class OccupancyGrid():
    """
    Siatka zajętości z KD-tree dla detekcji kolizji; aktualizowana z czujników ultradźwiękowych 
    """
    def __init__(self):
        
        self.car_width = C.CAR_WIDTH
        self.car_length = C.CAR_LENGTH
        self.wheel_base = C.WHEELBASE
        self.cg_to_rear_axle = np.array([1.2975, 0, 0.1]) # dane z Webots
        self.cg_x = self.cg_to_rear_axle[0]
        self.rear_bumper_to_axle = 1.0
        self.front_bumper_to_axle = C.CAR_LENGTH - self.rear_bumper_to_axle
        self.collision_radius = np.hypot(max(self.front_bumper_to_axle - self.cg_x,
                                             self.rear_bumper_to_axle - self.cg_x),
                                             0.5*self.car_width)

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
        self.l_free = -2.0
        
        self.obstacles = []
        self.spots = []
        self.detected_spots = []
        self.yolo_points_buffer = []
        self.yolo_x_pts = []
        self.yolo_y_pts = []
        self.ox = None
        self.oy = None

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
            print(f"[OccupancyGrid] Ustawiono z {len(obstacles)} przeszkodami (obiektami).")
        else:
            self.kd_tree = None
            self.max_obs_radius = 0
            print("[OccupancyGrid] Brak przeszkód.")
        
    
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
    
    def convert_to_polar(self,x_t,y_t,sensor_pose):
        x_s,y_s,theta_s = sensor_pose
        r_t = np.hypot(x_t-x_s, y_t-y_s)
        phi_t = np.arctan2(y_t-y_s, x_t-x_s)
        phi_t = wrap_angle(phi_t-theta_s)
        return r_t,phi_t

    def inverse_sensor_model(self,cell,dist,sensor_pose,params):
        # zwróć evidence dla danej klatki 
        print(sensor_pose)
        beta = params["aperture"] # kąt rozwarcia
        max_range = params["max_range"] 
        
        x_t,y_t = cell
        r_t,phi_t = self.convert_to_polar(x_t,y_t,sensor_pose) # dla łatwości przekształcenie do biegunowego układu
        alpha = 0.1
        
        if dist >= max_range or r_t > min(max_range,dist + alpha/2) or abs(phi_t) > beta/2:
            return self.l_0
        if dist < max_range and abs(r_t-dist) < alpha/2:
            return self.l_occ
        if r_t <= (dist - alpha/2):
            return self.l_free
        
        return self.l_0
    
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
        # 1. Pobierz parametry
        sensor_pose_rel = self.sensor_poses[sensor_name]
        
        sensor_global_x, sensor_global_y, sensor_global_theta = self.sensor_in_wrld(car_pose, sensor_pose_rel)
        
        params = self.params[sensor_name]
        max_range = params["max_range"]
        beta = params["aperture"]
        min_range = params["min_range"]
        if dist >= max_range - 0.15:
            effective_dist = max_range - 0.15
            is_hit = False # nie ma wskazania, bo czujnik zwraca domyślnie tą wartość
        else:
            effective_dist = dist
            is_hit = True
        
        x_min = sensor_global_x - max_range
        x_max = sensor_global_x + max_range
        y_min = sensor_global_y - max_range
        y_max = sensor_global_y + max_range
        i_min, j_min = self.make_cell((x_min, y_min)) or (0, 0) # Fallback na 0 jeśli None
        i_max, j_max = self.make_cell((x_max, y_max)) or (self.grid_width-1, self.grid_height-1)

        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.grid_width - 1, i_max)
        j_max = min(self.grid_height - 1, j_max)

        if i_min > i_max or j_min > j_max:
            return
        
        range_i = np.arange(i_min, i_max + 1)
        range_j = np.arange(j_min, j_max + 1)

        world_x = self.x_min + range_i * self.xy_resolution
        world_y = self.y_min + range_j * self.xy_resolution

        X, Y = np.meshgrid(world_x, world_y, indexing='ij')
        DX = X - sensor_global_x
        DY = Y - sensor_global_y
        
        R_grid = np.hypot(DX, DY)
        PHI_grid = wrap_angle(np.arctan2(DY, DX) - sensor_global_theta)
        #PHI_grid = (PHI_grid + np.pi) % (2 * np.pi) - np.pi

        mask_angle = np.abs(PHI_grid) <= (beta / 2.0)
        mask_greater_than_min = R_grid >= min_range
        thickness = 0.1
        buffer = 0.15
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

    # jaka jest najlepsza reprezentacja tej siatki? aby samochód zaczynał po środku mapy, to trzeba uwzględnić ujemne indeksy
    # ale jak liczyć indeksy, jeżeli poza pojazdu jest ciągła? 
    def interpret_readings(self,names_dists:dict,car_pose:tuple):
        # zrobić siatkę wokół samochodu na 
        #name = "distance sensor right side"
        #dist = names_dists[name]
        
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
            
            add_par = 1.5
            add_perp = 4.5
            #type = 'parallel' if obs_len > 2.5 else 'perpendicular'### if find_type == 'parallel' else add_perp
            add = add_par 
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

    def find_spots(self,car_pose,spot_type='parallel', side='right'):
        car_x, car_y, car_yaw = car_pose
            
        # Macierz rotacji do układu lokalnego (X-przód, Y-prawo dla ujemnych Y)
        c, s = np.cos(-car_yaw), np.sin(-car_yaw)
        R = np.array([[c, -s], [s, c]])
        filtered_obs = []

        for obs in self.obstacles:
            # Sprawdź typ przeszkody
            
            # Oblicz zakres X i Y przeszkody w układzie lokalnym
            cx, cy = obs['center']
            l, w = obs['length'], obs['width']
            ang = obs['angle']
            
            # Rogi przeszkody w układzie globalnym
            co, so = np.cos(ang), np.sin(ang)
            v_l = np.array([co * l/2, so * l/2])
            v_w = np.array([-so * w/2, co * w/2])
            
            corners_global = [
                np.array([cx, cy]) + v_l + v_w,
                np.array([cx, cy]) + v_l - v_w,
                np.array([cx, cy]) - v_l - v_w,
                np.array([cx, cy]) - v_l + v_w
            ]
            
            # Transformuj rogi do układu lokalnego
            local_corners = []
            for corner in corners_global:
                vec_corner = corner - np.array([car_x, car_y])
                local_corner = R @ vec_corner
                local_corners.append(local_corner)
            
            # Znajdź zakresy
            local_xs = [c[0] for c in local_corners]
            local_ys = [c[1] for c in local_corners]
            
            min_x = min(local_xs)
            max_x = max(local_xs)
            min_y = min(local_ys)
            max_y = max(local_ys)
            
            if side == 'right':
                if max_y > -0.5:
                    continue
            else:
                if max_y < 0.5:
                    continue

            # Zapisz dane
            obs['min_x'] = min_x
            obs['max_x'] = max_x
            obs['max_y'] = max_y
            obs['min_y'] = min_y
            obs['avg_y'] = (min_y + max_y) / 2.0
            
            filtered_obs.append(obs)

        # Sortuj po min_x (od tyłu do przodu)
        filtered_obs.sort(key=lambda o: o['min_x'])
        spots = []
        spacing = 0.4 if spot_type == 'parallel' else 0.3

        for i in range(len(filtered_obs) - 1):
            obs1 = filtered_obs[i]    
            obs2 = filtered_obs[i+1] 
            
            gap_start = obs1['max_x']
            gap_end = obs2['min_x']
            gap_size = gap_end - gap_start
            
            space_width = 5.0 if spot_type == 'parallel' else 2.2
            spot_positions_x = []    
            if gap_size < space_width:
                continue
            num_spots = int(gap_size/(space_width + spacing))
            if num_spots == 1:
                spot_positions_x.append((gap_start + gap_end) / 2.0)
            elif num_spots > 1:
                usable_gap = gap_size - 2 * spacing
                step = usable_gap / num_spots
                
                for j in range(num_spots):
                    spot_x = gap_start + spacing + step * (j + 0.5)
                    spot_positions_x.append(spot_x)
            shift = C.CAR_WIDTH/2 if spot_type == 'parallel' else C.CAR_LENGTH/2
            
            for spot_x in spot_positions_x:
                if side == 'right':
                    spot_local_y = obs1['max_y'] - shift if obs1['min_y'] > obs2['min_y'] else obs2['max_y'] - shift
                else:
                    spot_local_y = obs1['min_y'] + shift if obs1['max_y'] < obs2['max_y'] else obs2['min_y'] + shift
                vec_local = np.array([spot_x, spot_local_y])
                center_global = np.array([car_x, car_y]) + R.T @ vec_local
                
                angle_types = {('parallel','right'):0.0,('parallel','left'):0.0,('perpendicular','right'):np.pi/2,('perpendicular','left'):-np.pi/2,}
                angle = angle_types[(spot_type,side)] 
                offset = C.CAR_LENGTH/2 - 1
                target_x = center_global[0] - offset * np.cos(angle)
                target_y = center_global[1] - offset * np.sin(angle)
                
                spots.append({
                    'type': spot_type,
                    'side': side,
                    'center': center_global,
                    'target_rear_axle': np.array([target_x, target_y]),
                    'orientation': angle,
                    'length': gap_size,
                    'gap_start_x': gap_start,
                    'gap_end_x': gap_end,
                    'obs1': obs1,
                    'obs2': obs2
                })
    
        return spots
    
    def find_spots_scanning(self, car_pose, spot_type='parallel', side='right'):
        car_x, car_y, car_yaw = car_pose
        
        if spot_type == 'parallel':
            scan_depth_limit = 6.0
            required_len = 5.5
            min_spot_width = 2.5
        else: # perpendicular
            scan_depth_limit = 8.0
            required_len = 2.4
            min_spot_width = 5.3

        spacing = 0.3 

        c, s = np.cos(car_yaw), np.sin(car_yaw)
        R = np.array([[c, -s], [s, c]])
        
        all_obs = []
        cars = []   

        for obs in self.obstacles:
            l, w, ang = obs['length'], obs['width'], obs['angle']
            corners_glob = self.get_rect_corners(obs['center'], l, w, ang)
            corners_loc = (corners_glob - np.array([car_x, car_y])) @ R.T
            min_x, max_x = np.min(corners_loc[:,0]), np.max(corners_loc[:,0])
            min_y, max_y = np.min(corners_loc[:,1]), np.max(corners_loc[:,1])
            obs_data = {
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'avg_y': (min_y + max_y) / 2.0,
                'original': obs
            }
            all_obs.append(obs_data)
        
        cars = [o for o in all_obs if o['original'].get('is_car')]
        cars.sort(key=lambda c: c['min_x'])

        spots = []
        
        if len(cars) < 2:
            return []
        
        #print(f"ilosc samochodow: {len(cars)}, {[c['original']['center'] for c in cars]}")
        for i in range(len(cars) - 1):
            obs1 = cars[i]
            obs2 = cars[i+1]
            
            gap_start = obs1['max_x']
            gap_end = obs2['min_x']
            gap_size = gap_end - gap_start

            if gap_size < required_len:
                continue

            current_gap_depth = 20.0
            
            for obs in all_obs:
                if obs['max_x'] > gap_start + 0.2 and obs['min_x'] < gap_end - 0.2:
                    
                    dist_to_obs = abs(obs['max_y']) if side == 'right' else abs(obs['min_y'])
                    
                    if dist_to_obs < current_gap_depth:
                        current_gap_depth = dist_to_obs

            max_usable_depth = 3.0 if spot_type == "parallel" else 5.5
            available_depth = min(current_gap_depth, max_usable_depth)

            if available_depth < min_spot_width:
                continue 
            curb_detected = (current_gap_depth < max_usable_depth)
    
            curb_margin = 0.4
            target_spot_width = min(available_depth, 2.6 if spot_type=='parallel' else 5.0)
            if side == 'right':
                ref_line_y = (obs1['max_y'] + obs2['max_y']) / 2.0
                probable_center_y = ref_line_y - (target_spot_width / 2.0)
                spot_center_y = probable_center_y + curb_margin if curb_detected else probable_center_y
            else: # left
                ref_line_y = (obs1['min_y'] + obs2['min_y']) / 2.0
                if curb_detected:
                    spot_center_y = (current_gap_depth - curb_margin)
                else:
                    spot_center_y = ref_line_y + (target_spot_width / 2.0)
                    
            # Obliczenia liczby miejsc
            num_spots = int(gap_size / (required_len + spacing))
            
            spot_positions_x = []
            if num_spots == 1:
                spot_positions_x.append((gap_start + gap_end) / 2.0)
            elif num_spots > 1:
                step = gap_size / num_spots
                for j in range(num_spots):
                    spot_x = gap_start + step * (j + 0.5)
                    spot_positions_x.append(spot_x)
            
            for spot_x in spot_positions_x:
                vec_local = np.array([spot_x, spot_center_y])
                center_global = np.array([car_x, car_y]) + R.T @ vec_local
                
                angle_types = {
                    ('parallel','right'): 0.0,
                    ('parallel','left'): 0.0,
                    ('perpendicular','right'): np.pi/2, 
                    ('perpendicular','left'): -np.pi/2, 
                }
                final_angle = angle_types.get((spot_type, side), 0.0)
                
                offset = C.CAR_LENGTH/2 - 1.0 
                target_x = center_global[0] - offset * np.cos(final_angle)
                target_y = center_global[1] - offset * np.sin(final_angle)
                
                spots.append({
                    'type': spot_type,
                    'side': side,
                    'center': center_global,
                    'target_rear_axle': np.array([target_x, target_y]),
                    'orientation': final_angle,
                    'length': gap_size / num_spots,
                    'width': available_depth,      
                    'obs1': obs1['original'],
                    'obs2': obs2['original']
                })

        return spots
    
    def match_semantics_with_sat(self):
        if len(self.yolo_points_buffer) < 5 or not self.obstacles:
            return

        yolo_pts = np.array(self.yolo_points_buffer)
        sem_clustering = DBSCAN(eps=1.0, min_samples=3).fit(yolo_pts)
        sem_labels = sem_clustering.labels_
        yolo_clusters_rects = []
        for label in set(sem_labels):
            if label == -1: continue
            pts = yolo_pts[sem_labels == label]
            if pts is None or len(pts) < 4:
                continue
            pts = pts.astype(np.float32)
            try:
                rect = cv.minAreaRect(pts)
                corners = self.get_rect_corners(rect[0], rect[1][0], rect[1][1], np.radians(rect[2]))
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
class Path:
    """Reprezentuje ścieżkę (wynik planowania)"""
    def __init__(self, xs, ys, yaws, directions, costs=0.0):
        self.xs = xs       # Lista punktów
        self.ys = ys
        self.yaws = yaws
        self.costs = costs
        self.directions = directions
    
    # Metody pomocnicze:
    def __len__(self):          # len(path)
        return len(self.xs)
    def get_point(self, idx):
        return (self.xs[idx], self.ys[idx], self.yaws[idx])

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
        if existing is None or node.f_cost < existing.f_cost:
            self.heap[cell] = node.g_cost
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
        self.controller = controller
        self.expansionData.connect(self.controller.expansionUpdated)
        self.hmapData.connect(self.controller.hmapUpdated)

    def calc_actions(self):
        motion_actions = []
        steers = np.linspace(-C.MAX_WHEEL_ANGLE, C.MAX_WHEEL_ANGLE, self.n_steers)
        steers = np.unique(np.concatenate([steers, np.array([0.0])]))  
        for delta in steers:
            for direction in ["forward", "reverse"]:
                motion_actions.append((delta, direction))
        return motion_actions
    
    def discretize_state(self,cur_node_state) -> tuple:
        x, y, theta = cur_node_state
        return (
            int(round(x / self.xy_resolution)),
            int(round(y / self.xy_resolution)),
            int(round(wrap_angle(theta) / self.yaw_resolution))
        )
    
    def calculate_unconstrained_heuristic(self, start_pose, goal_pose, grid: OccupancyGrid):
        # 1. Pobierz listę przeszkód-prostokątów
        # Zakładam, że grid.obstacles to lista słowników z 'center', 'length', 'width', 'angle'
        obstacles = grid.obstacles 
        
        gx, gy, _ = goal_pose
        sx, sy, _ = start_pose

        all_x = [sx, gx]
        all_y = [sy, gy]

        collision_radius = 0.75
        obs_polygons = []
        for obs in obstacles:
            corners = grid.get_rect_corners(obs['center'],obs['length']+ 2 * collision_radius,obs['width']+ 2 * collision_radius,obs['angle'])
            obs_polygons.append(corners)
            for p in corners:
                all_x.append(p[0])
                all_y.append(p[1])

        
        pad = 2
        minx = int(np.floor(min(all_x) / self.xy_resolution)) - pad
        maxx = int(np.ceil(max(all_x) / self.xy_resolution)) + pad
        miny = int(np.floor(min(all_y) / self.xy_resolution)) - pad
        maxy = int(np.ceil(max(all_y) / self.xy_resolution)) + pad

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

        dist_map = distance_transform_edt(~obs_map) * self.xy_resolution
        
        potential_map = np.zeros_like(dist_map)
        
        safe_distance = 1.0 
        
        coll_zone = dist_map < safe_distance
        d_vals = np.maximum(dist_map[coll_zone], 0.05) 
        potential_map[coll_zone] = 1.0 / (2.0 * (d_vals ** 2))
        cost_map = 1.0 + potential_map * 2.0

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
                        cell_penalty = cost_map[nx,ny]
                        new_cost = cost + move_cost * cell_penalty
                        if new_cost < hmap[nx, ny]:
                            hmap[nx, ny] = new_cost
                            heapq.heappush(open_set, (new_cost, nx, ny))
        
        self.current_potential_map = potential_map
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
                h_a_star = h_map[pose_cell_x, pose_cell_y] * self.xy_resolution
                if np.isinf(h_a_star):
                    h_a_star = dist
        
        return max(h_a_star, h_rs) * C.H_COST
        
    def simulate_motion(self, state, delta, direction):
        x, y, theta = state
        d = 1.0 if direction == "forward" else -1.0
        theta_new = wrap_angle(theta + np.tan(delta)/C.WHEELBASE * self.step_size)
        theta = (theta + theta_new) / 2
        x_new = x + d * self.step_size * np.cos(theta)
        y_new = y + d * self.step_size * np.sin(theta)
        return (x_new, y_new, theta_new)
    
    def get_neighbours(self, node:Node, grid:OccupancyGrid) -> List[Node]:
        
        neighbours = []
        
        # 6 akcji sterujących
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
                cell=self.discretize_state(next_state),
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
        dx = to_state[0] - node.state[0]
        dy = to_state[1] - node.state[1]
        dist = np.hypot(dx, dy)
        
        cost = dist
        if direction == "forward":
            cost += C.FORWARD_COST
        if node.direction != direction:
            cost += C.GEAR_CHANGE_COST
        
        cost += C.STEER_ANGLE_COST * abs(delta)

        if abs(node.delta - delta) > 0.01:
            cost += C.STEER_CHANGE_COST * abs(node.delta - delta)

        if hasattr(self, 'current_potential_map'):
            mx = int(round(to_state[0] / self.xy_resolution)) - self.map_offset_x
            my = int(round(to_state[1] / self.xy_resolution)) - self.map_offset_y
            pmap = self.current_potential_map
            
            # Sprawdzenie zakresu mapy
            if 0 <= mx < pmap.shape[0] and 0 <= my < pmap.shape[1]:
                
                cost += pmap[mx, my] * C.OBS_COST * dist
        return cost

    def try_reeds_shepp(self,node:Node,goal_pose,grid:OccupancyGrid):
        rs_path = reeds_shepp.path_sample(node.state,goal_pose,C.MAX_RADIUS,self.xy_resolution)

        trajectory_risk_cost = 0.0
        # skręt w prawo - 1, linia prosta - 2, skręt w lewo - 3
        curve_dict = {1:-C.MAX_RADIUS,3:C.MAX_RADIUS}

        if not rs_path:
            return None,None
        for i in range(0,len(rs_path),1): 
            if grid.is_collision(*rs_path[i]):
                return None,None
            if hasattr(self, 'current_potential_map'):
                mx = int(round(rs_path[i][0] / self.xy_resolution)) - self.map_offset_x
                my = int(round(rs_path[i][1] / self.xy_resolution)) - self.map_offset_y
                
                # Pobieramy mapę potencjału
                pmap = self.current_potential_map
                
                # Jeśli punkt jest w granicach mapy, dodajemy jego "niebezpieczeństwo" do kosztu
                if 0 <= mx < pmap.shape[0] and 0 <= my < pmap.shape[1]:
                    trajectory_risk_cost += pmap[mx, my]

        
        path_cost = 0.0
        path_segments = reeds_shepp.path_type(
            node.state,
            goal_pose,
            C.MAX_RADIUS
        )
        for i in range(len(path_segments)-1):
            (ctype,length) = path_segments[i]
            path_cost += abs(length) * C.FORWARD_COST  if length >= 0.0 else length  
            path_cost += C.GEAR_CHANGE_COST if path_segments[i] * path_segments[i+1] < 0.0 else 0.0
            path_cost += C.STEER_ANGLE_COST * C.MAX_RADIUS if ctype != 2 else 0.0
            curr_steer = curve_dict[path_segments[i][0]]
            next_steer = curve_dict[path_segments[i+1][0]]
            path_cost += C.STEER_CHANGE_COST * abs(next_steer - curr_steer)

        total_risk_penalty = trajectory_risk_cost * C.OBS_COST * self.step_size

        goal_node = Node(
            cell=self.discretize_state(goal_pose),
            state=goal_pose,
            delta=0.0,
            direction=node.direction,
            g_cost=node.g_cost + path_cost + total_risk_penalty,  
            h_cost=self.calculate_hybrid_heuristic(goal_pose,goal_pose),
            parent=node
        )   
        return rs_path,goal_node
        
    def reconstruct_path(self,rs_path,start_node:Node,goal_node:Node):
        path_xs = []
        path_ys = []
        path_yaws = []
        dirs = []
        costs = []

        curr = goal_node.parent
        while curr is not None:
            path_xs.append(curr.state[0])
            path_ys.append(curr.state[1])
            path_yaws.append(curr.state[2])
            dirs.append(curr.direction)
            costs.append(curr.g_cost)
            curr = curr.parent

        path_xs = path_xs[::-1]
        path_ys = path_ys[::-1]
        path_yaws = path_yaws[::-1]
        
        dirs = dirs[::-1]
        costs = costs[::-1]

        for pt in rs_path:
            path_xs.append(pt[0])
            path_ys.append(pt[1])
            path_yaws.append(pt[2])
            dirs.append(goal_node.direction) 
            costs.append(goal_node.g_cost)
        
        return Path(path_xs, path_ys, path_yaws, dirs, costs)
    
    # TODO: powinien planować tak, że lepiej niech znajdzie reeds_shepp, a później rozszerza, jeżeli jest kolizja
    def hybrid_a_star_planning(self,start_pose,goal_pose,grid:OccupancyGrid):
        print("[Planner] Zaczęto planowanie")
        open_set = PriorityQueue()
        closed_set = {}

        self.hmap = self.calculate_unconstrained_heuristic(start_pose,goal_pose,grid)
        self.hmapData.emit(self.hmap)

        dx = goal_pose[0] - start_pose[0]
        dy = goal_pose[1] - start_pose[1]
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = abs(wrap_angle(angle_to_goal - start_pose[2]))
        init_dir = "forward" if angle_diff < np.pi/2 else "reverse"

        start_node = Node(
            cell = self.discretize_state(start_pose),
            state=start_pose,
            delta = 0.0,
            direction=init_dir,
            g_cost=0.0,
            h_cost = self.calculate_hybrid_heuristic(start_pose,goal_pose)
        )
        open_set.push(start_node)
        
        best_goal_node = None

        while not open_set.empty():
            
            self.expansion_counter += 1
            
            current_node = open_set.pop() 
            if best_goal_node is not None and current_node.f_cost >= best_goal_node.g_cost:
                print(f"[Planner] Znaleziono optymalną trasę! Koszt: {best_goal_node.g_cost:.2f}")
                
                return self.reconstruct_path(best_rs_path, start_node, best_goal_node)
            current_cell = current_node.cell
            if current_cell in closed_set:
                if closed_set[current_cell] <= current_node.g_cost:
                    continue # już byliśmy tu
            closed_set[current_cell] = current_node.g_cost
            
            rs_path, possible_goal_node = self.try_reeds_shepp(current_node, goal_pose, grid)
            if rs_path is not None and possible_goal_node is not None:
                if best_goal_node is None or possible_goal_node.g_cost < best_goal_node.g_cost:
                    best_goal_node = possible_goal_node
                    best_rs_path = rs_path

            neighbours = self.get_neighbours(current_node,grid)
            
            for neighbour in neighbours:
                neighbour_cell = neighbour.cell

                if neighbour_cell in closed_set and closed_set[neighbour_cell] <= neighbour.g_cost:
                    continue
                neighbour.h_cost = self.calculate_hybrid_heuristic(neighbour.state,goal_pose)
                neighbour.f_cost = neighbour.g_cost + neighbour.h_cost

                if best_goal_node is not None and neighbour.f_cost >= best_goal_node.g_cost:
                    continue

                if not open_set.contains(neighbour_cell):
                    open_set.push(neighbour)
                else:
                    existing = open_set.get_node(neighbour_cell)
                    if neighbour.f_cost < existing.f_cost:
                        open_set.push(neighbour)
            
            #self.expansionData.emit(current_node.state)
            print(f"[Planner] h_cost:{current_node.h_cost}, g_cost:{current_node.g_cost}, f_cost:{current_node.f_cost}")
            
        return None
    
class PlanningWorker(QtCore.QObject):

    stateData = QtCore.pyqtSignal(str)
    pathData = QtCore.pyqtSignal(object)  
    finished   = QtCore.pyqtSignal(bool)

    def __init__(self,controller,grid):
        super().__init__() 
        self.controller = controller
        self.grid = grid
        #self.stateData.connect(self.controller.stateUpdated)
        self.pathData.connect(self.controller.pathUpdated)
        
    @QtCore.pyqtSlot()
    def run(self):
        if self.controller.state == "waiting_for_planning":
            print("[PlannerWorker] Zaczynam planowanie ścieżki...")
            grid = self.grid
            start = self.controller.start_pose
            goal = self.controller.goal_pose

            planner = NewPlanner(self.controller)
            path = None
            try:
                path = planner.hybrid_a_star_planning(start,goal,grid)
            except:
                print("[PlanningWorker] Błąd w planowaniu!")
            if path is not None:
                self.pathData.emit(path)
                self.controller.state = "finished_planning"
                print("[PlannerWorker] Znaleziono ścieżkę.")
                self.finished.emit(True)
            else:
                self.controller.state = "searching"
                print("[PlannerWorker] Nie znaleziono ścieżki.")
                self.finished.emit(False)
        else:
            self.finished.emit(False)

class Kalman():
    def __init__(self,wheelbase):
        self.states = 5

        self.Q = np.diag([1e-6,1e-6,1e-6,1e-12,1e-12]) # macierz kowariancji szumu procesowego
        self.R = np.diag([(1e-8)**2]) # kowariancji szumu pomiarowego
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
        psi_model = wrap_angle(psi_model + dt * (v_model/self.wheelbase) * np.tan(delta_model))
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
        innov[0] = wrap_angle(innov[0]) # obetnij yaw, żeby był maks. 2pi
        S = self.H @ self.E @ self.H.T + self.R     # 3x3
        K = self.E @ self.H.T @ np.linalg.inv(S)    # 5x3
        x_upd = x_hat + K @ innov
        x_upd[2] = wrap_angle(x_upd[2])
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
            # koniec programu
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



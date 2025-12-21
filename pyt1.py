
import reeds_shepp
import numpy as np
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys
import heapq
from heapdict import heapdict
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
"""
Funkcje odpowiadające za tworzenie trajektorii Hybrid-A* z rozszerzeniami o ciągłej krzywiźnie,
wraz z tworzeniem mapy przeszkód
"""
    
def generate_parking_map(x_start=0.0, y_start = 4.0, # gdzie zacząć mapę
                            n_c=10, # liczba samochodów
                            l_c = 5,l_p = 2*4.85, # długość samochodów/miejsca parkingowego
                            w_c = 1.9): # szerokość samochodów
    cars = []
    x_center = x_start
    for _ in range(n_c): # liczba samochodów
        dist = np.random.uniform(low = 0.2,high = l_p)
        x_center += dist + l_c/2
        yaw = np.random.uniform(low=-0.05,high=0.05)
        y_start += np.random.uniform(low=-0.3,high=0.3)
        cars.append((x_center, y_start,l_c, w_c,yaw))  # 1 pas samochodów z losową odległością od siebie i orientacją
        x_center += l_c/2
    return cars

class VisController(QtCore.QObject):

    stateChanged = QtCore.pyqtSignal(str)  # "drawing", "planning", "executing"
    mapUpdated = QtCore.pyqtSignal(object)   # obstacles, start, goal
    pathUpdated = QtCore.pyqtSignal(object)  # path data
    carUpdated = QtCore.pyqtSignal(object)   # (x, y, yaw)
    expansionUpdated = QtCore.pyqtSignal(object)
    hmapUpdated = QtCore.pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.state = "drawing"  # drawing, planning, finished_planning, executing
        self.ox = []
        self.oy = []
        self.start_pose = None
        self.goal_pose = None
        self.obstacles = []

"""
Parametry samochodu
"""
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
    BACKWARD_COST = c_val*2.0
    GEAR_CHANGE_COST = c_val*3.0
    STEER_CHANGE_COST = c_val*1.5
    STEER_ANGLE_COST = c_val*0.8
    OBS_COST = c_val*5.0
    H_COST = 10.0
    # parametry dla próbkowania
    XY_RESOLUTION = 0.1 # m
    YAW_RESOLUTION = np.deg2rad(10)
    # parametry dla Stanley
    K_STANLEY = 0.5

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

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

class NewOccupancyGrid():
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
        self.offset = (self.car_length / 2.0) - self.rear_bumper_to_axle
        
    def setup_obstacles(self, obstacles: list):
        print("cos1")
        self.obstacles = []
        print(obstacles)
        centers = [] # Tu będą środki geometryczne
        if len(obstacles) > 0:
            
            for obs in obstacles:
                
                processed_obs = obs.copy()
                axle_x, axle_y = obs['center']
                ang = obs['angle']
                
                x = axle_x + self.offset * np.cos(ang)
                y = axle_y + self.offset * np.sin(ang)
                processed_obs['center'] = np.array([x, y])
                self.obstacles.append(processed_obs)
                centers.append([x, y])
                
            self.kd_tree = KDTree(np.array(centers))
            
            self.max_obs_radius = max([np.hypot(o['length'], o['width'])/2 for o in obstacles])
            print(f"[OccupancyGrid] Zainicjalizowano z {len(obstacles)} przeszkodami (obiektami).")
        else:
            self.kd_tree = None
            self.max_obs_radius = 0
            print("[OccupancyGrid] Brak przeszkód.")

    
    def get_rect_corners(self, center, length, width, angle):
        cx, cy = center
        
        c, s = np.cos(angle), np.sin(angle)
        
        dx_l = (length/2) * c
        dy_l = (length/2) * s
        
        dx_w = (width / 2) * -s
        dy_w = (width / 2) * c
        
        # kolejność: FL, FR, RR, RL
        p1 = (cx + dx_l + dx_w, cy + dy_l + dy_w)
        p2 = (cx + dx_l - dx_w, cy + dy_l - dy_w)
        p4 = (cx - dx_l + dx_w, cy - dy_l + dy_w)
        p3 = (cx - dx_l - dx_w, cy - dy_l - dy_w)
        
        
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
        
        box_center_x = x + self.offset * np.cos(yaw)
        box_center_y = y + self.offset * np.sin(yaw)
        
        rect = self.get_rect_corners((box_center_x, box_center_y), C.CAR_LENGTH, C.CAR_WIDTH,yaw)
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

class OccupancyGrid():
    """ Stara siatka zajętości, nowa jest NewOccupancyGrid"""
    def __init__(self, ox, oy):

        self.ox = np.array(ox)
        self.oy = np.array(oy)
        if len(ox) == len(oy) and len(ox) > 0:
            obstacle_points = np.column_stack([self.ox, self.oy])
            self.kd_tree = KDTree(obstacle_points)
            print(f"[OccupancyGrid] Inicjalizowano z ilością punktów przeskzód: {obstacle_points.shape}")
        else:
            self.kd_tree = None
            print("[OccupancyGrid] Możliwy błąd przy tworzeniu KD-Tree")
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
        
    def get_obst_indices(self,x,y,yaw):
        rel_x = x + self.cg_x * np.cos(yaw)
        rel_y = y + self.cg_x * np.sin(yaw)
        query_point = np.array([rel_x, rel_y], dtype=np.float64)
        return self.kd_tree.query_ball_point(query_point,self.collision_radius)

    def is_collision(self,x,y,yaw) -> bool:
        # sprawdzenie kolizji za pomocą kd-tree 
        if self.kd_tree is None or len(self.ox) == 0:
            return False
        indices = self.get_obst_indices(x,y,yaw)
        if not indices:
            return False
        rot_mat = np.array([[np.cos(yaw),-np.sin(yaw)],
                            [np.sin(yaw),np.cos(yaw)]])
        for index in indices:
            obs_x_ref = self.ox[index] - x
            obs_y_ref = self.oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot_mat, [obs_x_ref, obs_y_ref])
            safety = 0.2
            if (rotated_obs_x_ref <= self.front_bumper_to_axle + safety
                    and rotated_obs_x_ref >= -self.rear_bumper_to_axle - safety
                    and rotated_obs_y_ref <= 0.5 * self.car_width + safety
                    and rotated_obs_y_ref >= -0.5 * self.car_width - safety):
                return True

        return False  
    
    def belief(self,bel):
        pass
    def interpret_readings(self,names_dists):
        
        pass
        

class Path:
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


class NewPlanner(QtCore.QObject):
    """ 
    Planer Hybrid-A* nowy z separating axes theorem detekcją kolizji i 
    budowaniem heurystyki na podstawie prostokątnych przeszkód
    """
    expansionData = QtCore.pyqtSignal(object)
    hmapData = QtCore.pyqtSignal(object)
    def __init__(self,controller:VisController):
        super().__init__()
        print("cos")
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
    
    def calculate_unconstrained_heuristic(self, start_pose, goal_pose, grid: NewOccupancyGrid):

        obstacles = grid.obstacles 
        
        gx, gy, _ = goal_pose
        sx, sy, _ = start_pose

        all_x = [sx, gx]
        all_y = [sy, gy]

        collision_radius = 0.2
        obs_polygons = []
        for obs in obstacles:
            corners = grid.get_rect_corners(obs['center'],obs['length']+ 2 * collision_radius,obs['width']+ 2 * collision_radius,obs['angle'])
            obs_polygons.append(corners)
            for p in corners:
                all_x.append(p[0])
                all_y.append(p[1])

        
        pad = 10
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
        
        return max(h_a_star,h_rs)*C.H_COST
        
    def simulate_motion(self, state, delta, direction):
        x, y, theta = state
        d = 1.0 if direction == "forward" else -1.0
        theta_new = wrap_angle(theta + np.tan(delta)/C.WHEELBASE * self.step_size)
        theta = (theta + theta_new) / 2
        x_new = x + d * self.step_size * np.cos(theta)
        y_new = y + d * self.step_size * np.sin(theta)
        return (x_new, y_new, theta_new)
    
    def get_neighbours(self, node:Node, grid:NewOccupancyGrid) -> list[Node]:
        
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
        if node.direction != direction:
            cost += C.GEAR_CHANGE_COST
        #if direction == "backward":
            #cost += C.BACKWARD_COST

        cost += C.STEER_ANGLE_COST * abs(delta)
        
        if abs(node.delta - delta) >= self.yaw_resolution:
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
        rs_path = reeds_shepp.path_sample(node.state,goal_pose,C.MAX_RADIUS,self.step_size)

        trajectory_risk_cost = 0.0

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

        path_length = reeds_shepp.path_length(
            node.state,
            goal_pose,
            C.MAX_RADIUS
        )

        total_risk_penalty = trajectory_risk_cost * C.OBS_COST * self.step_size

        goal_node = Node(
            cell=self.discretize_state(goal_pose),
            state=goal_pose,
            delta=0.0,
            direction=node.direction,
            g_cost=node.g_cost + path_length + total_risk_penalty,  
            h_cost=self.calculate_hybrid_heuristic(goal_pose,goal_pose),
            parent=node
        )   
        return rs_path,goal_node
        
    def reconstruct_path(self, rs_path, start_node: Node, goal_node: Node):
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
        self.hmapData.emit(self.hmap) # wyślij do parkingmap

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

        # TODO: przemyśleć taki zapis do closed_set, żeby później można było sięgnąć do poprzedniego itemu czy następnego po cell lub
        while not open_set.empty():
            if self.controller.state != "planning":
                return None
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
            #dist_to_goal = np.hypot(current_node.state[0]-goal_pose[0], current_node.state[1]-goal_pose[1])
            
            # rs_path,goal_node = self.try_reeds_shepp(current_node,goal_pose,grid)
            # if rs_path is not None and goal_node is not None:
            #     return self.reconstruct_path(rs_path,start_node,goal_node)

            rs_path, possible_goal_node = self.try_reeds_shepp(current_node, goal_pose, grid)
            if rs_path is not None and possible_goal_node is not None:
                # Nie zwracamy od razu! Sprawdzamy, czy to najlepsze co mamy.
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
                    #jeśli nowy koszt jest lepszy aktualizuj
                    if neighbour.g_cost < existing.g_cost:
                        open_set.push(neighbour)
            
            self.expansionData.emit(current_node.state)
            print(f"[Planner] h_cost:{current_node.h_cost}, g_cost:{current_node.g_cost}, f_cost:{current_node.f_cost}")
            #print(f"[Planner] Długość O_set: {len(open_set.heap)}, długość C_set: {len(closed_set)}")
            
        return None
    
class Planner(QtCore.QObject):
    expansionData = QtCore.pyqtSignal(object)
    def __init__(self,controller:VisController):
        super().__init__()
        self.goal_tolerance = 0.5
        self.xy_resolution = C.XY_RESOLUTION
        self.yaw_resolution = C.YAW_RESOLUTION  
        self.expansion_counter = 0.0
        self.step_size = 0.1
        self.n_steers = 9
        self.actions = self.calc_actions()
        self.hmap = None
        self.controller = controller
        self.expansionData.connect(self.controller.expansionUpdated)

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
    
    def calculate_unconstrained_heuristic(self,start_pose,goal_pose,grid:OccupancyGrid):
        gx,gy,_ = goal_pose
        sx,sy,_ = start_pose
        cgx,cgy = int(gx/self.xy_resolution),int(gy/self.xy_resolution)
        csx,csy = int(sx/self.xy_resolution),int(sy/self.xy_resolution)
        if len(grid.ox) == 0:
            return None
            
        if len(grid.ox) > 0:
            all_x = np.concatenate((grid.ox, [sx, gx]))
            all_y = np.concatenate((grid.oy, [sy, gy]))
        else:
            all_x = [sx, gx]
            all_y = [sy, gy]

        minx = int(np.floor(min(all_x - 2) / self.xy_resolution)) 
        maxx = int(np.ceil(max(all_x + 2) / self.xy_resolution)) 
        miny = int(np.floor(min(all_y - 2) / self.xy_resolution)) 
        maxy = int(np.ceil(max(all_y + 2) / self.xy_resolution)) 

        xw,yw = maxx-minx,maxy-miny
        hmap = np.full((xw,yw),np.inf)
        
        # rozszerz o promień kolizji przeszkody
        obs_map = np.zeros((xw,yw),dtype=bool)
        collision_radius = 2.0
        collision_radius_grid = int(np.ceil(collision_radius/self.xy_resolution))

        for ox, oy in zip(grid.ox, grid.oy):
            ox_grid = int(round(ox / self.xy_resolution)) - minx
            oy_grid = int(round(oy / self.xy_resolution)) - miny
            
            # Rozszerz przeszkodę o promień samochodu
            for dx in range(-collision_radius_grid, collision_radius_grid + 1):
                for dy in range(-collision_radius_grid, collision_radius_grid + 1):
                    nx, ny = ox_grid + dx, oy_grid + dy
                    if 0 <= nx < xw and 0 <= ny < yw:
                        if np.hypot(dx, dy) * self.xy_resolution <= collision_radius:
                            obs_map[nx, ny] = True

        # dodaj końcową pozę jako początek dla poszukiwania kosztów
        start_idx = (cgx - minx, cgy - miny)
        open_set,closed_set = [],set()
        heapq.heappush(open_set,(0.0,start_idx))
        #hmap[start_idx[0],start_idx[1]] = 0.0
        # ruchy w 8 kierunków, (przód/tył, prawo/lewo, wprost lub po przekątnej)
        motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), 
            (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
        ]
        expansion_counter = 0
        while open_set:
            print(f"[Heurystyka] Trwa obliczenie {expansion_counter}")
            cost0,(cx,cy) = heapq.heappop(open_set)
            if (cx, cy) in closed_set:
                continue
            closed_set.add((cx, cy))

            for dir1,dir2,cost in motions:
                nx,ny = cx + dir1, cy + dir2
                if not (0 <= nx < xw and 0 <= ny < yw):
                    continue
                if obs_map[nx, ny]:
                    continue
                
                if (nx, ny) in closed_set:
                    continue
                
                new_cost = cost0 + cost
                if new_cost < hmap[nx,ny]:
                    hmap[nx,ny] = new_cost
                    heapq.heappush(open_set,(new_cost,(nx,ny)))
            expansion_counter += 1
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
        theta_new = wrap_angle(theta + np.tan(delta)/C.WHEELBASE * self.xy_resolution)
        theta = (theta + theta_new) / 2
        x_new = x + d * self.xy_resolution * np.cos(theta)
        y_new = y + d * self.xy_resolution * np.sin(theta)
        return (x_new, y_new, theta_new)
    
    def get_neighbours(self, node:Node, grid:NewOccupancyGrid) -> list[Node]:
        
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
        if direction == "reverse":
            cost *= C.BACKWARD_COST
            if node.direction != direction:
                cost += C.GEAR_CHANGE_COST
        
        cost += C.STEER_ANGLE_COST * abs(delta)

        if abs(node.delta - delta) > 0.01:
            cost += C.STEER_CHANGE_COST * abs(node.delta - delta)
        
        return cost
    
    def required_expansion(self,pose,goal_pose):
        dist = np.hypot(
            pose[0] - goal_pose[0],
            pose[1] - goal_pose[1]
        )
        return dist < 10.0

    def try_reeds_shepp(self,node:Node,goal_pose,grid:OccupancyGrid):
        rs_path = reeds_shepp.path_sample(node.state,goal_pose,C.MAX_RADIUS,self.step_size)
        if not rs_path:
            return None,None
        for i in range(0, len(rs_path), 5): # co 5 próbka
            if grid.is_collision(*rs_path[i]):
                return None,None
            
        path_length = reeds_shepp.path_length(
            node.state,
            goal_pose,
            C.MAX_RADIUS
        )
        goal_node = Node(
            cell=self.discretize_state(goal_pose),
            state=goal_pose,
            delta=0.0,
            direction=node.direction,
            g_cost=node.g_cost + path_length * 1.2,  # Lekka kara
            h_cost=self.calculate_hybrid_heuristic(goal_pose,goal_pose),
            parent=node
        )   
        return rs_path,goal_node
        
    def reconstruct_path(self,rs_path,start_node:Node,goal_node:Node):
        path_xs,path_ys,path_yaws,dirs,costs = [],[],[],[],[]
        
        for i in range(len(rs_path)):
            path_xs.append(rs_path[i][0])
            path_ys.append(rs_path[i][1])
            path_yaws.append(rs_path[i][2])

        cur = goal_node
        while True:
            if cur.cell == start_node.cell:
                break
            x,y,yaw = cur.state
            path_xs.append(x)
            path_ys.append(y)
            path_yaws.append(yaw)
            dirs.append(cur.direction)

            costs.append(cur.g_cost)
            cur = cur.parent
            

        return Path(path_xs,path_ys,path_yaws,dirs,costs)
    
    # TODO: powinien planować tak, że lepiej niech znajdzie reeds_shepp, a później rozszerza, jeżeli jest kolizja
    def hybrid_a_star_planning(self,start_pose,goal_pose,grid:OccupancyGrid):
        print("[Planner] Zaczęto planowanie")
        open_set = PriorityQueue()
        closed_set = {}

        self.hmap = self.calculate_unconstrained_heuristic(start_pose,goal_pose,grid)

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
        
        # TODO: przemyśleć taki zapis do closed_set, żeby później można było sięgnąć do poprzedniego itemu czy następnego po cell lub
        while not open_set.empty():
            if self.controller.state != "planning":
                return None
            self.expansion_counter += 1
            
            current_node = open_set.pop() 
            #if self.is_goal(current_node.state,goal_pose):
            #return self.reconstruct_path(current_node) # zrób ścieżkę, jeżeli dostatecznie blisko do celu
            current_cell = current_node.cell
            if current_cell in closed_set:
                if closed_set[current_cell] <= current_node.g_cost:
                    continue # już byliśmy tu
            closed_set[current_cell] = current_node.g_cost

            #if self.required_expansion(current_node.state,goal_pose):
            rs_path,goal_node = self.try_reeds_shepp(current_node,goal_pose,grid)
            if rs_path is not None and goal_node is not None:
                return self.reconstruct_path(rs_path,start_node,goal_node)
            neighbours = self.get_neighbours(current_node,grid)
            
            for neighbour in neighbours:
                neighbour_cell = neighbour.cell

                if neighbour_cell in closed_set and closed_set[neighbour_cell] <= neighbour.g_cost:
                    continue
                neighbour.h_cost = self.calculate_hybrid_heuristic(neighbour.state,goal_pose)
                neighbour.f_cost = neighbour.g_cost + neighbour.h_cost
                if not open_set.contains(neighbour_cell):
                    open_set.push(neighbour)
                #else:
                    #existing = open_set.get_node(neighbour_cell)
                    #jeśli nowy koszt jest lepszy aktualizuj
                    #if neighbour.g_cost < existing.g_cost:
                        #open_set.push(neighbour)
            self.expansionData.emit(current_node.state)
            print(f"[Planner] h_cost:{current_node.h_cost}, g_cost:{current_node.g_cost}, f_cost:{current_node.f_cost}")
            #print(f"[Planner] Długość O_set: {len(open_set.heap)}, długość C_set: {len(closed_set)}")
            
        return None


class ParkingMap(pg.GraphicsLayoutWidget):   
    def __init__(self,cont:VisController):
        super().__init__(title="Siatka zajętości dla parkingu")
        self.resize(1000, 800)
        self.setBackground((235, 235, 250))
        self.controller = cont
        self.ox = []
        self.oy = []
        self.rear_bumper_to_axle = 1.0
        self.front_bumper_to_axle = C.CAR_LENGTH - self.rear_bumper_to_axle
        # stan rysowania
        self.mode = "idle"
        
        # przechowywanie linii
        self.current_line_points = []  # [(x, y), ...]
        self.line_items = []           # lista narysowanych linii
        self.obstacle_items = []
        self.preview_line = None
        #
        self.start_item = None
        self.start_arrow = None
        self.goal_item = None
        self.goal_arrow = None
        self.car_item = None
        self.path_item = None
        # Orientacja start/goal
        self.temp_start_pos = None     # start bez odchylenia
        self.temp_goal_pos = None      # koniec bez odchylenia
        #
        self.start_pose = None
        self.goal_pose = None
        self.test_path = None
        # Setup UI
        self._setup_plot()
        #
        # Mysz 
        self.map.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.map.scene().sigMouseClicked.connect(self.on_mouse_click)
        # Nadpisz mouseDragEvent w ViewBox
        #original_drag = self.map.vb.mouseDragEvent
        """
        def custom_drag(ev):
            # Obsłuż nasze rysowanie
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                if ev.isStart():
                    self._handle_drag_start(ev)
                elif ev.isFinish():
                    self._handle_drag_finish(ev)
                else:
                    self._handle_drag_move(ev)
                ev.accept()  # Zaakceptuj event
            else:
                # Inne przyciski - domyślne zachowanie (np. PPM do przesuwania)
                original_drag(ev)
        """
        #self.map.vb.mouseDragEvent = custom_drag


        cont.stateChanged.connect(self.on_draw_state_changed)
        cont.pathUpdated.connect(self.draw_path)
        cont.carUpdated.connect(self.update_car)
        cont.expansionUpdated.connect(self.draw_expansion_cars)
        cont.hmapUpdated.connect(self.draw_hmap)

    """
    Funkcje dla UI
    """
    def _clear_plot(self):
        """Wyczyść wszystkie elementy z mapy"""
        # Wyczyść cały plot (usuwa wszystkie items i ploty)
        self.map.clear()
        # Kursor do śledzenia
        self.cursor_item = pg.TextItem("", anchor=(0.5, 0.5))
        self.cursor_item.setPos(0, 0)
        self.map.addItem(self.cursor_item)
        # Ścieżka
        self.route = self.map.plot([], [], pen='r', name="Ścieżka Reeds-Shepp")
        self.route.setZValue(1)
        # Resetuj referencje
        self.line_items = []
        self.obstacle_items = []
        self.current_line_points = []
        self.preview_line = None
        self.start_item = None
        self.start_arrow = None
        self.goal_item = None
        self.goal_arrow = None
        self.car_item = None
        self.path_item = None
        self.temp_start_pos = None
        self.temp_goal_pos = None
        self.temp_obst_pos = None
        self.temp_obst = None
        # Resetuj stan
        self.mode = "idle"
        self.is_drawing = False
        self.path_drawn = False
        self.tx = []
        self.ty = []
        self.controller.ox = []
        self.controller.oy = []
        self.controller.obstacles = []
        # Wyczyść dane w kontrolerze
        self.start_pose = None
        self.goal_pose = None
        self.start_car = None
        self.goal_car = None 
        # dla testu Reeds-Shepp
        self.test_path = None
        self.clear_debug_items()
        
    # Ustawienia wykresu
    def _setup_plot(self):
        # Bufory
        self.tx = []; self.ty = []
        # Wykres
        self.map = self.addPlot(lockAspect=True)
        self.map.setRange(xRange=[-50, 50], yRange=[-50, 50])
        self.map.setMouseEnabled(x=True, y=True)
        self.map.showGrid(x=True, y=True, alpha=0.3)
        self.map.addLegend()
        
        #
        self.map.setClipToView(True)
        self.map.setDownsampling(mode='peak')
        self.map.setMenuEnabled(False)      # Dla PlotItem
        self.map.vb.setMenuEnabled(False)   # Dla ViewBox
        #
        # Kursor do śledzenia
        self.cursor_item = pg.TextItem("", anchor=(0.5, 0.5))
        self.cursor_item.setPos(0, 0)
        self.map.addItem(self.cursor_item)
        # Ścieżka
        self.route = self.map.plot([], [], pen='r', name="Ścieżka Reeds-Shepp")
        self.route.setZValue(1)
        self.test_path = None
        self.mode = "idle"
        self.is_drawing = False
        self.path_drawn = False
        self._setup_instructions()
        # samochody
        self.start_car = None
        self.goal_car = None 
        self.debug_items = []

    # Instrukcje
    def _setup_instructions(self):
        """Dodaj tekst z instrukcjami"""
        self.instructions = pg.TextItem(
            "Rysuj przeszkody (przeciągnij LPM) | Start (klik LPM) | Koniec (klik PPM)",
            anchor=(0.5, 1),
            color=(50, 50, 150)
        )
        self.instructions.setPos(0, 100)
        self.map.addItem(self.instructions)    

    """
    Funkcje dla obsługi myszy
    """

    def on_mouse_move(self, pos):
        """Śledzenie kursora myszy"""
        if not self.map.sceneBoundingRect().contains(pos):
            return
        
        mouse_point = self.map.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Aktualizuj tekst kursora
        #if self.mode == "drawing_line":
            #self.cursor_item.setText(f"Rysowanie... ({x:.1f}, {y:.1f})")
        if self.mode == "idle":
            self.cursor_item.setText(f"Ustawianie przeszkód... ({x:.1f}, {y:.1f})")
        elif self.mode == "waiting_obst_orientation":
            self.cursor_item.setText(f"Wybierz kierunek przeszkody ({x:.1f}, {y:.1f})")
        elif self.mode == "waiting_start_orientation":
            self.cursor_item.setText(f"Wybierz kierunek STARTOWY ({x:.1f}, {y:.1f})")
        elif self.mode == "waiting_goal_orientation":
            self.cursor_item.setText(f"Wybierz kierunek KOŃCOWY ({x:.1f}, {y:.1f})")
        else:
            self.cursor_item.setText(f"({x:.1f}, {y:.1f})")
        
        self.cursor_item.setPos(x, y + 2)
            
    # po kliknięciu myszy przeszkoda  dodaje się do listy jako słownik  
    def on_mouse_click(self, event):
        """Obsługa kliknięć myszy"""
        if self.mode != "disabled":
            pos = event.scenePos()
            if not self.map.sceneBoundingRect().contains(pos):
                return
            mouse_point = self.map.vb.mapSceneToView(pos)
            
            x, y = mouse_point.x(), mouse_point.y()
            
                      
            if self.mode == "waiting_obst_orientation":
                if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                    obst = self._finish_obstacle_orientation(x, y)
                    self.make_obst_car(obst['center'][0],obst['center'][1],obst['angle'])
                    print(self.controller.obstacles)
                    self.controller.obstacles.append(obst)
                    print(self.controller.obstacles)
                    offset = C.CAR_LENGTH/2 - self.rear_bumper_to_axle
                    cx = obst['center'][0] + offset * np.cos(obst['angle'])
                    cy = obst['center'][1] + offset * np.sin(obst['angle'])
                    debug_shape = self.draw_debug_box(
                        (cx,cy), 
                        obst['length'], 
                        obst['width'], 
                        obst['angle'], 
                        color='m' # Magenta dla przeszkód
                    )
                    
                    self.debug_items.append(debug_shape)
            elif self.mode == "waiting_start_orientation":
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._finish_start_orientation(x, y)
                    self.controller.start_pose = self.start_pose
                    self.test_path = None
                    self.make_start_car()
                    #self.build_test_path()
                return
            
            elif self.mode == "waiting_goal_orientation":
                if event.button() == QtCore.Qt.MouseButton.RightButton:
                    self._finish_goal_orientation(x, y)
                    self.controller.goal_pose = self.goal_pose
                    self.test_path = None
                    self.make_goal_car()
                    #self.build_test_path()
                return
            elif self.mode == "idle":
                if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                    self._set_obstacle_point(x, y)
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._set_start_point(x, y)
                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    self._set_goal_point(x, y)

    def build_test_path(self):
        if self.start_pose is not None and self.goal_pose is not None and self.test_path is None:
            self.tx = []; self.ty = []
            self.test_path = reeds_shepp.path_sample(self.start_pose,self.goal_pose,C.MAX_RADIUS,0.05)
            types = reeds_shepp.path_type(self.start_pose,self.goal_pose,C.MAX_RADIUS)
            print(f"Typy ścieżek: {types}")
            for p in self.test_path:
                x,y,yaw = p
                self.tx.append(x); self.ty.append(y)
            self.route.setData(self.tx,self.ty)
        else:
            self.tx = []; self.ty = []
    
    def make_start_car(self):
        if self.start_car is None:
            if self.start_pose is not None:
                x,y,yaw = self.start_pose
                self.start_car = self.make_car(x,y,yaw)
                self.start_car.setBrush(QtGui.QBrush(QtGui.QColor(0,100,0,150)))
                self.map.addItem(self.start_car)
                self.start_car.setZValue(2)
        else:
            self.transform_car_item(self.start_car,self.start_pose)
            

    def make_goal_car(self):
        if self.goal_car is None:
            if self.goal_pose is not None:
                x,y,yaw = self.goal_pose
                self.goal_car = self.make_car(x,y,yaw)
                self.goal_car.setBrush(QtGui.QBrush(QtGui.QColor(100,0,0,150)))
                self.map.addItem(self.goal_car)
                self.goal_car.setZValue(2)
        else:
            self.transform_car_item(self.goal_car,self.goal_pose)
            
    
    def make_obst_car(self,x,y,yaw):
        obst_car = self.make_car(x,y,yaw)
        obst_car.setBrush(QtGui.QBrush(QtGui.QColor(20,0,0,150)))
        self.obstacle_items.append(obst_car)
        self.map.addItem(obst_car)
        obst_car.setZValue(2)
         
    """
    Rysowanie przeszkód
    """
    
    # def _handle_drag_start(self, ev):
    #     """Początek przeciągania"""
    #     if self.mode != "idle":
    #         return
        
    #     scene_pos = ev.scenePos()
    #     view_pos = self.map.vb.mapSceneToView(scene_pos)
    #     x, y = view_pos.x(), view_pos.y()
        
    #     self.is_drawing = True
    #     self.mode = "drawing_line"
    #     self.current_line_points = [(x, y)]
        
    #     # Stwórz preview line
    #     self.preview_line = self.map.plot(
    #         [x], [y],
    #         pen=pg.mkPen(color='b', width=2, style=QtCore.Qt.PenStyle.DashLine)
    #     )
        
    #     print(f"[Map] Start drawing line at ({x:.1f}, {y:.1f})")
    
    # def _handle_drag_move(self, ev):
    #     """Ruch podczas przeciągania"""
    #     if not self.is_drawing or self.mode != "drawing_line":
    #         return
        
    #     scene_pos = ev.scenePos()
    #     view_pos = self.map.vb.mapSceneToView(scene_pos)
    #     x, y = view_pos.x(), view_pos.y()
        
    #     # Sprawdź dystans od ostatniego punktu
    #     if len(self.current_line_points) > 0:
    #         x_last, y_last = self.current_line_points[-1]
    #         dist = np.hypot(x - x_last, y - y_last)
            
    #         if dist < 0.3:  # Minimalna odległość 30cm
    #             return
        
    #     # Dodaj punkt
    #     self.current_line_points.append((x, y))
        
    #     # Aktualizuj preview
    #     xs = [p[0] for p in self.current_line_points]
    #     ys = [p[1] for p in self.current_line_points]
    #     self.preview_line.setData(xs, ys)
    
    # def _handle_drag_finish(self, ev):
    #     """Koniec przeciągania"""
    #     if not self.is_drawing or self.mode != "drawing_line":
    #         return
        
    #     self.is_drawing = False
        
    #     if len(self.current_line_points) < 2:
    #         # Za krótka linia - usuń preview
    #         if self.preview_line:
    #             self.map.removeItem(self.preview_line)
    #             self.preview_line = None
    #         self.mode = "idle"
    #         self.current_line_points = []
    #         return
        
    #     # Usuń preview
    #     if self.preview_line:
    #         self.map.removeItem(self.preview_line)
    #         self.preview_line = None
        
    #     # Narysuj finalną linię
    #     xs = [p[0] for p in self.current_line_points]
    #     ys = [p[1] for p in self.current_line_points]
        
    #     final_line = self.map.plot(
    #         xs, ys,
    #         pen=pg.mkPen(color='k', width=3)
    #     )
    #     self.line_items.append(final_line)
        
    #     # Interpoluj do gęstych punktów
    #     ox_line, oy_line = self._interpolate_line(self.current_line_points, resolution=1.0)
    #     self.ox.extend(ox_line)
    #     self.oy.extend(oy_line)
    #     self.controller.ox = self.ox
    #     self.controller.oy = self.oy
    #     print(f"[Map] Finished line with {len(self.current_line_points)} vertices, "
    #           f"{len(ox_line)} interpolated points")
        
    #     # Reset
    #     self.mode = "idle"
    #     self.current_line_points = []

    # def _interpolate_line(self, points, resolution=1.0):
    #     """
    #     Interpoluj linię łamaną do gęstych punktów.
        
    #     Args:
    #         points: [(x0, y0), (x1, y1), ...]
    #         resolution: [m] odległość między punktami
        
    #     Returns:
    #         ox, oy: listy interpolowanych punktów
    #     """
    #     ox, oy = [], []
        
    #     for i in range(len(points) - 1):
    #         x0, y0 = points[i]
    #         x1, y1 = points[i + 1]
            
    #         # Długość segmentu
    #         dx = x1 - x0
    #         dy = y1 - y0
    #         length = np.hypot(dx, dy)
            
    #         # Liczba punktów
    #         n_points = max(int(length / resolution), 2)
            
    #         # Interpoluj
    #         for j in range(n_points):
    #             t = j / (n_points - 1)
    #             ox.append(x0 + t * dx)
    #             oy.append(y0 + t * dy)
        
    #     return ox, oy
    
    """
    Ustawienie punktów końcowych i startowych
    """

    def _set_obstacle_point(self, x, y):
        """Krok 1: Ustaw pozycję startu, czekaj na orientację"""
                
        # Zapisz tymczasową pozycję
        self.temp_obst_pos = (x, y)
        
        # Narysuj punkt (bez strzałki jeszcze)
        obst_point = pg.ScatterPlotItem(
            [x], [y],
            size=15,
            brush=pg.mkBrush(0, 100, 10, 200),
            symbol='o'
        )
        self.map.addItem(obst_point)
        
        # Flash
        self._flash_point(x, y, color='b')
        
        # Przejdź w tryb wyboru orientacji
        self.mode = "waiting_obst_orientation"
        self.instructions.setText("Kliknij LPM aby wybrać kierunek przeszkody")
         
        print(f"[Mapa] Pozycja przeszkody: ({x:.1f}, {y:.1f}). Czekam na orientację...")
    
    def _finish_obstacle_orientation(self, x_dir, y_dir):
        """Krok 2: Oblicz orientację z drugiego punktu"""
        if not self.temp_obst_pos:
            return
        
        x0, y0 = self.temp_obst_pos
        
        # Oblicz kąt: yaw = atan2(dy, dx)
        dx = x_dir - x0
        dy = y_dir - y0
        yaw = np.arctan2(dy, dx)
        
        # Reset trybu
        self.mode = "idle"
        
        self.temp_obst_pos = None
        self.instructions.setText("Rysuj przeszkody (przeciągnij LPM) | Start (klik LPM) | Koniec (klik PPM)")
        print(f"[Map] Orientacja przeszkody: {np.degrees(yaw):.1f}°")

        obst = {
            'center': np.array([x0,y0]),
            'angle': yaw,
            'length': C.CAR_LENGTH,
            'width': C.CAR_WIDTH,
        }

        return obst
    def _set_start_point(self, x, y):
        """Krok 1: Ustaw pozycję startu, czekaj na orientację"""
        # Usuń stary
        if self.start_item:
            self.map.removeItem(self.start_item)
        if self.start_arrow:
            for arrow_part in self.start_arrow:
                self.map.removeItem(arrow_part)
            self.start_arrow = None
                
        # Zapisz tymczasową pozycję
        self.temp_start_pos = (x, y)
        
        # Narysuj punkt (bez strzałki jeszcze)
        self.start_item = pg.ScatterPlotItem(
            [x], [y],
            size=15,
            brush=pg.mkBrush(0, 255, 0, 200),
            symbol='o'
        )
        self.map.addItem(self.start_item)
        
        # Flash
        self._flash_point(x, y, color='g')
        
        # Przejdź w tryb wyboru orientacji
        self.mode = "waiting_start_orientation"
        self.instructions.setText("Kliknij LPM aby wybrać kierunek START")
         
        print(f"[Map] Pozycja startowa: ({x:.1f}, {y:.1f}). Czekam na orientację...")
        
    def _finish_start_orientation(self, x_dir, y_dir):
        """Krok 2: Oblicz orientację z drugiego punktu"""
        if not self.temp_start_pos:
            return
        
        x0, y0 = self.temp_start_pos
        
        # Oblicz kąt: yaw = atan2(dy, dx)
        dx = x_dir - x0
        dy = y_dir - y0
        yaw = np.arctan2(dy, dx)
        
        # Narysuj strzałkę
        self._draw_arrow(x0, y0, yaw, color='g')
        
        # Reset trybu
        self.mode = "idle"
        self.start_pose = (x0,y0,yaw)
        self.temp_start_pos = None
        self.instructions.setText("Rysuj przeszkody (przeciągnij LPM) | Start (klik LPM) | Koniec (klik PPM)")
        print(f"[Map] Orientacja startowa: {np.degrees(yaw):.1f}°")
    
    # ========================================================================
    # GOAL POINT - Z ORIENTACJĄ
    # ========================================================================
    
    def _set_goal_point(self, x, y):
        """Krok 1: Ustaw pozycję celu, czekaj na orientację"""
        # Usuń stary
        if self.goal_item:
            self.map.removeItem(self.goal_item)
        if self.goal_arrow:
            for arrow_part in self.goal_arrow:
                self.map.removeItem(arrow_part)
            self.goal_arrow = None
        # Zapisz tymczasową pozycję
        self.temp_goal_pos = (x, y)
        # Narysuj punkt
        self.goal_item = pg.ScatterPlotItem(
            [x], [y],
            size=15,
            brush=pg.mkBrush(255, 0, 0, 200),
            symbol='s'
        )
        self.map.addItem(self.goal_item)
        
        # Flash
        self._flash_point(x, y, color='r')
        
        # Przejdź w tryb wyboru orientacji
        self.mode = "waiting_goal_orientation"
        self.instructions.setText("Kliknij PPM aby wybrać kierunek końcowy")
        
        print(f"[Map] Pozycja końcowa: ({x:.1f}, {y:.1f}). Czekam na orientację...")
    
    def _finish_goal_orientation(self, x_dir, y_dir):
        """Krok 2: Oblicz orientację z drugiego punktu"""
        if not self.temp_goal_pos:
            return
        
        x0, y0 = self.temp_goal_pos
        
        # Oblicz kąt
        dx = x_dir - x0
        dy = y_dir - y0
        yaw = np.arctan2(dy, dx)
        
        # Narysuj strzałkę
        self._draw_arrow(x0, y0, yaw, color='r')
        
        # Reset trybu
        self.mode = "idle"
        self.goal_pose = (x0,y0,yaw) 

        self.temp_goal_pos = None

        self.instructions.setText("Rysuj przeszkody (przeciągnij LPM) | Start (klik LPM) | Koniec (klik PPM)")
        print(f"[Map] Orientacja końcowa: {np.degrees(yaw):.1f}°")
    
    """
    Wizualne efekty
    """
    def _draw_arrow(self, x, y, yaw, color='g', length=3.0):
        """Narysuj strzałkę wskazującą orientację"""
        # Koniec strzałki
        x_end = x + length * np.cos(yaw)
        y_end = y + length * np.sin(yaw)
        
        # Linia
        arrow_line = self.map.plot(
            [x, x_end], [y, y_end],
            pen=pg.mkPen(color=color, width=3)
        )
        
        # Grot (mały trójkąt)
        arrow_size = 0.5
        angle_left = yaw + np.pi * 0.8
        angle_right = yaw - np.pi * 0.8
        
        x_left = x_end + arrow_size * np.cos(angle_left)
        y_left = y_end + arrow_size * np.sin(angle_left)
        x_right = x_end + arrow_size * np.cos(angle_right)
        y_right = y_end + arrow_size * np.sin(angle_right)
        
        arrow_head = self.map.plot(
            [x_left, x_end, x_right],
            [y_left, y_end, y_right],
            pen=pg.mkPen(color=color, width=3),
            brush=pg.mkBrush(color)
        )
        
        # Zapisz referencję
        if color == 'g':
            self.start_arrow = [arrow_line, arrow_head]
        else:
            self.goal_arrow = [arrow_line, arrow_head]
    
    def _flash_point(self, x, y, color='w', duration=300):
        flash = pg.ScatterPlotItem(
            [x], [y],
            size=30,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(color, width=3)
        )
        self.map.addItem(flash)
        QtCore.QTimer.singleShot(duration, lambda: self.map.removeItem(flash))

    def get_math_corners(self,center, length, width, angle):
        """
        Oblicza rogi prostokąta dokładnie tak, jak robi to OccupancyGrid.
        angle: kąt matematyczny (radiany, CCW)
        """
        cx, cy = center
        c, s = np.cos(angle), np.sin(angle)
        
        # Wektory od środka do krawędzi
        dx_l = (length/2) * c
        dy_l = (length/2) * s
        
        dx_w = (width/2) * -s
        dy_w = (width/2) * c
        
        # Kolejność: FL, FR, RR, RL
        p1 = (cx + dx_l + dx_w, cy + dy_l + dy_w)
        p2 = (cx + dx_l - dx_w, cy + dy_l - dy_w)
        p3 = (cx - dx_l - dx_w, cy - dy_l - dy_w)
        p4 = (cx - dx_l + dx_w, cy - dy_l + dy_w)
        
        return [p1, p2, p3, p4]

    def draw_debug_box(self, center, length, width, angle, color='y'):
        """
        Rysuje obrys kolizji (wireframe).
        """
        corners = self.get_math_corners(center, length, width, angle)
        # Dodajemy pierwszy punkt na koniec, żeby zamknąć pętlę
        corners.append(corners[0])
        
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        
        # Rysujemy przerywaną linią
        plot_item = self.map.plot(
            xs, ys, 
            pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        return plot_item

    def clear_debug_items(self):
        """Usuwa stare linie debugowania"""
        if hasattr(self, 'debug_items'):
            for item in self.debug_items:
                self.map.removeItem(item)
        self.debug_items = []

    """
    Funkcje dla rysowania obiektów
    """
    def make_car(self,x_center,y_center,yaw,l_c=C.CAR_LENGTH,w_c=C.CAR_WIDTH):
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
        t.translate(x_center, y_center) #
        t.rotate(np.degrees(yaw))   #, QtCore.Qt.ZAxis
        car.setTransform(t)
        pen = pg.mkPen(color=(0, 0, 0), width=1)
        car.setPen(pen)
        car.setBrush(QtGui.QBrush(QtGui.QColor(80,80,80,150)))
        
        return car 

    def transform_car_item(self,item,pose):
        x,y,yaw = pose
        t = QtGui.QTransform()
        t.translate(x, y)
        t.rotate(np.degrees(yaw))  
        item.setTransform(t)

    @QtCore.pyqtSlot(object)
    def update_car(self, car_pos):
        x, y, yaw = car_pos
        if self.car_item is None:
            self.car_item = self.make_car(x,y,yaw)
            self.map.addItem(self.car_item)
    
    @QtCore.pyqtSlot(object)
    def draw_expansion_cars(self,car_pos):
        x, y, yaw = car_pos
        car_item = self.make_car(x,y,yaw)
        self.transform_car_item(car_item,car_pos)
        self.map.addItem(car_item)

    """
    Dla obsługi stanów aplikacji
    """      
    @QtCore.pyqtSlot(str)
    def on_draw_state_changed(self, state):
        """Zmiana stanu aplikacji"""
        if state == "planning":
            self.mode = "disabled"
            self.instructions.setText("Planowanie ścieżki...")
        elif state == "executing":
            self.mode = "disabled"
            sx,sy,syaw = self.start_pose
            gx,gy,gyaw = self.goal_pose
            car_item_st = self.make_car(sx,sy,syaw)
            self.map.addItem(car_item_st)
            car_item_go = self.make_car(gx,gy,gyaw)
            self.map.addItem(car_item_go)
            self.instructions.setText("Wykonywanie ścieżki...")
        
    @QtCore.pyqtSlot(object)
    def draw_path(self, path: Path):
        """Rysuj zaplanowaną ścieżkę"""
        if self.path_drawn:
            return
        
        self.tx = path.xs[:] 
        self.ty = path.ys[:]
        self.route.setData(self.tx, self.ty)
        self.path_drawn = True
        
        print(f"[Map] Path drawn: {len(self.tx)} points")

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
        self.map.addItem(map)
        hmap_copy = hmap.copy()
        max_val = np.max(hmap_copy[~np.isinf(hmap_copy)]) if np.any(~np.isinf(hmap_copy)) else 10
        hmap_copy[np.isinf(hmap_copy)] = max_val * 1.5  # Ściany jako bardzo jasne/ciemne

        # Tworzenie ImageItem
        self.hmap_item = pg.ImageItem(hmap_copy)
        
        colormap = pg.colormap.get('viridis') 
        self.hmap_item.setLookupTable(colormap.getLookupTable())

        self.hmap_item.setOpacity(0.5)
        tr = QtGui.QTransform()
        
        origin_x = minx * C.XY_RESOLUTION
        origin_y = miny * C.XY_RESOLUTION
        
        tr.translate(origin_x, origin_y)
        tr.scale(C.XY_RESOLUTION, C.XY_RESOLUTION)
        
        self.hmap_item.setTransform(tr)
        
        # Dodajemy na sam spód (z-index niski)
        self.hmap_item.setZValue(-10)
        self.map.addItem(self.hmap_item)
        print("[Vis] Heatmapa heurystyki narysowana.")


class MainWindow(QtWidgets.QWidget):
    """Główne okno aplikacji"""
    
    mapData = QtCore.pyqtSignal(object)   # obstacles, start, goal
    carData = QtCore.pyqtSignal(object)   # (x, y, yaw)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ścieżki")
        self.resize(1200, 900)
        
        # Controller
        self.controller = VisController()
        # Layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        # Mapa
        self.map_widget = ParkingMap(self.controller)
        layout.addWidget(self.map_widget)
        # Przyciski
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btn_done = QtWidgets.QPushButton("Zaplanuj ścieżkę")
        self.btn_done.clicked.connect(self.on_done_clicked)
        button_layout.addWidget(self.btn_done)
        
        self.btn_clear = QtWidgets.QPushButton("Wyczyść mapę")
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        button_layout.addWidget(self.btn_clear)
        
        # NIEFUNKCJONALNE 
        self.btn_simulate = QtWidgets.QPushButton("Symuluj")
        self.btn_simulate.clicked.connect(self.on_simulate_clicked)
        self.btn_simulate.setEnabled(False)
        button_layout.addWidget(self.btn_simulate)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Stan: Rysowanie mapy")
        layout.addWidget(self.status_label)

        # Połącz sygnały
        self.controller.stateChanged.connect(self.ui_on_state_changed)
        self.mapData.connect(self.controller.mapUpdated)
        self.carData.connect(self.controller.carUpdated)

        # planowanie wątki
        self.planning_thread = None
        self.planning_worker = None

    # do planowania wątek, żeby odciążyć apkę
    def start_planning_thread(self):
        
        self.planning_worker = PlanningWorker(self.controller)
        self.planning_thread = QtCore.QThread()
        self.planning_worker.moveToThread(self.planning_thread)

        self.planning_thread.started.connect(self.planning_worker.run)
        self.planning_worker.finished.connect(self.planning_thread.quit)
        self.planning_worker.finished.connect(self.planning_worker.deleteLater)
        self.planning_thread.finished.connect(self.planning_thread.deleteLater)

        self.planning_worker.finished.connect(self.planning_finished)

        self.planning_thread.start()

    @QtCore.pyqtSlot(bool)
    def planning_finished(self,fin):
        if fin:
            self.btn_simulate.setEnabled(True)
        else:
            self.btn_simulate.setEnabled(False)

    # ___ DO PRZYCISKÓW _____________________________
    def on_done_clicked(self):
        self.start_pose = self.map_widget.start_pose
        self.goal_pose = self.map_widget.goal_pose
        #print(f"DEBUG: len(ox)={len(self.map_widget.ox)}, len(oy)={len(self.map_widget.oy)}")
        #print(f"DEBUG: ox[:5]={self.map_widget.ox[:5]}")
        #print(f"DEBUG: oy[:5]={self.map_widget.oy[:5]}")
        if self.finish_drawing():
            self.btn_done.setEnabled(False)
            self.btn_simulate.setEnabled(False)
            self.start_planning_thread()
    
    def on_clear_clicked(self):
        self.map_widget._clear_plot()
        #print(self.map_widget.test_path)
        self.btn_done.setEnabled(True)
        if self.controller.state != "drawing":
            self.set_state("drawing")
        print("[MainWindow] Mapa wyczyszczona")
    
    def on_simulate_clicked(self):
        if self.controller.state("finished_planning"):
            self.set_state("executing")
            
    # ___ _____________ _____________________________
       
    def set_state(self, new_state):
        self.controller.state = new_state
        self.controller.stateChanged.emit(new_state)
        print(f"[Controller] State changed: {new_state}")
    
    def finish_drawing(self):
        if self.start_pose is None or self.goal_pose is None:
            print("[Controller] BŁĄD: Brak startu lub celu!")
            return False
        
        print("[Controller] Ukończono rysowanie.")
        print(f"   Start: {self.start_pose}")
        print(f"   Koniec: {self.goal_pose}")
        self.set_state("planning")
        return True

    @QtCore.pyqtSlot(str)
    def ui_on_state_changed(self, state):
        """Aktualizuj UI na podstawie stanu"""
        state_text = {
            "drawing": "Stan: Rysowanie mapy",
            "planning": "Stan: Planowanie ścieżki...",
            "finished_planning": "Stan: Znaleziono ścieżkę!..",
            "executing": "Stan: Wykonywanie ścieżki..."
        }
        self.status_label.setText(state_text.get(state, "Stan: Nieznany"))
            
class PlanningWorker(QtCore.QObject):

    stateData = QtCore.pyqtSignal(str)
    pathData = QtCore.pyqtSignal(object) 
    finished = QtCore.pyqtSignal(bool)

    def __init__(self,controller:VisController):
        super().__init__() 
        self.controller = controller
        self.stateData.connect(self.controller.stateChanged)
        self.pathData.connect(self.controller.pathUpdated)
        

    @QtCore.pyqtSlot()
    def run(self):
        if self.controller.state == "planning":
            print("[PlannerWorker] Mapa gotowa, zaczynam planowanie ścieżki...")
            grid = NewOccupancyGrid()
            grid.setup_obstacles(self.controller.obstacles)
            start = self.controller.start_pose
            goal = self.controller.goal_pose

            planner = NewPlanner(self.controller)
            path = planner.hybrid_a_star_planning(start,goal,grid)
            
            
            if path:
                self.pathData.emit(path)
                self.stateData.emit("finished_planning")
                self.finished.emit(True)
            else:
                print("[PlannerWorker] Nie znaleziono ścieżki.")
                self.finished.emit(False)
        else:
            self.finished.emit(False)     

if __name__ == "__main__":
    # TEST REEDS-SHEPP
    """
    start = (0.0,0.0,0.0)
    goal = [(40.0,-60.0,-np.pi/4),(20.0,-10.0,-np.pi/4),(5.0,-80.0,-np.pi/2),(-20.0,25.0,-np.pi/6),
            (-80.0,100.0,-np.pi/4),(11.0,5.0,-np.pi/3),(3.0,10.0,-np.pi*3),(-10.0,5.0,-np.pi)]
    wheelbase = 2.995
    max_delta = 0.5
    radius = wheelbase/np.tan(max_delta)
    t0 = time.time()
    for _ in range(1000):
        for g in goal:
            s = reeds_shepp.path_sample(start,g,radius,step_size=0.1)
            l = reeds_shepp.path_length(start,g,radius)
            t = reeds_shepp.path_type(start,g,radius)

    tt = time.time()
    print(f"Czas: {tt-t0}")
    """
    # CZĘŚĆ DLA QT I HYBRID-A*
    
    app = pg.QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    

    sys.exit(app.exec())

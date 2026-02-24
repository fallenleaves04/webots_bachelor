import numpy as np
import cv2
#from ultralytics import YOLO
import math

"""
W tym pliku są pomocnicze funkcje, które zajmowały niepotrzebne miejsce
w pliku kontrolera. Odpowiadają za narysowanie brył i punktów pochodzących
ze stereowizji i z YOLO
"""

def calculate_intrinsic_matrix(width, height, fov_rad,fisheye=False):
    """
    Policz macierz kamery. W Webots nie ma zniekształćeń, tak że po prostu
    na podstawie parametrów obrazu i poziomego kątu widzenia
    """
    if not fisheye:
        fx = (width / 2) / math.tan(fov_rad / 2)
    else:
        fx = (width/2)/(fov_rad/2)
    fy = fx
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ],dtype=np.float32)

    return K

"""
Dwie poniższe funkcje są do zbudowania pozy kamery lub szachownicy
w postaci macierzy przekształcenia jednorodnego.
"""
def build_homogeneous_transform(R:np.ndarray, t:np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def build_pose_matrix(position:np.ndarray, yaw_deg):

    yaw_rad = np.deg2rad(yaw_deg)
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


"""
Funkcja pixel_to_world robi to, co wskazuje jej nazwa -
przelicza piksel na obrazie kamery do współrzędnych globalnych
na podstawie parametrów wewnętrznych i zewnętrznych kamery.
Parametry zewnętrzne są określone macierzą Tk_s -
położenie kamery w układzie samochodu; przekształca punkt w układzie kamery na punkt w układzie samochodu.
"""
def pixel_to_world(u, v, K:np.ndarray, Tk_s:np.ndarray):
    pixel = np.array([u, v, 1.0])  # Piksel w przestrzeni obrazu (homogeniczny)
    ray_camera = np.linalg.inv(K) @ pixel
    ray_camera = ray_camera / np.linalg.norm(ray_camera)  # Normalizowanie

    ray_world = Tk_s[:3, :3] @ ray_camera # a vector from the origin of camera frame transformed to the car frame
    camera_position = Tk_s[:3, 3] # przesunięcie kamery względem układu samochodu

    if ray_world[2] == 0:
        return None  # Promień równoległy do ziemi, brak przecięcia

    t = -camera_position[2] / ray_world[2] # 0 = camera_position[2] + t * ray_world[2] 
    point_on_ground = camera_position + t * ray_world

    return point_on_ground  # Punkt na ziemi (X, Y, 0)

"""
Przelicza punkty z układu globalnego 3D na obraz kamery 2D.
Posługuje się macierzą kamery oraz jej połozeniem
w układzie globalnym (samochodu).
"""
def project_points_world_to_image(points_world, Tk_s:np.ndarray, K:np.ndarray): # Tk_s - kamera w układzie samochodu
    projected_points = []

    # Inverse -> Camera <- World
    Ts_k = np.linalg.inv(Tk_s)

    for point_s in points_world:
        point_h = np.append(point_s, 1)  # homogeneous
        point_c = Ts_k @ point_h # P_k = Ts_k @ P_s
        Xc, Yc, Zc = point_c[:3]

        if Zc <= 0:
            continue  # behind camera

        # Project
        p_image = K @ np.array([Xc, Yc, Zc])
        u = p_image[0] / p_image[2]
        v = p_image[1] / p_image[2]
        projected_points.append((int(u), int(v)))

    return projected_points

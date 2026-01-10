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
Parametry zewnętrzne są określone macierzą T_center_to_camera -
położenie kamery w układzie samochodu.
"""
def pixel_to_world(u, v, K:np.ndarray, T_center_to_camera:np.ndarray):
    pixel = np.array([u, v, 1.0])  # Piksel w przestrzeni obrazu (homogeniczny)
    ray_camera = np.linalg.inv(K) @ pixel
    ray_camera = ray_camera / np.linalg.norm(ray_camera)  # Normalizowanie

    ray_world = T_center_to_camera[:3, :3] @ ray_camera
    camera_position = T_center_to_camera[:3, 3]

    if ray_world[2] == 0:
        return None  # Promień równoległy do ziemi, brak przecięcia

    t = -camera_position[2] / ray_world[2]
    point_on_ground = camera_position + t * ray_world

    return point_on_ground  # Punkt na ziemi (X, Y, 0)

"""
Przelicza punkty z układu globalnego 3D na obraz kamery 2D.
Posługuje się macierzą kamery oraz jej połozeniem
w układzie globalnym (samochodu).
"""
def project_points_world_to_image(points_world, T_world_to_camera:np.ndarray, K:np.ndarray):
    projected_points = []

    # Inverse -> Camera <- World
    T_camera_to_world = np.linalg.inv(T_world_to_camera)

    for point in points_world:
        point_h = np.append(point, 1)  # homogeneous
        point_in_camera = T_camera_to_world @ point_h
        Xc, Yc, Zc = point_in_camera[:3]

        if Zc <= 0:
            continue  # behind camera

        # Project
        p_image = K @ np.array([Xc, Yc, Zc])
        u = p_image[0] / p_image[2]
        v = p_image[1] / p_image[2]
        projected_points.append((int(u), int(v)))

    return projected_points

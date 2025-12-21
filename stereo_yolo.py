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
Poniższa funkcja jest dla przeliczenia punktu na płaszczyznie
drogi zgodnie z położeniem kamery w układzie samochodu -
przelicza kliknięty punkt w pikselach na punkt
w metrach odnośnie samochodu.
"""
def get_click_position(event, x, y, flags, param):
    global click_position,image

    T_center_to_camera,K = param
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        print(f"Kliknięto na pozycji: {click_position}")

        # Rysowanie punktu na obrazie
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("kamerka", image)
        point_on_ground = pixel_to_world(x, y, K, T_center_to_camera)
        print("Punkt na ziemi w układzie BEV:", point_on_ground)

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

click_position = None
global image

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


"""
Automatyczne utworzenie punktów podstawy górnej oraz dolnej
bryły o danym punkcie zaczepienia i ustalonych wymiarach.
Pierwsza próba wizualizacji brył na obiektach z YOLO.
"""
def create_3d_box(anchor_point, side,length=4.5, width=1.8, height=1.8):
    x, y, z = anchor_point

    dx = length
    dy = width

    if side == 'right':
        base = [
            [x, y, 0],
            [x, y+dy, 0],
            [x + dx, y+dy, 0],
            [x+dx, y, 0]
        ]
    elif side == 'left':
        base = [
            [x, y, 0],
            [x , y-dy, 0],
            [x + dx, y - dy, 0],
            [x +dx, y, 0]
        ]

    top = [[px, py, height] for (px, py, _) in base]
    return base + top  # 8 punktów

"""
Funkcja do oszacowania odległości w Y w układzie samochodu i
wyznaczenia w jakiej stronie znajduje się obiekt względem kamery.
Do starego podejścia z sztywnymi bryłami YOLO
"""
def classify_object_position_and_anchor(bbox, K, T_center_to_camera,camera_name):
    x, y, w, h = bbox

    # Dolny lewy punkt (lewa strona)
    pt_left = pixel_to_world(x, y + h, K, T_center_to_camera)

    # Dolny prawy punkt (prawa strona)
    pt_right = pixel_to_world(x + w, y + h, K, T_center_to_camera)
    # Jeśli którykolwiek z punktów jest None – pomijamy
    if pt_left is None or pt_right is None:
        return None, None

    if pt_right[1] < 0:  # ujemne Y → prawa strona
        anchor = pt_right
        side = "right"
    else:
        anchor = pt_left
        side = "left"

    if camera_name == "camera_front_top" or camera_name == "camera_rear":
        anchor = anchor
        side = side
    elif camera_name == "camera_right_pillar":
        anchor = pt_right
        side = "right"
    elif camera_name == "camera_left_pillar":
        anchor = pt_left
        side = "left"
    elif camera_name == "camera_right_fender":
        anchor = pt_left
        side = "left"
    elif camera_name == "camera_left_fender":
        anchor = pt_right
        side = "right"


    return anchor, side
"""
Pomocnicza funkcja, która nie pozwala na rysowanie
punktów za obrazem (dla pominięcia błędów).
Działa tylko w sprzężeniu z pozostałymi funkcjami,
inaczej nie ma sensu.
"""
def safe_line(img, pts, i, j, color, thickness=2):
    if i < len(pts) and j < len(pts):
        cv2.line(img, pts[i], pts[j], color, thickness)


def disp_to_cam3D(u, v, d, K, B):
    if d <= 0:
        return None  # brak danych

    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    Z = f * B / d
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    return np.array([X, Y, Z])  # w układzie kamery


def get_valid_disparity(disp_map, x, y, window=30):
        h, w = disp_map.shape
        x0 = max(0, x - window)
        x1 = min(w, x + window + 1)
        y0 = max(0, y - window)
        y1 = min(h, y + window + 1)
        patch = disp_map[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return np.median(valid)

def points_from_mask_to_3D(mask_resized, filtered_disp, K, baseline, T_cam_to_car):
    """
    Given a segmentation mask and filtered disparity map, compute 3D coordinates of the left and right extreme points of the object.

    Args:
        mask_resized (np.ndarray): Binary mask of shape (H, W)
        filtered_disp (np.ndarray): Disparity map (float32) of shape (H, W)
        K (np.ndarray): Intrinsic matrix of the right (or left) camera
        baseline (float): Stereo baseline in meters (e.g., 0.03 m)
        T_cam_to_car (np.ndarray): 4x4 transformation matrix from camera to car frame

    Returns:
        tuple: Two 3D points (np.ndarray) in car coordinate frame or (None, None) if no points found
    """

    # Step 1: Find leftmost and rightmost mask points

    ys, xs = np.where(mask_resized > 0)
    if len(xs) == 0:
        return None, None

    left_x = np.min(xs)
    right_x = np.max(xs)

    left_y = int(np.median(ys[xs == left_x]))
    right_y = int(np.median(ys[xs == right_x]))


    
    # Step 2: Get disparity at these points

    disp_left = get_valid_disparity(filtered_disp, left_x, left_y)
    disp_right = get_valid_disparity(filtered_disp, right_x, right_y)

    if disp_left is None or disp_right is None:
        return None, None
    if disp_left <= 0 or disp_right <= 0:
        return None, None

    # Step 3: Reproject using pinhole stereo model
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # Point 1 (left)
    Z1 = f * baseline / disp_left
    X1 = (left_x - cx) * Z1 / f
    Y1 = (left_y - cy) * Z1 / f

    point_cam1 = np.array([X1, Y1, Z1, 1.0])

    # Point 2 (right)
    Z2 = f * baseline / disp_right
    X2 = (right_x - cx) * Z2 / f
    Y2 = (right_y - cy) * Z2 / f

    point_cam2 = np.array([X2, Y2, Z2, 1.0])

    # Step 4: Transform to car frame
    point_car1 = T_cam_to_car @ point_cam1
    point_car2 = T_cam_to_car @ point_cam2

    return point_car1[:3], point_car2[:3]

# DLA ODNALEZIENIA KRAWĘŻNIKA, SKOPIOWANE Z https://github.com/Arun-purakkatt/medium_repo/blob/main/road_lane_detection%20(1).py

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            if len(line) == 1:  
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)),
                    (255, 0, 0), 5)
    return lines_image

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        #print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    #print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

#TO NIŻEJ SKOPIOWAĆ DO PĘTLI MAIN W SEKCJI GDZIE JEST
#INTERVAL PRZETWARZANIA OBRAZÓW

"""

T_center_to_camera = front_top_T
name_right = "camera_front_top"
name_left = "camera_front_top_add"

# Obiekt stereo matcher
stereo_left = cv2.StereoSGBM_create(minDisparity=0,
numDisparities=32,
blockSize=20,
disp12MaxDiff=1,
uniquenessRatio=10,
speckleWindowSize=100,
speckleRange=8)
# matcher dla prawego obrazu - trzeba użyć createRightMatcher z ximgproc
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)


name_right = "camera_front_top"
name_left = "camera_front_top_add"
img_right = names_images[name_right]
img_left = names_images[name_left]
right_copy = img_right.copy()
results = model(img_right,half=True,device = 0,classes = [2,5,7,10],conf=0.6,verbose=False,imgsz=(1280,960))
if results and results[0].masks is not None:

K_right = cam_matrices[name_right]
K_left = cam_matrices[name_left]
f_left = K_left[0][0]
f_right = K_right[0][0]
# Zamień na grayscale

grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)


#TUTAJ DALEJ ODFILTROWANE DISPARITY
# oblicz disparity z lewej i prawej kamery
disp_left = stereo_left.compute(grayL, grayR).astype(np.float32) / 16.0
disp_right = stereo_right.compute(grayR, grayL).astype(np.float32) / 16.0

# utwórz filtr WLS
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.9)

# filtruj disparity
filtered_disp = wls_filter.filter(disp_left, grayL, None, disp_right)

disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.nan_to_num(disp_vis, nan=0.0, posinf=0.0, neginf=0.0)
disp_vis = np.uint8(disp_vis)
cv2.namedWindow("Disparity WLS filtered",cv2.WINDOW_NORMAL)
cv2.imshow("Disparity WLS filtered", disp_vis)




#annotated_frame = results[0].plot()
masks = results[0].masks.data.cpu().numpy()  # shape: (num_detections, H, W)
orig_h, orig_w = img_right.shape[:2]

for i, mask in enumerate(masks):
    # Resize do rozmiaru obrazu
    mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Kolor losowy
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

    # Nałóż maskę
    colored = np.zeros_like(right_copy, dtype=np.uint8)
    for c in range(3):
        colored[:, :, c] = color[c] * mask_resized

    # Przezroczyste nałożenie
    alpha = 0.6
    right_copy = cv2.addWeighted(right_copy, 1.0, colored, alpha, 0)

    filtered_disp_clean = np.nan_to_num(filtered_disp, nan=0.0, posinf=0.0, neginf=0.0)
    disparity_masked = filtered_disp_clean * mask_resized

    # Znajdź indeks punktu z największą disparity (czyli najmniejszą odległością)
    # W masce disparity może być 0 tam gdzie brak danych, więc pomijamy
    # Pobierz disparity tylko w masce i >0
    valid_disparities = disparity_masked[(mask_resized > 0) & (disparity_masked > 0)]

    if len(valid_disparities) == 0:
        continue

    mean_disp = valid_disparities.mean()
    depth_m = f_right * 0.03 / mean_disp




    p1_3d, p2_3d = sy.points_from_mask_to_3D(mask_resized, filtered_disp, K_right, 0.03, T_center_to_camera)

    h, w = right_copy.shape[:2]
    if p1_3d is not None and p2_3d is not None:
        #print(f"Punkt 1: {p1_3d}")
        #print(f"Punkt 2: {p2_3d}")


        p1_3d = np.append(p1_3d, 1.0)  # -> [X, Y, Z, 1]

        p1_3d = p1_3d[:3]
        p1_3d[2] = 0
        p2_3d = np.append(p2_3d, 1.0)  # -> [X, Y, Z, 1]

        p2_3d = p2_3d[:3]
        p2_3d[2]=0


        # Rzut na obraz
        pts = sy.project_points_world_to_image([p1_3d,p2_3d], T_center_to_camera, K_right)


        (u1, v1), (u2, v2) = pts[0], pts[1]



        color_tuple = tuple(color.tolist())
        if 0 <= u1 < w and 0 <= v1 < h:
            cv2.circle(right_copy, (u1, v1), 6, color_tuple, 5)
            cv2.putText(right_copy, f"PT1", (u1 + 5, v1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

        if 0 <= u2 < w and 0 <= v2 < h:
            cv2.circle(right_copy, (u2, v2), 6, color_tuple, 5)
            cv2.putText(right_copy, "PT2", (u2+5, v2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

        x1 = p1_3d[0]; x2 = p2_3d[0]
        x_min, x_max = min(x1, x2), max(x1, x2)


        ranges.append((x_min, x_max, i))


            #print("-----------------------------")
            #print("_____________________________")

        ys, xs = np.where(mask_resized > 0)
        if len(xs) == 0:
            return
        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))

        cv2.putText(right_copy, str(i), (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)






cv2.namedWindow("Maski z punktami najblizszymi",cv2.WINDOW_NORMAL)
cv2.imshow("Maski z punktami najblizszymi", right_copy)
"""

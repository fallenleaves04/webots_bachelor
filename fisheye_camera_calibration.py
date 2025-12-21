import cv2 as cv
import numpy as np
import os
import sys

CAMERA_HEIGHT=1080
CAMERA_WIDTH=1920

import visualise as vis


def average_chessboard_size(corners, pattern_size):
    """
    Funkcja instrumentalna licząca wymiary kwadratów szachownicy.
    Wykorzystywana przez funkcję "solve_chess_size".
    Zlicza wszystkie narożniki i na podstawie tego wylicza średnią.
    """
    cols, rows = pattern_size
    corners = corners.reshape(-1, 2)

    horizontal_lengths = []
    vertical_lengths = []

    for row in range(rows):
        for col in range(cols - 1):
            idx1 = row * cols + col
            idx2 = idx1 + 1
            dist = np.linalg.norm(corners[idx1] - corners[idx2])
            horizontal_lengths.append(dist)

    for row in range(rows - 1):
        for col in range(cols):
            idx1 = row * cols + col
            idx2 = idx1 + cols
            dist = np.linalg.norm(corners[idx1] - corners[idx2])
            vertical_lengths.append(dist)

    width_lengths = []
    for row in range(rows):
        start_idx = row * cols
        end_idx = start_idx + cols - 1
        width = np.linalg.norm(corners[start_idx] - corners[end_idx])
        width_lengths.append(width)

    height_lengths = []
    for col in range(cols):
        start_idx = col
        end_idx = start_idx + (rows - 1) * cols
        height = np.linalg.norm(corners[start_idx] - corners[end_idx])
        height_lengths.append(height)

    avg_width = np.mean(horizontal_lengths)
    avg_height = np.mean(vertical_lengths)
    total_width = np.mean(width_lengths)
    total_height = np.mean(height_lengths)

    return avg_width, avg_height, total_width, total_height

def solve_camera_pose(image, pattern_size, K, camera_name,show=True):
    """
    Zwraca pozę kamery, R i tvec, później można zbudować macierz jednorodną.
    Przyjmuje obraz z kamery, rozmiar szachownicy, jej parametry wewnętrzne.
    Nazwa camera_name do wizualizacji i debugowania.

    wszystkie punkty wymiarowania szachownic są liczone
    od lewego górnego odnalezionego przez algorytm - w metrach
    przeliczane

    """
    # Define 3D object points (0,0,0), (1,0,0), ..., in chessboard frame
    objp = np.zeros((4,2),dtype=np.float32)
    if camera_name == "camera_front_bumper":
        objp = np.array([[0.0,0.0],[0.0,-0.25*6],[-0.25*4,-0.25*6],[-0.25*4,0.0]]).astype(np.float32)
        #objp = np.array([[0.0,0.0],[0.25*3,0.0],[0.25*3,0.25*2],[0.0,0.25*2]]).astype(np.float32)
    elif camera_name == "camera_rear":
        objp = np.array([[0.0,0.0],[0,-0.25*6],[-0.25*4,-0.25*6],[-0.25*4,0.0]]).astype(np.float32)
    elif camera_name == "camera_front_left" or camera_name == "camera_front_right":
        objp = np.array([[0.0,0.0],[0.0,-0.25*3],[-0.25*2,-0.25*3],[-0.25*2,0.0]]).astype(np.float32)
    elif camera_name == "camera_left_mirror":
        objp = np.array([[0.0,0.0],[0.25*6,0.0],[0.25*6,-0.25*4],[0.0,-0.25*4]]).astype(np.float32)
    elif camera_name == "camera_right_mirror":
        objp = np.array([[0.0,0.0],[-0.25*6,0.0],[-0.25*6,0.25*4],[0.0,0.25*4]]).astype(np.float32)
    #elif camera_name == "camera_front_bumper":
        #objp = np.array([[0.0,0.0],[0.25*3,0.0],[0.25*3,0.25*2],[0.0,0.25*2]]).astype(np.float32)

    objp_fixed = np.zeros((4, 3), dtype=np.float32)
    objp_fixed[:, :2] = objp   #dodajemy Z=0


    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size)

    if not ret:
        print(f"[{camera_name}] chessboard not found.")
        return None, None

    # Subpixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 120, 0.00001)
    corners_refined = cv.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)


    # Assume no distortion
    distCoeffs = np.zeros((4,1),dtype=np.float32)
    cols, rows = pattern_size

    top_left = corners_refined[0]
    top_right = corners_refined[cols - 1]
    bottom_left = corners_refined[(rows - 1) * cols]
    bottom_right = corners_refined[rows * cols - 1]

    chessboard_corners_4 = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)
    # Solve PnP
    success, rvec, tvec = cv.fisheye.solvePnP(objp_fixed, chessboard_corners_4, K, distCoeffs)

    if not success:
        print(f"[{camera_name}] solvePnP failed.")
        return None, None

    #R, _ = cv.Rodrigues(rvec)

    #print(f"\n[{camera_name}] === Camera Pose ===")
    #print("Rotation matrix:\n", R)
    #print("Translation vector (in meters):\n", tvec.ravel())

    if show:


        
        #  oś X (czerwona), Y (zielona), Z (niebieska)
        axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,0.5]]).reshape(-1, 1, 3)  # 20 cm osie
        imgpts, _ = cv.fisheye.projectPoints(axis, rvec, tvec, K, distCoeffs)

        origin = tuple(chessboard_corners_4[0].ravel().astype(int))
        vis = cv.line(image, origin, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)  # X
        vis = cv.line(image, origin, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)  # Y
        vis = cv.line(image, origin, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)  # Z
        vis = cv.drawChessboardCorners(image, pattern_size, corners_refined, ret)
        cv.namedWindow(f"Chessboard - {camera_name}",cv.WINDOW_NORMAL)
        cv.imshow(f"Chessboard - {camera_name}", vis)

    """
    #TO TUTAJ KOD DO ODNALEZIENIA KAMERY W ŚWIECIE WZGLĘDEM ŚRODKA TYLNEJ OSI SAMOCHODU
    #SKOPIOWAĆ DO PĘTLI GŁÓWNEJ PO LIŚCIE NAMES_IMAGES
                   if name == "camera_front_bumper_wide":
                       chessboard_position = np.array([3.66-0.425+0.4,0.6,0.0]).astype(np.float32)
                   elif name == "camera_rear":
                       chessboard_position = np.array([-3.09-0.425-0.4,-0.6,0.0]).astype(np.float32)
                   elif name == "camera_left_fender":
                       chessboard_position = np.array([-0.2-0.425+0.6,-2.39+0.4,0.0]).astype(np.float32)
                       pattern_size = (5,7)
                   elif name == "camera_right_fender":
                       chessboard_position = np.array([-0.2-0.425+0.6,-2.39+0.4,0.0]).astype(np.float32)
                       pattern_size = (5,7)
                   elif name == "camera_left_pillar":
                       chessboard_position = np.array([-0.2-0.425+0.6,-2.39+0.4,0.0]).astype(np.float32)
                       pattern_size = (5,7)
                   elif name == "camera_right_pillar":
                       chessboard_position = np.array([-0.2-0.425-0.6,-2.39-0.4,0.0]).astype(np.float32)
                       pattern_size = (5,7)
                   elif name == "camera_front_top":
                       chessboard_position = np.array([2.97-0.425-0.4,-0.6,1.1085]).astype(np.float32)
                       pattern_size = (4,3)
                   chessboard_yaw = 0  # degrees
                   rvec,tvec = cc.solve_camera_pose(img,pattern_size,cam_matrices[name],name)
                   if rvec is not None and tvec is not None:
                       T_center_to_chessboard = build_pose_matrix(chessboard_position, chessboard_yaw)


                       #R, _ = cv.Rodrigues(rvec)
                       T_camera_to_chessboard = build_homogeneous_transform(rvec, tvec)

                       # Combine to get rear axle → camera
                       T_center_to_camera = T_center_to_chessboard @ np.linalg.inv(T_camera_to_chessboard)

                       print(f"[{name}] pose wrt rear axle (T_center_to_camera):\n", T_center_to_camera)
    cc.save_homo(T_rearaxle_to_camera,f"{name}_T_global")

    bbox_world = np.array([
             [-2.49, -0.6, 0],   # bottom front right
             [-2.49,0.6, 0],  # bottom front left
             [-1.69, 0.6, 0],  # bottom rear left
             [-1.69, -0.6, 0],   # bottom rear right
             [-2.49, -0.6, 0.2], # top front right
             [-2.49,0.6, 0.2],# top front left
             [-1.69, 0.6, 0.2],# top rear left
             [-1.69, -0.6, 0.2]  # top rear right
     ])

     K = cam_matrices[name]

     # Project bbox
     image_points = project_points_world_to_image(bbox_world, T_rearaxle_to_camera, K)



     # Draw bottom rectangle
     for i in range(4):
         pt1 = image_points[i]
         pt2 = image_points[(i + 1) % 4]
         cv2.line(image, pt1, pt2, (0, 255, 0), 2)

     # Draw top rectangle
     for i in range(4, 8):
         pt1 = image_points[i]
         pt2 = image_points[4 + (i + 1) % 4]
         cv2.line(image, pt1, pt2, (0, 0, 255), 2)

     # Draw vertical lines
     for i in range(4):
         pt1 = image_points[i]
         pt2 = image_points[i + 4]
         cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    cv2.namedWindow(f"Projected 3D BBox {name}",cv2.WINDOW_NORMAL)
    cv2.imshow(f"Projected 3D BBox {name}", image)
    """
    return rvec, tvec


def detect_apriltags(image,name):
    # 1. Konwersja na odcienie szarości
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 2. Wybór słownika (Musi pasować do tego w Webots!)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_25h9)
    parameters = cv.aruco.DetectorParameters()
    
    # Opcjonalnie: poprawa detekcji małych tagów
    parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

    # 3. Detekcja
    # corners: lista znalezionych narożników [(tl, tr, br, bl)]
    # ids: lista znalezionych ID
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        print(f"Znaleziono {len(ids)} tagów")
        
        # Wizualizacja (rysowanie ramek)
        vis_image = image.copy()
        vis_image = cv.cvtColor(vis_image, cv.COLOR_BGR2RGB)
        cv.aruco.drawDetectedMarkers(vis_image, corners, ids)
        cv.namedWindow(f"Detected Tags {name}",cv.WINDOW_NORMAL)
        cv.imshow(f"Detected Tags {name}", vis_image)
        #cv.imwrite("camera_front_right_apriltag.png",vis_image)
    else:
        print("Nie znaleziono tagów")
    return corners, ids

def estimate_tag_pose(corners, ids, K, D, tag_size_meters):
    
    if ids is None:
        return {}
        
    
    half_size = tag_size_meters / 2.0
    obj_points = np.array([
    [ half_size,  half_size, 0], # 0: Top-Left  
    [ half_size, -half_size, 0], # 1: Top-Right 
    [ -half_size, -half_size, 0], # 2: Bottom-Right
    [ -half_size,  half_size, 0]  # 3: Bottom-Left 
    ], dtype=np.float32)

    poses = {}

    for i in range(len(ids)):
        tag_corners_px = corners[i].reshape(4, 1, 2).astype(np.float32)
        obj_points = obj_points.reshape(4, 1, 3)
        # albo fisheye solvepnp
        success, rvec, tvec = cv.solvePnP(
            obj_points, 
            tag_corners_px, 
            K, 
            D, 
            #flags=cv.SOLVEPNP_IPPE_SQUARE 
        )
        
        if success:
            poses[ids[i][0]] = (rvec, tvec)
            
    return poses


def translate_images(img1, img2, homography):
    """
    Nakłada img2 na img1 używając tylko translacji (ignorując rotację i perspektywę) z homografii.
    Dynamicznie wyznacza minimalną kanwę zawierającą oba obrazy.
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Narożniki obu obrazów
    corners1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # Przesuń tylko translacją z homografii
    tx = homography[0, 2]
    ty = homography[1, 2]
    translation = np.array([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]], dtype=np.float32)

    translated_corners2 = cv.perspectiveTransform(corners2, translation)

    all_corners = np.concatenate((corners1, translated_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    warp_trans = np.array([[1, 0, translation_dist[0]],
                           [0, 1, translation_dist[1]],
                           [0, 0, 1]], dtype=np.float32)

    canvas = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

    # Umieść img1
    canvas[translation_dist[1]:translation_dist[1] + rows1,
           translation_dist[0]:translation_dist[0] + cols1] = img1

    # Umieść img2 przesunięty tylko o translację
    M = warp_trans @ translation
    img2_translated = cv.warpPerspective(img2, M, (x_max - x_min, y_max - y_min))

    # Nakładanie dla nieczarnych obszarów
    mask = img2_translated > 0
    canvas[mask] = img2_translated[mask]

    return canvas

def alt_warp_images(img1, img2, homography):
    """
    Prostuje img1 na img2 zamiast jak jest w "warp_images". Pomocna, jeżeli
    się chce odwrócić kolejność tworzenia homografii. Dodatkowa funkcja,
    rzadko wykorzystywana
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Narożniki
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    points1_transformed = cv.perspectiveTransform(points1, homography)

    all_points = np.concatenate((points1_transformed, points2), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
    translation = [-x_min, -y_min]

    # Translacja i rozmiar kanwy
    H_trans = np.array([[1, 0, translation[0]],
                        [0, 1, translation[1]],
                        [0, 0, 1]], dtype=np.float32)
    output_size = (x_max - x_min, y_max - y_min)

    # Przekształcenie img1
    warped_img1 = cv.warpPerspective(img1, H_trans @ homography, output_size)
    mask1 = cv.warpPerspective(np.ones((rows1, cols1), dtype=np.uint8) * 255, H_trans @ homography, output_size)

    # img2 jako tło
    shifted_img2 = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    mask2 = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    x_offset, y_offset = translation
    shifted_img2[y_offset:y_offset+rows2, x_offset:x_offset+cols2] = img2
    mask2[y_offset:y_offset+rows2, x_offset:x_offset+cols2] = 255

    # Blender
    blender = cv.detail_FeatherBlender()
    blender.prepare((0, 0, output_size[0], output_size[1]))
    blender.feed(shifted_img2.astype(np.int16), mask2, (0, 0))  # now base
    blender.feed(warped_img1.astype(np.int16), mask1, (0, 0))  # warped img1
    result, _ = blender.blend(None, None)

    return np.clip(result, 0, 255).astype(np.uint8)


def warp_images(img1, img2, homography):
    """
    Przekształca obrazy homografią i wykorzystuje feather-blender dla dopasowania na krańcach.
    Trochę dorobiona wersja z github repo 360ls/stitcher
    Nie jest najszybsza, ponieważ na CPU.
    Na jej podstawie przerobiono funkcję warp_and_blend_gpu w pliku visualise.py
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Narożniki
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    points2_transformed = cv.perspectiveTransform(points2, homography)

    all_points = np.concatenate((points1, points2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)
    translation = [-x_min, -y_min]

    # Translacja i rozmiar kanwy
    H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=np.float32)
    output_size = (x_max - x_min, y_max - y_min)

    # Przekształcenia
    warped_img2 = cv.warpPerspective(img2, H_trans @ homography, output_size)
    mask2 = cv.warpPerspective(np.ones((rows2, cols2), dtype=np.uint8) * 255, H_trans @ homography, output_size)

    shifted_img1 = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    mask1 = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    x_offset, y_offset = translation
    shifted_img1[y_offset:y_offset+rows1, x_offset:x_offset+cols1] = img1
    mask1[y_offset:y_offset+rows1, x_offset:x_offset+cols1] = 255

    # Blender
    blender = cv.detail_FeatherBlender()
    blender.prepare((0, 0, output_size[0], output_size[1]))
    blender.feed(shifted_img1.astype(np.int16), mask1, (0, 0))
    blender.feed(warped_img2.astype(np.int16), mask2, (0, 0))
    result, _ = blender.blend(None, None)

    #albo tutaj poniżej, jeżeli wystarczy po prostu uśrednić
    #result = mean_blend_1(shifted_img1,wapred_img2)
    return np.clip(result, 0, 255).astype(np.uint8)


def mean_blend_1(img1, img2, blend_width=15):
    """
    Funkcja pomocna przy zmieszaniu obrazów.
    Nie jest najlepszym rozwiązaniem, ale nie wymaga dużych zasobów CPU.
    """
    assert img1.shape == img2.shape
    h, w = img1.shape[:2]
    blended = np.zeros_like(img1)

    blended[:, :w - blend_width] = np.where(
        np.any(img1[:, :w - blend_width] >1, axis=2, keepdims=True),
        img1[:, :w - blend_width],
        img2[:, :w - blend_width]
    )

    # Blend pasma
    for i in range(blend_width):
        alpha = 0.2 * (1 - np.cos(np.pi * i / blend_width))  # przejście cosinusem
        col = w - blend_width + i

        pixel1 = img1[:, col].astype(np.float32)
        pixel2 = img2[:, col].astype(np.float32)

        valid1 = np.any(pixel1 > 1, axis=1, keepdims=True)
        valid2 = np.any(pixel2 > 1, axis=1, keepdims=True)

        blended[:, col] = np.where(
            valid1 & valid2,
            ((1 - alpha) * pixel1 + alpha * pixel2).astype(np.uint8),
            np.where(valid1, pixel1, pixel2)
        )

    return blended


def homography(img1, img2):
    """
    Homografia pomocna dla sklejania obrazów na podstawie punktów charakterystycznych.
    Jej wygoda w sprawie z widokiem z lotu ptaka jest wątpliwa.
    Zato wykorzystuje nieopatentowane rozwiązania typu ORB i BFMatcher.
    Dopasowanie obrazów może się polepszyć, jeżeli się odpowiednio
    zgra liczba nfeatures i ratio test (w warunkach gdzie porównuje się dystanse, często 0.77 - 0.8).
    """
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray1 = cv.GaussianBlur(gray1,(5,5),0)
    gray2 = cv.GaussianBlur(gray2,(5,5),0)

    orb = cv.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Nie znaleziono deskryptorów.")
        return None, None


    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.77* n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        print("Mało dopasowań.")
        return None, None


    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)

    if H is None:
        print("Nie policzono macierzy.")
        return None, None


    h, w = img2.shape[:2]
    warped_img = cv.warpPerspective(img1, H, (w, h))


    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv.imshow("Dopasowania", img_matches)
    cv.imshow("Wyprostowany obraz", warped_img)

    return H, warped_img


#-----------------------------------------------------------------------------------------------------------------------------------------

def gaussian_blur_cuda(image, ksize, sigma):
    """
    Funkcja do nałożenia gaussowskiego rozmycia.
    Jak w poprzedniej funkcji, najlepiej skopiować do innej funkcji,
    aby można było cały proces robić na gpu i nie przenosić danych zbytnio.
    """
    if len(image.shape) == 2:
        src_type = cv.CV_8UC1
    elif image.shape[2] == 3:
        src_type = cv.CV_8UC3
    else:
        raise ValueError("Obraz musi być 1- lub 3-kanałowy")

    gpu_img = cv.cuda_GpuMat()
    gpu_img.upload(image)

    gaussian_filter = cv.cuda.createGaussianFilter(srcType=src_type,
                                                   dstType=src_type,
                                                   ksize=ksize,
                                                   sigma1=sigma)

    gpu_blurred = gaussian_filter.apply(gpu_img)
    blurred = gpu_blurred.download()
    return blurred

def expand_bbox_fixed(x, y, w, h, margin, image_shape):
    """
    Funkcja pomocnicza do rozszerzenia prostokąta o jakiś margin.
    Pomocne dla zdefiniowania maski i robienia homografii -
    jeżeli np. chcemy usunąć z ROI (region of interest) narożniki jednej szachownicy,
    a znaleźć zamiast tego drugą
    """
    x_new = max(x - margin, 0)
    y_new = max(y - margin, 0)
    x2 = min(x + w + margin, image_shape[1])
    y2 = min(y + h + margin, image_shape[0])
    return x_new, y_new, x2 - x_new, y2 - y_new

def apply_mask_to_image(img, bbox):
    """
    Nałóż maskę aby tylko prostokąt był widoczny (bbox)
    """
    mask = np.zeros_like(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 255
    return mask

def mask_out_chessboard(img, bbox, margin=0):
    """
    Nakłada czarne piksele na szachownicę - albo dowolny bounding box.
    Zdefiniowane również opcjonalnie odstępem margin
    """
    x, y, w, h = bbox
    x, y, w, h = expand_bbox_fixed(x, y, w, h, margin, img.shape)
    out = img.copy()
    out[y:y+h, x:x+w] = 0
    return out
def compute_reprojection_error(H, src_points, dst_points):
    """
    Na podstawie homografii, docelowych i źródłowych punktów liczy
    błąd euklidesowy (średniokwadratowy) dla wyznaczonego przekształcenia.
    Pomocne przy wyłonieniu najlepiej dopasowanego wyprostowania.
    """
    # Project the source points using the homography
    projected_points = cv.perspectiveTransform(src_points, H)

    # Compute the L2 distance between projected and actual destination points
    errors = np.linalg.norm(projected_points - dst_points, axis=2)
    mean_error = np.mean(errors)

    return mean_error, errors

def chess_homography(img1, img2, pattern_size,margin=200):
    """
    Na podstawie znalezionej szachownicy na jednym obrazie
    oraz drugim pozwala wyznaczyć homografię. Wyposażona w zabezpieczenie przez znalezieniem
    innych szachownic po wykryciu, czyli
    mocuje się na jednej już znalezionej.
    Na wejście wchodzą dwa obrazy: źródłowy i docelowy; wymiary szachownicy; opcjonalnie
    odstęp dla zamaskowania obszaru wokół szachownicy.
    Jest przecyzyjniejszym sposobem, ponieważ nie musimy znać wymiarów szachownicy w rzeczywistym
    układzie.
    """

    gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    gray1 = gaussian_blur_cuda(gray1, (5, 5), 0)
    gray2 = gaussian_blur_cuda(gray2, (5, 5), 0)

    pattern_size_corrected = (pattern_size[0], pattern_size[1])

    ret1, corners1 = cv.findChessboardCorners(gray1, pattern_size_corrected,
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    ret2, corners2 = cv.findChessboardCorners(gray2, pattern_size_corrected,
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

    if not (ret1 and ret2):
       print("Nie udało się znaleźć narożników szachownicy na jednym z obrazów.")
       return None,None
    # Oblicz bounding box i rozszerz
    x1, y1, w1, h1 = cv.boundingRect(corners1)
    x2, y2, w2, h2 = cv.boundingRect(corners2)

    x1_crop, y1_crop, w1_crop, h1_crop = expand_bbox_fixed(x1, y1, w1, h1, margin, gray1.shape)
    x2_crop, y2_crop, w2_crop, h2_crop = expand_bbox_fixed(x2, y2, w2, h2, margin, gray2.shape)

    # Przytnij obrazy
    gray1_crop = gray1[y1_crop:y1_crop+h1_crop, x1_crop:x1_crop+w1_crop]
    gray2_crop = gray2[y2_crop:y2_crop+h2_crop, x2_crop:x2_crop+w2_crop]

    # Znajdź narożniki na przyciętych
    ret1, corners1 = cv.findChessboardCorners(gray1_crop, pattern_size)
    ret2, corners2 = cv.findChessboardCorners(gray2_crop, pattern_size)

    if not (ret1 and ret2):
        print("Nie znaleziono narożników po przycięciu")
        return None,None

    # Dodaj przesunięcie do narożników (od przycięcia)
    corners1 += np.array([[x1_crop, y1_crop]], dtype=np.float32)
    corners2 += np.array([[x2_crop, y2_crop]], dtype=np.float32)


    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    corners1 = cv.cornerSubPix(gray1, corners1, (21, 21), (-1, -1), criteria)
    corners2 = cv.cornerSubPix(gray2, corners2, (21, 21), (-1, -1), criteria)

    # homografia
    H, _ = cv.findHomography(corners1, corners2,cv.RANSAC,2.0)

    H = H.astype(np.float32)
    h, w = img2.shape[:2]

    warped_img = vis.warp_with_cuda(img1,H,"frrrr",h,w,stream=None,gpu=False)
    print(H)

    img1_draw = img1.copy()
    img2_draw = img2.copy()
    cv.drawChessboardCorners(img1_draw, pattern_size_corrected, corners1, ret1)
    cv.drawChessboardCorners(img2_draw, pattern_size_corrected, corners2, ret2)

    # Błąd reprojekcji
    reproj_error, per_point_errors = compute_reprojection_error(H, corners1, corners2)
    print(f"Średni błąd reprojekcji: {reproj_error:.3f}")
    # Pokaz obrazów
    cv.namedWindow("Image 1 - detected corners", cv.WINDOW_NORMAL)
    cv.imshow("Image 1 - detected corners", img1_draw)
    #cv.imwrite("img111.png",img1_draw)
    cv.namedWindow("Image 2 - detected corners", cv.WINDOW_NORMAL)
    cv.imshow("Image 2 - detected corners", img2_draw)
    #cv.imwrite("img222.png",img2_draw)
    cv.namedWindow("Warped image 1 to image 2", cv.WINDOW_NORMAL)
    cv.imshow("Warped image 1 to image 2", cv.cvtColor(warped_img,cv.COLOR_BGR2RGB))
    #cv.imwrite("img333.png",cv.cvtColor(warped_img,cv.COLOR_BGR2RGB))

    return H, warped_img

def chess_homography_multiple_boards(img1, img2, pattern_size,second_patt_size,margin=200):
    """
    Druga funkcja pozwalająca de-fakto na wyznaczenie homografii na podstawie dwóch szachownic.
    Jedną musi znaleźć, zamaskować ją, a później na zamaskowanym obrazie znaleźć druga.
    Ilość punktów do dopasowania może się powiększyć nawet więcej niż 2 razy.
    Najlepiej wykorzystywać różne szachownice w różnych obszarach obu obrazów.

    """
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    #gray1 = gaussian_blur_cuda(gray1, (5, 5), 0)
    #gray2 = gaussian_blur_cuda(gray2, (5, 5), 0)

    pattern_size_corrected = (pattern_size[0], pattern_size[1])
    first = True
    ret1, corners1 = cv.findChessboardCorners(gray1, pattern_size_corrected,
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    ret2, corners2 = cv.findChessboardCorners(gray2, pattern_size_corrected,
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

    if not (ret1 and ret2):
       print("Nie udało się znaleźć narożników szachownicy na jednym z obrazów.")
       return None,None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    corners1 = cv.cornerSubPix(gray1, corners1, (21, 21), (-1, -1), criteria)
    corners2 = cv.cornerSubPix(gray2, corners2, (21, 21), (-1, -1), criteria)
    # Przytnij obraz i zamaskuj
    x1, y1, w1, h1 = cv.boundingRect(corners1)
    x2, y2, w2, h2 = cv.boundingRect(corners2)

    x1_crop, y1_crop, w1_crop, h1_crop = expand_bbox_fixed(x1, y1, w1, h1, margin, gray1.shape)
    x2_crop, y2_crop, w2_crop, h2_crop = expand_bbox_fixed(x2, y2, w2, h2, margin, gray2.shape)

    img1_masked = mask_out_chessboard(gray1.copy(), (x1_crop, y1_crop, w1_crop, h1_crop), margin=100)
    img2_masked = mask_out_chessboard(gray2.copy(), (x2_crop, y2_crop, w2_crop, h2_crop), margin=100)
    pattern_size_corrected = second_patt_size

    ret1_, corners1_ = cv.findChessboardCorners(img1_masked, pattern_size_corrected,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    ret2_, corners2_ = cv.findChessboardCorners(img2_masked, pattern_size_corrected,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    corners1_combined = []
    corners2_combined = []
    # Jeśli druga szachownica się różni – połącz dane
    # po subpix i boundingRect/ maskowaniu…
    if ret1_ and ret2_ and not np.array_equal(corners1, corners1_) and not np.array_equal(corners2, corners2_):
        corners1_combined = np.vstack((corners1,  corners1_))
        corners2_combined = np.vstack((corners2,  corners2_))
        corners1_combined = cv.cornerSubPix(gray1, corners1_combined, (21, 21), (-1, -1), criteria)
        corners2_combined = cv.cornerSubPix(gray2, corners2_combined, (21, 21), (-1, -1), criteria)
    else:
        # jeśli nie ma drugiej — użyj tylko pierwszej
        print("Nie znaleziono drugiej szachownicy")
        corners1_combined = corners1.copy()
        corners2_combined = corners2.copy()

    # Homografia
    H, _ = cv.findHomography(
    corners1_combined,
    corners2_combined,
    method=cv.RANSAC,
    ransacReprojThreshold=6.0,
    )

    H = H.astype(np.float32)
    h, w = img2.shape[:2]

    warped_img = vis.warp_with_cuda(img1,H,"frrrr",h,w,stream=None,gpu=False)
    print(H)

    img1_draw = img1.copy()
    img2_draw = img2.copy()
    # Narysuj pierwszy zestaw
    cv.drawChessboardCorners(img1_draw, pattern_size, corners1[:pattern_size[0]*pattern_size[1]], ret1)
    cv.drawChessboardCorners(img2_draw, pattern_size, corners2[:pattern_size[0]*pattern_size[1]], ret2)

    # Narysuj drugi zestaw jeśli jest
    if ret1_ and ret2_:
        cv.drawChessboardCorners(img1_draw, pattern_size_corrected, corners1_[...], ret1_)
        cv.drawChessboardCorners(img2_draw, pattern_size_corrected, corners2_[...], ret2_)

    # Błąd reprojekcji
    reproj_error, per_point_errors = compute_reprojection_error(H, corners1, corners2)
    print(f"Średni błąd reprojekcji: {reproj_error:.3f}")
    # Pokaz obrazów
    cv.namedWindow("Image 1 - detected corners", cv.WINDOW_NORMAL)
    cv.imshow("Image 1 - detected corners", img1_draw)

    cv.namedWindow("Image 2 - detected corners", cv.WINDOW_NORMAL)
    cv.imshow("Image 2 - detected corners", img2_draw)

    cv.namedWindow("Warped image 1 to image 2", cv.WINDOW_NORMAL)
    cv.imshow("Warped image 1 to image 2", warped_img)

    return H, warped_img


def save_homo(homography, homography_filename):
    """
    Zapis homografii do pliku. Będzie zapisane w tym samym folderze, co i plik kontrolera
    (najlepiej jeżeli wszystkie pliki .py są w tym samym folderze).
    """
    np.save(homography_filename, homography)
    print(f"Matrix saved as {homography_filename}")

def solve_chess(image, name,pattern_size,pattern_size2=None,second=False):
    """
    Policz rozmiar szachownicy i zwróc punkty. Pozwala na drugą szachownicę równiez.
    Pomocne dla policzenia pozy szachownicy.
    """

    #h,w = shape #rozmiar docelowego obrazu
    #warped = warp_with_cuda(image, homography, name, h, w, pattern_size)
    #w_m,h_m = chess_real_size + cv.CALIB_CB_NORMALIZE_IMAGE
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size,
                                              cv.CALIB_CB_ADAPTIVE_THRESH )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    if ret:
        corners_new = cv.cornerSubPix(gray, corners, (30, 30), (-1, -1), criteria)
        #print(corners)

        img_draw = cv.drawChessboardCorners(image,pattern_size, corners_new, ret)
        cols, rows = pattern_size

        top_left = corners_new[0]
        top_right = corners_new[cols - 1]
        bottom_left = corners_new[(rows - 1) * cols]
        bottom_right = corners_new[rows * cols - 1]

        corners_4 = np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ], dtype=np.float32)

        if second:
            x1, y1, w1, h1 = cv.boundingRect(corners_4)
            x1_crop, y1_crop, w1_crop, h1_crop = expand_bbox_fixed(x1, y1, w1, h1, 200, gray.shape)
            img_masked = mask_out_chessboard(gray.copy(), (x1_crop, y1_crop, w1_crop, h1_crop), margin=100)
            pattern_size_corrected = pattern_size2 #TUTAJ ZMIENIAMY
            # Spróbuj znaleźć kolejną szachownicę
            ret_, corners_ = cv.findChessboardCorners(img_masked, pattern_size_corrected,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

            corners_combined = []
            if ret_ and (not np.array_equal(corners_new, corners_)):
                #narożniki dla drugiej szachownicy
                corners_ = cv.cornerSubPix(img_masked, corners_, (21, 21), (-1, -1), criteria)
                cols, rows = pattern_size_corrected

                top_left = corners_[0]
                top_right = corners_[cols - 1]
                bottom_left = corners_[(rows - 1) * cols]
                bottom_right = corners_[rows * cols - 1]

                corners_4_ = np.array([
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ], dtype=np.float32)

                #dodaj pierwsze i drugie do siebie i narysuj punkty też
                corners_combined = np.concatenate((corners_4, corners_4_), axis=0)
                img2_draw = cv.drawChessboardCorners(img_draw,pattern_size_corrected, corners_, ret_)
                top_left_min_ = min(corners_4_, key=lambda x: (x[0][0], x[0][1]))  # Pierwszy element to współrzędne x,y
                # narysuj punkt skrajny, od którego liczy się wszystkie narożniki - dla debugowania kolejności
                cv.circle(img2_draw, (int(top_left_min_[0][0]), int(top_left_min_[0][1])), 20, (0, 0, 255), -1)
                cv.namedWindow(f"detected corners {name}", cv.WINDOW_NORMAL)
                cv.imshow(f"detected corners {name}", cv.cvtColor(img2_draw,cv.COLOR_BGR2RGB))

                return corners_combined
            else:
                print(f"Nie znaleziono drugiej szachownicy lub jest identyczna {name}")
                return corners_4
        cv.namedWindow(f"detected corners {name}", cv.WINDOW_NORMAL)
        cv.imshow(f"detected corners {name}", cv.cvtColor(img_draw,cv.COLOR_BGR2RGB))
        return corners_4


    else:
         print(f"Nie znaleziono szachownicy {name}")
         return None

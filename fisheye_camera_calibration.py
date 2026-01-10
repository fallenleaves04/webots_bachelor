import cv2 as cv
import numpy as np

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
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
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
    [ -half_size,  -half_size, 0], # 0: Top-Left  
    [ -half_size, half_size, 0], # 1: Top-Right 
    [ half_size, half_size, 0], # 2: Bottom-Right
    [ half_size,  -half_size, 0]  # 3: Bottom-Left 
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



def save_homo(homography, homography_filename):
    """
    Zapis homografii do pliku. Będzie zapisane w tym samym folderze, co i plik kontrolera
    (najlepiej jeżeli wszystkie pliki .py są w tym samym folderze).
    """
    np.save(homography_filename, homography)
    print(f"Matrix saved as {homography_filename}")

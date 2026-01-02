import numpy as np
import cv2
from controller import (Robot, Keyboard, Supervisor, Display)
from vehicle import Car
import sys
import visualise as vis
from camera_calibration import save_homo
from park_algo import TrajStateMachine,Kalman,OccupancyGrid,wrap_angle,mod2pi,C,PlanningWorker,Path
from fisheye_camera_calibration import estimate_tag_pose,detect_apriltags
import stereo_yolo as sy
from ultralytics import YOLO
import pandas as pd
import os
import time
from copy import deepcopy
sys.path.append(r"D:\\User Files\\BACHELOR DIPLOMA\\Kod z Github (rozne algorytmy)")

#from PIL import Image
# --------------------- Stałe ---------------------
TIME_STEP = 64
NUM_DIST_SENSORS = 12
NUM_CAMERAS = 8
MAX_SPEED = 250.0
CAMERA_HEIGHT = 2160
CAMERA_WIDTH = 3840

SENSOR_INTERVAL = 0.064
IMAGE_INTERVAL  = 0.2
KEYBOARD_INTERVAL = 0.04

# --------------------- Zmienne globalne ---------------------

robot = Robot()
driver = Car()
supervisor = Supervisor()

display = Display('display')
keyboard = Keyboard()
keyboard.enable(TIME_STEP)
cameras = []
camera_names = []
cam_matrices = {}
images =[]
front_sensors = []
rear_sensors = []
right_side_sensors = []
left_side_sensors = []

front_sen_apertures = {}
rear_sen_apertures = {}
right_side_sen_apertures = {}
left_side_sen_apertures = {}

steering_angle = 0.0
manual_steering = 0

path_to_models = r"D:\\User Files\\BACHELOR DIPLOMA\\Modele sieci\\"


#DETECTRON2 DLA PANOPTIC ORAZ DEEPLABV3+

"""
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.projects.deeplab import add_deeplab_config


cfg_panoptic = get_cfg()
add_panoptic_deeplab_config(cfg_panoptic)
cfg_panoptic.merge_from_file(r"C:\\Users\\fbiwa\\detectron2\\projects\\Panoptic-DeepLab\\configs\\Cityscapes-PanopticSegmentation\\panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml")
cfg_panoptic.MODEL.WEIGHTS = path_to_models + "panoptic-deeplab.pkl"
cfg_panoptic.MODEL.DEVICE = "cuda"
cfg_panoptic.MODEL.PANOPTIC_DEEPLAB.INSTANCES = True
cfg_panoptic.freeze()


cfg_deeplab = get_cfg()
add_deeplab_config(cfg_deeplab)
cfg_deeplab.merge_from_file(r"C:\\Users\\fbiwa\\detectron2\\projects\\DeepLab\\configs\\Cityscapes-SemanticSegmentation\\deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml")
cfg_deeplab.MODEL.WEIGHTS = path_to_models + "deeplabv3+.pkl"
cfg_deeplab.MODEL.DEVICE = "cuda"
cfg_deeplab.freeze()

predictor_panoptic = DefaultPredictor(cfg_panoptic)
"""


# --------------------- Helper Functions ---------------------
def print_help():
    print("Samochód teraz jeździ.")
    print("Proszę użyć klawiszy UP/DOWN dla zwiększenia prędkości lub LEFT/RIGHT dla skrętu")
    print("Naciśnij klawisz P, aby rozpocząć poszukiwanie miejsca")
    print("Podczas parkowania, wciśnij Q aby szukać z prawej strony")
    print("albo E aby szukać miejsca z lewej strony")

def set_speed(kmh,driver):
    global speed
    speed = min(kmh, MAX_SPEED)
    driver.setCruisingSpeed(speed)
    print(f"Ustawiono prędkość {speed} km/h")

def set_steering_angle(wheel_angle,driver):
    global steering_angle
    # Clamp steering angle to [-0.5, 0.5] radians (per vehicle constraints)
    wheel_angle = max(min(wheel_angle, C.MAX_WHEEL_ANGLE), -C.MAX_WHEEL_ANGLE)
    steering_angle = wheel_angle
    driver.setSteeringAngle(steering_angle)
    print(f"Skręcam {steering_angle} rad")

def change_manual_steering_angle(inc,driver):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if -25.0 <= new_manual_steering <= 25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02,driver)


#----------------------Sensor functions-----------------

camera_names = [
        
        #"camera_rear","camera_front_bumper","camera_front_left",
        "camera_front_right",
        "camera_left_mirror","camera_right_mirror"
    ]

def get_camera_image(camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img = camera.getImage()
    if img is None:
        return None

    img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))[:, :, :3]
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array


front_sensor_names = ["distance sensor front right", "distance sensor front righter",
"distance sensor front lefter", "distance sensor front left"
]
rear_sensor_names = ["distance sensor right", "distance sensor righter",
"distance sensor lefter", "distance sensor left"
]
left_side_sensor_names = ["distance sensor left front side","distance sensor left side"]
right_side_sensor_names = ["distance sensor right front side","distance sensor right side"]

def process_distance_sensors(sen):
    l_dist = sen.getLookupTable()
    a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
    b_dist = l_dist[3]-l_dist[4]*a_dist
    value = sen.getValue()
    distance = a_dist*value+b_dist
    sigma = l_dist[2]
    noisy_distance = distance + np.random.normal(0, sigma)
    return noisy_distance


# --------------------- Main Controller Loop ---------------------

speed = 0.0

rear_T = np.load("matrices/camera_rear_T_global.npy").astype(np.float32)
front_right_T = np.load("matrices/camera_front_right_T_global.npy").astype(np.float32)
front_left_T = np.load("matrices/camera_front_left_T_global.npy").astype(np.float32)
right_mirror_T = np.load("matrices/camera_right_mirror_T_global.npy").astype(np.float32)
left_mirror_T = np.load("matrices/camera_left_mirror_T_global.npy").astype(np.float32)

# dla czujników ultradźwiękowych
dists = []
max_min_dict = {}

# dla wizualizacji, pokazywania okien i przesyłania danych
class VisController(vis.QtCore.QObject):

    parkingToggled = vis.QtCore.pyqtSignal(bool)  # rozpoczęty proces parkowania
    sensorUpdated = vis.QtCore.pyqtSignal(object) # aktualizowane sensory (dalej wspólne sloty dla wizualizacji i wątku symulacji)
    locUpdated = vis.QtCore.pyqtSignal(object)    # dane dla 
    trajUpdated = vis.QtCore.pyqtSignal(object) 
    pathUpdated = vis.QtCore.pyqtSignal(object)
    speedUpdated = vis.QtCore.pyqtSignal(object)      
    angleUpdated = vis.QtCore.pyqtSignal(object)  
    stateUpdated = vis.QtCore.pyqtSignal(str)  # "searching","planning","executing"
    expansionUpdated = vis.QtCore.pyqtSignal(object)
    hmapUpdated = vis.QtCore.pyqtSignal(object)
    pathCarUpdated = vis.QtCore.pyqtSignal(object) # dla aktualizacji względem ścieżki
    
    def __init__(self):
        super().__init__()
        self.parking = False
        self.start_pose = None
        self.goal_pose = None
        self.state = "inactive"  
        self.planning_active = False  
        self.stopped = False
        self.path : Path = None
        self.timer = 0.0

    @vis.QtCore.pyqtSlot()
    def toggle_parking(self):
        self.parking = not self.parking
        self.timer = 0.0
        if self.state == "inactive":
            self.state = "searching"
        else:
            self.planning_active = False
            self.stopped = False
            self.path = None
            self.pathUpdated.emit(Path([], [], [], [], [], []))
            self.state = "inactive"
        self.parkingToggled.emit(self.parking)
   

# IMPLEMENTACJA MAIN ALE W QTHREAD, ŻEBY ZROBIĆ WIZUALIZACJĘ DOBRĄ ; CAŁE RUN MOŻNA PRZENIEŚĆ DO DEF MAIN(), JEŻELI QT NIEPOTRZEBNE


class MainWorker(vis.QtCore.QObject):
    
    sensorData  = vis.QtCore.pyqtSignal(object) # dane z czujników ultradźwiękowych
    poseData    = vis.QtCore.pyqtSignal(object) # pomiary z Webots
    trajData    = vis.QtCore.pyqtSignal(object) # trajektoria dla rysowania plus przeszkody i miejsca
    pathData    = vis.QtCore.pyqtSignal(object) # dane o ścieżce
    speedData   = vis.QtCore.pyqtSignal(object) # wykres prędkości
    angleData   = vis.QtCore.pyqtSignal(object) # wykresy skrętu i odchylenia
    stateData   = vis.QtCore.pyqtSignal(str)    # stan parkowania - poszukiwanie, planowanie, wykonywanie
    pathCarData = vis.QtCore.pyqtSignal(object) # dla przesyłania bieżącego śledzonego punktu trajektorii
    finished    = vis.QtCore.pyqtSignal(bool)   # ukończenie życia wątku
    

    def __init__(self,supervisor,controller:VisController):
        super().__init__()
        self.controller = controller
        
        self.first_call_pose = True
        self.first_call_traj = True
        self.first = True # żeby przy pierwszej próbie symulacji inicjalizować wszystko zależne od supervisora
        self.node = supervisor.getSelf()
        self.front_sensor_field = self.node.getField("sensorsSlotFront")
        self.rear_sensor_field = self.node.getField("sensorsSlotRear")
        self.center_sensor_field = self.node.getField("sensorsSlotCenter")
        
        self.front_ultrasonic_sensor_poses = {}
        self.rear_ultrasonic_sensor_poses = {}
        self.cameras_poses = {}
        self.chessboards = {}
        
        self.writeParkingPose = False
        self.planning_thread = None

    def _load_or_save_pose(self, name, Tp_s, target_dict):
        path = f"sensor_poses/{name}.npy"
        if not os.path.exists(path):
            target_dict[name] = Tp_s.astype(np.float32)
            save_homo(Tp_s, "sensor_poses/" + name)
        else:
            target_dict[name] = np.load(path).astype(np.float32)

    def _process_sensor_node(self, node, Tw_p, ultrasonic_dict=None):
        sensor_type = node.getTypeName()
        name_field = node.getField("name")

        if name_field is None:
            return
        name = name_field.getSFString()

        Tw_s = sy.build_homogeneous_transform(
            np.array(node.getOrientation()).reshape(3, 3),
            np.array(node.getPosition()).reshape(1, 3)
        )
        Tp_s = np.linalg.inv(Tw_p) @ Tw_s

        if sensor_type == "Camera":
            self._load_or_save_pose(name, Tp_s, self.cameras_poses)
        elif sensor_type == "DistanceSensor" and ultrasonic_dict is not None:
            self._load_or_save_pose(name, Tp_s, ultrasonic_dict)
        elif sensor_type == "Floor":
            chess_name = name + "_chessboard"
            self._load_or_save_pose(chess_name, Tp_s, self.chessboards)

    def init_sensors_poses(self):
        self.front_sensor_field = self.node.getField("sensorsSlotFront")
        self.rear_sensor_field = self.node.getField("sensorsSlotRear")
        self.center_sensor_field = self.node.getField("sensorsSlotCenter")

        self.front_ultrasonic_sensor_poses = {}
        self.rear_ultrasonic_sensor_poses = {}
        self.cameras_poses = {}
        self.chessboards = {}

        Tw_p = sy.build_homogeneous_transform(
            np.array(self.node.getOrientation()).reshape(3, 3),
            np.array(self.node.getPosition()).reshape(1, 3)
        )

        for i in range(self.front_sensor_field.getCount()):
            node = self.front_sensor_field.getMFNode(i)
            self._process_sensor_node(node, Tw_p, self.front_ultrasonic_sensor_poses)

        for i in range(self.rear_sensor_field.getCount()):
            node = self.rear_sensor_field.getMFNode(i)
            self._process_sensor_node(node, Tw_p, self.rear_ultrasonic_sensor_poses)

        for i in range(self.center_sensor_field.getCount()):
            node = self.center_sensor_field.getMFNode(i)
            self._process_sensor_node(node, Tw_p)
                        

    def start_planning_thread(self,ogm):
        # if self.planning_thread is not None and self.planning_thread.isRunning():
        #     print("[MainWorker] Wątek planowania już istnieje.")
        #     return
        
        print("[MainWorker] Uruchamiam wątek planowania")
        self.planning_worker = PlanningWorker(self.controller,ogm)
        self.planning_thread = vis.QtCore.QThread()
        self.planning_worker.moveToThread(self.planning_thread)

        self.planning_thread.started.connect(self.planning_worker.run)
        self.planning_worker.finished.connect(self.planning_thread.quit)
        self.planning_worker.finished.connect(self.planning_worker.deleteLater)
        self.planning_thread.finished.connect(self.planning_thread.deleteLater)

        self.planning_worker.finished.connect(self.planning_finished) 

        self.planning_thread.start()
        
    @vis.QtCore.pyqtSlot(bool)
    def planning_finished(self,success):
        self.controller.planning_active = False
        if success:
            self.controller.state = "executing"
            #self.planned_path = self.pathData or Path([],[],[],[],[])
        else:
            self.controller.state = "searching"
            

    @vis.QtCore.pyqtSlot()
    def run(self):
        
        self.init_sensors_poses()
        for name in front_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                front_sensors.append(sen)
                l_dist = sen.getLookupTable()
                front_sen_apertures[name] = sen.getAperture()
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]

        for name in rear_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                rear_sensors.append(sen)
                l_dist = sen.getLookupTable()
                rear_sen_apertures[name] = sen.getAperture()
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]
        for name in left_side_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                left_side_sensors.append(sen)
                l_dist = sen.getLookupTable()
                left_side_sen_apertures[name] = sen.getAperture()
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]

        for name in right_side_sensor_names:
            sen = robot.getDevice(name)
            if sen:
                sen.enable(TIME_STEP)
                right_side_sensors.append(sen)
                l_dist = sen.getLookupTable()
                right_side_sen_apertures[name] = sen.getAperture()
                a_dist = (l_dist[0]-l_dist[3])/(l_dist[1]-l_dist[4])
                b_dist = l_dist[3]-l_dist[4]*a_dist

                max_dist = a_dist*sen.getMinValue()+b_dist
                min_dist = a_dist*sen.getMaxValue()+b_dist
                max_min_dict[name] = [min_dist,max_dist]
       
        # Inicjalizuj kamery
        for name in camera_names:
            cam = robot.getDevice(name)
            if cam:
                cam.enable(TIME_STEP)
                cameras.append(cam)
                width = cam.getWidth()
                height = cam.getHeight()
                fov_rad = cam.getFov()
                Kmat = sy.calculate_intrinsic_matrix(width, height, fov_rad)
                cam_matrices[name] = Kmat
                
        # GPS inicjalizacja
        gps = robot.getDevice("gps")
        if gps:
            gps.enable(TIME_STEP)
        #Inicjalizacja IMU
        imu = robot.getDevice("inertial unit")
        if imu:
            imu.enable(TIME_STEP)
        #Inicjalizacja żyroskopu
        gyro = robot.getDevice("gyro")
        if gyro:
            gyro.enable(TIME_STEP)
        #Inicjalizacja akcelerometru
        acc = robot.getDevice("accelerometer")
        if acc:
            acc.enable(TIME_STEP)
        # Wydrukuj wskazówki
        print_help()
        last_key_time = robot.getTime()
        park_poses = {}

        def check_keyboard(cont:VisController):
            
            nonlocal last_key_time
            key = keyboard.getKey()
            #gear = driver.getGear()
            
            if key == Keyboard.UP:
                set_speed(speed + 0.5,driver)
                # if gear >= 0:
                #     if speed >= 0.0 and speed < 6.0:
                #         driver.setGear(1)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 6.0 and speed < 24.0:
                #         driver.setGear(2)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 24.0 and speed < 40.0:
                #         driver.setGear(3)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                #     elif speed >= 40.0 and speed < 70.0:
                #         driver.setGear(4)
                #         driver.setThrottle(1.0)
                #         driver.setBrakeIntensity(0.0)
                # elif gear == -1:
                #     driver.setThrottle(0.0)
                #     driver.setBrakeIntensity(1.0)

            elif key == Keyboard.DOWN:
                set_speed(speed - 0.5,driver)

                # if gear == -1:
                #     driver.setThrottle(1.0)
                #     driver.setBrakeIntensity(0.0)
                # elif gear > 0:
                #     driver.setThrottle(0.0)
                #     driver.setBrakeIntensity(1.0)


            elif key == Keyboard.RIGHT:
                change_manual_steering_angle(+5,driver)
            elif key == Keyboard.LEFT:
                change_manual_steering_angle(-5,driver)
            elif key in (ord('p'),ord('P')):

                cont.toggle_parking()
                if cont.parking:

                    print("Rozpoczęto parking")
                else:
                    cv2.destroyAllWindows()
                    print("Ukończono parking")
                
            elif key in (ord('e'),ord('E')):
                
                if cont.parking and self.controller.state == "waiting_for_confirm_start" and self.controller.stopped:
                    self.controller.timer = 0.0
                    self.controller.state = "planning"
            # elif key in (ord('F'),ord('f')):
            #     driver.setGear(1)
            #     print("Napęd do przodu")
            # elif key in (ord('R'),ord('r')):
            #     driver.setGear(-1)
            #     print("Napęd do tyłu")
            #else:
            #driver.setThrottle(0.0)
            #driver.setBrakeIntensity(0.0)
            
            elif not cont.parking:
                self.controller.state == "inactive"
                if key==ord('l') or key==ord('L'):
                    if len(park_poses) == 0:
                        print("Brak danych do zapisania.")
                        return
                    path = r"pozy.csv"
                    df = pd.DataFrame([{
                        "x_odo": p["x_odo"],
                        "y_odo": p["y_odo"],
                        "psi_odo": p["psi"],
                        "x_webots": p["node_pos"][0],
                        "y_webots": p["node_pos"][1],
                        "psi_webots": p["yaw_webots"],
                    } for p in park_poses])

                    file_exists = os.path.exists(path)
                    with open(path, mode=("a" if file_exists else "w"), encoding="utf-8", newline="") as f:
                        df.to_csv(f, index=False, header=not file_exists)
                        f.write("\n")

                    print(f"CSV zapisany: {path}, N={len(df)}")
            
            elif cont.parking:
                if key==ord('l') or key==ord('L'):
                    self.writeParkingPose = not self.writeParkingPose

        model = YOLO(path_to_models + "yolov8m-seg.pt")
        
        prev_time = driver.getTime()
        prev_real = time.time()
        # do supervisora - POZYCJA SAMOCHODU W WEBOTS
        node_pos0 = np.zeros(3)

        # do odometrii
        im0 = [0.0,0.0,0.0]
        gp0 = [0.0,0.0,0.0]
        x_odo = 0.0
        y_odo = 0.0
        sp_odo = 0.0
        psi = 0.0
        delta = 0.0
        yaw_est = 0.0
        yaw_real = 0.0

        # parametry samochodu
        front_radius = driver.getFrontWheelRadius()
        rear_radius = driver.getRearWheelRadius()
        wheelbase = driver.getWheelbase()
        # dla enkoderów i prędkości z odometrii
        curr_enc = np.zeros(4)
        enc0 = np.zeros(4)
        wheel_speeds = np.zeros(4)
        # przeszłe enkodery
        prev_enc = np.zeros(4)
        enc_ok = False
        
        def get_speed_odo(dt):
            raw_enc = [driver.getWheelEncoder(i) for i in range(4)]
            #liczymy enkodery
            for i in range(4):
                curr_enc[i] = raw_enc[i] - enc0[i]
                wheel_speeds[i] = (raw_enc[i] - prev_enc[i]) / dt
                prev_enc[i] = raw_enc[i]

            #speed = 0.5 * (wheel_speeds[0] + wheel_speeds[1]) * front_radius
            # speed = 0.25 * (wheel_speeds[0] + wheel_speeds[1]) * front_radius + \
            #         0.25 * (wheel_speeds[2] + wheel_speeds[3]) * rear_radius
            speed = 0.5 * (wheel_speeds[2] + wheel_speeds[3]) * rear_radius
                
            return speed,curr_enc
        
        def webots_to_odom_xy(dx,dy,yaw0):
            c = np.cos(yaw0); s = np.sin(yaw0)
            # obrót o -yaw0
            x_loc =  c*dx + s*dy
            y_loc = -s*dx + c*dy
            return x_loc, y_loc

        def R_xyz(yaw, pitch, roll):
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            return np.array([
                [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                [-sp,    cp*sr,             cp*cr]
            ])
        
        def get_pose_kalman(dt,kalman):
            
            nonlocal yaw_est,x_odo,y_odo,sp_odo,delta,yaw_real,gp0,im0,node_pos0,enc0
            if self.first_call_pose:
                x_odo = 0.0
                y_odo = 0.0
                gp0 = gps.getValues()
                im0 = imu.getRollPitchYaw()
                for i in range(4):
                    enc0[i] = driver.getWheelEncoder(i)
                    prev_enc[i] = 0.0
                node_pos0 = self.node.getPosition()
                
                self.first_call_pose = False
                yaw_real = 0.0
            # gps
            node_pos = self.node.getPosition()
            node_vel = self.node.getVelocity()
            # imu
            rpy = imu.getRollPitchYaw()
            yaw_imu = wrap_angle(rpy[2] - im0[2])
            im = [rpy[0] - im0[0], rpy[1] - im0[1], yaw_imu]
            # supervisor
            R = R_xyz(rpy[2], rpy[1], rpy[0])   # ZYX
            node_vel_xyz = R.T @ np.array(node_vel[:3])
            node_vel_x = node_vel_xyz[0]  
            # żyroskop
            gyr = gyro.getValues()
            yaw_est = wrap_angle(yaw_est + gyr[2] * dt)
            # odometria
            sp_odo_meas,encoders = get_speed_odo(dt)
            # kąt skrętu kół [rad]
            delta_meas = -driver.getSteeringAngle()
            x_pred = kalman.predict(np.array([x_odo,y_odo,yaw_real,sp_odo_meas,delta_meas]),dt)
            x_upd = kalman.update(x_pred,np.array([yaw_est]))
            x_odo = x_upd[0]
            y_odo = x_upd[1]
            yaw_real = x_upd[2]
            sp_odo = x_upd[3]
            delta = x_upd[4]
            # akcelerometr
            accer = acc.getValues()
            # node position
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]
            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":yaw_real,"dt":dt,"x_odo":x_odo,"y_odo":y_odo,"encoders":encoders,"node_pos":node_pos,"acc":accer,"node_vel":node_vel_xyz,"node_vel_x":node_vel_x}

        R0 = np.eye(3)
        def dxdys(state,v,delta):
            _,_,yaw = state
            dxdt = v * np.cos(yaw)
            dydt = v * np.sin(yaw)
            dpsidt = v/C.WHEELBASE * np.tan(delta)
            return np.array([dxdt,dydt,dpsidt])
        
        def runge_kutta_odo(x,y,yaw,v,delta,dt):
            # policz runge-kutta 
            state_k1 = np.array([x,y,yaw])
            k1 = dxdys(state_k1,v,delta)
            state_k2 = state_k1 + k1 * dt/2
            k2 = dxdys(state_k2,v,delta)
            state_k3 = state_k1 + k2 * dt/2
            k3 = dxdys(state_k3,v,delta)
            state_k4 = state_k1 + k3 * dt
            k4 = dxdys(state_k4,v,delta)
            f = 1/6*(k1 + 2*k2 + 2*k3 + k4)
            new_xyyaw = state_k1 + f*dt
            x_odo,y_odo,yaw_odo = new_xyyaw
            #yaw_odo = mod2pi(yaw_odo)
            return x_odo,y_odo,yaw_odo
        
        def get_pose(dt):
            # liczy z odometrii pozę samochodu, 
            nonlocal psi,x_odo,y_odo,gp0,im0,node_pos0,yaw_est,R0
            # nadanie wartości początkowych
            if self.first_call_pose:
                x_odo = 0.0
                y_odo = 0.0
                psi = 0.0
                gp0 = gps.getValues()
                im0 = imu.getRollPitchYaw()
                raw_enc = [driver.getWheelEncoder(i) for i in range(4)]
                for i in range(4):
                    enc0[i] = raw_enc[i]
                    prev_enc[i] = raw_enc[i]
                node_pos0 = self.node.getPosition()
                
                R0 = R_xyz(im0[2], im0[1], im0[0])   # orientacja początkowa
                self.first_call_pose = False
                return {
                    "sp_odo": 0.0,
                    "im": [0.0, 0.0, 0.0],
                    "delta": 0.0,
                    "psi": 0.0,
                    "dt": dt,
                    "x_odo": 0.0,
                    "y_odo": 0.0,
                    "encoders": [0.0]*4,
                    "node_pos": [0.0, 0.0, 0.0],
                    "acc": [0.0, 0.0, 0.0],
                    "node_vel": [0.0, 0.0, 0.0],
                    "node_vel_x": 0.0
                }
            # supervisor
            node_pos = self.node.getPosition()
            node_vel = self.node.getVelocity()
            # imu
            rpy = imu.getRollPitchYaw()
            yaw_imu = mod2pi(rpy[2] - im0[2])
            #yaw_imu = rpy[2]
            im = [mod2pi(rpy[0] - im0[0]), mod2pi(rpy[1] - im0[1]), yaw_imu]
            
            
            R = R_xyz(rpy[2], rpy[1], rpy[0])   # ZYX
            
            node_vel_xyz = R.T @ np.array(node_vel[:3])
            #node_vel_xyz = node_vel
            node_vel_x = node_vel_xyz[0]  
            #odometria, przednie koła
            sp_odo,encoders = get_speed_odo(dt)
            delta = -driver.getSteeringAngle()  # kąt skrętu kół [rad]
            gyr = gyro.getValues()
            dpsi_gyr = gyr[2] * dt
            # aktualizacja z uśrednieniem psi, później się dodaje
            dpsi_odo = (sp_odo * np.tan(delta) / wheelbase) * dt
            # akcelerometr
            accer = acc.getValues()
            # mid_psi = psi + dpsi_odo/2
            # # mid_psi = psi + dpsi_odo
            # mid_psi = mod2pi(mid_psi)
            # x_odo += sp_odo * dt * np.cos(mid_psi)
            # y_odo += sp_odo * dt * np.sin(mid_psi)
           
            # psi += dpsi_odo
            # yaw_real = mod2pi(psi)
            # yaw_real = psi

            # komplelentarny filtr żyroskopu i odometrii
            alpha = 0.7
            
            old_x,old_y,old_yaw = x_odo,y_odo,psi
            new_x,new_y,new_yaw = runge_kutta_odo(old_x,old_y,old_yaw,sp_odo,delta,dt)
            x_odo,y_odo,psi_odo = new_x,new_y,new_yaw
            psi = wrap_angle(alpha * psi_odo + (1-alpha)*(old_yaw + dpsi_gyr))
            yaw_real = psi
            # supervisor
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]

            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":yaw_real,"dt":dt,"x_odo":x_odo,
                    "y_odo":y_odo,"encoders":encoders,"node_pos":node_pos,"acc":accer,
                    "node_vel":node_vel_xyz,"node_vel_x":node_vel_x}

        # do wyswietlania punktów z YOLO
        Mtr = np.eye(4)
        Mtr[0,3] = -C.CAR_LENGTH + 1
        name = "camera_front_right"
        T_center_to_front = Mtr @ front_right_T 
        T_center_to_camera = T_center_to_front
        ptt = sy.project_points_world_to_image(np.array([[3.85, 0.0, 0.0]], dtype=np.float32),front_right_T,cam_matrices[name])
        # obwód koła
        circ = 2*np.pi*(front_radius+rear_radius)*0.5
        #circ *= 0.5 
        #print(f"Obwód koła: {circ}")

        kalman = Kalman(wheelbase)
        v_kmh = 4.0      
        tsm = TrajStateMachine(driver,self)  
        name = "camera_left_mirror"
        
        
        ogm = OccupancyGrid()
        ogm.setup_sensors(self.front_ultrasonic_sensor_poses,
                            self.rear_ultrasonic_sensor_poses,
                            [front_sensor_names,rear_sensor_names,right_side_sensor_names,left_side_sensor_names],
                            [front_sen_apertures,rear_sen_apertures,right_side_sen_apertures,left_side_sen_apertures],
                            max_min_dict)

        
        # kalibracja fisheye
        name = "camera_left_mirror"
        # K_left_mirror = cam_matrices["camera_left_mirror"]
        # T_left_mirror = self.cameras_poses["camera_left_mirror"]
        # K_front_bumper = cam_matrices["camera_front_bumper"]
        # K_right_mirror = cam_matrices["camera_right_mirror"]
        # K_rear = cam_matrices["camera_rear"]
        def init_ipm_maps(K_fisheye, D_fisheye, rvec, tvec, out_w, out_h, meters_per_pixel):
            
            grid_u, grid_v = np.meshgrid(np.arange(out_w), np.arange(out_h))
            # poprawić dalej, bo X do przodu i Y w lewo
            world_x = (out_h/2 - grid_v) * meters_per_pixel
            world_y = (out_w/2 - grid_u) * meters_per_pixel
            world_z = np.zeros_like(world_x) # Ziemia jest płaska

            #tvec = np.array([[0], [1.14], [0]], dtype=np.float32)
            # Punkty 3D w świecie (N, 1, 3)
            # fisheye. dodać do project points
            object_points = np.stack([world_x, world_y, world_z], axis=-1).reshape(-1, 1, 3)
            distorted_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, K_fisheye, D_fisheye
            )
            
            # 5. Konwersja na format mapy dla remap
            map_x = distorted_points[:, 0, 0].reshape(out_h, out_w).astype(np.float32)
            map_y = distorted_points[:, 0, 1].reshape(out_h, out_w).astype(np.float32)
            
            return map_x, map_y
        
        w,h = 1000,1000
        mpx = 0.01
        
        tag_position = np.array([-0.6,2.8,0.0]).astype(np.float32)
        tag_yaw = 0.0
        block_size = 1.5
        #tag_position[0] += 2*block_size
        #tag_position[1] -= 2.5*block_size
        
        model(np.ones((4,1,3)),half=True,device = 0,conf=0.6,verbose=False,imgsz=(640,480))

        # PLANER HYBRID A*
        p1 = None
        p2 = None
        import reeds_shepp
        path_built = False
        path = None
        ref_path = None
        self.path_index = int(0.0)
        delta_prev = 0.0
        while robot.step(64) != -1:
            check_keyboard(cont)
            #print(self.controller.state)
            vis.QtCore.QCoreApplication.processEvents()
            now = driver.getTime()
            now_real = time.time()
            dt_real = now_real - prev_real
            
            now = driver.getTime()
            dt_sim = now - prev_time
            pose_measurements = get_pose(dt_sim)
            prev_real = now
            prev_time = now
            names_images = dict(zip(camera_names, [get_camera_image(c) for c in cameras]))
            image = names_images[name].copy()
            
            if cont.parking:
                
                #tsm.update(now_real,dt_sim)
                
                x_odo = pose_measurements["x_odo"]
                y_odo = pose_measurements["y_odo"]
                yaw_odo = pose_measurements["psi"]
                delta_meas = pose_measurements["delta"]
                sp_odo = pose_measurements["sp_odo"]
                node_vel_x = pose_measurements["node_vel_x"]
                node_pos = pose_measurements["node_pos"]
                yaw_webots = pose_measurements["im"][2]
                
                

                if self.writeParkingPose:
                    self.writeParkingPose = not self.writeParkingPose
                    #park_poses.update({"x_odo":x_odo,"y_odo":y_odo,"node_pos_x":node_pos[0],"node_pos_y":node_pos[1],"psi_odo":yaw_odo,"psi_webots":yaw_webots})
                    #print("Zapisano miejsce parkingowe")
                
                front_dists = [process_distance_sensors(s) for s in front_sensors]
                rear_dists = [process_distance_sensors(s) for s in rear_sensors]
                right_side_dists = [process_distance_sensors(s) for s in right_side_sensors]
                left_side_dists = [process_distance_sensors(s) for s in left_side_sensors]

                front_names_dists = dict(zip(front_sensor_names, front_dists))
                rear_names_dists = dict(zip(rear_sensor_names, rear_dists))
                left_side_names_dists = dict(zip(left_side_sensor_names, left_side_dists))
                right_side_names_dists = dict(zip(right_side_sensor_names, right_side_dists))
                # if not path_built:
                #     point1 = (x_odo,y_odo,yaw_odo)
                #     point2 = (10.0,30.0,np.pi/2)
                #     path_xs = []
                #     path_ys = []
                #     path_yaws = []
                #     path_dirs = []
                #     path_curvs = []
                #     path = reeds_shepp.path_sample(point1,point2,C.MAX_RADIUS,0.05)
                #     for p in path:
                #         path_xs.append(p[0])
                #         path_ys.append(p[1])
                #         path_yaws.append(p[2])
                #         path_dirs.append(np.sign(p[4]))
                #         path_curvs.append(-p[3])
                #     ref_path = Path(path_xs,path_ys,path_yaws,path_dirs,path_curvs,[])
                #     self.pathData.emit(ref_path)
                #     path_built = True


                # if ref_path is not None and path_built:
                #     x = x_odo
                #     y = y_odo
                #     yaw = yaw_odo
                #     v = sp_odo
                    
                #     dir_ref = ref_path.directions[self.path_index]   
                #     v_eff = dir_ref * abs(v)

                #     delta,ind = ref_path.rear_wheel_feedback_control(x,y,v_eff,yaw)
                #     self.path_index = ind
                #     delta = delta
                #     #print(f"delta: {delta}, ind: {ind}")
                    
                #     dist_to_goal = ref_path.s[-1] - ref_path.s[ind]
                #     target_v = dir_ref * C.MAX_SPEED
                #     if ind > 0 and ref_path.directions[ind] != ref_path.directions[ind-1]:
                #         target_v = 0.0
                #     if dist_to_goal <= 3.0:
                #         target_v = 0.0
                #     v_set = ref_path.pid_control(target_v,v,dt_sim)
                    
                #     tracked_pose = (ref_path.xs[ind],ref_path.ys[ind],ref_path.yaws[ind])
                #     self.pathCarData.emit(tracked_pose)
                #     max_rate = 0.5
                #     delta_cmd = np.clip(delta, delta_prev - max_rate*dt_sim, delta_prev + max_rate*dt_sim)
                #     delta_prev = delta_cmd
                #     #delta_cmd = max(-C.MAX_WHEEL_ANGLE, min(C.MAX_WHEEL_ANGLE, delta))
                #     driver.setSteeringAngle(min(max(-C.MAX_WHEEL_ANGLE,delta_cmd),C.MAX_WHEEL_ANGLE))     
                    
                

                if not (self.controller.planning_active or self.controller.state == "executing"):
                    
                    
                    #if self.controller.state not in ["planning","executing"]:
                    ogm.interpret_readings({**front_names_dists,**rear_names_dists,**left_side_names_dists,**right_side_names_dists},(x_odo,y_odo,yaw_odo))
                    # mamy macierz grid tych wielkości; z nich trzeba przemnożyć na xy_resolution te indeksy, aby otrzymać właściwe pozycje przeszkód, są większe lub równe od 0
                    ox,oy = ogm.extract_obstacles()
                    #ox,oy = [],[]
                    find_type = 'parallel'
                    side = 'right'
                    #,"camera_left_mirror"
                    names_yolo = ["camera_front_right","camera_left_mirror","camera_right_mirror"]
                    for name in names_yolo:  
                        image = names_images[name].copy()
                        results = model(image,half=True,device = 0,conf=0.6,verbose=False,imgsz=(640,480))
                        if results is not None:
                            for box in results[0].boxes.xyxy.cpu().numpy():  # [x1,y1,x2,y2]
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                                x,y = (x1+x2)/2,y2
                                T_center_to_camera = right_mirror_T if name == "camera_right_mirror" \
                                    else left_mirror_T if name == "camera_left_mirror" \
                                    else front_right_T if name == "camera_front_right" else None

                                pt = sy.pixel_to_world(x,y,cam_matrices[name],T_center_to_camera=T_center_to_camera) # punkty w układzie samochodu
                                if abs(pt[0]) < C.CAR_LENGTH - 0.9 and abs(pt[1]) < C.CAR_WIDTH/2 + 0.2:
                                    continue
                                if pt is not None:
                                    # transformować punkty yolo do układu samochodu
                                    pt_x, pt_y = pt[0], pt[1]
                                    pt_homogeneous = np.array([pt_x, pt_y, 1.0])
                                    c, s = np.cos(yaw_odo), np.sin(yaw_odo)
                                    transf = np.array([
                                        [c, -s, x_odo],
                                        [s,  c, y_odo],
                                        [0,  0, 1.0]
                                    ], dtype=np.float32)
                                    pt_global = transf @ pt_homogeneous
                                    ogm.yolo_points_buffer.append(pt_global[:2])
                                    ogm.yolo_x_pts.append(pt_global[0])
                                    ogm.yolo_y_pts.append(pt_global[1])
                                
                            cv2.namedWindow(f"yolo {name}", cv2.WINDOW_NORMAL)
                            cv2.imshow(f"yolo {name}", image)
                    
                    spots = []
                    if len(ox) > 0:
                        cls = ogm.analyze_clusters((x_odo,y_odo,yaw_odo)) 
                        #if len(cls_old) < len(cls):
                        #cls_old = cls
                        ogm.setup_obstacles(cls)
                        ogm.match_semantics_with_sat()
                        spots = ogm.find_spots_scanning((x_odo,y_odo,yaw_odo),spot_type = find_type, side = side)
                        #spot = ogm.find_first_good_spot((x_odo,y_odo,yaw_odo),spot_type = find_type, side = side)
                        #if spot is not None: spots.append(spot)
                        spots.extend(ogm.find_spots_scanning((x_odo,y_odo,yaw_odo),spot_type = 'perpendicular', side = 'left'))
                        ogm.spots = spots
                    else:
                        ox,oy,cls,spots = [],[],[],[]
                #ox,oy,cls,spots = ogm.ox,ogm.oy,ogm.obstacles,ogm.spots
                

                p1 = (x_odo,y_odo,yaw_odo)
                p2,_ = ogm.choose_spot(p1)
                
                if self.controller.state == "searching" and p2 is not None:
                    self.controller.state = "found_spot"
                    target_spot = p2
                    print("[MainWorker] Znaleziono miejsce! Proszę się zatrzymać.")

                elif self.controller.state == "found_spot" and p2 is not None:
                    tc = target_spot['center']
                    pc = p2['center']
                    dist = np.hypot(pc[0] - tc[0], pc[1] - tc[1])

                    if dist > 1.0:
                        target_spot = p2
                        print("[MainWorker] Znaleziono nowe miejsce! Proszę się zatrzymać.")

                if self.controller.state == "found_spot":
                    if abs(sp_odo) <= 1e-2:
                        self.controller.timer += dt_real
                    self.controller.stopped = True if self.controller.timer >= 3.0 and abs(sp_odo) <= 1e-4 else False
                    if self.controller.stopped:  
                        self.controller.state = "waiting_for_confirm_start"     
                        print("[MainWorker] Proszę wcisnąć przycisk E aby rozpocząć parkowanie.")

                if not self.controller.planning_active and self.controller.state == "planning" and self.controller.stopped:
                    #grid = deepcopy(ogm)
                    self.controller.start_pose = p1
                    #self.controller.state = "planning"
                    self.controller.planning_active = True
                    self.controller.goal_pose = (target_spot['target_rear_axle'][0],
                                                    target_spot['target_rear_axle'][1],
                                                    target_spot['orientation'])
                    try:
                        self.start_planning_thread(ogm)
                    except:
                        print("[MainWorker] Nie udało się uruchomić planowania.")   

                if self.controller.state == "executing":

                    x = x_odo
                    y = y_odo
                    yaw = yaw_odo
                    v = sp_odo
                    self.planned_path = self.controller.path or Path([],[],[],[],[])
                    theta_e, er, k, yaw, ind = self.planned_path.calc_theta_e_and_er(x,y,yaw)
                    
                    #v_eff = self.planned_path.directions[ind] * abs(v)
                    v_eff = v
                    delta,ind = self.planned_path.rear_wheel_feedback_control(x,y,v_eff,yaw)
                    delta = -delta
                    self.path_index = ind
                    #print(f"delta: {delta}, ind: {ind}")
                    
                    dist_to_goal = self.planned_path.s[-1] - self.planned_path.s[ind]
                    if self.planned_path.directions[ind] < 0:
                        target_v = -C.MAX_SPEED
                    else:
                        target_v = C.MAX_SPEED
                    if ind > 0 and self.planned_path.directions[ind] != self.planned_path.directions[ind-1]:
                        target_v = 0.0
                    #v_set = self.planned_path.pid_control(target_v,v,dt_sim)
                    if abs(v) < 0.05:
                        delta = 0.0
                    tracked_pose = (self.planned_path.xs[ind],self.planned_path.ys[ind],self.planned_path.yaws[ind])
                    self.pathCarData.emit(tracked_pose)
                    max_rate = 1.5
                    delta_cmd = np.clip(delta, delta_prev - max_rate*dt_sim, delta_prev + max_rate*dt_sim)
                    delta_prev = delta_cmd
                    driver.setSteeringAngle(min(max(-C.MAX_WHEEL_ANGLE,delta_cmd),C.MAX_WHEEL_ANGLE))     
                    #driver.setCruisingSpeed(v_set)
                #ogm.yolo_y_pts = []
                traj_to_send = [[x_odo,y_odo,yaw_odo],node_pos,[ogm.ox,ogm.oy],ogm.obstacles,ogm.spots,[ogm.yolo_x_pts,ogm.yolo_y_pts]]
                
                traj_data = traj_to_send 
                speed_data = [now,sp_odo,node_vel_x]
                angle_data = [now,delta_meas,psi,yaw_webots]
                self.sensorData.emit([front_names_dists,rear_names_dists,
                                      left_side_names_dists,right_side_names_dists,
                                      max_min_dict])
                self.poseData.emit(pose_measurements)
                self.trajData.emit(traj_data)
                self.speedData.emit(speed_data)
                self.angleData.emit(angle_data)
                

                # image = names_images["camera_left_mirror"].copy()
                
                # corners,ids = detect_apriltags(image,"camera_left_mirror")
                # print(f"corners: {corners}, ids: {ids}")
                # distcoeffs = np.zeros(4).astype(np.float32)
                # poses = estimate_tag_pose(corners,ids,cam_matrices["camera_left_mirror"],distcoeffs,1.5)
                # if poses is not None and len(poses)>0:
                #     id = list(poses.keys())[0]
                #     rvec, tvec = poses[id]
                #     vis_image = image.copy()
                #     vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    
                #     # Definicja osi: origin (0,0,0) + 3 punkty na końcach osi
                #     axis = np.float32([[block_size/2,0,0], [0,block_size/2,0], [0,0,block_size/2]]).reshape(-1,3)  # 20 cm osie
                #     imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrices[name], distcoeffs)
                    
                #     # Origin to pierwszy punkt
                #     #origin = tuple(imgpts[0])
                #     corn = corners[0].reshape(-1,2)
                #     # Rysuj osie z origin do każdego punktu
                #     #cv2.line(vis_image, tuple(corn[0]), tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)   # X - CZERWONY
                #     #cv2.line(vis_image, tuple(corn[0]), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)   # Y - ZIELONY
                #     #cv2.line(vis_image, tuple(corn[0]), tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)   # Z - NIEBIESKI

                #     # cv2.line(vis_image, tuple(corn[0]), tuple(corn[1]), (0, 0, 255), 3)   # X - CZERWONY
                #     # cv2.line(vis_image, tuple(corn[0]), tuple(corn[2]), (0, 255, 0), 3)   # Y - ZIELONY
                #     # cv2.line(vis_image, tuple(corn[0]), tuple(corn[3]), (255, 0, 0), 3)   # Z - NIEBIESKI
                #     # Opcjonalnie: dodaj etykiety
                #     cv2.putText(vis_image, "X", tuple(imgpts[0].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #     cv2.putText(vis_image, "Y", tuple(imgpts[1].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #     cv2.putText(vis_image, "Z", tuple(imgpts[2].ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #     cv2.drawFrameAxes(vis_image,cam_matrices[name],distcoeffs,rvec,tvec,0.75,3)
                #     cv2.namedWindow("Tag",cv2.WINDOW_NORMAL)
                #     cv2.imshow("Tag", vis_image) 
                #     #cv2.imwrite(f"camera_front_right_axes.png",vis_image)
                #     obj_points = np.float32([[-block_size/2,-block_size/2,0], [-block_size/2,block_size/2,0], [block_size/2,block_size/2,0.0],[block_size/2,-block_size/2,0.0]]).reshape(-1,3)
                #     projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, cam_matrices[name], distcoeffs)

                #     # 2. Obliczamy błąd euklidesowy między punktami wykrytymi (corners) a rzutowanymi
                #     # corners[i] ma kształt (1, 4, 2) lub (4, 1, 2), upewnij się że kształty pasują
                    
                #     projected_points = projected_points.reshape(-1, 2)

                #     error = cv2.norm(corn, projected_points, cv2.NORM_L2)
                #     rms_error = error / len(projected_points) # Błąd średni na punkt w pikselach

                #     print(f"Błąd reprojekcji (root mean square): {rms_error:.4f} px")

                #     if name == "camera_left_mirror":
                #         chessboard_position = tag_position
                #         chessboard_yaw = 0  # degrees
                #         #rvec,tvec = cc.solve_camera_pose(image,pattern_size,cam_matrices[name],name)
                #         #if R is not None and tvec is not None:
                #         T_chessboard_to_center = sy.build_pose_matrix(chessboard_position, chessboard_yaw)

                #         R, _ = cv2.Rodrigues(rvec)
                #         T_camera_to_chessboard = np.linalg.inv(sy.build_homogeneous_transform(R, tvec))

                #         # Combine to get rear axle → camera
                #         T_center_to_camera = T_chessboard_to_center @ T_camera_to_chessboard
                        
                #         # Project bbox
                #         bbox_world = np.array([
                #             [-0.6+block_size, 2.8+block_size, 0],   # bottom front right
                #             [-0.6+block_size,2.8-block_size, 0],  # bottom front left
                #             [-0.6-block_size, 2.8-block_size, 0],  # bottom rear left
                #             [-0.6-block_size, 2.8+block_size, 0],   # bottom rear right
                #             [-0.6+block_size, 2.8+block_size, 1.0],   # bottom front right
                #             [-0.6+block_size,2.8-block_size, 1.0],  # bottom front left
                #             [-0.6-block_size, 2.8-block_size, 1.0],  # bottom rear left
                #             [-0.6-block_size, 2.8+block_size, 1.0],   # bottom rear right
                #         ])
                #         image_points = sy.project_points_world_to_image(bbox_world, T_center_to_camera, cam_matrices[name])
                #         # Draw bottom rectangle
                #         for i in range(4):
                #             pt1 = image_points[i]
                #             pt2 = image_points[(i + 1) % 4]
                #             cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

                #         # Draw top rectangle
                #         for i in range(4, 8):
                #             pt1 = image_points[i]
                #             pt2 = image_points[4 + (i + 1) % 4]
                #             cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)

                #         # Draw vertical lines
                #         for i in range(4):
                #             pt1 = image_points[i]
                #             pt2 = image_points[i + 4]
                #             cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)
                #         cv2.namedWindow(f"Projected 3D BBox {name}",cv2.WINDOW_NORMAL)
                #         cv2.imshow(f"Projected 3D BBox {name}", vis_image)
                #         print(f"[{name}] pose wrt rear axle (T_center_to_camera):\n", T_center_to_camera)
                #         save_homo(T_center_to_camera,f"{name}_T_global")
                    

                
                cv2.waitKey(1)
            elif not cont.parking:
                self.first_call_pose = True
                self.first_call_traj = True
                #ogm.ox,ogm.oy,ogm.obstacles,ogm.spots = [],[],[],[]
                ogm = OccupancyGrid()
                ogm.setup_sensors(self.front_ultrasonic_sensor_poses,
                                    self.rear_ultrasonic_sensor_poses,
                                    [front_sensor_names,rear_sensor_names,right_side_sensor_names,left_side_sensor_names],
                                    [front_sen_apertures,rear_sen_apertures,right_side_sen_apertures,left_side_sen_apertures],
                                    max_min_dict)
                #prev_time = driver.getTime()
                #prev_real = time.time()
                
                
            #if now - last_key_time >= KEYBOARD_INTERVAL:
            
            #last_key_time = now
            

        app.quit()

        self.finished.emit()


if __name__ == "__main__":
    app = vis.pg.QtWidgets.QApplication(sys.argv)
    cont = VisController()
    
    thread = vis.QtCore.QThread()
    worker = MainWorker(supervisor,cont)
    worker.moveToThread(thread)

    # sygnały z mainworker
    thread.started.connect(worker.run)
    worker.sensorData.connect(cont.sensorUpdated)
    worker.poseData.connect(cont.locUpdated)
    worker.trajData.connect(cont.trajUpdated)
    worker.speedData.connect(cont.speedUpdated)
    worker.angleData.connect(cont.angleUpdated)
    worker.stateData.connect(cont.stateUpdated)
    worker.pathData.connect(cont.pathUpdated)
    worker.pathCarData.connect(cont.pathCarUpdated)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()

    win  = vis.SensorView(cont)
    win.hide()

    win1 = vis.SpeedView(cont)
    win1.hide()

    win2 = vis.AngleView(cont)
    win2.hide()

    win3 = vis.TrajView(cont)
    win3.hide()
    
    sys.exit(app.exec())


#results = model(names_images[name],half=True,device = 0,conf=0.6)

#annotated_frame = results[0].plot()

#cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
#cv2.imshow("yolo", annotated_frame)
#cv2.waitKey(1)

# Wyniki:
# out["panoptic_seg"] -> (panoptic_map[H,W] (int32), segments_info[list])
# out["sem_seg"]      -> [C,H,W] (logity)
# out["instances"]    -> obiekty (jeśli włączone)

# img = names_images[name]  # HxWx3 (RGB)
# out = predictor_panoptic(img)
# panoptic, segments_info = out["panoptic_seg"]
# panoptic = panoptic.to("cpu").numpy()
# cv2.namedWindow("panoptic", cv2.WINDOW_NORMAL)
# cv2.imshow("panoptic", panoptic)

# Define the codec and create VideoWriter object

"""
if first_call:
    print("Zapis filmiku z kamery.")
    first_call = False
cv2.imwrite(img_paths + f"img_{i}.png",cv2.cvtColor(names_images[name],cv2.COLOR_BGR2RGB))
"""


"""
input_image = names_images[name]
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model_deeplab.to('cuda')
with torch.no_grad():
    output = eff_ps(input_batch)['out'][0]
# Wyniki z modelu
output_predictions = output.argmax(0)

"""


"""
predictions, _, _ = predictor.numpy_image(names_images[name])



out = names_images[name].copy()
for ann in predictions:
    kps = ann.data
    skeleton = ann.skeleton
    #print(ann)
    # rysuj krawędzie
    for (s,e) in skeleton:
        x1,y1,v1 = kps[s-1]
        x2,y2,v2 = kps[e-1]
        if v1 > 0 and v2 > 0:

            thickness = 2
            col = (0,255,0)
            if v1 > 0.5 and v2 > 0.5:
                cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), col, thickness)
            else:
                # przerywana: narysuj kilka krótkich segmentów
                n = 8
                pts = np.linspace([x1,y1],[x2,y2],n)
                for p,q in zip(pts[:-1], pts[1:]):
                    cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), col, 1)

    # rysuj punkty
    for (x,y,v) in kps:
        if v > 0:
            radius = 3
            color = (0,0,255)            # czerwone kropki
            cv2.circle(out, (int(x),int(y)), radius, color, -1)
out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
cv2.imshow('Vehicle Pose', out)

cv2.waitKey(1)
"""






#vis.alt_collect_homo(names_images, homographies, car, streams)
"""
images = [get_camera_image(c) for c in cameras]
names_images = dict(zip(camera_names, images))
name = "camera_front_top"
#sens_queue.put(dists)

results = model(names_images[name])[0]
img = names_images[name].copy()
for result in results:
    bbox = result.boxes.xyxy[0].tolist()

    for kpt in result.keypoints.data[0].tolist():
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

# Display the image with keypoints
cv2.namedWindow("yolo",cv2.WINDOW_NORMAL)
cv2.imshow("yolo",img)
"""
"""
# 2) inference
predictions, _, _ = predictor.numpy_image(names_images[name])

 # 4) rysowanie w OpenCV
out = names_images[name].copy()
for ann in predictions:
    kps = ann.data                      # shape (K,3): x,y,confidence
    skeleton = ann.skeleton             # lista par 1-based

    # rysuj krawędzie
    for (s,e) in skeleton:
        x1,y1,v1 = kps[s-1]
        x2,y2,v2 = kps[e-1]
        if v1 > 0 and v2 > 0:
            # solid jeśli oba > threshold, inaczej przerywana
            thickness = 2
            col = (0,255,0)             # tu możesz wymyślić paletę
            if v1 > 0.5 and v2 > 0.5:
                cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), col, thickness)
            else:
                # przerywana: narysuj kilka krótkich segmentów
                n = 8
                pts = np.linspace([x1,y1],[x2,y2],n)
                for p,q in zip(pts[:-1], pts[1:]):
                    cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), col, 1)

    # rysuj punkty
    for (x,y,v) in kps:
        if v > 0:
            radius = 3
            color = (0,0,255)            # czerwone kropki
            cv2.circle(out, (int(x),int(y)), radius, color, -1)
out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
cv2.imshow('Vehicle Pose', out)

"""
"""
if first_call:
    yaw_init = imu.getRollPitchYaw()[2]
    first_call = False
# automaty parkowania
yaw = imu.getRollPitchYaw()[2] - yaw_init
"""






"""
                image = names_images["camera_left_mirror"].copy()
                img_mapped = names_images["camera_left_mirror"].copy()
                corners,ids = fcc.detect_apriltags(image,"camera_left_mirror")
                poses = fcc.estimate_tag_pose(corners,ids,K_left_mirror,np.zeros(4).astype(np.float32),1.5)
                if poses is not None and len(poses) == 1:
                    rvec,tvec = poses[0]
                    mapx,mapy = init_ipm_maps(K_left_mirror,np.zeros(4).astype(np.float32),rvec,tvec,w,h,mpx)
                    img_mapped = cv2.remap(img_mapped,mapx,mapy,cv2.INTER_LINEAR)
                    cv2.imshow("img_left",img_mapped)
                image = names_images["camera_right_mirror"].copy()
                img_mapped = names_images["camera_right_mirror"].copy()
                corners,ids = fcc.detect_apriltags(image,"camera_right_mirror")
                poses = fcc.estimate_tag_pose(corners,ids,K_right_mirror,np.zeros(4).astype(np.float32),1.5)
                if poses is not None and len(poses) == 1:
                    rvec,tvec = poses[0]
                    mapx,mapy = init_ipm_maps(K_right_mirror,np.zeros(4).astype(np.float32),rvec,tvec,w,h,mpx)
                    img_mapped = cv2.remap(img_mapped,mapx,mapy,cv2.INTER_LINEAR)
                    cv2.imshow("img_right",img_mapped)

                
                image = names_images["camera_front_bumper"].copy()
                img_mapped = names_images["camera_front_bumper"].copy()
                corners,ids = fcc.detect_apriltags(image,"camera_front_bumper")
                poses = fcc.estimate_tag_pose(corners,ids,K_front_bumper,np.zeros(4).astype(np.float32),1.5)
                if poses is not None and len(poses) == 1:
                    rvec,tvec = poses[0]
                    mapx,mapy = init_ipm_maps(K_front_bumper,np.zeros(4).astype(np.float32),rvec,tvec,w,h,mpx)
                    img_mapped = cv2.remap(img_mapped,mapx,mapy,cv2.INTER_LINEAR)
                    cv2.imshow("img_front",img_mapped)

                
                image = names_images["camera_rear"].copy()
                img_mapped = names_images["camera_rear"].copy()
                corners,ids = fcc.detect_apriltags(image,"camera_rear")
                poses = fcc.estimate_tag_pose(corners,ids,K_rear,np.zeros(4).astype(np.float32),1.5)
                if poses is not None and len(poses) == 1:
                    rvec,tvec = poses[0]
                    mapx,mapy = init_ipm_maps(K_rear,np.zeros(4).astype(np.float32),rvec,tvec,w,h,mpx)
                    img_mapped = cv2.remap(img_mapped,mapx,mapy,cv2.INTER_LINEAR)
                    cv2.imshow("img_rear",img_mapped)
"""
                
                

                    
"""    
                    img_right = names_images[name_right]
                    orig_h, orig_w = img_right.shape[:2]
                    
                    img_left = names_images[name_left]
                    right_copy = img_right.copy()
                    
                    grayL = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                    grayR = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)


                    #TUTAJ DALEJ ODFILTROWANE DISPARITY
                    # oblicz disparity z lewej i prawej kamery
                    disp_left = stereo_left.compute(grayL, grayR).astype(np.float32) / 16.0
                    disp_right = stereo_right.compute(grayR, grayL).astype(np.float32) / 16.0

                    # filtruj disparity
                    filtered_disp = wls_filter.filter(disp_left, grayL, None,disp_right)

                    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
                    disp_vis = np.nan_to_num(disp_vis, nan=0.0, posinf=0.0, neginf=0.0)
                    disp_vis = np.uint8(disp_vis)
                    cv2.namedWindow("Disparity WLS filtered",cv2.WINDOW_NORMAL)
                    cv2.imshow("Disparity WLS filtered", disp_vis)

                    masks = results[0].masks.data.cpu().numpy()  # shape: (num_detections, H, W)
                    

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

                        p1_3d, p2_3d = sy.points_from_mask_to_3D(mask_resized, filtered_disp, K_right, 0.05, T_center_to_camera)

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


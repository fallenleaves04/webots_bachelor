import numpy as np
import cv2
from controller import (Robot, Keyboard, Supervisor, Display)
from vehicle import Car
import sys
import visualise as vis
from camera_calibration import save_homo
from park_algo import TrajStateMachine,Kalman,OccupancyGrid,mod2pi,C,PlanningWorker,Path
#from fisheye_camera_calibration import estimate_tag_pose,detect_apriltags
import stereo_yolo as sy
from ultralytics import YOLO
import pandas as pd
import os
import time
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

#sys.path.append(r"D:\\User Files\\BACHELOR DIPLOMA\\Kod z Github (rozne algorytmy)")

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
    global steering_angle,manual_steering
    wheel_angle = max(min(wheel_angle, C.MAX_WHEEL_ANGLE + 0.05), -C.MAX_WHEEL_ANGLE - 0.05)
    steering_angle = wheel_angle
    driver.setSteeringAngle(steering_angle)
    manual_steering = steering_angle / 0.02 
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

class ExpRecorder:
    # dla zapisywania do tabel, plików, plotów *.png
    def __init__(self, controller, output_dir="obr_do_spr\\proby parkowania NOWE"):
        
        self.ctrl = controller
        self.output_dir = output_dir

        self.run_id = 1
        self.metrics_runs = []

        # foldery
        self.plots_dir = os.path.join(output_dir, "wykresy")
        self.tables_dir = os.path.join(output_dir, "tabele")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

    def register_views(self, *, traj=None, speed=None, angle_yaw=None, angle_delta=None, yaw_kappa=None):
        self.traj_view:vis.TrajView = traj
        self.speed_view:vis.SpeedView = speed
        self.angle_yaw:vis.AngleView1 = angle_yaw
        self.angle_delta:vis.AngleView2 = angle_delta
        self.yaw_kappa:vis.YawKappaView = yaw_kappa

    def _save_plot(self, plot_item: pg.PlotItem, filename, size_px=1600):
        vb = plot_item.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        vb.autoRange(padding=0.05)

        exporter = ImageExporter(plot_item)
        exporter.parameters()['width'] = size_px
        exporter.parameters()['height'] = size_px
        exporter.parameters()['antialias'] = True

        exporter.export(filename)
    
    def save_plots(self):
        tag = f"run_{self.run_id}"

        if self.traj_view:
            self._save_plot(
                self.traj_view.view_trajectory,
                f"{self.plots_dir}/trajectory_{tag}.png"
            )

        if self.speed_view:
            self._save_plot(
                self.speed_view.view_speed,
                f"{self.plots_dir}/speed_{tag}.png"
            )

        if self.angle_yaw:
            self._save_plot(
                self.angle_yaw.view_angle,
                f"{self.plots_dir}/angle_{tag}.png"
            )

        if self.angle_delta:
            self._save_plot(
                self.angle_delta.view_angle,
                f"{self.plots_dir}/angle_{tag}.png"
            )

        if self.yaw_kappa:
            self._save_plot(
                self.yaw_kappa.error_plot,
                f"{self.plots_dir}/error_s_run_{self.run_id}.png"
            )
        print(f"[Recorder] Zapisano wykresy dla {tag}")
    
    def add_metrics(self, metrics: dict):
        self.metrics_runs.append(metrics)
        self._write_latex_table()
        self.run_id += 1

    def _write_latex_table(self):
        cols = len(self.metrics_runs)

        def fmt(x, nd=3):
            if isinstance(x, float):
                return f"{x:.{nd}f}".replace('.', ',')
            return str(x)

        rows = [
            ("$L_s$ [m]", "path_length"),
            ("$N_s$", "segments"),
            ("$\\over\\{e_r\\}$ [m]", "mean_er"),
            ("$\\Delta x$ [m]", "x_err"),
            ("$\\Delta y$ [m]", "y_err"),
            ("$\\Delta \\psi$ [$^\\circ$]",
             lambda m: np.degrees(m["yaw_err"])),
            ("$T_p$ [s]", "time"),
            ("$\\over{v}$ [m/s]", "mean_speed"),
        ]

        path = f"{self.tables_dir}/wyniki_tabela.tex"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\\begin{table}[H]\n\\centering\n")
            f.write("\\caption{Wyniki parkowania}\n")
            f.write("\\label{tab:parking}\n")

            f.write("\\begin{tabular}{p{6.5cm}" + " c"*cols + "}\n")
            f.write("\\toprule\n")

            f.write("\\textbf{Wielkość}")
            for i in range(cols):
                f.write(f" & \\textbf{{Próba {i+1}}}")
            f.write("\\\\\n\\midrule\n")

            for label, key in rows:
                f.write(label)
                for run in self.metrics_runs:
                    val = key(run) if callable(key) else run[key]
                    f.write(f" & {fmt(val)}")
                f.write("\\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        print(f"[Recorder] Zapisano tabelę LaTeX ({cols} prób)")


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
    funcCarUpdated = vis.QtCore.pyqtSignal(object) # dla wykresów yaw, kappa i delta - weryfikacja ścieżki
    segmentProgressUpdated = vis.QtCore.pyqtSignal(object) # dla strzałki na wizualizacji, progres segmentu
    clearDataPlots = vis.QtCore.pyqtSignal() # do czyszczenia wykresów
    sensorStats = vis.QtCore.pyqtSignal(dict,dict)
    sensorConeDraw = vis.QtCore.pyqtSignal(str,float,float,float,float,float) # sensor_name, sensor_global_x, sensor_global_y, sensor_global_theta, max_range, beta
    yoloRects = vis.QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.parking = False
        self.start_pose = None
        self.goal_pose = None
        self.state = "inactive"  
        self.planning_active = False  
        self.stopped = False
        self.path : Path = None
        # timer dla zatrzymania się przed planowaniem
        self.timer = 0
        # strona i typ miejsca
        self.side = "right"
        self.find_type = "parallel"
        # timer dla ukończenia parkingu
        self.finish_timer = 0.0
        self.parking_finished = False
        # szukanie miejsc
        self.found_spot = False
        self.found_another_spot = False
        # żeby nie ustawiać czujników kilka razy i nie przekazywać słowników
        self.sensors_set = False
        self.sensor_names = None
        self.sensor_poses = None
        # dla zapisu logów i plotów
        self.recorder = None
        # sygnały
        self.sensorStats.connect(self.receive_sensor_stats)

    @vis.QtCore.pyqtSlot(dict, dict)
    def receive_sensor_stats(self,params,poses):
        self.sensor_params = params
        self.sensor_poses = poses
        self.sensors_set = True

    @vis.QtCore.pyqtSlot()
    def toggle_parking(self):
        self.parking = not self.parking
        self.timer = 0.0
        self.v_set = 0.0
        if self.state == "inactive":
            self.state = "inactive_waiting"
        else:
            self.planning_active = False
            self.stopped = False
            self.path = None
            self.pathUpdated.emit(Path([], [], [], [], [], []))
            self.state = "inactive"
            self.sensors_set = False
            
        self.parkingToggled.emit(self.parking)
        self.stateUpdated.emit(self.state)
   

# IMPLEMENTACJA MAIN ALE W QTHREAD, ŻEBY ZROBIĆ WIZUALIZACJĘ DOBRĄ ; CAŁE RUN MOŻNA PRZENIEŚĆ DO DEF MAIN(), JEŻELI QT NIEPOTRZEBNE


class MainWorker(vis.QtCore.QObject):
    
    global manual_steering

    sensorData  = vis.QtCore.pyqtSignal(object) # dane z czujników ultradźwiękowych
    poseData    = vis.QtCore.pyqtSignal(object) # pomiary z Webots
    trajData    = vis.QtCore.pyqtSignal(object) # trajektoria dla rysowania plus przeszkody i miejsca
    pathData    = vis.QtCore.pyqtSignal(object) # dane o ścieżce
    speedData   = vis.QtCore.pyqtSignal(object) # wykres prędkości
    angleData   = vis.QtCore.pyqtSignal(object) # wykresy skrętu i odchylenia
    stateData   = vis.QtCore.pyqtSignal(str)    # stan parkowania - poszukiwanie, planowanie, wykonywanie
    pathCarData = vis.QtCore.pyqtSignal(object) # dla przesyłania bieżącego śledzonego punktu trajektorii
    funcCarData = vis.QtCore.pyqtSignal(object) # dla przesyłania yaw, kappa, delta - wykresy weryfikacji ścieżki
    segmentProgressData = vis.QtCore.pyqtSignal(object) # dla przesyłania progresu przejazdu segmentu ścieżki
    clearDataPlots = vis.QtCore.pyqtSignal() # do czyszczenia wykresów
    finished    = vis.QtCore.pyqtSignal(bool)   # ukończenie życia wątku
    

    def __init__(self,supervisor,controller:VisController,main_window:vis.MainWindow):
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
        self.delta_tracked = 0.0
        self.executing_maneuver = False
        self.steer_hold_timer = 0.0
        self.STEER_HOLD_TIME = 0.2
        self.v_set = 0.0

        self.main_window = main_window

        self.prev_time = driver.getTime()
        self.prev_real = time.time()

        self.t_plan_1 = driver.getTime()
        self.t_plan_2 = driver.getTime()

        self.t_real_plan_1 = time.time()
        self.t_real_plan_2 = time.time()

        
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
        
        print("[MainWorker] Uruchamiam wątek planowania")
        self.t_plan_1 = driver.getTime()
        self.t_real_plan_1 = time.time()
        self.planning_worker = PlanningWorker(self.controller,ogm)
        self.planning_thread = vis.QtCore.QThread()
        self.planning_worker.moveToThread(self.planning_thread)

        self.planning_thread.started.connect(self.planning_worker.run)
        self.planning_worker.finished.connect(self.planning_thread.quit)
        self.planning_worker.finished.connect(self.planning_worker.deleteLater)
        self.planning_thread.finished.connect(self.planning_thread.deleteLater)

        self.planning_worker.finished.connect(self.planning_finished) 

        self.planning_thread.start()
        self.clearDataPlots.emit()

    def set_state(self, new_state):
        self.controller.state = new_state
        self.stateData.emit(new_state)
        #print(f"[MainWorker] Zmieniony stan kontrolera: {new_state}")
            
    @vis.QtCore.pyqtSlot(bool)
    def planning_finished(self,success):
        self.controller.planning_active = False
        self.executing_maneuver = success
        if success:
            self.t_plan_2 = driver.getTime()
            self.t_real_plan_2 = time.time()
            print(f"[PlanningWorker] Czas na planowanie driver: {self.t_plan_2 - self.t_plan_1} s, real: {self.t_real_plan_2 - self.t_real_plan_1} s") 
            print(f"[PlanningWorker] Długość ścieżki: {self.controller.path.s[-1]}")
            self.set_state("stop_for_change")
        else:
            
            self.set_state("searching")
            

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
            if key == Keyboard.UP:
                set_speed(speed + 0.5,driver)

            elif key == Keyboard.DOWN:
                set_speed(speed - 0.5,driver)

            if key == Keyboard.RIGHT:
                change_manual_steering_angle(+2.0,driver)

            elif key == Keyboard.LEFT:
                change_manual_steering_angle(-2.0,driver)
            
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
            nonlocal psi,x_odo,y_odo,gp0,im0,node_pos0,yaw_est
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
            # gps
            node_pos = self.node.getPosition()
            node_vel = self.node.getVelocity()
            # imu
            rpy = imu.getRollPitchYaw()
            yaw_imu = mod2pi(rpy[2] - im0[2])
            im = [rpy[0] - im0[0], rpy[1] - im0[1], yaw_imu]
            # supervisor
            R = R_xyz(rpy[2], rpy[1], rpy[0])   # ZYX
            node_vel_xyz = R.T @ np.array(node_vel[:3])
            node_vel_x = node_vel_xyz[0]  
            # żyroskop
            gyr = gyro.getValues()
            dpsi_gyr = gyr[2] * dt
            yaw_est = mod2pi(yaw_est + dpsi_gyr)
            # odometria
            sp_odo_meas,encoders = get_speed_odo(dt)
            # kąt skrętu kół [rad]
            delta_meas = -driver.getSteeringAngle()
            old_yaw = psi
            x_pred = kalman.predict(np.array([x_odo,y_odo,psi,sp_odo_meas,delta_meas]),dt)
            x_upd = kalman.update(x_pred,np.array([yaw_est]))
            x_odo = x_upd[0]
            y_odo = x_upd[1]
            psi = x_upd[2]
            sp_odo = x_upd[3]
            delta = x_upd[4]
            new_yaw = psi
            yaw_rate = (new_yaw - old_yaw) / dt
            # akcelerometr
            accer = acc.getValues()
            # node position
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]
            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":psi,"yaw_rate":yaw_rate,"dt":dt,"x_odo":x_odo,
                    "y_odo":y_odo,"encoders":encoders,"node_pos":node_pos,"acc":accer,
                    "node_vel":node_vel_xyz,"node_vel_x":node_vel_x}

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
                self.prev_real = time.time()
                self.prev_time = driver.getTime()
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
            # aktualizacja z uśrednieniem psi, później się dodaje
            dpsi_odo = (sp_odo * np.tan(delta) / wheelbase) * dt
            # akcelerometr
            accer = acc.getValues()
            
            old_yaw = psi
            x_odo += sp_odo * dt * np.cos(psi)
            y_odo += sp_odo * dt * np.sin(psi)
            psi += dpsi_odo
            psi = mod2pi(psi)
            new_yaw = psi
            
            # old_x,old_y,old_yaw = x_odo,y_odo,psi
            # new_x,new_y,new_yaw = runge_kutta_odo(old_x,old_y,old_yaw,sp_odo,delta,dt)
            # x_odo,y_odo,psi = new_x,new_y,new_yaw
            yaw_rate = (new_yaw - old_yaw) / dt
            # psi = wrap_angle(alpha * psi_odo + (1-alpha)*(old_yaw + dpsi_gyr))
            # supervisor
            x_node,y_node = webots_to_odom_xy(node_pos[0] - node_pos0[0],node_pos[1] - node_pos0[1],im0[2])
            node_pos = [x_node,y_node,node_pos[2] - node_pos0[2]]

            # yaw-rate dla obliczenia krzywizny
            

            return {"sp_odo":sp_odo,"im":im,"delta":delta,"psi":psi,"yaw_rate":yaw_rate,"dt":dt,"x_odo":x_odo,
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
        
        
        ogm = OccupancyGrid(self.controller)
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

        # pomocnicze DLA PLANERA I REGULATORA
        p1 = None
        p2 = None
        # test ścieżki
        import reeds_shepp
        path_built = False
        path = None
        ref_path = None
        
        # dla miejsc
        old_type = None
        
        # dla ścieżki
        init_maneuver = True
        delta_prev = 0.0
        ind_la = None

        
         

        class ExpLogger:
            def __init__(self):
                self.reset()

            def reset(self):
                self.active = False
                self.t0 = None

                self.t = []
                self.delta = []

                self.yaw_webots = []
                self.yaw_odo = []

                self.x = []
                self.y = []

                self.v = []
                self.er = []

                self.x_webots = []
                self.y_webots = []
                self.v_webots = []
            def start(self, sim_time):
                self.reset()
                self.active = True
                self.t0 = sim_time

            def stop(self):
                self.active = False

            def log(self, sim_time, *, delta, yaw_webots, yaw_odo, x, y, v, x_webots, y_webots, v_webots, er=None):
                if not self.active:
                    return

                t_rel = sim_time - self.t0

                self.t.append(t_rel)
                self.delta.append(delta)
                self.yaw_webots.append(yaw_webots)
                self.yaw_odo.append(yaw_odo)
                self.x.append(x)
                self.y.append(y)
                self.v.append(v)
                self.x_webots.append(x_webots)
                self.y_webots.append(y_webots)
                self.v_webots.append(v_webots)
                self.er.append(er)
            
        def compute_metrics(log: ExpLogger, planned_path: Path):
            t = np.array(log.t)

            # odometria
            x_odo = np.array(log.x)
            y_odo = np.array(log.y)
            v_odo = np.array(log.v)
            yaw_odo = np.array(log.yaw_odo)

            # webots (ground truth)
            x_web = np.array(log.x_webots)
            y_web = np.array(log.y_webots)
            v_web = np.array(log.v_webots)
            yaw_web = np.array(log.yaw_webots)

            er = np.array([e for e in log.er if e is not None])

            total_time = t[-1] - t[0]
            mean_speed = np.mean(v_odo)
            mean_speed_err = np.mean(np.abs(v_odo - v_web))

            dx = np.diff(x_odo)
            dy = np.diff(y_odo)
            path_length = np.sum(np.hypot(dx, dy))

            num_segments = len(planned_path.segments)

            mean_er = np.mean(np.abs(er)) if len(er) > 0 else np.nan

            pos_err = np.hypot(x_odo - x_web, y_odo - y_web)
            mean_pos_err = np.mean(pos_err)
            mean_x_err = np.mean(np.abs(x_odo - x_web))
            mean_y_err = np.mean(np.abs(y_odo - y_web))

            yaw_err = np.array([mod2pi(a - b) for a, b in zip(yaw_odo, yaw_web)])
            mean_yaw_err = np.mean(np.abs(yaw_err))

            x_end_err = x_odo[-1] - planned_path.goal[0]
            y_end_err = y_odo[-1] - planned_path.goal[1]
            yaw_end_err = mod2pi(yaw_odo[-1] - planned_path.goal[2])

            return {
                "path_length": path_length,
                "segments": num_segments,
                "time": total_time,

                # śledzenie
                "mean_er": mean_er,

                # Webots vs Odometria
                "mean_pos_err": mean_pos_err,
                "mean_x_err_webots": mean_x_err,
                "mean_y_err_webots": mean_y_err,
                "mean_speed_err": mean_speed_err,
                "mean_yaw_err": mean_yaw_err,

                # końcowe
                "x_err": x_end_err,
                "y_err": y_end_err,
                "yaw_err": yaw_end_err,
                "mean_speed": mean_speed,
            }

        self.exp_logger = ExpLogger()
        write_logs = True

        def plot_kdtree_query_debug(obstacles, kd_tree, car_pose, car_L, car_W,
                            max_obs_radius, margin=0.1, out_path="kdtree_query.png"):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
            """
            obstacles: lista słowników z polami:
                - center: (cx, cy)
                - length, width
                - yaw (opcjonalnie) -> jeśli brak, rysujemy AABB
            car_pose: (x, y, yaw) - najlepiej środek auta albo rear-axle (spójnie z Twoim plannerem)
            """
            x, y, yaw = car_pose

            # promień auta jako półprzekątna (bezpieczny)
            car_radius = 0.5 * np.hypot(car_L, car_W)
            rq = car_radius + max_obs_radius + margin

            # broad-phase query
            cand_idx = []
            if kd_tree is not None and len(obstacles) > 0:
                cand_idx = kd_tree.query_ball_point([x, y], rq)

            fig, ax = plt.subplots(figsize=(8, 5))

            # okrąg zapytania
            ax.add_patch(Circle((x, y), rq, fill=False, linestyle="--", linewidth=2))
            ax.plot([x], [y], marker="o")  # punkt zapytania

            # auto jako prostokąt (osiowo, dla debug; jeśli chcesz obrót, też da się)
            car_rect = Rectangle((x - car_L/2, y - car_W/2), car_L, car_W,
                                    fill=False, linewidth=2)
            ax.add_patch(car_rect)

            # przeszkody
            for i, obs in enumerate(obstacles):
                cx, cy = obs["center"]
                L = obs.get("length", 1.0)
                W = obs.get("width", 1.0)

                is_cand = i in cand_idx
                rect = Rectangle((cx - L/2, cy - W/2), L, W,
                                    fill=True, alpha=0.35 if is_cand else 0.15,
                                    linewidth=1.5)
                ax.add_patch(rect)
                ax.plot([cx], [cy], marker=".", markersize=6)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title("KD-tree broad-phase: query_ball_point()")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, alpha=0.25)

            # dopasuj widok
            pad = rq + 2.0
            ax.set_xlim(x - pad, x + pad)
            ax.set_ylim(y - pad, y + pad)

            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            return out_path, len(cand_idx), rq

        while robot.step(64) != -1:
            check_keyboard(cont)
            vis.QtCore.QCoreApplication.processEvents()

            now_real = time.time()
            dt_real = now_real - self.prev_real
            self.prev_real = now_real
            
            now = driver.getTime()
            dt_sim = now - self.prev_time
            self.prev_time = now

            pose_measurements = get_pose(dt_sim)
            
            names_images = dict(zip(camera_names, [get_camera_image(c) for c in cameras]))
            image = names_images[name].copy()
            

            front_dists = [process_distance_sensors(s) for s in front_sensors]
            rear_dists = [process_distance_sensors(s) for s in rear_sensors]
            right_side_dists = [process_distance_sensors(s) for s in right_side_sensors]
            left_side_dists = [process_distance_sensors(s) for s in left_side_sensors]

            front_names_dists = dict(zip(front_sensor_names, front_dists))
            rear_names_dists = dict(zip(rear_sensor_names, rear_dists))
            left_side_names_dists = dict(zip(left_side_sensor_names, left_side_dists))
            right_side_names_dists = dict(zip(right_side_sensor_names, right_side_dists))
            
            self.sensorData.emit([front_names_dists,rear_names_dists,
                                  left_side_names_dists,right_side_names_dists,
                                  max_min_dict])
            
            if cont.parking:
                x_odo = pose_measurements["x_odo"]
                y_odo = pose_measurements["y_odo"]
                yaw_odo = pose_measurements["psi"]
                yaw_rate = pose_measurements.get("yaw_rate", 0.0)
                delta_meas = pose_measurements["delta"]
                sp_odo = pose_measurements["sp_odo"]
                node_vel_x = pose_measurements["node_vel_x"]
                node_pos = pose_measurements["node_pos"]
                yaw_webots = pose_measurements["im"][2]
                
                
                if not self.executing_maneuver:
                    if not (self.controller.planning_active or self.controller.state in ["executing","drive_forward","drive_backward","stop_for_change","stop_at_end","parking_finished"]):
                        if self.main_window.progress_arrow.progress > 0.0:
                            self.main_window.progress_arrow.set_progress(0.0)
                        ogm.interpret_readings({**front_names_dists,**rear_names_dists,**left_side_names_dists,**right_side_names_dists},(x_odo,y_odo,yaw_odo))
                        # mamy macierz grid tych wielkości; z nich trzeba przemnożyć na xy_resolution te indeksy, aby otrzymać właściwe pozycje przeszkód, są większe lub równe od 0
                        ox,oy = ogm.extract_obstacles()
                        #ox,oy = [],[]
                        find_type = self.controller.find_type
                        side = self.controller.side
                        type_changed = old_type != find_type
                        if type_changed:
                            self.set_state(f"searching_{side}_{find_type}")
                            old_type = find_type
                        
                        if self.controller.side == "right":
                            names_yolo = ["camera_front_right","camera_right_mirror"]
                        else:
                            names_yolo = ["camera_front_right","camera_left_mirror"]
                        #names_yolo = ["camera_front_right","camera_left_mirror","camera_right_mirror"]
                        for name in names_yolo:  
                            image = names_images[name].copy()
                            results = model(image,half=True,device = 0,classes=[2,5,7],conf=0.6,verbose=False,imgsz=(640,480))
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
                                    
                                #cv2.namedWindow(f"yolo {name}", cv2.WINDOW_NORMAL)
                                #cv2.imshow(f"yolo {name}", cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                        
                        spots = []
                        if len(ox) > 0:
                            cls = ogm.analyze_clusters((x_odo,y_odo,yaw_odo)) 
                            #if len(cls_old) < len(cls):
                            #cls_old = cls
                            ogm.setup_obstacles(cls)
                            #plot_kdtree_query_debug(ogm.obstacles,ogm.kd_tree,(x_odo,y_odo,yaw_odo),C.CAR_LENGTH,C.CAR_WIDTH,ogm.collision_radius + ogm.max_obs_radius)
                            ogm.match_semantics_with_sat()
                            spots = ogm.find_spots_scanning((x_odo,y_odo,yaw_odo), find_type, side)
                            #spots.extend(ogm.find_spots_scanning((x_odo,y_odo,yaw_odo), "parallel", "left"))
                            ogm.spots = spots
                        else:
                            ox,oy,cls,spots = [],[],[],[]
                    #ox,oy,cls,spots = ogm.ox,ogm.oy,ogm.obstacles,ogm.spots
                    

                    


                    # dalej planowanie i sterowanie 


                    p1 = (x_odo,y_odo,yaw_odo)
                    p2,_ = ogm.choose_spot(p1)
                    
                    if 'searching' in self.controller.state and p2 is not None:
                        self.controller.found_spot = True
                        target_spot = p2
                        self.set_state("found_spot")
                        print("[MainWorker] Znaleziono miejsce! Proszę się zatrzymać.")
                    elif self.controller.state not in ["executing","stop_for_change","stop_at_end","drive_forward","drive_backward","parking_finished"] and self.controller.found_spot and p2 is None:
                        self.set_state("return_searching")
                        self.controller.found_spot = False
                        target_spot = None
                        print("[MainWorker] Miejsce porzucone. Poszukiwanie dalsze...")
                    if self.controller.state in ["return_searching"] and p2 is not None:
                        tc = target_spot['center'] or (0,0)
                        pc = p2['center']
                        dist = np.hypot(pc[0] - tc[0], pc[1] - tc[1])
                        if dist > 2.0:
                            target_spot = p2
                            self.set_state("found_another_spot")
                            print("[MainWorker] Znaleziono nowe miejsce! Proszę się zatrzymać.")

                    if (self.controller.found_spot or self.controller.found_another_spot) and self.controller.state in ["found_spot","found_another_spot"]:
                        if abs(sp_odo) <= 1e-4:
                            self.controller.timer += dt_real
                        self.controller.stopped = True if self.controller.timer >= 1.0 and abs(sp_odo) <= 1e-4 else False
                        if self.controller.stopped:  
                            self.set_state("waiting_for_confirm_start")
                            print("[MainWorker] Proszę wcisnąć odpowiedni przycisk, aby rozpocząć parkowanie.")

                    if not self.controller.planning_active and self.controller.state == "planning" and self.controller.stopped:
                        p1 = (x_odo,y_odo,yaw_odo)
                        target_spot,_ = ogm.choose_spot(p1)
                        self.exp_logger.start(driver.getTime())
                        self.controller.start_pose = p1
                        self.controller.planning_active = True
                        self.set_state("planning")
                        self.controller.goal_pose = (target_spot['target_rear_axle'][0],
                                                        target_spot['target_rear_axle'][1],
                                                        target_spot['orientation'])
                        try:
                            
                            self.start_planning_thread(ogm)
                        except:
                            print("[MainWorker] Nie udało się uruchomić planowania.")   

                # dać takie stany, żeby samochód się zatrzymał dopiero kiedy błąd wzdłużny przekroczy jakiś próg
                # potem po zatrzymaniu się samochód czeka na przekręcenie kół i dopiero wtedy zmienia segment

                if self.controller.state != "parking_finished":
                    traj_to_send = [[x_odo,y_odo,yaw_odo],node_pos,[ogm.ox,ogm.oy],ogm.obstacles,ogm.spots,[ogm.yolo_x_pts,ogm.yolo_y_pts]]
                    
                    traj_data = traj_to_send 
                    
                    angle_data = [now,delta_meas,psi,yaw_webots]
                    
                    self.poseData.emit(pose_measurements)
                    self.trajData.emit(traj_data)
                    #speed_data = [now,sp_odo,node_vel_x]
                    #self.speedData.emit(speed_data)
                    self.angleData.emit(angle_data)

                
                def rate_limit(target, prev, rate, dt):
                    return np.clip(target, prev - rate*dt, prev + rate*dt)

                LD_PP = 0.9
                STEER_RATE = 1.25
                END_TOLER = 0.05
                V_STOP_TOL = 5e-2
                DELTA_TOL = 0.02
                NEXT_SEG_PEEK = 5 


                if self.executing_maneuver:

                    x, y, yaw, v = x_odo, y_odo, yaw_odo, sp_odo
                    v_webots = node_vel_x
                    x_webots = node_pos[0] 
                    y_webots = node_pos[1]
                    yaw_webots = yaw_webots

                    self.planned_path = self.controller.path or Path([], [], [], [], [])

                    # ===== wspólne =====
                    seg_start, seg_end = self.planned_path.get_segment_bounds()
                    theta_e, er, _, _, ind_nearest = self.planned_path.calc_theta_e_and_er(x, y, yaw)

                    s_start = self.planned_path.s[seg_start]
                    s_end = self.planned_path.s[seg_end]
                    s_now = self.planned_path.s[ind_nearest]
                    dir_seg = self.planned_path.directions[seg_start]

                    # progress
                    progress = (s_now - s_start) / max(abs(s_end - s_start), 1e-6)
                    progress = np.clip(progress, 0.0, 1.0)
                    self.main_window.progress_arrow.set_progress(progress)

                    self.exp_logger.log(
                        driver.getTime(),
                        delta=delta_meas,
                        yaw_webots=yaw_webots,
                        yaw_odo=yaw_odo,
                        x=x_odo,
                        y=y_odo,
                        v=sp_odo,
                        x_webots=x_webots,
                        y_webots=y_webots,
                        v_webots=v_webots,
                        er=er
                    )

                    speed_data = [now,sp_odo,node_vel_x,self.v_set]
                    self.speedData.emit(speed_data)
                    
                    if self.controller.state in ["drive_forward", "drive_backward"]:

                        # --- steering: PURE PURSUIT ---
                        delta_pp, ind_la = self.planned_path.pure_pursuit(x, y, yaw, v, ld=LD_PP)
                        delta_pp = np.clip(delta_pp, -C.MAX_WHEEL_ANGLE - 0.05, C.MAX_WHEEL_ANGLE + 0.05)

                        delta_cmd = rate_limit(delta_pp, delta_prev, STEER_RATE, dt_sim)
                        delta_prev = delta_cmd
                        set_steering_angle(delta_cmd, driver)   

                        self.delta_tracked = delta_pp

                        # --- speed: NIE DOTYKAMY ---
                        dist_to_end = max(0.0, s_end - s_now)
                        v_cmd = dir_seg * C.MAX_SPEED
                        if abs(er) > 0.02:
                            v_cmd *= 0.5

                        self.v_set = self.planned_path.speed_control(
                            v_cmd, v, dist_to_end, 0.0, self.delta_tracked, dt_sim
                        )

                        
                        set_speed(self.v_set, driver)

                        kappa = np.nan if abs(v) < 1e-3 else yaw_rate / abs(v)
                        stats = [er, delta_meas, kappa, ind_nearest] # statystyki dla wysłania na widget
                        self.funcCarData.emit(stats)

                        # tracked_pose = (self.planned_path.xs[ind_la], self.planned_path.ys[ind_la], self.planned_path.yaws[ind_la]) 
                        # self.pathCarData.emit(tracked_pose)

                        at_end_geom = (seg_end - ind_nearest) <= 0 or abs(s_end - s_now) <= END_TOLER

                        if at_end_geom:
                            print("[FSM] Koniec segmentu -> STOP_AT_END")
                            self.delta_hold = self.delta_tracked   # ZAMRAŻAMY skręt
                            self.set_state("stop_at_end")

                            print(
                                f"[FSM->STOP_AT_END] "
                                f"delta_tracked={self.delta_tracked:.3f}, "
                                f"delta_prev={delta_prev:.3f}, "
                                f"delta_meas={delta_meas:.3f}"
                            )
                            
                        continue

                    if self.controller.state == "stop_at_end":

                        # skręt ZAMROŻONY
                        delta_cmd = rate_limit(self.delta_hold, delta_prev, STEER_RATE, dt_sim)
                        delta_prev = delta_cmd
                        set_steering_angle(delta_cmd, driver)

                        dist_to_end = max(0.0, s_end - s_now)
                        v_cmd = dir_seg * C.MAX_SPEED
                        if abs(er) > 0.02:
                            v_cmd *= 0.5

                        self.v_set = self.planned_path.speed_control(
                            v_cmd/3.6, v, dist_to_end, 0.0, self.delta_tracked, dt_sim
                        )
                        self.v_set *= 3.6
                        
                        set_speed(self.v_set, driver)

                        last_segment = (self.planned_path.active_segment == len(self.planned_path.segments) - 1)

                        print(
                            f"[STOP_AT_END] v={sp_odo:.3f}, "
                            f"v_cmd={self.v_set:.3f}, "
                            f"delta_hold={self.delta_hold:.3f}, "
                            f"delta_prev={delta_prev:.3f}"
                        )
                        # if ind_la is not None:
                        #     tracked_pose = (self.planned_path.xs[ind_la], self.planned_path.ys[ind_la], self.planned_path.yaws[ind_la]) 
                        #     self.pathCarData.emit(tracked_pose)
                        
                        if abs(sp_odo) < V_STOP_TOL:
                            if last_segment:
                                self.controller.finish_timer += dt_real  
                                if self.controller.finish_timer >= 1.5:
                                    print("[FSM] Ostatni segment -> PARKING_FINISHED (timer ok)")
                                    self.v_set = 0.0
                                    set_speed(self.v_set, driver)
                                    self.set_state("parking_finished")
                                    self.controller.parking_finished = True
                                    self.found_spot = False
                                    self.found_another_spot = False
                            else:
                                # nie ostatni segment: przejście do obracania kół
                                self.controller.finish_timer = 0.0
                                print("[FSM] Stoi -> STOP_FOR_CHANGE")
                                self.v_set = 0.0
                                set_speed(self.v_set, driver)
                                self.set_state("stop_for_change")
                        else:
                            # nie stoi, reset timera
                            if last_segment:
                                self.controller.finish_timer = 0.0

                        continue
                    
                    if self.controller.state == "stop_for_change":
                        self.v_set = 0.0
                        set_speed(self.v_set, driver)

                        # 1) wybierz punkt celowania: current segment dla INIT, next segment dla NEXT
                        if init_maneuver:
                            cur_start, cur_end = self.planned_path.get_segment_bounds()
                            look_idx = min(cur_start + NEXT_SEG_PEEK, cur_end)
                        else:
                            next_start, next_end = self.planned_path.get_next_segment_bounds()
                            look_idx = min(next_start + NEXT_SEG_PEEK, next_end)

                        tx = self.planned_path.xs[look_idx]
                        ty = self.planned_path.ys[look_idx]

                        # 2) policz delta_target (proste PP do punktu tx,ty)
                        alpha = np.arctan2(ty - y, tx - x) - yaw
                        delta_target = np.arctan2(
                            2.0 * C.WHEELBASE * np.sin(alpha),
                            max(LD_PP, 1e-3)
                        )
                        delta_target = np.clip(delta_target, -C.MAX_WHEEL_ANGLE - 0.05, C.MAX_WHEEL_ANGLE + 0.05)

                        # 3) rate limit i komenda
                        delta_cmd = rate_limit(delta_target, delta_prev, STEER_RATE, dt_sim)
                        delta_prev = delta_cmd
                        set_steering_angle(delta_cmd, driver)

                        # 4) błąd - UWAGA: NIE rób delta_meas *= -1 (to niszczy zmienną!)
                        delta_meas_eff = -delta_meas  # jeśli w Webots masz odwrócony znak pomiaru
                        delta_error = abs(delta_meas_eff - delta_target)

                        mode = "INIT" if init_maneuver else "NEXT"
                        print(f"[FSM:{mode}] err={delta_error:.3f}, meas={delta_meas_eff:.3f}, target={delta_target:.3f}, look_idx={look_idx}")

                        if delta_error < DELTA_TOL:
                            if init_maneuver:
                                init_maneuver = False
                                cur_start, _ = self.planned_path.get_segment_bounds()
                                dir_next = self.planned_path.directions[cur_start]
                            else:
                                self.planned_path.advance_segment()
                                seg_start2, _ = self.planned_path.get_segment_bounds()
                                dir_next = self.planned_path.directions[seg_start2]

                            self.planned_path.v_cmd_prev = 0.0

                            if dir_next > 0:
                                self.set_state("drive_forward")
                            else:
                                self.set_state("drive_backward")

                        continue
                

                if self.controller.state == "parking_finished" and self.controller.parking_finished:
                    self.exp_logger.stop()
                    if write_logs:
                        logs = compute_metrics(self.exp_logger,self.planned_path)
                        #self.controller.recorder.save_plots()
                        #self.controller.recorder.add_metrics(logs)
                        print(f"Logi po ukończeniu parkowania: {logs}")
                        write_logs = False
                    max_rate = 1.0
                    delta_tracked = 0.0
                    delta_cmd = np.clip(delta_tracked, delta_prev - max_rate*dt_sim, delta_prev + max_rate*dt_sim)
                    delta_prev = delta_cmd
                    set_steering_angle(delta_cmd,driver)
                    self.executing_maneuver = False
                    self.main_window.progress_arrow.set_progress(0.0)


                

                cv2.waitKey(1)
            elif not cont.parking:
                self.first_call_pose = True
                self.first_call_traj = True
                self.executing_maneuver = False
                ogm = OccupancyGrid(self.controller)
                ogm.setup_sensors(self.front_ultrasonic_sensor_poses,
                                    self.rear_ultrasonic_sensor_poses,
                                    [front_sensor_names,rear_sensor_names,right_side_sensor_names,left_side_sensor_names],
                                    [front_sen_apertures,rear_sen_apertures,right_side_sen_apertures,left_side_sen_apertures],
                                    max_min_dict)
                old_type = None
                write_logs = True


        app.quit()

        self.finished.emit()


if __name__ == "__main__":
    app = pg.QtWidgets.QApplication(sys.argv)
    

    cont = VisController()
    
    win = vis.MainWindow(cont)
    win.show()

    thread = vis.QtCore.QThread()
    worker = MainWorker(supervisor,cont,win)
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
    worker.funcCarData.connect(cont.funcCarUpdated)
    worker.segmentProgressData.connect(cont.segmentProgressUpdated)
    worker.clearDataPlots.connect(cont.clearDataPlots)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()

    speed_view = vis.SpeedView(cont)
    speed_view.hide()

    angle_view_1 = vis.AngleView1(cont)
    angle_view_1.hide()

    angle_view_2 = vis.AngleView2(cont)
    angle_view_2.hide()


    traj_view = vis.TrajView(cont)
    traj_view.hide()
    
    yaw_kappa_view = vis.YawKappaView(cont)
    yaw_kappa_view.hide()

    recorder = ExpRecorder(cont)
    recorder.register_views(
        traj=traj_view,
        speed=speed_view,
        angle_yaw=angle_view_1,
        angle_delta=angle_view_2,
        yaw_kappa=yaw_kappa_view
    )
    cont.recorder = recorder

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
    

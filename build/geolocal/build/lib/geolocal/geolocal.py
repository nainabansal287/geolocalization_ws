import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu, NavSatFix

from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import math

from ament_index_python.packages import get_package_share_directory
import os

class Geolocalization(Node):

    def __init__(self):
        super().__init__('Geolocalization')
        
        # --- Parameters ---
        self.declare_parameter('confidence', 0.4) 
        self.declare_parameter('output_csv', 'detections.csv')
        
        # --- TUNING PARAMETERS (Fix your ground roll/pitch here!) ---
        # If the drone says it has 3 deg roll on the ground, put -3.0 here.
        self.roll_offset_deg = 0.0
        self.pitch_offset_deg = 0.0

        self.conf_thresh = self.get_parameter('confidence').value
        self.output_csv = self.get_parameter('output_csv').value

        # --- Model Setup ---
        try:
            pkg_share = get_package_share_directory('geolocal')
            self.model_path = os.path.join(pkg_share, 'models', 'best1.pt')
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Loaded YOLO model from: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}. Loading 'yolov8n.pt' as fallback.")
            self.model = YOLO("yolov8n.pt") 

        self.bridge = CvBridge()

        # --- State Variables ---
        self.current_quat = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.base_altitude = None
        
        self.frame_id = 0
        self.csv_rows = []

        # --- Camera Intrinsics ---
        self.intrinsic_matrix = np.array([
            [380.0, 0.0, 320.080],
            [0.0, 380.0, 245.482],
            [0.0, 0.0, 1.0]
        ])

        # --- Camera to Body Transform ---
        # CASE 1: Standard Downward Camera (Top of image = Forward)
        # self.R_Cam_Body = np.array([
        #     [0, -1, 0],  # Cam X (Right) -> Body -Y (Right)
        #     [-1, 0, 0],  # Cam Y (Down)  -> Body -X (Back)
        #     [0,  0, -1]  # Cam Z (Fwd)   -> Body -Z (Down)
        # ])

        # CASE 2: Rotated 180 (Top of image = Backward)
        # ** TRY THIS IF TARGET MOVES OPPOSITE TO DRONE **
        self.R_Cam_Body = np.array([
            [0,  1, 0],   # Cam X (Right) -> Body Y (Left)
            [1,  0, 0],   # Cam Y (Down)  -> Body X (Forward)
            [0,  0, -1]   # Cam Z (Fwd)   -> Body -Z (Down)
        ])
        
        # --- Subscribers ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_image = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        
        self.sub_imu = self.create_subscription(
            Imu,
            '/mavros/imu/data', 
            self.imu_callback, 
            qos
        )

        self.sub_gps = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.pos_callback,
            qos
        )

    # --- Callbacks ---

    def pos_callback(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        
        if self.base_altitude is None: 
            self.base_altitude = msg.altitude
        
        self.altitude = msg.altitude - self.base_altitude
        
        if self.altitude < 0.5:
            self.altitude = 0.5
    
    def imu_callback(self, msg):
        self.current_quat = msg.orientation

    def image_callback(self, msg):
        self.frame_id += 1

        if self.current_quat is None or self.altitude is None:
            return

        # Snapshot state
        q_now = self.current_quat
        alt_now = self.altitude
        lat_now = self.latitude
        lon_now = self.longitude

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # YOLO Tracking
        results = self.model.track(
            frame,
            persist=True,
            classes=[0], 
            conf=self.conf_thresh,
            verbose=False
        )

        # Draw Overlay Data (for debugging)
        roll_deg, pitch_deg, yaw_deg = self.get_euler_deg(q_now)
        debug_text = f"R:{roll_deg:.1f} P:{pitch_deg:.1f} Y:{yaw_deg:.1f}"
        cv2.putText(frame, debug_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                
                u_center = (x1 + x2) / 2.0
                v_center = (y1 + y2) / 2.0

                # CALCULATION
                target_lat, target_lon = self.calculate_geolocation(
                    u_center, v_center, q_now, alt_now, lat_now, lon_now
                )

                if target_lat is not None:
                    self.get_logger().info(f"ID {track_id} | Alt: {alt_now:.1f}m | Tgt: {target_lat:.7f}, {target_lon:.7f}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Lat:{target_lat:.6f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    self.csv_rows.append({
                        'frame': self.frame_id,
                        'id': track_id,
                        'drone_lat': lat_now,
                        'drone_lon': lon_now,
                        'drone_alt': alt_now,
                        'drone_roll': roll_deg,
                        'drone_pitch': pitch_deg,
                        'drone_yaw': yaw_deg,
                        'target_lat': target_lat,
                        'target_lon': target_lon
                    })

        cv2.imshow("YOLO Drone View", frame)
        cv2.waitKey(1)

    # --- Core Math ---

    def get_euler_deg(self, q):
        """Helper to see what the drone thinks its orientation is."""
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)

    def get_corrected_rotation_matrix(self, q):
        """
        Rebuilds the Rotation Matrix from Euler angles, 
        applying the manual OFFSETS to fix ground-level errors.
        """
        # 1. Get raw Euler
        roll_rad, pitch_rad, yaw_rad = self.get_euler_from_quat(q)

        # 2. Apply Offsets (Fixing the sensor error)
        roll_rad += np.deg2rad(self.roll_offset_deg)
        pitch_rad += np.deg2rad(self.pitch_offset_deg)

        # 3. Rebuild Matrix (Z-Y-X sequence)
        c_y = np.cos(yaw_rad)
        s_y = np.sin(yaw_rad)
        c_p = np.cos(pitch_rad)
        s_p = np.sin(pitch_rad)
        c_r = np.cos(roll_rad)
        s_r = np.sin(roll_rad)

        # R_Body_to_World
        R = np.array([
            [c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
            [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r],
            [-s_p,    c_p*s_r,               c_p*c_r]
        ])
        return R

    def get_euler_from_quat(self, q):
        """Standard conversion without deg conversion."""
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def calculate_geolocation(self, u, v, q, alt, lat, lon):
        
        # 1. Intrinsics
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]

        ray_cam = np.array([
            (u - cx) / fx,
            (v - cy) / fy,
            1.0
        ])

        # 2. Get Corrected Rotation Matrix
        R_Body_World = self.get_corrected_rotation_matrix(q)

        # 3. Transform Ray
        ray_body = self.R_Cam_Body @ ray_cam
        ray_world = R_Body_World @ ray_body

        # 4. Intersect Ground
        if ray_world[2] >= -0.01: # Check if looking up
            return None, None

        t = -alt / ray_world[2]

        # 5. Get Offsets (ENU)
        rel_x_east = t * ray_world[0]
        rel_y_north = t * ray_world[1]

        # 6. Convert to GPS
        return self.local_to_global(rel_x_east, rel_y_north, lat, lon)

    def local_to_global(self, x_east, y_north, lat, lon):
        R_earth = 6378137.0
        lat_rad = np.deg2rad(lat)

        dlat = y_north / R_earth
        dlon = x_east / (R_earth * np.cos(lat_rad))

        new_lat = lat + np.rad2deg(dlat)
        new_lon = lon + np.rad2deg(dlon)
        
        return new_lat, new_lon

    def destroy_node(self):
        if self.csv_rows:
            df = pd.DataFrame(self.csv_rows)
            df.to_csv(self.output_csv, index=False)
            self.get_logger().info(f"✅ Saved detections to {self.output_csv}")
        
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args = None):
    rclpy.init(args = args)
    node = Geolocalization()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
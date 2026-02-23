"""
Person Localization Node

This node detects persons using YOLOv8 and calculates their global GPS coordinates
using drone position and attitude data. Detections are logged to a CSV file.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import csv
import os
from datetime import datetime
from scipy.spatial.transform import Rotation

from .yolo_detector import YOLODetector
from .coordinate_transformer import CoordinateTransformer

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from haversine import haversine, Unit


class PersonLocalizationNode(Node):

    def __init__(self):
        super().__init__('person_localization_node')

        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('csv_output_path', 'detections.csv')

        yolo_model = self.get_parameter('yolo_model').value
        confidence_threshold = self.get_parameter('confidence_threshold').value
        csv_path = self.get_parameter('csv_output_path').value

        self.bridge = CvBridge()

        self.intrinsic_matrix = np.array([
            [393.12779563, 0.0, 321.48263787],
            [0.0, 394.76440916, 241.58044155],
            [0.0, 0.0, 1.0]
        ])

        self.get_logger().info(f'Loading YOLO model: {yolo_model}')
        self.detector = YOLODetector(
            model_name=yolo_model,
            confidence_threshold=confidence_threshold
        )
        self.get_logger().info('YOLO model loaded successfully')

        self.transformer = CoordinateTransformer(self.intrinsic_matrix)

        self.current_gps = None
        self.altitude = None
        self.current_imu = None
        self.current_compass = None

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # CSV setup
        self.csv_path = csv_path
        self._init_csv()

        # NEW: second CSV for drone-person distance error
        self.error_csv_path = 'drone_calc_error.csv'
        self._init_csv2()

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_callback,
            mavros_qos
        )

        self.altitude_sub = self.create_subscription(
            Float64,
            '/mavros/global_position/rel_alt',
            self.altitude_callback,
            mavros_qos
        )

        self.compass_sub = self.create_subscription(
            Float64,
            '/mavros/global_position/compass_hdg',
            self.compass_callback,
            mavros_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            mavros_qos
        )

        self.person_gps_pub = self.create_publisher(
            NavSatFix,
            '/detected_person/global_position',
            10
        )

        self.visualization_pub = self.create_publisher(
            Image,
            '/detected_person/visualization',
            10
        )

        self.get_logger().info('Person Localization Node initialized')
        self.get_logger().info(f'Logging detections to: {os.path.abspath(self.csv_path)}')
        self.get_logger().info('Waiting for drone to take off to set ground reference altitude...')

    # ================= CSV FUNCTIONS =================

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_sec',
                'person_idx',
                'person_lat',
                'person_lon',
                'drone_lat',
                'drone_lon',
                'drone_altitude_agl',
                'roll_deg',
                'pitch_deg',
                'yaw_deg',
                'confidence',
                'num_detections_in_frame',
                'altitude'
            ])

    # NEW
    def _init_csv2(self):
        with open(self.error_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_sec',
                'person_lat',
                'person_lon',
                'drone_lat',
                'drone_lon',
                'error_meters'
            ])

    # NEW
    def log_detection_to_csv2(self, timestamp_sec,
                              person_lat, person_lon,
                              drone_lat, drone_lon):

        error = haversine(
            (person_lat, person_lon),
            (drone_lat, drone_lon),
            unit=Unit.METERS
        )

        with open(self.error_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f'{timestamp_sec:.9f}',
                f'{person_lat:.9f}',
                f'{person_lon:.9f}',
                f'{drone_lat:.9f}',
                f'{drone_lon:.9f}',
                f'{error:.4f}'
            ])

    def _log_detection_to_csv(self, timestamp_sec, person_idx, person_lat, person_lon,
                               drone_lat, drone_lon, drone_alt_agl,
                               confidence, num_detections):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f'{timestamp_sec:.9f}',
                person_idx,
                f'{person_lat:.9f}',
                f'{person_lon:.9f}',
                f'{drone_lat:.9f}',
                f'{drone_lon:.9f}',
                f'{drone_alt_agl:.3f}',
                f'{np.rad2deg(self.roll):.4f}',
                f'{np.rad2deg(self.pitch):.4f}',
                f'{np.rad2deg(self.yaw):.4f}',
                f'{confidence:.4f}',
                num_detections,
            ])

    # ================= CALLBACKS =================

    def gps_callback(self, msg):
        self.current_gps = msg

    def compass_callback(self, msg):
        self.current_compass = msg.data

    def altitude_callback(self, msg):
        self.altitude = msg.data

    def imu_callback(self, msg):
        self.current_imu = msg
        quat = msg.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = r.as_euler('ZYX', degrees=False)
        self.yaw = euler[0]
        self.pitch = euler[1]
        self.roll = euler[2]

    def image_callback(self, msg):
        if self.current_gps is None:
            self.get_logger().warn('No GPS data received yet', throttle_duration_sec=5.0)
            return
        if self.altitude is None:
            self.get_logger().warn('No altitude data received yet', throttle_duration_sec=5.0)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = self.detector.detect_persons(cv_image)

            stamp = msg.header.stamp
            timestamp_sec = stamp.sec + stamp.nanosec * 1e-9

            drone_altitude_agl = self.altitude

            if len(detections) > 0:
                self.get_logger().info(f'Detected {len(detections)} person(s)')

                gps_coords = []

                for idx, detection in enumerate(detections):
                    cx, cy = detection['center']
                    confidence = detection['confidence']

                    result = self.transformer.pixel_to_gps(
                        u=cx,
                        v=cy,
                        drone_lat=self.current_gps.latitude,
                        drone_lon=self.current_gps.longitude,
                        drone_altitude=drone_altitude_agl,
                        roll=self.roll,
                        pitch=self.pitch,
                        yaw=self.yaw,
                        ground_altitude=0.0
                    )

                    if result is not None:
                        target_lat, target_lon = result
                        gps_coords.append((target_lat, target_lon))

                        self._log_detection_to_csv(
                            timestamp_sec=timestamp_sec,
                            person_idx=idx + 1,
                            person_lat=target_lat,
                            person_lon=target_lon,
                            drone_lat=self.current_gps.latitude,
                            drone_lon=self.current_gps.longitude,
                            drone_alt_agl=drone_altitude_agl,
                            confidence=confidence,
                            num_detections=len(detections),
                        )

                        # NEW CALL
                        self.log_detection_to_csv2(
                            timestamp_sec,
                            target_lat,
                            target_lon,
                            self.current_gps.latitude,
                            self.current_gps.longitude
                        )

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PersonLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
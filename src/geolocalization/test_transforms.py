#!/usr/bin/env python3
"""
Test script for coordinate transformation calculations.
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'person_localization_ros2'))

from person_localization_ros2.coordinate_transformer import CoordinateTransformer


def test_coordinate_transformation():
    """
    Test the coordinate transformation with sample data.
    """
    print("=" * 60)
    print("Testing Coordinate Transformation")
    print("=" * 60)
    
    # Camera intrinsic matrix
    intrinsic_matrix = np.array([
        [380.0, 0.0, 320.080],
        [0.0, 380.0, 245.482],
        [0.0, 0.0, 1.0]
    ])
    
    # Create transformer
    transformer = CoordinateTransformer(intrinsic_matrix)
    
    # Test case 1: Person directly below drone (image center)
    print("\nTest 1: Person directly below drone")
    print("-" * 60)
    
    # Image center pixel
    u, v = 320, 245
    
    # Drone parameters
    drone_lat = 37.7749  # San Francisco latitude
    drone_lon = -122.4194  # San Francisco longitude
    drone_altitude = 50.0  # 50 meters above ground
    
    # Drone orientation (level flight)
    roll = 0.0
    pitch = 0.0
    yaw = np.deg2rad(45)  # 45 degrees heading
    
    result = transformer.pixel_to_gps(
        u, v, drone_lat, drone_lon, drone_altitude,
        roll, pitch, yaw
    )
    
    if result:
        target_lat, target_lon = result
        print(f"Drone GPS: ({drone_lat:.6f}, {drone_lon:.6f})")
        print(f"Drone Altitude: {drone_altitude} m AGL")
        print(f"Pixel: ({u}, {v})")
        print(f"Person GPS: ({target_lat:.6f}, {target_lon:.6f})")
        print(f"Difference: Lat={abs(target_lat-drone_lat):.8f}, Lon={abs(target_lon-drone_lon):.8f}")
        print("Expected: Should be very close to drone GPS (person directly below)")
    else:
        print("ERROR: Could not calculate GPS")
    
    # Test case 2: Person offset from center
    print("\n\nTest 2: Person offset from center")
    print("-" * 60)
    
    # Offset pixel
    u, v = 420, 245  # 100 pixels to the right
    
    result = transformer.pixel_to_gps(
        u, v, drone_lat, drone_lon, drone_altitude,
        roll, pitch, yaw
    )
    
    if result:
        target_lat, target_lon = result
        print(f"Drone GPS: ({drone_lat:.6f}, {drone_lon:.6f})")
        print(f"Drone Altitude: {drone_altitude} m AGL")
        print(f"Pixel: ({u}, {v})")
        print(f"Person GPS: ({target_lat:.6f}, {target_lon:.6f})")
        
        # Calculate approximate distance
        lat_diff = target_lat - drone_lat
        lon_diff = target_lon - drone_lon
        
        # Rough distance calculation
        meters_per_deg_lat = 111132.92
        meters_per_deg_lon = 111412.84 * np.cos(np.deg2rad(drone_lat))
        
        north_dist = lat_diff * meters_per_deg_lat
        east_dist = lon_diff * meters_per_deg_lon
        total_dist = np.sqrt(north_dist**2 + east_dist**2)
        
        print(f"Offset: North={north_dist:.2f}m, East={east_dist:.2f}m")
        print(f"Total distance: {total_dist:.2f}m")
    else:
        print("ERROR: Could not calculate GPS")
    
    # Test case 3: Different altitude
    print("\n\nTest 3: Higher altitude (100m)")
    print("-" * 60)
    
    u, v = 320, 245
    drone_altitude = 100.0
    
    result = transformer.pixel_to_gps(
        u, v, drone_lat, drone_lon, drone_altitude,
        roll, pitch, yaw
    )
    
    if result:
        target_lat, target_lon = result
        print(f"Drone GPS: ({drone_lat:.6f}, {drone_lon:.6f})")
        print(f"Drone Altitude: {drone_altitude} m AGL")
        print(f"Pixel: ({u}, {v})")
        print(f"Person GPS: ({target_lat:.6f}, {target_lon:.6f})")
        print("Expected: Should still be close to drone GPS (center pixel)")
    else:
        print("ERROR: Could not calculate GPS")
    
    # Test case 4: With roll and pitch
    print("\n\nTest 4: Drone tilted (roll=5°, pitch=10°)")
    print("-" * 60)
    
    u, v = 320, 245
    drone_altitude = 50.0
    roll = np.deg2rad(5)
    pitch = np.deg2rad(10)
    yaw = 0.0
    
    result = transformer.pixel_to_gps(
        u, v, drone_lat, drone_lon, drone_altitude,
        roll, pitch, yaw
    )
    
    if result:
        target_lat, target_lon = result
        print(f"Drone GPS: ({drone_lat:.6f}, {drone_lon:.6f})")
        print(f"Drone Altitude: {drone_altitude} m AGL")
        print(f"Orientation: Roll={np.rad2deg(roll):.1f}°, Pitch={np.rad2deg(pitch):.1f}°, Yaw={np.rad2deg(yaw):.1f}°")
        print(f"Pixel: ({u}, {v})")
        print(f"Person GPS: ({target_lat:.6f}, {target_lon:.6f})")
        
        lat_diff = target_lat - drone_lat
        lon_diff = target_lon - drone_lon
        north_dist = lat_diff * meters_per_deg_lat
        east_dist = lon_diff * meters_per_deg_lon
        
        print(f"Offset: North={north_dist:.2f}m, East={east_dist:.2f}m")
        print("Expected: Should be offset due to drone tilt")
    else:
        print("ERROR: Could not calculate GPS")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_coordinate_transformation()

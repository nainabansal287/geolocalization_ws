"""
Launch file for person localization node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate launch description for person localization node.
    """
    
    # Declare launch arguments
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='yolov8n.pt',
        description='YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Minimum confidence threshold for person detection (0.0-1.0)'
    )
    
    # Node
    person_localization_node = Node(
        package='person_localization_ros2',
        executable='person_localization_node',
        name='person_localization_node',
        output='screen',
        parameters=[{
            'yolo_model': LaunchConfiguration('yolo_model'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
        }],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        yolo_model_arg,
        confidence_threshold_arg,
        person_localization_node,
    ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import random

def generate_launch_description():
    # Генерируем случайный параметр для передачи в узел ouster_pedestrian_detector
    random_parameter_value = random.uniform(0.0, 1.0)

    return LaunchDescription([
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
        ),
        Node(
            package='ouster_pedestrian_detector',
            executable='detector',
            name='ouster_pedestrian_detector',
            output='screen',
            parameters=[{
                'random_param': random_parameter_value  # Передаем случайный параметр
            }]
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/ruslan/Desktop/Skoltech/YOLO_LIDAR/code/ros2_ws/src/pedestrian_detector/pedestrian_detector/rosbag/rosbag2_2024_08_19-16_41_01_0.db3'],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['foxglove-studio'],
            output='screen',
        ),
    ])
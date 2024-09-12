from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
        ),
        Node(
            package='ouster_pedestrian_detector',
            executable='detector',
            name='ouster_pedestrian_detector',
            parameters=[{'Tracker': True,
                         'Tracker/distance_threshold': 0.9,
                         'Tracker/lost_time': 10,
                         'Detector/conf_threshold': 0.5,
                         'Detector/angle_offset': 0,
                         'Detector/center_radius': 3,}]
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/ruslan/Desktop/Skoltech/YOLO_LIDAR/code/ros2_ws/src/pedestrian_detector/pedestrian_detector/rosbag/rosbag2_2024_08_19-16_41_01_0.db3'],
        ),
        ExecuteProcess(
            cmd=['foxglove-studio'],
            output='screen',
        ),
    ])
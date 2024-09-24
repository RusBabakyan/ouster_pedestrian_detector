from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    # PedestrianDetectorNode
    tracker_enable = LaunchConfiguration('tracker_enable', default=True)
    marker_enable = LaunchConfiguration('marker_enable', default=True)
    name_publisher_quantity = LaunchConfiguration('name_publisher_quantity', default='/pedestrians/quantity')
    name_publisher_pose = LaunchConfiguration('name_publisher_pose', default='/pedestrians/pose')
    name_publisher_marker = LaunchConfiguration('name_publisher_marker', default='/pedestrians/marker')
    name_publisher_tracker_pose = LaunchConfiguration('name_publisher_tracker_pose', default='/pedestrians/tracker/pose')
    name_publisher_tracker_marker = LaunchConfiguration('name_publisher_tracker_marker', default='/pedestrians/tracker/marker')
    frame_id = LaunchConfiguration('frame_id', default='os_lidar')

    # PedestrianDetector
    conf_threshold = LaunchConfiguration('conf_threshold', default=0.5)
    angle_offset = LaunchConfiguration('angle_offset', default=0)
    center_radius = LaunchConfiguration('center_radius', default=3)
    device = LaunchConfiguration('device', default='cuda')

    # Tracker
    distance_threshold = LaunchConfiguration('distance_threshold', default=0.5)
    lost_time = LaunchConfiguration('lost_time', default=10)

    return LaunchDescription([
        Node(
            package='ouster_pedestrian_detector',
            executable='detector',
            name='ouster_pedestrian_detector',
            parameters=[{   'tracker_enable': tracker_enable,
                            'marker_enable': marker_enable,
                            'name_publisher_quantity': name_publisher_quantity,
                            'name_publisher_pose': name_publisher_pose,
                            'name_publisher_marker': name_publisher_marker,
                            'name_publisher_tracker_pose': name_publisher_tracker_pose,
                            'name_publisher_tracker_marker': name_publisher_tracker_marker,
                            'frame_id': frame_id,
                            
                            'conf_threshold': conf_threshold,
                            'angle_offset': angle_offset,
                            'center_radius': center_radius,
                            'device': device,

                            'distance_threshold': distance_threshold,
                            'lost_time': lost_time,}],
        )
    ])
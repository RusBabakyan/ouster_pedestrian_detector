import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from message_filters import Subscriber, TimeSynchronizer
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Int32
from visualization_msgs.msg import Marker, MarkerArray

from .DetectorClass import PedestrianDetector
from .Tracker import Tracker

package_name = "ouster_pedestrian_detector"

default_dict = {'PedestrianDetectorNode':   {'tracker_enable': True,
                                             'marker_enable': True, 
                                             'name_publisher_quantity': '/pedestrians/quantity',
                                             'name_publisher_pose': '/pedestrians/pose',
                                             'name_publisher_marker': '/pedestrians/marker',
                                             'name_publisher_tracker_pose': '/pedestrians/tracker/pose',
                                             'name_publisher_tracker_marker': '/pedestrians/tracker/marker',
                                             'frame_id': 'os_lidar'},
                'PedestrianDetector':       {'conf_threshold': 0.5,
                                             'angle_offset': 0,
                                             'center_radius': 3,
                                             'device': 'cuda'}, 
                'Tracker':                  {'distance_threshold': 0.5,
                                             'lost_time': 10}}


class ParametersProcessor():
    def __init__(self, node_obj=None, **default_dict):
        self.dict = default_dict
        if node_obj:
            self._declare_parameters(node_obj)
        
    def _declare_parameters(self, node_obj):
        assert isinstance(node_obj, Node), f"{node_obj} is not a subclass of Node"
        name = node_obj.__class__.__name__

        for class_name, default_class_dict in self.dict.items():
            for param, default_value in default_class_dict.items():
                node_obj.declare_parameter(param, default_value)
                default_class_dict[param] = node_obj.get_parameter(param).value
                if name == class_name:
                    setattr(node_obj, param, default_class_dict[param])

    def __call__(self, param_class):
        name = param_class.__name__
        assert name in self.dict.keys(), f"{name} is not defined in {self.dict.keys()}"
        return self.dict[name]
    
class OptionalPublisher():
    def __init__(self, node_obj, msg_type, topic: str,  queue, enabled=True):
        assert isinstance(node_obj, Node), f"{node_obj} is not a subclass of Node"
        self.enabled = enabled
        if self.enabled:
            self.pubslisher = node_obj.create_publisher(msg_type, topic, queue)
            self.publish = self.__publish
        else:
            self.publish = self.__empty

    @staticmethod
    def __empty(msg):
        pass

    def __publish(self, msg):
        self.pubslisher.publish(msg)
        


class PedestrianDetectorNode(Node):
    def __init__(self):
        # Call parent class initializer
        super().__init__(node_name = package_name)
        parameter_processor = ParametersProcessor(self, **default_dict)

        # Set up QoS profile for subscribers
        scan_sub_qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        # Create subscribers for reflection and range images
        self.reflec_subscriber = Subscriber(self, Image, "/ouster/reflec_image", qos_profile=scan_sub_qos)
        self.range_subscriber = Subscriber(self, Image, "/ouster/range_image", qos_profile=scan_sub_qos)

        # Create publishers for pedestrian data
        self.quantity_publisher = OptionalPublisher(self, Int32, self.name_publisher_quantity, 10)
        self.pose_publisher = OptionalPublisher(self, PoseArray, self.name_publisher_pose , 10)
        self.marker_publisher = OptionalPublisher(self, MarkerArray, self.name_publisher_marker, 10, self.marker_enable)

        if self.tracker_enable:
            self.tracker = Tracker(**parameter_processor(Tracker))
            self.tracker_publisher = OptionalPublisher(self, MarkerArray, self.name_publisher_tracker_marker, 10, self.marker_enable)

        # Synchronize messages from reflection and range subscribers
        self.sync = TimeSynchronizer(
            [self.reflec_subscriber, self.range_subscriber], 10
        )
        self.sync.registerCallback(self.MessageCallback)

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Set the model path for the pedestrian detector
        self.model_path = os.path.join(
            get_package_share_directory(package_name), "best.pt"
        )

        # Initialize pedestrian detector
        self.detector = PedestrianDetector(
            self.model_path, **parameter_processor(PedestrianDetector))

        # Log that the node has started
        self.get_logger().info(f"Node started")

    def MessageCallback(self, reflec_msg, range_msg):
        # self.get_logger().info("MessageCallback")
        reflec_img = self.Messagge2Image(reflec_msg)
        reflec_img = self.detector.process_image(reflec_img, "/ouster/reflec_image")
        range_img = self.Messagge2Image(range_msg)
        range_img = self.detector.process_image(range_img, "/ouster/range_image")

        time_stamp = self.get_clock().now().to_msg()

        people = self.detector.find_people(reflec_img, range_img)
        if people:
            self.get_logger().info(f"Found {people.quantity} people")
            msg = Int32()
            msg.data = people.quantity
            self.quantity_publisher.publish(msg)
            self.PublishPoseArray(people, time_stamp)
            self.PublishMarkerArray(people, time_stamp)

        if self.tracker_enable:
            tracked_people = self.tracker.track(people)
            self.PublishTrackedMarkerArray(tracked_people, time_stamp)


    def Messagge2Image(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        return img


    def PublishPoseArray(self, people, time_stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = time_stamp
        pose_array.header.frame_id = self.frame_id
        for cart_position in people.cart_position:
            pose = Pose()
            pose.position = Point(x=float(cart_position[0]), y=float(cart_position[1]))
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pose_array.poses.append(pose)
        self.pose_publisher.publish(pose_array)

    def PublishMarkerArray(self, people, time_stamp):
        marker_array = MarkerArray()
        for index, (cart_position, confidence) in enumerate(
            zip(people.cart_position, people.conf)
        ):
            marker = Marker()
            marker.id = index
            marker.header.stamp = time_stamp
            marker.header.frame_id = self.frame_id
            marker.type = 1
            marker.scale = Vector3(x=0.5, y=0.5, z=1.5)
            marker.pose = Pose(
                position=Point(x=float(cart_position[0]), y=float(cart_position[1]))
            )
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=float(confidence))
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker.frame_locked = True
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

    def PublishTrackedMarkerArray(self, people, time_stamp):
        if not people:
            return
        marker_array = MarkerArray()

        for cart_position, id, lost in zip(people.cart_position, people.id, people.lost):
            marker = Marker()
            marker.id = int(id)
            marker.header.stamp = time_stamp
            marker.header.frame_id = self.frame_id
            marker.type = 1
            marker.scale = Vector3(x=0.5, y=0.5, z=1.5)
            marker.pose = Pose(
                position=Point(x=float(cart_position[0]), y=float(cart_position[1]))
            )
            if lost:
                g = 0.0
                b = 1.0
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            else:
                g = 1.0
                b = 0.0
            marker.color = ColorRGBA(r=0.0, g=g, b=b, a=0.8)
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker.frame_locked = True
            marker_array.markers.append(marker)
        self.tracker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    detector = PedestrianDetectorNode()

    rclpy.spin(detector)

    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

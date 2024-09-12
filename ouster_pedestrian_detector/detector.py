import os
from pprint import pprint, pformat

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from message_filters import Subscriber, TimeSynchronizer
from pkg_resources import resource_stream
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from .DetectorClass import PedestrianDetector
from .Tracker import Tracker

package_name = "ouster_pedestrian_detector"


class PedestrianDetectorNode(Node):
    def __init__(self):
        # Call parent class initializer
        super().__init__(node_name = "ouster_pedestrian_detector")

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
        self.quantity_publisher = self.create_publisher(Int32, "pedestrians/quantity", 10)
        self.pose_publisher = self.create_publisher(PoseArray, "pedestrians/pose", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "pedestrians/marker", 10)

        self.declare_parameter("Tracker", True)
        self.declare_parameter("Tracker/distance_threshold", 0.5)
        self.declare_parameter("Tracker/lost_time", 10)
        self.declare_parameter("Detector/conf_threshold", 0.5)
        self.declare_parameter("Detector/angle_offset", 0)
        self.declare_parameter("Detector/center_radius", 3)

        self.tracker_enabled = self.get_parameter("Tracker").value
        distance_threshold = self.get_parameter("Tracker/distance_threshold").value
        lost_time = self.get_parameter("Tracker/lost_time").value
        conf_threshold = self.get_parameter("Detector/conf_threshold").value
        angle_offset = self.get_parameter("Detector/angle_offset").value
        center_radius = self.get_parameter("Detector/center_radius").value


        if self.tracker_enabled:
            self.tracker = Tracker(distance_threshold=distance_threshold, lost_time=lost_time)
            self.tracker_publisher = self.create_publisher(MarkerArray, "pedestrians/tracker", 10)

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
            self.model_path, conf_threshold=conf_threshold, angle_offset=angle_offset, center_radius=center_radius
        )

        # Set the frame ID used in messages
        self.frame_id = "os_lidar"

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

        if self.tracker_enabled:
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
        # self.get_logger().info(pformat(people))

        for cart_position, id, lost in zip(people.cart_position, people.id, people.lost):
            marker = Marker()
            marker.id = int(id)
            marker.header.stamp = time_stamp
            marker.header.frame_id = self.frame_id
            marker.type = 1
            marker.scale = Vector3(x=0.5, y=0.5, z=1.5)
            # self.get_logger().info(pformat(cart_position))
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
        pass


def main(args=None):
    rclpy.init(args=args)

    detector = PedestrianDetectorNode()

    rclpy.spin(detector)

    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

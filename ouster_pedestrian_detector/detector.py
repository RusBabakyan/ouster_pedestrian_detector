import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32, ColorRGBA
from rclpy.duration import Duration
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

from pkg_resources import resource_stream

from .DetectorClass import PedestrianDetector
import numpy as np

package_name = 'ouster_pedestrian_detector'

import os
from ament_index_python.packages import get_package_share_directory


class PedestrianDetectorNode(Node):
    def init(self):
        # Call parent class initializer
        super().init(package_name)
        
        # Set up QoS profile for subscribers
        scan_sub_qos = rclpy.qos.QoSProfile(depth=10,
                                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                durability=rclpy.qos.DurabilityPolicy.VOLATILE)

        # Create subscribers for reflection and range images
        self.reflec_subscriber = Subscriber(self, Image, "/ouster/reflec_image", qos_profile=scan_sub_qos)
        self.range_subscriber  = Subscriber(self, Image, "/ouster/range_image", qos_profile=scan_sub_qos)

        # Create publishers for pedestrian data
        self.quantity_publisher = self.create_publisher(Int32, 'pedestrians/quantity', 10)
        self.pose_publisher = self.create_publisher(PoseArray, 'pedestrians/pose', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'pedestrians/marker', 10)

        self.declare_parameter('Tracker', False)
        self.tracker_enabled = self.get_parameter('Tracker').value

        # Synchronize messages from reflection and range subscribers
        self.sync = TimeSynchronizer([self.reflec_subscriber, self.range_subscriber], 10)
        self.sync.registerCallback(self.MessageCallback)

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Set the model path for the pedestrian detector
        self.model_path = os.path.join(get_package_share_directory(package_name), 'best.pt')
        
        # Initialize pedestrian detector
        self.detector = PedestrianDetector(self.model_path, conf_threshold=0.6, angle_offset=0, center_radius=3)
        
        # Set the frame ID used in messages
        self.frame_id = 'os_lidar'
        
        # Log that the node has started
        self.get_logger().info(f"Node started")


    def MessageCallback(self, reflec_msg, range_msg):
        # self.get_logger().info("MessageCallback")
        reflec_img = self.Messagge2Image(reflec_msg)
        reflec_img = self.detector.process_image(reflec_img, "/ouster/reflec_image")
        range_img  = self.Messagge2Image(range_msg)
        range_img  = self.detector.process_image(range_img, "/ouster/range_image")

        people = self.detector.find_people(reflec_img, range_img)
        if people:
            self.get_logger().info(f"Found {people.quantity} people")
            msg = Int32()
            msg.data = people.quantity
            self.quantity_publisher.publish(msg)
            self.PublishAllData(people)


    def Messagge2Image(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        return img
    
    def PublishAllData(self, people):
        time_stamp = self.get_clock().now().to_msg() 
        self.PublishPoseArray(people, time_stamp)
        self.PublishMarkerArray(people, time_stamp)
        pass

    def PublishPoseArray(self, people, time_stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = time_stamp
        pose_array.header.frame_id = self.frame_id
        for cart_position in people.cart_position:
            pose = Pose()
            pose.position = Point(x=float(cart_position[0]), y=float(cart_position[1]))
            pose.orientation = Quaternion(x=0.,y=0.,z=0.,w=1.)
            pose_array.poses.append(pose)
        self.pose_publisher.publish(pose_array)

    def PublishMarkerArray(self, people, time_stamp):
        marker_array = MarkerArray()
        for index, (cart_position, confidence) in enumerate(zip(people.cart_position, people.conf)):
            marker = Marker()
            marker.id = index
            marker.header.stamp = time_stamp
            marker.header.frame_id = self.frame_id
            marker.type = 1
            marker.scale = Vector3(x=0.5,y=0.5,z=1.5)
            marker.pose = Pose(position=Point(x=float(cart_position[0]), y=float(cart_position[1])))
            marker.color = ColorRGBA(r=0.,g=1.,b=0.,a=float(confidence))
            marker.lifetime = Duration(seconds=0.1).to_msg()
            marker.frame_locked = True
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = PedestrianDetectorNode()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# Ouster Pedestrian Detector

## Description

The Ouster Pedestrian Detector is a ROS2 Python node for detecting and tracking people using LiDAR data and the YOLOv8n computer vision model. This project integrates advanced computer vision techniques and sensor data processing to facilitate real-time pedestrian detection and tracking.

## Table of Contents

- Installation
- Configuration
- Running the Program
- Command Breakdown
- Launch File Parameters
- Dependencies
- Topics

## Installation

To install the Ouster Pedestrian Detector, please follow these steps:

### Step 1: Clone the Repository

Clone this repository to your ROS2 workspace:
``` bash
git clone https://github.com/yourusername/ouster_pedestrian_detector.git
cd ouster_pedestrian_detector
```

### Step 2: Install Dependencies

Make sure to install the required dependencies. While most dependencies will be installed automatically, you will need to install ultralytics manually:
``` bash
rosdep install -i --from-path src --rosdistro <your_ros_distro> -y
pip install ultralytics
```

### Step 3: Build the Package

Build the package with colcon:
``` bash
colcon build
```
or
``` bash
colcon build --symlink-install
```

### Step 4: Source the Setup Script

Source the setup script to overlay this workspace on top of your current environment:
``` bash
source install/setup.bash
```

## Running the Program

After you've installed and built the package, you can run the program using the following command:
``` bash
ros2 launch ouster_pedestrian_detector launch/with_foxglow.py
```

This command will launch the necessary nodes and processes as defined in the launch file.

## Command Breakdown

1. Build the Package:
   - Command: ```colcon build```
   - Description: This command builds your ROS2 package and its dependencies, preparing everything for execution.

2. Activate the Environment:
   - Command: ```source install/setup.bash```
   - Description: This command activates the ROS2 environment, allowing you to access the built packages and execute ROS2 commands.

3. Launch the Node:
   - Command: ```ros2 launch ouster_pedestrian_detector launch/with_foxglow.py```
   - Description: This command starts up the node and any associated components specified in the ```with_foxglow.py``` launch file.

## Launch File Parameters

The launch file with_foxglow.py includes parameters that influence the behavior of the tracker and detector. Here are the parameters defined in the launch file:

- ```Tracker```: Enables or disables the tracker. (True to enable)
- ```Tracker/distance_threshold```: The distance threshold for the tracker (default: 0.9)
- ```Tracker/lost_time```: The time in seconds after which the tracker considers an object lost (default: 10)
- ```Detector/conf_threshold```: The confidence threshold for the detector (default: 0.5)
- ```Detector/angle_offset```: Offset angle for the detector (default: 0)
- ```Detector/center_radius```: Center radius for the detector (default: 3)

### Example of Parameters:
``` python
parameters = [
    {
        'Tracker': True,
        'Tracker/distance_threshold': 0.9,
        'Tracker/lost_time': 10,
        'Detector/conf_threshold': 0.5,
        'Detector/angle_offset': 0,
        'Detector/center_radius': 3,
    }
]
```

## Dependencies

The project includes the following dependencies:

- ```setuptools```: For package setup and installation.
- ```rclpy```: The ROS2 client library for Python.
- ```ultralytics```: For the YOLOv8n mod

## Topics

The Ouster Pedestrian Detector node interacts with the ROS2 ecosystem through various topics:

### Subscribed Topics
- ```/ouster/reflec_image```: This topic receives reflection images from the LiDAR sensor, which are essential for detecting pedestrians.
- ```/ouster/range_image```: This topic receives range images from the LiDAR sensor, providing spatial data required for accurate tracking.

### Published Topics
- ```pedestrians/quantity```: This topic publishes the number of detected pedestrians, represented as an Int32.
- ```pedestrians/pose```: This topic publishes an array of poses for the detected pedestrians, represented as a PoseArray.
- ```pedestrians/marker```: This topic publishes visualization markers for the detected pedestrians, represented as a MarkerArray.
-```pedestrians/tracker```: This topic publishes visualization markers for the tracked pedestrians, represented as a MarkerArray.

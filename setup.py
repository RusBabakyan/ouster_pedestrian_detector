from setuptools import find_packages, setup

package_name = 'pedestrian_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test'], include=['pedestrian_detector', 'pedestrian_detector.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['pedestrian_detector/best.pt'])],
    install_requires=['setuptools', 'rclpy', 'message_filters',
                        'opencv-python',
                        'numpy',
                        'torch',
                        'matplotlib',
                        'ultralytics'],
    zip_safe=True,
    maintainer='RBabakyan',
    maintainer_email='rusiklaim@gmail.com',
    description='Visual pedestrian detection by LIDAR',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = pedestrian_detector.detector:main'
        ],
    },
)

cmake_minimum_required(VERSION 3.0.2)
project(phase_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/train_phases.py
    scripts/test_phases.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

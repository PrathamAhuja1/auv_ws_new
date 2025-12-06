cmake_minimum_required(VERSION 3.8)
project(auv_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# Install directories
install(DIRECTORY src launch rviz worlds models config
  DESTINATION share/${PROJECT_NAME})

install(FILES src/description/orca4_description.urdf
  DESTINATION share/${PROJECT_NAME}/urdf)

install(PROGRAMS
  scripts/bluerov_teleop_ign.py
  scripts/gate_detector_node.py
  scripts/gate_navigator_node.py
  scripts/safety_monitor_node.py
  scripts/diagnostic_node.py
  scripts/simple_thruster_mapper.py
  scripts/qualification_detector_node.py
  scripts/qualification_navigator_node.py
  scripts/thruster_configuration.py
  DESTINATION lib/${PROJECT_NAME})


# ============================================================================
# INSTALL LAUNCH AND CONFIG
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
    install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
    install(DIRECTORY config
        DESTINATION share/${PROJECT_NAME}/)
endif()

ament_package()
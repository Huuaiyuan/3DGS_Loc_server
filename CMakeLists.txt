cmake_minimum_required(VERSION 3.0.2)
project(image_localization_3dgs)

find_package(catkin REQUIRED)

catkin_package()

# The nodes run under a conda env (see README), so they are installed as plain
# scripts rather than through catkin_python_setup().
catkin_install_python(PROGRAMS
  image_localization_ros1_server.py
  test_image_publisher.py
  selftest_localization.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

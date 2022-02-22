cmake_minimum_required(VERSION 3.14)
project(tcvl)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(
  beam REQUIRED 
  COMPONENTS 
    calibration
    cv
    utils
    matching
    mapping
    filtering
    optimization
)

find_package(
  catkin REQUIRED
  COMPONENTS
    fuse_core
    fuse_graphs
    fuse_variables
    fuse_constraints
    fuse_loss
    pluginlib
    roscpp
    tf2
)

find_package(OpenCV 4.5.2 REQUIRED COMPONENTS)
find_package(Ceres REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS 
    fuse_core
    fuse_graphs
    fuse_variables
    fuse_constraints
    fuse_loss
  DEPENDS
    Boost
    CERES
    EIGEN3
    tf2
)

include_directories(
  include
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)
add_library(
  ${PROJECT_NAME}
  src/utils.cpp
  src/constraints/reprojection_constraint.cpp
)
add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    include
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(
  ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    beam::utils
    beam::matching
    beam::filtering
    beam::mapping
    beam::calibration
    beam::cv
    beam::optimization
)

## Add executables
add_executable(${PROJECT_NAME}_vl_study_main
	src/vl_study_main.cpp
)
add_dependencies(${PROJECT_NAME}_vl_study_main
  ${catkin_EXPORTED_TARGETS}
)
target_include_directories(${PROJECT_NAME}_vl_study_main
  PUBLIC
    ${PROJECT_NAME}
)
target_link_libraries(${PROJECT_NAME}_vl_study_main
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)
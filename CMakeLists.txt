cmake_minimum_required(VERSION 3.0.2)
project(pallet_detection)

## 設定 C++14 標準
add_compile_options(-std=c++14)

## 尋找 catkin 套件
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

## 尋找 PCL
find_package(PCL 1.8 REQUIRED)

## 尋找 VTK (用於視覺化)
find_package(VTK REQUIRED)

###################################
## catkin 專案設定 ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    sensor_msgs 
    pcl_conversions
    pcl_ros
)

###########
## Build ##
###########

## 指定額外的標頭檔位置
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
)

## 連結 PCL 目錄
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## VFH
add_executable(vfh_node 
  src/pointcloud_vfh_visualizer.cpp
)
target_link_libraries(vfh_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${VTK_LIBRARIES}
)

## 地面移除節點
add_executable(ground_removal_node 
  src/ground_removal_node.cpp
)
target_link_libraries(ground_removal_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${VTK_LIBRARIES}
)
## 地面移除節點
add_executable(ground_removal_with_box_node 
  src/ground_removal_with_box_node.cpp
)
target_link_libraries(ground_removal_with_box_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${VTK_LIBRARIES}
)

#############
## Install ##
#############

## 安裝執行檔
install(TARGETS vfh_node ground_removal_node ground_removal_with_box_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <iostream>

class PCDViewerNode {
private:
    ros::NodeHandle nh_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    std::string pcd_filename_;
    
public:
    PCDViewerNode(const std::string& filename) : pcd_filename_(filename) {
        // 初始化可視化器
        viewer_.reset(new pcl::visualization::PCLVisualizer("PCD Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();
        
        ROS_INFO("PCD Viewer Node initialized");
        ROS_INFO("Loading file: %s", pcd_filename_.c_str());
    }
    
    bool loadAndVisualize() {
        // 載入點雲
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filename_, *cloud) == -1) {
            ROS_ERROR("Failed to load PCD file: %s", pcd_filename_.c_str());
            return false;
        }
        
        ROS_INFO("✓ Successfully loaded point cloud");
        ROS_INFO("  Total points: %zu", cloud->size());
        
        // 檢查點雲有效性
        validatePointCloud(cloud);
        
        // 計算點雲邊界
        printCloudBounds(cloud);
        
        // 視覺化
        visualizeCloud(cloud);
        
        return true;
    }
    
    void validatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        if (cloud->size() == 0) {
            ROS_WARN("Point cloud is empty!");
            return;
        }
        
        // 檢查無效點
        int invalid_count = 0;
        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                invalid_count++;
            }
        }
        
        if (invalid_count > 0) {
            ROS_WARN("Found %d invalid points (NaN/Inf)", invalid_count);
        } else {
            ROS_INFO("✓ All points are valid");
        }
    }
    
    void printCloudBounds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        if (cloud->size() == 0) return;
        
        float x_min = cloud->points[0].x, x_max = cloud->points[0].x;
        float y_min = cloud->points[0].y, y_max = cloud->points[0].y;
        float z_min = cloud->points[0].z, z_max = cloud->points[0].z;
        
        for (const auto& point : cloud->points) {
            if (std::isfinite(point.x)) {
                x_min = std::min(x_min, point.x);
                x_max = std::max(x_max, point.x);
            }
            if (std::isfinite(point.y)) {
                y_min = std::min(y_min, point.y);
                y_max = std::max(y_max, point.y);
            }
            if (std::isfinite(point.z)) {
                z_min = std::min(z_min, point.z);
                z_max = std::max(z_max, point.z);
            }
        }
        
        ROS_INFO("Point cloud bounds:");
        ROS_INFO("  X: [%.3f, %.3f] (range: %.3f m)", x_min, x_max, x_max - x_min);
        ROS_INFO("  Y: [%.3f, %.3f] (range: %.3f m)", y_min, y_max, y_max - y_min);
        ROS_INFO("  Z: [%.3f, %.3f] (range: %.3f m)", z_min, z_max, z_max - z_min);
    }
    
    void visualizeCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        // 依據 Z 軸高度著色
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> 
            height_color(cloud, "z");
        
        viewer_->addPointCloud(cloud, height_color, "cloud");
        viewer_->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
        
        // 添加資訊文字
        std::stringstream ss;
        ss << "File: " << pcd_filename_ << " (" << cloud->size() << " points)";
        viewer_->addText(ss.str(), 10, 10, 16, 1.0, 1.0, 1.0, "info_text");
        
        ROS_INFO("Visualization ready. Close window to exit.");
    }
    
    void spin() {
        while (ros::ok() && !viewer_->wasStopped()) {
            viewer_->spinOnce(100);
            ros::Duration(0.1).sleep();
        }
        
        ROS_INFO("Viewer closed. Shutting down...");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pcd_viewer_node");
    
    // 檢查命令列參數
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <pcd_filename>" << std::endl;
        std::cout << "Example: " << argv[0] << " pallet_scan.pcd" << std::endl;
        return -1;
    }
    
    std::string pcd_filename = argv[1];
    
    PCDViewerNode node(pcd_filename);
    
    if (node.loadAndVisualize()) {
        node.spin();
    } else {
        ROS_ERROR("Failed to load point cloud. Exiting...");
        return -1;
    }
    
    return 0;
}

/*
# 基本用法
rosrun your_package pcd_viewer_node pallet_scan.pcd

# 使用相對路徑
rosrun your_package pcd_viewer_node ./data/pallet_front.pcd

# 使用絕對路徑
rosrun your_package pcd_viewer_node /home/user/workspace/pallet_scan.pcd

# 如果忘記檔名，會顯示使用說明
rosrun your_package pcd_viewer_node
# 輸出：
# Usage: /path/to/pcd_viewer_node <pcd_filename>
# Example: /path/to/pcd_viewer_node pallet_scan.pcd
*/
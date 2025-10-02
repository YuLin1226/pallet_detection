#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <sstream>
#include <iomanip>

class GroundRemovalNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    
    bool processed_;
    
    // 地面分割參數
    double distance_threshold_;
    int max_iterations_;
    
public:
    GroundRemovalNode() : processed_(false) {
        // 讀取參數
        nh_.param("distance_threshold", distance_threshold_, 0.02);
        nh_.param("max_iterations", max_iterations_, 1000);
        
        // 訂閱點雲主題
        cloud_sub_ = nh_.subscribe("/depth_camera/depth/points", 1, 
            &GroundRemovalNode::cloudCallback, this);
        
        // 初始化可視化器
        viewer_.reset(new pcl::visualization::PCLVisualizer("Ground Removal Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();
        
        ROS_INFO("Ground Removal Node initialized");
        ROS_INFO("Distance threshold: %.4f", distance_threshold_);
        ROS_INFO("Max iterations: %d", max_iterations_);
        ROS_INFO("Waiting for point cloud data...");
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (processed_) {
            return;  // 已經處理過了，忽略後續訊息
        }
        
        ROS_INFO("Received point cloud, processing...");
        
        // 轉換 ROS 訊息到 PCL 格式
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        
        ROS_INFO("Original cloud size: %zu points", cloud->size());
        
        // 執行地面分割
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground = removeGround(cloud);
        
        ROS_INFO("After ground removal: %zu points", cloud_no_ground->size());
        ROS_INFO("Removed: %zu points", cloud->size() - cloud_no_ground->size());
        
        // 視覺化結果
        visualizeResult(cloud, cloud_no_ground);
        
        processed_ = true;
        ROS_INFO("Processing complete. Visualization is running.");
        ROS_INFO("Close the visualization window to exit.");
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeGround(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        
        // 使用 RANSAC 平面分割來找到地面
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        // 建立分割物件
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(max_iterations_);
        seg.setDistanceThreshold(distance_threshold_);
        
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() == 0) {
            ROS_WARN("Could not estimate a planar model for the given dataset.");
            return cloud;
        }
        
        // 輸出平面方程式
        ROS_INFO("Ground plane model: %.4fx + %.4fy + %.4fz + %.4f = 0",
                 coefficients->values[0], coefficients->values[1],
                 coefficients->values[2], coefficients->values[3]);
        
        // 提取非地面點雲
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);  // 提取非地面點
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_filtered);
        
        return cloud_filtered;
    }
    
    void visualizeResult(const pcl::PointCloud<pcl::PointXYZ>::Ptr& original,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered) {
        
        // 只顯示移除地面後的點雲（彩色，依高度著色）
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> 
            height_color(filtered, "z");
        viewer_->addPointCloud(filtered, height_color, "filtered_cloud");
        viewer_->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "filtered_cloud");
        
        // 計算移除比例
        float removal_ratio = (1.0 - (float)filtered->size() / original->size()) * 100.0;
        
        // 添加文字說明
        std::stringstream ss;
        ss << "Ground Removed (kept " << filtered->size() << " / " << original->size() 
           << " points, removed " << std::fixed << std::setprecision(1) << removal_ratio << "%)";
        viewer_->addText(ss.str(), 10, 10, 16, 1.0, 1.0, 1.0, "info_text");
        
        ROS_INFO("Visualization initialized:");
        ROS_INFO("  Showing only non-ground points (colored by height)");
        ROS_INFO("  Ground removal ratio: %.1f%%", removal_ratio);
    }
    
    void spin() {
        ros::Rate rate(10);
        
        while (ros::ok() && !viewer_->wasStopped()) {
            if (!processed_) {
                ros::spinOnce();
            }
            
            viewer_->spinOnce(100);
            rate.sleep();
        }
        
        ROS_INFO("Shutting down...");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ground_removal_node");
    
    GroundRemovalNode node;
    node.spin();
    
    return 0;
}
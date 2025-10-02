#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <boost/thread/thread.hpp>

class PointCloudVFHVisualizer {
private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    pcl::visualization::PCLHistogramVisualizer::Ptr hist_viewer_;
    
    bool cloud_updated_;
    bool has_processed_once_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;
    
public:
    PointCloudVFHVisualizer() : cloud_updated_(false) , has_processed_once_(false){
        latest_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        
        // 訂閱點雲主題
        cloud_sub_ = nh_.subscribe("/depth_camera/depth/points", 1, 
            &PointCloudVFHVisualizer::cloudCallback, this);
        
        // 初始化可視化器
        viewer_.reset(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();
        
        hist_viewer_.reset(new pcl::visualization::PCLHistogramVisualizer());
        hist_viewer_->setBackgroundColor(1, 1, 1);
        
        ROS_INFO("PointCloud VFH Visualizer initialized");
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        
        if(has_processed_once_) return;
        
        // 轉換 ROS 訊息到 PCL 格式
        pcl::fromROSMsg(*msg, *latest_cloud_);
        cloud_updated_ = true;
        
        ROS_INFO("Received point cloud with %zu points", latest_cloud_->size());
    }
    
    void computeVFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                    pcl::PointCloud<pcl::VFHSignature308>::Ptr& vfh_features) {
        // 計算法向量
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(0.03);  // 可依據點雲尺度調整
        ne.compute(*normals);
        
        // 計算 VFH 特徵
        pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
        vfh.setInputCloud(cloud);
        vfh.setInputNormals(normals);
        vfh.setSearchMethod(tree);
        
        vfh_features.reset(new pcl::PointCloud<pcl::VFHSignature308>);
        vfh.compute(*vfh_features);
        
        ROS_INFO("VFH feature computed with %zu features", vfh_features->size());
    }
    
    void updateVisualization() {
        if (!cloud_updated_) {
            return;
        }
        
        if (latest_cloud_->empty()) {
            ROS_WARN("Empty point cloud received");
            return;
        }
        
        // 更新點雲顯示
        if (!viewer_->updatePointCloud(latest_cloud_, "cloud")) {
            viewer_->addPointCloud(latest_cloud_, "cloud");
            viewer_->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        }
        
        // 計算並顯示 VFH
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_features;
        computeVFH(latest_cloud_, vfh_features);
        
        if (vfh_features && !vfh_features->empty()) {
            // 顯示 VFH 直方圖
            hist_viewer_->addFeatureHistogram(*vfh_features, 308, "VFH Histogram");
        }
        
        cloud_updated_ = false;
        has_processed_once_ = true;
    }
    
    void spin() {
        ros::Rate rate(30);  // 30 Hz
        
        while (ros::ok() && !viewer_->wasStopped()) {
            ros::spinOnce();
            
            updateVisualization();
            
            viewer_->spinOnce(10);
            hist_viewer_->spinOnce(10);
            
            rate.sleep();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_vfh_visualizer");
    
    PointCloudVFHVisualizer visualizer;
    visualizer.spin();
    
    return 0;
}
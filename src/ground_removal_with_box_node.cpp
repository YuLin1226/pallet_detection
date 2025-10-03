#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
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
    
    // 棧板參數 (根據你的 SDF)
    const double PALLET_LENGTH = 1.22;  // x方向
    const double PALLET_WIDTH = 0.8;    // y方向
    const double PALLET_HEIGHT = 0.145; // z方向
    const double PALLET_X = 2.0;        // 中心點 x 座標
    const double PALLET_Y = 0.0;        // 中心點 y 座標
    
    // 相機位置 (根據你的 URDF，相機在 (0,0,2)，向下看)
    const double CAMERA_HEIGHT = 2.0;
    
    // 棧板在相機座標系中的 z 座標
    double pallet_z_in_camera_;
    
    // 選擇顯示模式
    bool filter_mode_;  // true: 只顯示方框內的點雲, false: 顯示所有點雲+方框
    
public:
    GroundRemovalNode() : processed_(false) {
        // 讀取參數
        nh_.param("distance_threshold", distance_threshold_, 0.02);
        nh_.param("max_iterations", max_iterations_, 1000);
        nh_.param("filter_mode", filter_mode_, false);
        
        // 計算棧板在相機座標系中的 z 座標
        // 相機在 z=2.0 向下看，棧板中心高度是 PALLET_HEIGHT/2
        // 在相機座標系中，z 軸向前（因為相機倒置），實際是向下
        pallet_z_in_camera_ = CAMERA_HEIGHT - PALLET_HEIGHT / 2.0;
        
        // 訂閱點雲主題
        cloud_sub_ = nh_.subscribe("/depth_camera/depth/points", 1, 
            &GroundRemovalNode::cloudCallback, this);
        
        // 初始化可視化器
        viewer_.reset(new pcl::visualization::PCLVisualizer("Ground Removal with Pallet Box"));
        viewer_->setBackgroundColor(0, 0, 0);
        viewer_->addCoordinateSystem(1.0);
        viewer_->initCameraParameters();
        
        ROS_INFO("Ground Removal Node initialized");
        ROS_INFO("Distance threshold: %.4f", distance_threshold_);
        ROS_INFO("Max iterations: %d", max_iterations_);
        ROS_INFO("Filter mode: %s", filter_mode_ ? "Only show points inside box" : "Show all points with box");
        ROS_INFO("Pallet dimensions: %.2f x %.2f x %.2f m", PALLET_LENGTH, PALLET_WIDTH, PALLET_HEIGHT);
        ROS_INFO("Pallet center position: (%.2f, %.2f, %.4f) in camera frame", 
                 PALLET_X, PALLET_Y, pallet_z_in_camera_);
        ROS_INFO("Waiting for point cloud data...");
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (processed_) {
            return;
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
        
        // 根據模式選擇要顯示的點雲
        pcl::PointCloud<pcl::PointXYZ>::Ptr display_cloud;
        if (filter_mode_) {
            display_cloud = filterPointsInBox(cloud_no_ground);
            ROS_INFO("Points inside pallet box: %zu", display_cloud->size());
        } else {
            display_cloud = cloud_no_ground;
        }
        
        // 視覺化結果
        visualizeResult(cloud, display_cloud);
        
        processed_ = true;
        ROS_INFO("Processing complete. Visualization is running.");
        ROS_INFO("Close the visualization window to exit.");
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeGround(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
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
        
        ROS_INFO("Ground plane model: %.4fx + %.4fy + %.4fz + %.4f = 0",
                 coefficients->values[0], coefficients->values[1],
                 coefficients->values[2], coefficients->values[3]);
        
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_filtered);
        
        return cloud_filtered;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterPointsInBox(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        
        pcl::CropBox<pcl::PointXYZ> crop_box;
        crop_box.setInputCloud(cloud);
        
        // 定義棧板邊界框 (在相機座標系中)
        Eigen::Vector4f min_point(
            PALLET_X - PALLET_LENGTH / 2.0,
            PALLET_Y - PALLET_WIDTH / 2.0,
            pallet_z_in_camera_ - PALLET_HEIGHT / 2.0,
            1.0
        );
        
        Eigen::Vector4f max_point(
            PALLET_X + PALLET_LENGTH / 2.0,
            PALLET_Y + PALLET_WIDTH / 2.0,
            pallet_z_in_camera_ + PALLET_HEIGHT / 2.0,
            1.0
        );
        
        crop_box.setMin(min_point);
        crop_box.setMax(max_point);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        crop_box.filter(*cloud_filtered);
        
        return cloud_filtered;
    }
    
    void drawPalletBox() {
        // 計算棧板的8個頂點
        double x_min = PALLET_X - PALLET_LENGTH / 2.0;
        double x_max = PALLET_X + PALLET_LENGTH / 2.0;
        double y_min = PALLET_Y - PALLET_WIDTH / 2.0;
        double y_max = PALLET_Y + PALLET_WIDTH / 2.0;
        double z_min = pallet_z_in_camera_ - PALLET_HEIGHT / 2.0;
        double z_max = pallet_z_in_camera_ + PALLET_HEIGHT / 2.0;
        
        pcl::PointXYZ p1(x_min, y_min, z_min);
        pcl::PointXYZ p2(x_max, y_min, z_min);
        pcl::PointXYZ p3(x_max, y_max, z_min);
        pcl::PointXYZ p4(x_min, y_max, z_min);
        pcl::PointXYZ p5(x_min, y_min, z_max);
        pcl::PointXYZ p6(x_max, y_min, z_max);
        pcl::PointXYZ p7(x_max, y_max, z_max);
        pcl::PointXYZ p8(x_min, y_max, z_max);
        
        // 繪製底面的4條邊
        viewer_->addLine(p1, p2, 0.0, 1.0, 0.0, "line1");
        viewer_->addLine(p2, p3, 0.0, 1.0, 0.0, "line2");
        viewer_->addLine(p3, p4, 0.0, 1.0, 0.0, "line3");
        viewer_->addLine(p4, p1, 0.0, 1.0, 0.0, "line4");
        
        // 繪製頂面的4條邊
        viewer_->addLine(p5, p6, 0.0, 1.0, 0.0, "line5");
        viewer_->addLine(p6, p7, 0.0, 1.0, 0.0, "line6");
        viewer_->addLine(p7, p8, 0.0, 1.0, 0.0, "line7");
        viewer_->addLine(p8, p5, 0.0, 1.0, 0.0, "line8");
        
        // 繪製連接底面和頂面的4條邊
        viewer_->addLine(p1, p5, 0.0, 1.0, 0.0, "line9");
        viewer_->addLine(p2, p6, 0.0, 1.0, 0.0, "line10");
        viewer_->addLine(p3, p7, 0.0, 1.0, 0.0, "line11");
        viewer_->addLine(p4, p8, 0.0, 1.0, 0.0, "line12");
        
        // 設置線條粗細
        for (int i = 1; i <= 12; i++) {
            std::string line_id = "line" + std::to_string(i);
            viewer_->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, line_id);
        }
        
        ROS_INFO("Pallet bounding box drawn (green lines)");
    }
    
    void visualizeResult(const pcl::PointCloud<pcl::PointXYZ>::Ptr& original,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered) {
        
        // 繪製棧板邊界框
        drawPalletBox();
        
        // 顯示點雲（依高度著色）
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> 
            height_color(filtered, "z");
        viewer_->addPointCloud(filtered, height_color, "filtered_cloud");
        viewer_->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "filtered_cloud");
        
        // 計算移除比例
        float removal_ratio = (1.0 - (float)filtered->size() / original->size()) * 100.0;
        
        // 添加文字說明
        std::stringstream ss;
        if (filter_mode_) {
            ss << "Points inside pallet box: " << filtered->size() 
               << " (Ground removed: " << std::fixed << std::setprecision(1) 
               << removal_ratio << "%)";
        } else {
            ss << "Ground Removed (kept " << filtered->size() << " / " << original->size() 
               << " points, removed " << std::fixed << std::setprecision(1) 
               << removal_ratio << "%)";
        }
        viewer_->addText(ss.str(), 10, 10, 16, 1.0, 1.0, 1.0, "info_text");
        
        // 添加棧板資訊
        std::stringstream ss2;
        ss2 << "Pallet: " << PALLET_LENGTH << "m x " << PALLET_WIDTH << "m x " 
            << PALLET_HEIGHT << "m at (" << PALLET_X << ", " << PALLET_Y << ", " 
            << std::fixed << std::setprecision(3) << pallet_z_in_camera_ << ")";
        viewer_->addText(ss2.str(), 10, 30, 14, 0.0, 1.0, 0.0, "pallet_info");
        
        ROS_INFO("Visualization initialized");
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
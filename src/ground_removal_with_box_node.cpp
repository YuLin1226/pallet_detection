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
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <boost/thread/thread.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>

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
    const double PALLET_LENGTH = 1.22;  // 對應 z 方向
    const double PALLET_WIDTH = 0.8;    // 對應 x 方向
    const double PALLET_HEIGHT = 0.145; // 對應 y 方向
    
    // 棧板中心位置（在旋轉後的座標系中）
    double pallet_center_x_;  // 對應寬度方向
    double pallet_center_y_;  // 對應高度方向
    double pallet_center_z_;  // 對應長度方向
    
    // 選擇顯示模式
    bool filter_mode_;  // true: 只顯示方框內的點雲, false: 顯示所有點雲+方框
    
    // 儲存相關參數
    std::string output_filename_;
    bool auto_save_;
    
public:
    GroundRemovalNode() : processed_(false) {
        // 讀取參數
        nh_.param("distance_threshold", distance_threshold_, 0.02);
        nh_.param("max_iterations", max_iterations_, 1000);
        nh_.param("filter_mode", filter_mode_, false);
        
        // 設定棧板中心位置（需要根據實際旋轉後的座標系調整）
        nh_.param("pallet_center_x", pallet_center_x_, 0.0);
        nh_.param("pallet_center_y", pallet_center_y_, 0.05);
        nh_.param("pallet_center_z", pallet_center_z_, 2.0);
        
        // 儲存參數
        nh_.param<std::string>("output_filename", output_filename_, "pallet_scan");
        nh_.param("auto_save", auto_save_, true);
        
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
        ROS_INFO("Pallet dimensions: Length(z)=%.2fm, Width(x)=%.2fm, Height(y)=%.2fm", 
                 PALLET_LENGTH, PALLET_WIDTH, PALLET_HEIGHT);
        ROS_INFO("Pallet center position: (x=%.2f, y=%.2f, z=%.2f)", 
                 pallet_center_x_, pallet_center_y_, pallet_center_z_);
        
        if (auto_save_) {
            ROS_INFO("Auto-save enabled: %s.pcd", output_filename_.c_str());
        } else {
            ROS_INFO("Auto-save disabled");
        }
        
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
        
        // 對 X 軸做 180 度旋轉（與第一份程式碼相同）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
        rotateAroundX(cloud, cloud_rotated, 180.0);
        
        ROS_INFO("Applied 180 degree rotation around X-axis");
        
        // 執行地面分割
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground = removeGround(cloud_rotated);
        
        ROS_INFO("After ground removal: %zu points", cloud_no_ground->size());
        ROS_INFO("Removed: %zu points", cloud_rotated->size() - cloud_no_ground->size());
        
        // 根據模式選擇要顯示和儲存的點雲
        pcl::PointCloud<pcl::PointXYZ>::Ptr display_cloud;
        if (filter_mode_) {
            display_cloud = filterPointsInBox(cloud_no_ground);
            ROS_INFO("Points inside pallet box: %zu", display_cloud->size());
        } else {
            display_cloud = cloud_no_ground;
        }
        
        // 自動儲存點雲
        if (auto_save_) {
            savePointCloud(display_cloud);
        }
        
        // 視覺化結果
        visualizeResult(cloud_rotated, display_cloud);
        
        processed_ = true;
        ROS_INFO("Processing complete. Visualization is running.");
        ROS_INFO("Close the visualization window to exit.");
    }
    
    void rotateAroundX(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                       double angle_degrees) {
        // 將角度轉換為弧度
        double angle_rad = angle_degrees * M_PI / 180.0;
        
        // 建立旋轉矩陣（繞 Z 軸旋轉）
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitZ()));
        
        // 執行轉換
        pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
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
        
        // 定義棧板邊界框（在旋轉後的座標系中）
        // x 對應寬度，y 對應高度，z 對應長度
        Eigen::Vector4f min_point(
            pallet_center_x_ - PALLET_WIDTH / 2.0,   // x 方向：寬度
            pallet_center_y_ - PALLET_HEIGHT / 2.0,  // y 方向：高度
            pallet_center_z_ - PALLET_LENGTH / 2.0,  // z 方向：長度
            1.0
        );
        
        Eigen::Vector4f max_point(
            pallet_center_x_ + PALLET_WIDTH / 2.0,   // x 方向：寬度
            pallet_center_y_ + PALLET_HEIGHT / 2.0,  // y 方向：高度
            pallet_center_z_ + PALLET_LENGTH / 2.0,  // z 方向：長度
            1.0
        );
        
        crop_box.setMin(min_point);
        crop_box.setMax(max_point);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        crop_box.filter(*cloud_filtered);
        
        return cloud_filtered;
    }
    
    void savePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        // 先過濾無效點
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        for (const auto& point : cloud->points) {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                cloud_filtered->push_back(point);
            }
        }
        
        ROS_INFO("Filtered invalid points: %zu -> %zu (removed %zu invalid points)",
                cloud->size(), cloud_filtered->size(), 
                cloud->size() - cloud_filtered->size());
        
        // 檢查過濾後的點雲
        if (cloud_filtered->size() == 0) {
            ROS_ERROR("All points are invalid! Cannot save empty point cloud.");
            return;
        }
        
        if (cloud_filtered->size() < 100) {
            ROS_WARN("Point cloud after filtering is very small: %zu points", 
                    cloud_filtered->size());
        }
        
        std::string filename = output_filename_ + ".pcd";
        
        // 使用二進位格式儲存（最快、最小檔案）
        if (pcl::io::savePCDFileBinary(filename, *cloud_filtered) == 0) {
            ROS_INFO("✓ Successfully saved %zu valid points to %s", 
                    cloud_filtered->size(), filename.c_str());
            
            // 顯示檔案大小
            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                double size_mb = file.tellg() / (1024.0 * 1024.0);
                ROS_INFO("  File size: %.2f MB", size_mb);
                file.close();
            }
        } else {
            ROS_ERROR("✗ Failed to save point cloud to %s", filename.c_str());
        }
    }

    bool validatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        bool is_valid = true;
        
        // 檢查點數量
        if (cloud->size() == 0) {
            ROS_WARN("Point cloud is empty!");
            return false;
        }
        
        // 檢查是否有 NaN 或 Inf
        int invalid_count = 0;
        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                invalid_count++;
            }
        }
        
        if (invalid_count > 0) {
            double invalid_ratio = (double)invalid_count / cloud->size() * 100.0;
            ROS_WARN("Found %d invalid points (%.1f%% of total)", invalid_count, invalid_ratio);
            is_valid = false;
        } else {
            ROS_INFO("✓ All points are valid");
        }
        
        return is_valid;
    }
    
    void drawPalletBox() {
        // 計算棧板的8個頂點
        // x 對應寬度方向
        double x_min = pallet_center_x_ - PALLET_WIDTH / 2.0;
        double x_max = pallet_center_x_ + PALLET_WIDTH / 2.0;
        
        // y 對應高度方向
        double y_min = pallet_center_y_ - PALLET_HEIGHT / 2.0;
        double y_max = pallet_center_y_ + PALLET_HEIGHT / 2.0;
        
        // z 對應長度方向
        double z_min = pallet_center_z_ - PALLET_LENGTH / 2.0;
        double z_max = pallet_center_z_ + PALLET_LENGTH / 2.0;
        
        // 定義8個頂點
        pcl::PointXYZ p1(x_min, y_min, z_min);
        pcl::PointXYZ p2(x_max, y_min, z_min);
        pcl::PointXYZ p3(x_max, y_max, z_min);
        pcl::PointXYZ p4(x_min, y_max, z_min);
        pcl::PointXYZ p5(x_min, y_min, z_max);
        pcl::PointXYZ p6(x_max, y_min, z_max);
        pcl::PointXYZ p7(x_max, y_max, z_max);
        pcl::PointXYZ p8(x_min, y_max, z_max);
        
        // 繪製底面的4條邊 (z = z_min)
        viewer_->addLine(p1, p2, 0.0, 1.0, 0.0, "line1");
        viewer_->addLine(p2, p3, 0.0, 1.0, 0.0, "line2");
        viewer_->addLine(p3, p4, 0.0, 1.0, 0.0, "line3");
        viewer_->addLine(p4, p1, 0.0, 1.0, 0.0, "line4");
        
        // 繪製頂面的4條邊 (z = z_max)
        viewer_->addLine(p5, p6, 0.0, 1.0, 0.0, "line5");
        viewer_->addLine(p6, p7, 0.0, 1.0, 0.0, "line6");
        viewer_->addLine(p7, p8, 0.0, 1.0, 0.0, "line7");
        viewer_->addLine(p8, p5, 0.0, 1.0, 0.0, "line8");
        
        // 繪製連接底面和頂面的4條邊 (沿 z 方向)
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
        ROS_INFO("Box range - X: [%.3f, %.3f], Y: [%.3f, %.3f], Z: [%.3f, %.3f]",
                 x_min, x_max, y_min, y_max, z_min, z_max);
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
        ss2 << "Pallet: Width(x)=" << PALLET_WIDTH << "m, Height(y)=" << PALLET_HEIGHT 
            << "m, Length(z)=" << PALLET_LENGTH << "m at (" 
            << pallet_center_x_ << ", " << pallet_center_y_ << ", " 
            << std::fixed << std::setprecision(3) << pallet_center_z_ << ")";
        viewer_->addText(ss2.str(), 10, 30, 14, 0.0, 1.0, 0.0, "pallet_info");
        
        // 添加儲存狀態資訊
        if (auto_save_) {
            std::stringstream ss3;
            ss3 << "Saved to: " << output_filename_ << ".pcd";
            viewer_->addText(ss3.str(), 10, 50, 14, 1.0, 1.0, 0.0, "save_info");
        }
        
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
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <iostream>
#include <vector>
#include <string>

struct PointCloudConfig {
    std::string filename;
    double rotation_degree;  // 旋轉角度（度）
};

class PointCloudMerger {
private:
    std::string config_file_;
    std::string output_file_;
    
    // 全局 offset
    double offset_x_;
    double offset_y_;
    double offset_z_;
    
    // 降採樣參數
    bool enable_downsampling_;
    double voxel_size_;
    
    std::vector<PointCloudConfig> cloud_configs_;
    
public:
    PointCloudMerger(const std::string& config_file) 
        : config_file_(config_file),
          enable_downsampling_(true),
          voxel_size_(0.005) {  // 預設 5mm
    }
    
    bool loadConfig() {
        try {
            boost::property_tree::ptree pt;
            boost::property_tree::ini_parser::read_ini(config_file_, pt);
            
            // 讀取全局設定
            output_file_ = pt.get<std::string>("global.output_file", "merged_pallet.pcd");
            offset_x_ = pt.get<double>("global.offset_x", 0.0);
            offset_y_ = pt.get<double>("global.offset_y", 0.0);
            offset_z_ = pt.get<double>("global.offset_z", 0.0);
            enable_downsampling_ = pt.get<bool>("global.enable_downsampling", true);
            voxel_size_ = pt.get<double>("global.voxel_size", 0.005);
            
            ROS_INFO("Configuration loaded:");
            ROS_INFO("  Output file: %s", output_file_.c_str());
            ROS_INFO("  Global offset: (%.3f, %.3f, %.3f)", offset_x_, offset_y_, offset_z_);
            ROS_INFO("  Downsampling: %s (voxel size: %.4f m)", 
                     enable_downsampling_ ? "enabled" : "disabled", voxel_size_);
            
            // 讀取點雲列表
            int cloud_count = pt.get<int>("global.cloud_count", 0);
            
            if (cloud_count == 0) {
                ROS_ERROR("No point clouds specified in config file!");
                return false;
            }
            
            ROS_INFO("  Number of clouds to merge: %d", cloud_count);
            
            for (int i = 0; i < cloud_count; i++) {
                std::string section = "cloud" + std::to_string(i);
                PointCloudConfig config;
                
                config.filename = pt.get<std::string>(section + ".filename");
                config.rotation_degree = pt.get<double>(section + ".rotation_degree", 0.0);
                
                cloud_configs_.push_back(config);
                
                ROS_INFO("  Cloud %d: %s (rotation: %.1f deg)", 
                         i, config.filename.c_str(), config.rotation_degree);
            }
            
            return true;
            
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to load config file: %s", e.what());
            return false;
        }
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    double rotation_degree) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
        
        // 步驟 1：先平移（offset）
        Eigen::Affine3f translation = Eigen::Affine3f::Identity();
        translation.translation() << offset_x_, offset_y_, offset_z_;
        pcl::transformPointCloud(*cloud, *temp_cloud, translation);
        
        // 步驟 2：再旋轉（繞 Y 軸）
        double rotation_rad = rotation_degree * M_PI / 180.0;
        Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
        rotation.rotate(Eigen::AngleAxisf(rotation_rad, Eigen::Vector3f::UnitY()));
        pcl::transformPointCloud(*temp_cloud, *transformed, rotation);
        
        return transformed;
    }
    
    bool mergePointClouds() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        ROS_INFO("\n=== Starting point cloud merging ===");
        
        for (size_t i = 0; i < cloud_configs_.size(); i++) {
            const auto& config = cloud_configs_[i];
            
            ROS_INFO("\nProcessing cloud %zu/%zu: %s", 
                     i + 1, cloud_configs_.size(), config.filename.c_str());
            
            // 載入點雲
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(config.filename, *cloud) == -1) {
                ROS_ERROR("  ✗ Failed to load: %s", config.filename.c_str());
                continue;
            }
            
            ROS_INFO("  ✓ Loaded: %zu points", cloud->size());
            
            // 變換點雲
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed = 
                transformCloud(cloud, config.rotation_degree);
            
            ROS_INFO("  ✓ Transformed (rotation: %.1f deg, offset: %.3f, %.3f, %.3f)",
                     config.rotation_degree, offset_x_, offset_y_, offset_z_);
            
            // 合併
            *merged_cloud += *transformed;
            
            ROS_INFO("  ✓ Merged. Total points: %zu", merged_cloud->size());
        }
        
        if (merged_cloud->size() == 0) {
            ROS_ERROR("\nMerged cloud is empty! No valid point clouds loaded.");
            return false;
        }
        
        ROS_INFO("\n=== Merge complete ===");
        ROS_INFO("Total points before filtering: %zu", merged_cloud->size());
        
        // 降採樣（可選）
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud;
        if (enable_downsampling_) {
            final_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
            
            ROS_INFO("\nApplying voxel grid filter (voxel size: %.4f m)...", voxel_size_);
            
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(merged_cloud);
            voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
            voxel_filter.filter(*final_cloud);
            
            ROS_INFO("  ✓ After downsampling: %zu points", final_cloud->size());
            ROS_INFO("  Reduction: %.1f%%", 
                     (1.0 - (double)final_cloud->size() / merged_cloud->size()) * 100.0);
        } else {
            final_cloud = merged_cloud;
        }
        
        // 儲存結果
        ROS_INFO("\nSaving merged point cloud to: %s", output_file_.c_str());
        
        if (pcl::io::savePCDFileBinary(output_file_, *final_cloud) == 0) {
            ROS_INFO("✓ Successfully saved %zu points", final_cloud->size());
            
            // 顯示檔案大小
            std::ifstream file(output_file_, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                double size_mb = file.tellg() / (1024.0 * 1024.0);
                ROS_INFO("  File size: %.2f MB", size_mb);
                file.close();
            }
            
            return true;
        } else {
            ROS_ERROR("✗ Failed to save merged point cloud!");
            return false;
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
        
        ROS_INFO("\nMerged point cloud bounds:");
        ROS_INFO("  X: [%.3f, %.3f] (range: %.3f m)", x_min, x_max, x_max - x_min);
        ROS_INFO("  Y: [%.3f, %.3f] (range: %.3f m)", y_min, y_max, y_max - y_min);
        ROS_INFO("  Z: [%.3f, %.3f] (range: %.3f m)", z_min, z_max, z_max - z_min);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "point_cloud_merger_node");
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.ini>" << std::endl;
        std::cout << "\nExample config.ini format:" << std::endl;
        std::cout << "[global]" << std::endl;
        std::cout << "output_file = merged_pallet.pcd" << std::endl;
        std::cout << "offset_x = 0.0" << std::endl;
        std::cout << "offset_y = 0.05" << std::endl;
        std::cout << "offset_z = 2.0" << std::endl;
        std::cout << "cloud_count = 4" << std::endl;
        std::cout << "enable_downsampling = true" << std::endl;
        std::cout << "voxel_size = 0.005" << std::endl;
        std::cout << "\n[cloud0]" << std::endl;
        std::cout << "filename = pallet_front.pcd" << std::endl;
        std::cout << "rotation_degree = 0.0" << std::endl;
        std::cout << "\n[cloud1]" << std::endl;
        std::cout << "filename = pallet_left.pcd" << std::endl;
        std::cout << "rotation_degree = 90.0" << std::endl;
        std::cout << "\n..." << std::endl;
        return -1;
    }
    
    std::string config_file = argv[1];
    
    PointCloudMerger merger(config_file);
    
    if (!merger.loadConfig()) {
        ROS_ERROR("Failed to load configuration. Exiting...");
        return -1;
    }
    
    if (merger.mergePointClouds()) {
        ROS_INFO("\n=== Merge completed successfully! ===");
        return 0;
    } else {
        ROS_ERROR("\n=== Merge failed! ===");
        return -1;
    }
}


/*
使用方法：
rosrun pallet_detection merge_point_clouds src/pallet_detection/config/merge_info.ini 

上面這個指令我是在 pallet_ws/ 下執行，因為 .pcd 都在這邊，我沒有上傳到github，避免佔容量。

*/
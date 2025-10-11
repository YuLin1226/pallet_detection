#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

class ICPAligner {
private:
    std::string config_file_;
    
    // 點雲檔案路徑
    std::string source_file_;
    std::string target_file_;
    
    // 初始估計值
    double init_x_, init_y_, init_z_;
    double init_roll_, init_pitch_, init_yaw_;
    
    // ICP 參數
    int max_iterations_;
    double transformation_epsilon_;
    double euclidean_fitness_epsilon_;
    double max_correspondence_distance_;
    
    // 點雲
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    
    // 結果
    Eigen::Matrix4f final_transformation_;
    bool has_converged_;
    double fitness_score_;
    
public:
    ICPAligner(const std::string& config_file) 
        : config_file_(config_file),
          max_iterations_(50),
          transformation_epsilon_(1e-6),
          euclidean_fitness_epsilon_(1e-6),
          max_correspondence_distance_(0.05),
          has_converged_(false),
          fitness_score_(0.0) {
        
        source_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        target_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        aligned_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }
    
    bool loadConfig() {
        try {
            boost::property_tree::ptree pt;
            boost::property_tree::ini_parser::read_ini(config_file_, pt);
            
            // 讀取檔案路徑
            source_file_ = pt.get<std::string>("files.source_cloud");
            target_file_ = pt.get<std::string>("files.target_cloud");
            
            // 讀取初始估計值
            init_x_ = pt.get<double>("initial_estimate.x", 0.0);
            init_y_ = pt.get<double>("initial_estimate.y", 0.0);
            init_z_ = pt.get<double>("initial_estimate.z", 0.0);
            init_roll_ = pt.get<double>("initial_estimate.roll", 0.0);
            init_pitch_ = pt.get<double>("initial_estimate.pitch", 0.0);
            init_yaw_ = pt.get<double>("initial_estimate.yaw", 0.0);
            
            // 讀取 ICP 參數（可選）
            max_iterations_ = pt.get<int>("icp.max_iterations", 50);
            transformation_epsilon_ = pt.get<double>("icp.transformation_epsilon", 1e-6);
            euclidean_fitness_epsilon_ = pt.get<double>("icp.euclidean_fitness_epsilon", 1e-6);
            max_correspondence_distance_ = pt.get<double>("icp.max_correspondence_distance", 0.05);
            
            std::cout << "Configuration loaded:" << std::endl;
            std::cout << "  Source cloud: " << source_file_ << std::endl;
            std::cout << "  Target cloud: " << target_file_ << std::endl;
            std::cout << "  Initial estimate:" << std::endl;
            std::cout << "    Position: (" << init_x_ << ", " << init_y_ << ", " << init_z_ << ")" << std::endl;
            std::cout << "    Orientation: roll=" << init_roll_ << "°, pitch=" << init_pitch_ 
                      << "°, yaw=" << init_yaw_ << "°" << std::endl;
            std::cout << "  ICP parameters:" << std::endl;
            std::cout << "    Max iterations: " << max_iterations_ << std::endl;
            std::cout << "    Transformation epsilon: " << transformation_epsilon_ << std::endl;
            std::cout << "    Fitness epsilon: " << euclidean_fitness_epsilon_ << std::endl;
            std::cout << "    Max correspondence distance: " << max_correspondence_distance_ << " m" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to load config file: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool loadPointClouds() {
        std::cout << "\nLoading point clouds..." << std::endl;
        
        // 載入 source 點雲
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_file_, *source_cloud_) == -1) {
            std::cerr << "✗ Failed to load source cloud: " << source_file_ << std::endl;
            return false;
        }
        std::cout << "✓ Source cloud loaded: " << source_cloud_->size() << " points" << std::endl;
        
        // 載入 target 點雲
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_file_, *target_cloud_) == -1) {
            std::cerr << "✗ Failed to load target cloud: " << target_file_ << std::endl;
            return false;
        }
        std::cout << "✓ Target cloud loaded: " << target_cloud_->size() << " points" << std::endl;
        
        return true;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr applyInitialTransform() {
        std::cout << "Applying initial transformation..." << std::endl;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // 步驟 1：先平移
        Eigen::Affine3f translation = Eigen::Affine3f::Identity();
        translation.translation() << init_x_, init_y_, init_z_;
        pcl::transformPointCloud(*target_cloud_, *temp_cloud, translation);
        
        std::cout << "  ✓ Applied translation: (" << init_x_ << ", " << init_y_ << ", " << init_z_ << ")" << std::endl;
        
        // 步驟 2：再旋轉（ZYX 順序）
        double roll_rad = init_roll_ * M_PI / 180.0;
        double pitch_rad = init_pitch_ * M_PI / 180.0;
        double yaw_rad = init_yaw_ * M_PI / 180.0;
        
        Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
        rotation.rotate(Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()));
        rotation.rotate(Eigen::AngleAxisf(pitch_rad, Eigen::Vector3f::UnitY()));
        rotation.rotate(Eigen::AngleAxisf(roll_rad, Eigen::Vector3f::UnitX()));
        
        pcl::transformPointCloud(*temp_cloud, *transformed_cloud, rotation);
        
        std::cout << "  ✓ Applied rotation: roll=" << init_roll_ << "°, pitch=" << init_pitch_ 
                << "°, yaw=" << init_yaw_ << "°" << std::endl;
        
        return transformed_cloud;
    }
    
    bool performICP() {
        std::cout << "\n=== Starting ICP alignment ===" << std::endl;
        
        // 使用初始估計值變換 target 點雲
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_transformed = applyInitialTransform();
        
        // 設定 ICP
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(target_transformed);
        icp.setInputTarget(source_cloud_);
        
        // ICP 參數
        icp.setMaximumIterations(max_iterations_);
        icp.setTransformationEpsilon(transformation_epsilon_);
        icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance_);
        
        std::cout << "Running ICP..." << std::endl;
        
        // 執行 ICP
        icp.align(*aligned_cloud_);
        
        // 取得結果
        has_converged_ = icp.hasConverged();
        fitness_score_ = icp.getFitnessScore();
        final_transformation_ = icp.getFinalTransformation();
        
        // 顯示結果
        std::cout << "\n=== ICP Results ===" << std::endl;
        if (has_converged_) {
            std::cout << "✓ ICP converged" << std::endl;
        } else {
            std::cout << "✗ ICP did not converge" << std::endl;
        }
        std::cout << "Fitness score: " << fitness_score_ << std::endl;
        
        // 顯示 ICP 變換矩陣（相對於初始估計的修正）
        std::cout << "\nICP transformation matrix (refinement):" << std::endl;
        std::cout << final_transformation_ << std::endl;
        
        // 提取 ICP 修正的位姿變化
        extractPoseFromMatrix(final_transformation_);
        
        return has_converged_;
    }
    
    void extractPoseFromMatrix(const Eigen::Matrix4f& matrix) {
        // 提取平移
        float x = matrix(0, 3);
        float y = matrix(1, 3);
        float z = matrix(2, 3);
        
        // 提取旋轉（轉換為歐拉角）
        Eigen::Matrix3f rotation = matrix.block<3, 3>(0, 0);
        Eigen::Vector3f euler = rotation.eulerAngles(2, 1, 0); // ZYX 順序
        
        float yaw = euler[0] * 180.0 / M_PI;
        float pitch = euler[1] * 180.0 / M_PI;
        float roll = euler[2] * 180.0 / M_PI;
        
        std::cout << "\nFinal pose:" << std::endl;
        std::cout << "  Position: (" << x << ", " << y << ", " << z << ") m" << std::endl;
        std::cout << "  Orientation: roll=" << roll << "°, pitch=" << pitch << "°, yaw=" << yaw << "°" << std::endl;
        
        // 顯示與初始估計的差異
        std::cout << "\nDelta from initial estimate:" << std::endl;
        std::cout << "  ΔPosition: (" << (x - init_x_) << ", " << (y - init_y_) << ", " << (z - init_z_) << ") m" << std::endl;
        std::cout << "  ΔOrientation: Δroll=" << (roll - init_roll_) << "°, Δpitch=" << (pitch - init_pitch_) 
                  << "°, Δyaw=" << (yaw - init_yaw_) << "°" << std::endl;
    }
    
    void visualize() {
        std::cout << "\nStarting visualization..." << std::endl;
        
        pcl::visualization::PCLVisualizer::Ptr viewer(
            new pcl::visualization::PCLVisualizer("ICP Alignment Result"));
        
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(0.5);
        viewer->initCameraParameters();
        
        // 顯示 source 點雲（紅色）
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
            source_color(source_cloud_, 255, 0, 0);
        viewer->addPointCloud(source_cloud_, source_color, "source");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");
        
        // 顯示對齊後的點雲（綠色）
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
            aligned_color(aligned_cloud_, 0, 255, 0);
        viewer->addPointCloud(aligned_cloud_, aligned_color, "aligned");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned");
        
        // 顯示初始位置的 target 點雲（藍色）
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_initial = applyInitialTransform();
        
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
            initial_color(target_initial, 0, 0, 255);
        viewer->addPointCloud(target_initial, initial_color, "initial");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "initial");
        
        // 添加圖例
        viewer->addText("Red: Source (reference)", 10, 70, 14, 1.0, 0.0, 0.0, "legend1");
        viewer->addText("Blue: Target (initial estimate)", 10, 50, 14, 0.0, 0.0, 1.0, "legend2");
        viewer->addText("Green: Target (after ICP)", 10, 30, 14, 0.0, 1.0, 0.0, "legend3");
        
        // 添加結果資訊
        std::stringstream ss;
        ss << "Fitness score: " << std::fixed << std::setprecision(6) << fitness_score_;
        viewer->addText(ss.str(), 10, 10, 14, 1.0, 1.0, 1.0, "fitness");
        
        std::cout << "Visualization ready. Close window to exit." << std::endl;
        std::cout << "Red: Source cloud (reference)" << std::endl;
        std::cout << "Blue: Target cloud (initial estimate)" << std::endl;
        std::cout << "Green: Target cloud (after ICP alignment)" << std::endl;
        
        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void saveResult(const std::string& output_file) {
        if (pcl::io::savePCDFileBinary(output_file, *aligned_cloud_) == 0) {
            std::cout << "✓ Saved aligned cloud to: " << output_file << std::endl;
        } else {
            std::cerr << "✗ Failed to save aligned cloud" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config.ini>" << std::endl;
        std::cout << "\nExample config.ini format:" << std::endl;
        std::cout << "[files]" << std::endl;
        std::cout << "source_cloud = pallet_source.pcd" << std::endl;
        std::cout << "target_cloud = pallet_target.pcd" << std::endl;
        std::cout << "\n[initial_estimate]" << std::endl;
        std::cout << "x = 0.1" << std::endl;
        std::cout << "y = 0.0" << std::endl;
        std::cout << "z = 0.0" << std::endl;
        std::cout << "roll = 0.0" << std::endl;
        std::cout << "pitch = 0.0" << std::endl;
        std::cout << "yaw = 5.0" << std::endl;
        std::cout << "\n[icp]  # Optional parameters" << std::endl;
        std::cout << "max_iterations = 50" << std::endl;
        std::cout << "transformation_epsilon = 1e-6" << std::endl;
        std::cout << "euclidean_fitness_epsilon = 1e-6" << std::endl;
        std::cout << "max_correspondence_distance = 0.05" << std::endl;
        return -1;
    }
    
    std::string config_file = argv[1];
    
    ICPAligner aligner(config_file);
    
    if (!aligner.loadConfig()) {
        return -1;
    }
    
    if (!aligner.loadPointClouds()) {
        return -1;
    }
    
    if (aligner.performICP()) {
        std::cout << "\n=== Alignment successful! ===" << std::endl;
        
        // 可選：儲存結果
        // aligner.saveResult("aligned_result.pcd");
        
        // 視覺化
        aligner.visualize();
        
        return 0;
    } else {
        std::cout << "\n=== Alignment failed or did not converge ===" << std::endl;
        
        // 仍然顯示結果供檢查
        aligner.visualize();
        
        return -1;
    }
}

// rosrun pallet_detection icp_alignment icp_info.ini
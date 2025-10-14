#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <random>
#include <string>

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd> [noise_stddev]" << std::endl;
        std::cerr << "  noise_stddev: standard deviation in meters (default: 0.005)" << std::endl;
        return -1;
    }

    std::string input_file = argv[1];
    double noise_stddev = 0.005;  // Default 5mm
    
    if (argc >= 3) {
        noise_stddev = std::atof(argv[2]);
    }

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
        std::cerr << "Failed to load: " << input_file << std::endl;
        return -1;
    }

    std::cout << "Loaded " << cloud->points.size() << " points" << std::endl;
    std::cout << "Applying Gaussian noise with stddev = " << noise_stddev << "m" << std::endl;

    // Setup random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noise_stddev);

    // Add noise to each point
    for (auto& point : cloud->points) {
        point.x += dist(gen);
        point.y += dist(gen);
        point.z += dist(gen);
    }

    // Generate output filename
    std::string output_file = input_file;
    size_t dot_pos = output_file.find_last_of('.');
    if (dot_pos != std::string::npos) {
        output_file.insert(dot_pos, "_noisy");
    } else {
        output_file += "_noisy";
    }

    // Save noisy point cloud
    pcl::io::savePCDFileBinary(output_file, *cloud);
    std::cout << "Saved to: " << output_file << std::endl;

    return 0;
}

/*
使用方法: 檔名、誤差參數(Optional)
./add_noise input.pcd
./add_noise input.pcd 0.01
*/
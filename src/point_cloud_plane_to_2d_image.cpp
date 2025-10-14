#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

// Check if plane normal is vertical (perpendicular to ground/y-axis)
bool isVerticalPlane(const Eigen::Vector3f& normal, double angle_threshold_deg = 5.0) {
    Eigen::Vector3f y_axis(0, 1, 0);  // Y 軸是垂直方向
    double angle = std::acos(std::abs(normal.dot(y_axis))) * 180.0 / M_PI;
    return angle > (90.0 - angle_threshold_deg) && angle < (90.0 + angle_threshold_deg);
}

// 矩形候選資訊結構
struct RectangleCandidate {
    cv::RotatedRect rect;
    double area;
    double aspect_ratio;
    double rectangularity;  // 緊湊度
    cv::Point2f center;
};

// 檢測矩形區域（基於灰階密度圖）
std::vector<RectangleCandidate> detectRectangles(const cv::Mat& density_image, 
                                                   cv::Mat& visualization) {
    std::vector<RectangleCandidate> candidates;
    
    std::cout << "\n=== Rectangle Detection Started ===" << std::endl;
    std::cout << "Input image size: " << density_image.cols << "x" << density_image.rows << std::endl;
    
    // 確保輸入是 8 位灰階
    cv::Mat gray;
    if (density_image.type() != CV_8U) {
        density_image.convertTo(gray, CV_8U);
    } else {
        gray = density_image.clone();
    }
    
    // 1. 膨脹操作：填補凹陷區域內的小洞
    cv::Mat dilated;
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(gray, dilated, kernel_dilate, cv::Point(-1, -1), 2);  // 膨脹 2 次
    
    cv::imshow("After Dilation", dilated);
    std::cout << "Applied dilation (5x5 kernel, 2 iterations)" << std::endl;
    
    // 2. 二值化：使用 Otsu 自動閾值（或固定閾值）
    cv::Mat binary;
    double thresh_value = cv::threshold(dilated, binary, 0, 255, 
                                        cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << "Otsu threshold: " << thresh_value << std::endl;
    
    // 如果 Otsu 失敗（閾值太低），使用固定閾值
    if (thresh_value < 10) {
        std::cout << "Otsu threshold too low, using fixed threshold: 30" << std::endl;
        cv::threshold(dilated, binary, 30, 255, cv::THRESH_BINARY);
    }
    
    cv::imshow("Binary Image", binary);
    
    // 3. 反轉：讓凹陷（暗）變成前景（白）
    cv::Mat binary_inv;
    cv::bitwise_not(binary, binary_inv);
    cv::imshow("Binary Inverted", binary_inv);
    std::cout << "Inverted binary image (dark regions = foreground)" << std::endl;
    
    // 4. 形態學閉運算：連接鄰近區域
    cv::Mat closed;
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(binary_inv, closed, cv::MORPH_CLOSE, kernel_close);
    cv::imshow("After Closing", closed);
    std::cout << "Applied closing operation" << std::endl;
    
    // 5. 開運算：去除小雜點
    cv::Mat opened;
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel_open);
    cv::imshow("After Opening", opened);
    std::cout << "Applied opening operation" << std::endl;
    
    // 6. 找輪廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(opened, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::cout << "Found " << contours.size() << " contours" << std::endl;
    
    // 計算影像總面積
    double total_area = density_image.rows * density_image.cols;
    
    // 7. 分析每個輪廓
    for (size_t i = 0; i < contours.size(); ++i) {
        double contour_area = cv::contourArea(contours[i]);
        
        // 面積過濾：至少 0.3%，最多 25%
        if (contour_area < total_area * 0.003 || contour_area > total_area * 0.25) {
            std::cout << "Contour #" << i << ": area=" << contour_area 
                      << " (" << (contour_area/total_area*100) << "%) - REJECTED (area out of range)" << std::endl;
            continue;
        }
        
        // 獲取最小外接矩形
        cv::RotatedRect min_rect = cv::minAreaRect(contours[i]);
        double rect_area = min_rect.size.width * min_rect.size.height;
        
        // 計算幾何特性
        double rectangularity = contour_area / rect_area;
        double width = std::min(min_rect.size.width, min_rect.size.height);
        double height = std::max(min_rect.size.width, min_rect.size.height);
        double aspect_ratio = height / width;
        
        // 幾何過濾
        bool is_valid = true;
        std::string reject_reason = "";
        
        // 長寬比：1.2 ~ 5.0（放寬範圍）
        if (aspect_ratio < 1.2 || aspect_ratio > 5.0) {
            is_valid = false;
            reject_reason = "aspect ratio out of range";
        }
        
        // 緊湊度：> 0.65（稍微放寬）
        if (rectangularity < 0.65) {
            is_valid = false;
            reject_reason = "low rectangularity";
        }
        
        // 輸出分析
        std::cout << "\nContour #" << i << ":" << std::endl;
        std::cout << "  Area: " << contour_area << " pixels (" 
                  << (contour_area / total_area * 100) << "% of image)" << std::endl;
        std::cout << "  Aspect ratio: " << aspect_ratio << std::endl;
        std::cout << "  Rectangularity: " << rectangularity << std::endl;
        std::cout << "  Center: (" << min_rect.center.x << ", " << min_rect.center.y << ")" << std::endl;
        std::cout << "  Size: " << width << " x " << height << " pixels" << std::endl;
        
        if (is_valid) {
            std::cout << "  ✓ ACCEPTED" << std::endl;
            
            RectangleCandidate candidate;
            candidate.rect = min_rect;
            candidate.area = contour_area;
            candidate.aspect_ratio = aspect_ratio;
            candidate.rectangularity = rectangularity;
            candidate.center = min_rect.center;
            
            candidates.push_back(candidate);
            
            // 繪製綠色框
            cv::Point2f vertices[4];
            min_rect.points(vertices);
            for (int j = 0; j < 4; j++) {
                cv::line(visualization, vertices[j], vertices[(j+1)%4], 
                        cv::Scalar(0, 255, 0), 3);
            }
            
            cv::circle(visualization, min_rect.center, 5, cv::Scalar(0, 255, 0), -1);
            
            std::string label = "#" + std::to_string(candidates.size());
            cv::putText(visualization, label, min_rect.center, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
        }
    }
    
    std::cout << "\n=== Detection Summary ===" << std::endl;
    std::cout << "Total candidates: " << candidates.size() << std::endl;
    
    // 配對分析
    if (candidates.size() >= 2) {
        std::cout << "\n=== Analyzing Pairs ===" << std::endl;
        for (size_t i = 0; i < candidates.size(); ++i) {
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                double area_ratio = candidates[i].area / candidates[j].area;
                if (area_ratio < 1.0) area_ratio = 1.0 / area_ratio;
                
                double y_diff = std::abs(candidates[i].center.y - candidates[j].center.y);
                double x_distance = std::abs(candidates[i].center.x - candidates[j].center.x);
                
                std::cout << "Pair #" << (i+1) << " and #" << (j+1) << ":" << std::endl;
                std::cout << "  Area ratio: " << area_ratio << std::endl;
                std::cout << "  Y diff: " << y_diff << " px, X dist: " << x_distance << " px" << std::endl;
                
                bool similar = (area_ratio < 1.5);
                bool aligned = (y_diff < density_image.rows * 0.15);
                bool separated = (x_distance > density_image.cols * 0.1);
                
                if (similar && aligned && separated) {
                    std::cout << "  ⭐ Likely slot pair!" << std::endl;
                    cv::line(visualization, candidates[i].center, candidates[j].center,
                            cv::Scalar(0, 255, 255), 2);
                }
            }
        }
    }
    
    return candidates;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd>" << std::endl;
        return -1;
    }

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1) {
        std::cerr << "Failed to load: " << argv[1] << std::endl;
        return -1;
    }
    std::cout << "Loaded " << cloud->points.size() << " points" << std::endl;

    // RANSAC plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations(1000);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *remaining_cloud);
    
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    
    // Find vertical plane
    bool found_vertical = false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3f plane_normal;
    
    for (int i = 0; i < 10; ++i) {
        if (remaining_cloud->points.size() < 100) break;
        
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < 100) break;
        
        plane_normal = Eigen::Vector3f(coefficients->values[0], 
                                       coefficients->values[1], 
                                       coefficients->values[2]);
        plane_normal.normalize();
        
        std::cout << "Plane " << i << ": normal = (" 
                  << plane_normal.x() << ", " 
                  << plane_normal.y() << ", " 
                  << plane_normal.z() << "), inliers = " 
                  << inliers->indices.size() << std::endl;
        
        if (isVerticalPlane(plane_normal)) {
            std::cout << "Found vertical plane!" << std::endl;
            extract.setInputCloud(remaining_cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane_cloud);
            found_vertical = true;
            break;
        }
        
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
    }
    
    if (!found_vertical) {
        std::cerr << "No vertical plane found!" << std::endl;
        return -1;
    }
    
    std::cout << "Vertical plane has " << plane_cloud->points.size() << " points" << std::endl;
    
    // Build local coordinate system
    Eigen::Vector3f z_local = plane_normal;
    
    Eigen::Vector3f world_y(0, 1, 0);
    Eigen::Vector3f x_local = world_y.cross(z_local);
    x_local.normalize();
    
    Eigen::Vector3f y_local = z_local.cross(x_local);
    y_local.normalize();
    
    std::cout << "Local coordinate system:" << std::endl;
    std::cout << "  X: (" << x_local.x() << ", " << x_local.y() << ", " << x_local.z() << ")" << std::endl;
    std::cout << "  Y: (" << y_local.x() << ", " << y_local.y() << ", " << y_local.z() << ")" << std::endl;
    std::cout << "  Z: (" << z_local.x() << ", " << z_local.y() << ", " << z_local.z() << ")" << std::endl;
    
    // Project points to 2D
    std::vector<Eigen::Vector3f> projected_points;
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    
    for (const auto& point : plane_cloud->points) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        
        float x_proj = p.dot(x_local);
        float y_proj = p.dot(y_local);
        float z_proj = p.dot(z_local);
        
        projected_points.push_back(Eigen::Vector3f(x_proj, y_proj, z_proj));
        
        min_x = std::min(min_x, x_proj);
        max_x = std::max(max_x, x_proj);
        min_y = std::min(min_y, y_proj);
        max_y = std::max(max_y, y_proj);
        min_z = std::min(min_z, z_proj);
        max_z = std::max(max_z, z_proj);
    }
    
    std::cout << "Projection bounds:" << std::endl;
    std::cout << "  X: [" << min_x << ", " << max_x << "] = " << (max_x - min_x) << "m" << std::endl;
    std::cout << "  Y: [" << min_y << ", " << max_y << "] = " << (max_y - min_y) << "m" << std::endl;
    std::cout << "  Z: [" << min_z << ", " << max_z << "] = " << (max_z - min_z) << "m" << std::endl;
    
    // Create 2D image
    const float resolution = 0.001;  // 1mm per pixel
    int width = static_cast<int>((max_x - min_x) / resolution) + 1;
    int height = static_cast<int>((max_y - min_y) / resolution) + 1;
    
    std::cout << "Image size: " << width << " x " << height << " pixels" << std::endl;
    
    if (width > 5000 || height > 5000) {
        std::cerr << "Warning: Image too large!" << std::endl;
        return -1;
    }
    
    cv::Mat count_image = cv::Mat::zeros(height, width, CV_32S);
    
    for (const auto& p : projected_points) {
        int px = static_cast<int>((p.x() - min_x) / resolution);
        int py = static_cast<int>((p.y() - min_y) / resolution);
        
        if (px >= 0 && px < width && py >= 0 && py < height) {
            count_image.at<int>(py, px) += 1;
        }
    }
    
    // Create density-based grayscale image
    cv::Mat density_image = cv::Mat::zeros(height, width, CV_8U);
    
    // 統計每個 pixel 的最大點數（用於正規化）
    int max_count = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            max_count = std::max(max_count, count_image.at<int>(y, x));
        }
    }
    
    std::cout << "Max point count per pixel: " << max_count << std::endl;
    
    // 將點數轉換為灰階值 (0-255)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int count = count_image.at<int>(y, x);
            if (count > 0) {
                // 線性映射：count -> 0-255
                density_image.at<uchar>(y, x) = static_cast<uchar>(
                    std::min(255.0, (double)count / max_count * 255.0));
            }
        }
    }
    
    std::cout << "Generated density-based grayscale image" << std::endl;
    
    // 顯示原始灰階圖
    cv::imshow("Original Density Image", density_image);
    
    // 矩形檢測（使用灰階密度圖）
    cv::Mat visualization = density_image.clone();
    cv::cvtColor(visualization, visualization, cv::COLOR_GRAY2BGR);  // 轉為彩色以便繪製
    std::vector<RectangleCandidate> rectangles = detectRectangles(density_image, visualization);
    
    // Display
    cv::imshow("Rectangle Detection Result", visualization);
    
    std::cout << "\nPress any key to close..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}
// /*
// 使用方法：./project_to_2d input.pcd
// */
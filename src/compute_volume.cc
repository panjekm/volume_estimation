#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  LOG(INFO) << "Starting the awesome volume calculator...";

  const std::string kRgbImageName =
      "/home/panjekm/volume_ws/src/volume_estimation/test_data/image.jpg";
  const std::string kDepthImageName =
      "/home/panjekm/volume_ws/src/volume_estimation/test_data/depth_map.png";
  const std::string kMaskImageName =
      "/home/panjekm/volume_ws/src/volume_estimation/test_data/mask.png";

  cv::Mat rgb_image = cv::imread(kRgbImageName, cv::IMREAD_COLOR);
  cv::Mat depth_image = cv::imread(kDepthImageName, cv::IMREAD_ANYDEPTH);
  cv::Mat mask_image = cv::imread(kMaskImageName, cv::IMREAD_GRAYSCALE);
  cv::Mat mask = cv::Mat::zeros(mask_image.size(), CV_8U);
  cv::threshold(mask_image, mask, 1, 255, cv::THRESH_BINARY);

  // cv::imshow("RGB image", rgb_image);
  // cv::imshow("depth image", depth_image);
  // cv::imshow("mask", mask);
  // cv::waitKey();

  // Camera intrinsics.
  constexpr float kFx = 620.078 / 2;
  constexpr float kFy = kFx;
  constexpr float kCx = 311.509 / 2;
  constexpr float kCy = 238.185 / 2;
  cv::Mat intrinsics = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
  intrinsics.at<float>(0, 0) = kFx;
  intrinsics.at<float>(1, 1) = kFy;
  intrinsics.at<float>(0, 2) = kCx;
  intrinsics.at<float>(1, 2) = kCy;
  intrinsics.at<float>(2, 2) = 1.0;

  // Assuming fixed distance from table.
  // TODO(panjekm): fit a plane to the table or, better, estimate plane from
  // around the segmentation mask on the actual plate.
  constexpr float kZMax = 0.400;

  // Extract only depth at masked place.
  cv::Mat object_depth = cv::Mat::zeros(depth_image.size(), CV_16U);
  depth_image.copyTo(object_depth, mask);

  // Convert to floats.
  cv::rgbd::rescaleDepth(depth_image, CV_32FC1, depth_image);

  // Compute 3d points.
  cv::Mat object_pointcloud;
  cv::rgbd::depthTo3d(depth_image, intrinsics, object_pointcloud);

  float object_volume = 0.0;
  float object_area = 0.0;

  for (size_t u = 0u; u < depth_image.rows; ++u) {
    for (size_t v = 0u; v < depth_image.cols; ++v) {
      if (mask.at<uint8_t>(u, v) != 0u) {
        const float kDepthValue = object_pointcloud.at<cv::Point3f>(u, v).z;
        const float kDepthDiff = kZMax - kDepthValue;
        const float kXSize =
            std::abs(std::abs(object_pointcloud.at<cv::Point3f>(u, v).x) -
                     std::abs(object_pointcloud.at<cv::Point3f>(u, v + 1).x));
        const float kYSize =
            std::abs(std::abs(object_pointcloud.at<cv::Point3f>(u, v).y) -
                     std::abs(object_pointcloud.at<cv::Point3f>(u + 1, v).y));
        const float kObjectPixelArea = kXSize * kYSize;

        // Compute pixel area at kZMax depth.
        // NOTE(panjekm): This is a WIP, still some code missing to get to area
        // of object at table depth. But it would go something like this...
        // const float kXValueTable = static_cast<float>(u - kCx) * kZMax / kFx;

        const float kVolumeIncrement = kObjectPixelArea * kDepthDiff;
        object_area += kObjectPixelArea;
        object_volume += kVolumeIncrement;
      }
    }
  }

  // Convert to cubic centimeters.
  LOG(INFO) << "OBJECT area is: " << object_area * 1e4 << " cm2.";
  LOG(INFO) << "OBJECT volume is: " << object_volume * 1e6 << " cm3.";

  return 0;
}

#include "ORBextractor.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main()
{
    std::string root_path    = "/home/lucas/catkin_ws/src/ORB_SLAM3/datasets/";
    std::string pic_pattern  = "test.png";
    std::string feature_type = "orb";

    std::string img_names = root_path + pic_pattern;
    
    cv::Ptr<cv::FeatureDetector> finder;
    if (feature_type == "orb") {
        finder = cv::ORB::create(1000);
    }
    //opencv orb检测函数
    cv::Mat img = cv::imread(img_names);
    if (img.empty()) {
        cout << "failed to load image : " << img_names;
        return -1;
    }
    std::vector<cv::detail::ImageFeatures> features(2);
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        finder->detectAndCompute(img, cv::Mat(), features[0].keypoints,
                                 features[0].descriptors, false);
    }
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"cv_orb extract ORB cost = "<<time_used.count()<<"seconds."<<endl;
    features[0].img_idx = 0;
    std::vector<cv::Mat> feature_img(2);
    cv::drawKeypoints(img, features[0].keypoints, feature_img[0]);
    cout << "cv orb feature num: " << features[0].keypoints.size()<<endl;

    t1=chrono::steady_clock::now();
    ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(
            1000, 1.2, 8, 20, 7);
    std::vector<int> vlapp{30};
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < 10; i++) {
        orb_extractor(gray_img, cv::Mat(), features[1].keypoints,
                      features[1].descriptors, vlapp);
    }
    t2=chrono::steady_clock::now();
    time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"slam_orb extract ORB cost = "<<time_used.count()<<"seconds."<<endl;

    cv::drawKeypoints(img, features[1].keypoints, feature_img[1]);
    cout << "slam orb feature num: " << features[1].keypoints.size()<<endl;

    cv::imshow("cv_orb", feature_img[0]);
    cv::imwrite("cv_orb.png", feature_img[0]);
    cv::imshow("slam_orb", feature_img[1]);
    cv::imwrite("slam_orb.png", feature_img[1]);
    cv::waitKey(0);
    return 0;
}

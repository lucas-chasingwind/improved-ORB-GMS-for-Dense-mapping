#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ORBextractor.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步：检测Oriented FAST角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(1000, 1.2, 8, 20, 7);
  std::vector<int> vlapp{30};
  cv::Mat gray_img_1,gray_img_2;
  cv::cvtColor(img_1, gray_img_1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_2, gray_img_2, cv::COLOR_BGR2GRAY);
  orb_extractor(gray_img_1, cv::Mat(), keypoints_1,descriptors_1, vlapp);
  orb_extractor(gray_img_2, cv::Mat(), keypoints_2,descriptors_2, vlapp);
  

  //detector->detect(img_1,keypoints_1);
  //detector->detect(img_2,keypoints_2);

  //-- 第二步：根据角点位置计算BRIEF描述子
  descriptor->compute(img_1,keypoints_1,descriptors_1);  
  descriptor->compute(img_2,keypoints_2,descriptors_2);  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<" extract ORB cost = "<<time_used.count()<<"seconds."<<endl;

  Mat outimg1,outimg2;
  drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  drawKeypoints(img_2,keypoints_2,outimg2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
  //imshow("ORB features 1",outimg1);
  //cv::imwrite("1_ORB.png",outimg1);
  //imshow("ORB features 2",outimg2);
  //cv::imwrite("2_ORB.png",outimg2);

  //-- 第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1,descriptors_2,matches);
  t2 = chrono::steady_clock::now();
  time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<" match ORB cost = "<<time_used.count()<<"seconds."<<endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  int num=good_matches.size();
  std::cout<<"matches count:"<<num<<endl;
  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  cv::imwrite("all_matched.png",img_match);
  imshow("good matches", img_goodmatch);
  cv::imwrite("good_matched.png",img_goodmatch);
  waitKey(0);

  return 0;
}

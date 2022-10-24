#include "gms_matcher.h"
#include <chrono>
#include "/home/lucas/catkin_ws/src/ORB_SLAM3/include/ORBextractor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

//#define USE_GPU 
#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
using cuda::GpuMat;
#endif

void GmsMatch(Mat &img1, Mat &img2);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

/*
void runImagePair() {
	Mat img1 = imread("../data/1.png");
	Mat img2 = imread("../data/3.png");

	GmsMatch(img1, img2);
}
*/

int main(int argc, char **argv)
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU
	if (argc != 3) {
		cout << "usage: feature_extraction img1 img2" << endl;
	return 1;
	}
	//-- 读取图像
	Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	assert(img1.data != nullptr && img2.data != nullptr);

	GmsMatch(img1,img2);

	return 0;
}

void GmsMatch(Mat &img1, Mat &img2) {
	vector<KeyPoint> kp1, kp2;  //特征点
	Mat d1, d2;		    //描述子
	vector<DMatch> matches_all, matches_gms;

	//Ptr<ORB> orb = ORB::create(10000);
	//orb->setFastThreshold(0);

	//orb->detectAndCompute(img1, Mat(), kp1, d1);
	//orb->detectAndCompute(img2, Mat(), kp2, d2);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();  //add
	ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(1000, 1.2, 8, 20, 7);
	std::vector<int> vlapp{30};
	cv::Mat gray_img_1,gray_img_2;
	cv::cvtColor(img1, gray_img_1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, gray_img_2, cv::COLOR_BGR2GRAY);
	orb_extractor(gray_img_1, cv::Mat(), kp1,d1, vlapp);
	orb_extractor(gray_img_2, cv::Mat(), kp2,d2, vlapp);
	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();  //add
        double mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        std::cout << "ORB extract time cost = " << mTimeORB_Ext << " milliseconds. " << endl; 

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	t1=std::chrono::steady_clock::now(); 
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	t2=std::chrono::steady_clock::now(); 
	double mTimeBF=std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        std::cout << "BF matches time cost = " << mTimeBF << " milliseconds. " << endl; 
#endif
	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();  //add
	// GMS filter
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
	//std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();  //add
        //double mTimeORB_Mat = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t4 - t3).count();
        //std::cout << "ORB-GMS matches time cost = " << mTimeORB_Mat << " milliseconds. " << endl; 

	//std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();  //add
	// int num_inliers = gms.GetInlierMask(vbInliers, false, false);	
	int num_inliers = gms.GetInlierMask(vbInliers, true, true);
	cout << "Get total " << num_inliers << " matches." << endl;
	std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();  //add
        double mTimeNum_inliers = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t6 - t3).count();
        std::cout << "GMS getInlierMask time cost = " << mTimeNum_inliers << " milliseconds. " << endl; 

	// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
 
	// draw matching
	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	imshow("show", show);
	//cv::imwrite("ORB-GMS.png",show);画出来是黄线，与shiyan_3不一致
	Mat img_match;
	drawMatches(img1, kp1, img2, kp2, matches_gms, img_match);
	imshow("ours", img_match);
	cv::imwrite("ours.png",img_match);
	waitKey();
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}

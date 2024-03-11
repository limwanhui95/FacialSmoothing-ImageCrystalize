#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#pragma once

class face_detector
{
public:
	face_detector();
	void detect_and_bilateralfilter(cv::Mat& src,cv::Mat& dst, double sigmaSpace, double sigmaweight);
private:
	cv::CascadeClassifier face_cascade;
	void HistogramEqualization(cv::Mat& src, cv::Mat& dst);
	int newPixelKValue(int v, int min, int max, int L, int Cdf[]);
	void MyBilateralfilter(cv::Mat& src, cv::Mat& dst, int nsize, double sigmaSpace, double sigmaWeight, std::vector<cv::Rect>& faces);
};


#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "face_detector.h"
#include "SLIC_processor.h"

face_detector myFaceDetector;
SLIC_processor my_SLIC;

cv::Mat image;
cv::Mat image2;
cv::Mat image3;

int sigmaspace_slider;
const int sigmaspace_slider_max = 100;
int sigmaweight_slider;
const int sigmaweight_slider_max = 100;
int k_super_pixel_slider;
const int k_super_pixel_slider_max = 3000;
int m_space_slider;
const int m_space_slider_max = 40;

static void bi_filter_on_trackbar(int, void*)
{
	image.convertTo(image2, -1, 1, 0);
	myFaceDetector.detect_and_bilateralfilter(image2, image2, double(sigmaspace_slider), double(sigmaweight_slider));
	cv::imshow("Face Bilateral_filter", image2);
}


static void superpixel_on_trackbar(int, void*)
{
	image.convertTo(image3, -1, 1, 0);
	my_SLIC.super_pixel(image3, k_super_pixel_slider, m_space_slider);
	cv::imshow("SLIC super pixel", image3);
}


int main(int argc, char** argv)
{
	std::string ImagePath;
	std::cout << "Please enter the image path:" << std::endl;
	std::getline(std::cin, ImagePath);
	// Read the image file
	image = cv::imread(ImagePath);

	if (image.empty()) // Check for failure
	{
		std::cout << "Could not open or find the image" << std::endl;
		system("pause"); //wait for any key press
		return -1;
	}

	// initial value for each parameter
	sigmaspace_slider = 50;
	sigmaweight_slider = 50;
	k_super_pixel_slider = 1024;
	m_space_slider = 15;

	char sigmaspace_trackbar_name[50] = "S_Space";
	char sigmaweight_trackbar_name[50] = "S_Weight";
	char k_superpixel_trackbar_name[50] = "K";
	char m_space_trackbar_name[50] = "M";

	cv::String window_2_Name = "Face Bilateral_filter";
	cv::namedWindow(window_2_Name);
	cv::String window_3_Name = "SLIC super pixel";
	cv::namedWindow(window_3_Name);


	cv::createTrackbar(sigmaspace_trackbar_name, window_2_Name, &sigmaspace_slider, sigmaspace_slider_max, bi_filter_on_trackbar);
	cv::createTrackbar(sigmaweight_trackbar_name, window_2_Name, &sigmaweight_slider, sigmaweight_slider_max, bi_filter_on_trackbar);
	bi_filter_on_trackbar(0, 0);
	
	cv::createTrackbar(k_superpixel_trackbar_name, window_3_Name, &k_super_pixel_slider, k_super_pixel_slider_max, superpixel_on_trackbar);
	cv::createTrackbar(m_space_trackbar_name, window_3_Name, &m_space_slider, m_space_slider_max, superpixel_on_trackbar);
	superpixel_on_trackbar(0, 0);

	cv::String windowName = "Original Image"; //Name of the window
	cv::namedWindow(windowName); // Create a window
	cv::imshow(windowName, image); // Show our image inside the created window.

	

	

	cv::waitKey(0); // Wait for any keystroke in the window
	cv::destroyAllWindows();



	

	return 0;
}


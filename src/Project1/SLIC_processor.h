#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

struct pixel
{
	uchar l;
	uchar a;
	uchar b;
	int x, y;
	int intensity;
	int label;
	double distance;
};
struct Cluster
{
	int l;
	int a;
	int b;
	int x;
	int y;
	std::vector<pixel> pixel_in_cluster;
};

class SLIC_processor
{
public:
	void super_pixel(cv::Mat& src, int k, int m);
	
private:
	void in_iterative();
	void initialize_cluster();
	void perturb_center();
	void update_center();
	void set_pixel_to_cluster_color();
	double cal_gradient(int pixel_index);
	double cal_distance(int cluster_index, int pixel_index);
	cv::Mat Lab_src;
	int space;
	int k_num;
	int m_paramter;
	int rows;
	int cols;
	int offset;
	std::vector<Cluster> my_cluster;
	std::vector<pixel> my_pixel;
};


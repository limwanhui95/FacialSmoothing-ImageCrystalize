#include "SLIC_processor.h"

void SLIC_processor::super_pixel(cv::Mat& src, int k, int m)
{
	k == 0 ? k_num = 1 : k_num = k;
	space = std::sqrt(src.rows * src.cols / k_num);
	m_paramter = m;
	cv::cvtColor(src, Lab_src, cv::COLOR_BGR2Lab);
	rows = Lab_src.rows;
	cols = Lab_src.cols * Lab_src.channels();
	offset = Lab_src.channels();
	for (int y = 0; y < rows; y++)			
	{
		uchar* PLabsrc = Lab_src.ptr<uchar>(y);
		for (int x = 0; x < cols; x += offset)
		{
			pixel tis_pixel;
			// initialize label and distance for each pixel
			tis_pixel.label = -1;
			tis_pixel.distance = 99999.9;	 
			for (int i = 0; i < offset; i++)
			{
				if (i == 0)
				{
					tis_pixel.l = PLabsrc[x + i];
				}
				else if (i == 1)
				{
					tis_pixel.a = PLabsrc[x + i];
				}
				else if (i == 2)
				{
					tis_pixel.b = PLabsrc[x + i];
				}
			}
			tis_pixel.intensity = tis_pixel.l + tis_pixel.a + tis_pixel.b;
			tis_pixel.x = x;
			tis_pixel.y = y;
			my_pixel.push_back(tis_pixel);
		}
	}
	initialize_cluster();
	perturb_center();
	in_iterative();
	set_pixel_to_cluster_color();
	// set lab color space back to bgr and pass the value to cv::mat src
	cv::cvtColor(Lab_src, src, cv::COLOR_Lab2BGR);

	// clear all vector in slic
	my_cluster.clear();
	my_pixel.clear();
}

void SLIC_processor::in_iterative()
{
	// iterative for 10 times
	for (int i = 0; i < 10; i++)
	{
		int cluster_index = 0;
		// for each cluster
		for (std::vector<Cluster>::iterator it = my_cluster.begin(); it != my_cluster.end(); it++)
		{
			// clear pixel in cluster vector
			it->pixel_in_cluster.clear();
			int start_x, end_x, start_y, end_y;
			// examine border
			start_x = ((it->x - 2 * space * offset) > 0) ? (it->x - 2 * space * offset) : 0;
			end_x = ((it->x + 2 * space * offset) < (cols - offset)) ? (it->x + 2 * space * offset) : (cols - offset);
			start_y = ((it->y - 2 * space) > 0) ? (it->y - 2 * space) : 0;
			end_y = ((it->y + 2 * space) < (rows - 1)) ? (it->y + 2 * space) : (rows-1);
			// for each pixel in a 2s*2s region around cluster
			for (int y = start_y; y <= end_y; y++)
			{
				for (int x = start_x; x <= end_x; x += offset)
				{
					int pixel_index = (y * (cols) + x) / 3;
					double temp_dis;
					temp_dis = cal_distance(cluster_index, pixel_index);
					if (temp_dis < my_pixel[pixel_index].distance)
					{
						my_pixel[pixel_index].distance = temp_dis;
						my_pixel[pixel_index].label = cluster_index;
						// record pixel into cluster
						it->pixel_in_cluster.push_back(my_pixel[pixel_index]);
					}
				}
			}
			cluster_index++;
		}
		update_center();
	}
}

void SLIC_processor::initialize_cluster()
{
	for (int y = space; y <= rows - space; y += space)
	{
		for (int x = offset * space; x <= cols - offset * space; x += offset * space)
		{
			// a new cluster
			Cluster ck;
			ck.y = y;
			ck.x = x;
			int index = (y * cols + x) / 3;
			ck.l = my_pixel[index].l;
			ck.a = my_pixel[index].a;
			ck.b = my_pixel[index].b;
			my_cluster.push_back(ck);
		}
	}
	// reset the number of k
	k_num = my_cluster.size();
}

void SLIC_processor::perturb_center()
{	
	
	for (int n = 0; n < k_num; n++)
	{
		int y_coord = my_cluster[n].y;
		int x_coord = my_cluster[n].x;
		double min_gradient = 99999.9; // set to a big amount at first step
		int temp_x, temp_y;
		// search in a neighbour of 3x3
		for (int i = -1; i < 2; i++)
		{
			uchar* PLabsrc = Lab_src.ptr<uchar>(y_coord + i);
			for (int j = -offset; j <= offset; j+= offset)
			{
				double temp_g;
				int index = ((y_coord + i) * (cols) + (x_coord + j)) / offset;
				temp_g = cal_gradient(index);
				if (temp_g < min_gradient)
				{
					// pick the lowest gradient pixel to be the cluster point
					min_gradient = temp_g;
					temp_x = x_coord;
					temp_y = y_coord;
				}
			}
		}
		// set the new cluster to the min gradient pixel
		my_cluster[n].y = temp_y;
		my_cluster[n].x = temp_x;
		int index = (temp_y * (cols) + temp_x) / 3;
		my_cluster[n].l = my_pixel[index].l;
		my_cluster[n].a = my_pixel[index].a;
		my_cluster[n].b = my_pixel[index].b;
		
	}
}

void SLIC_processor::update_center()
{
	// update the cluster center for each cluster
	for (std::vector<Cluster>::iterator it = my_cluster.begin(); it != my_cluster.end(); it++)
	{
		int center_x = 0;
		int center_y = 0;
		for (std::vector<pixel>::iterator pit = it->pixel_in_cluster.begin(); pit != it->pixel_in_cluster.end(); pit++)
		{
			center_x += pit->x;
			center_y += pit->y;
		}
		center_x = center_x / it->pixel_in_cluster.size();
		center_y = center_y / it->pixel_in_cluster.size();
	}
	// move center point in 3x3 region
	perturb_center();
}

void SLIC_processor::set_pixel_to_cluster_color()
{
	// for every pixel, set its color value equal to it's cluster color value
	for (std::vector<pixel>::iterator it = my_pixel.begin(); it != my_pixel.end(); it++)
	{
		int i = it->y;
		int j = it->x;
		uchar* Psrc = Lab_src.ptr<uchar>(i);
		Psrc[j] = my_cluster[it->label].l;
		Psrc[j+1] = my_cluster[it->label].a;
		Psrc[j+2] = my_cluster[it->label].b;
	}
}

double SLIC_processor::cal_gradient(int pixel_index)
{
	double dIdX, dIdY;
	dIdX = double(my_pixel[pixel_index + 1].intensity) - double(my_pixel[pixel_index].intensity);
	dIdY = double(my_pixel[pixel_index + cols / offset].intensity) - double(my_pixel[pixel_index].intensity);
	double gradient = std::sqrt(dIdX * dIdX + dIdY * dIdY);
	return gradient;
}

double SLIC_processor::cal_distance(int cluster_index, int pixel_index)
{
	double d_c, d_s;
	d_c =
		((double)my_cluster[cluster_index].l - (double)my_pixel[pixel_index].l) * ((double)my_cluster[cluster_index].l - (double)my_pixel[pixel_index].l) +
		((double)my_cluster[cluster_index].a - (double)my_pixel[pixel_index].a) * ((double)my_cluster[cluster_index].a - (double)my_pixel[pixel_index].a) +
		((double)my_cluster[cluster_index].b - (double)my_pixel[pixel_index].b) * ((double)my_cluster[cluster_index].b - (double)my_pixel[pixel_index].b);
	d_s =
		((double)my_cluster[cluster_index].x - (double)my_pixel[pixel_index].x) * ((double)my_cluster[cluster_index].x - (double)my_pixel[pixel_index].x) +
		((double)my_cluster[cluster_index].y - (double)my_pixel[pixel_index].y) * ((double)my_cluster[cluster_index].y - (double)my_pixel[pixel_index].y);

	double distance;
	distance = std::sqrt(d_c + (d_s) * (double(m_paramter) * double(m_paramter)) / (double(space) * double(space)));
	return distance;
}


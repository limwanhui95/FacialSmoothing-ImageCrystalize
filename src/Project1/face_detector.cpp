#include "face_detector.h"

face_detector::face_detector()
{
	cv::String face_cascade_name = "./classifier/haarcascade_frontalface_alt.xml";
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "--(!)Error loading face cascade\n";
	};
}

void face_detector::detect_and_bilateralfilter(cv::Mat& src, cv::Mat& dst, double sigmaSpace, double sigmaweight)
{
	std::vector<cv::Rect> faces;
	cv::Mat gray_src;
	cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
	HistogramEqualization(gray_src, gray_src);
	cv::equalizeHist(gray_src, gray_src);
	face_cascade.detectMultiScale(gray_src, faces);
	MyBilateralfilter(src, dst, 7, sigmaSpace, sigmaweight, faces);
}


int face_detector::newPixelKValue(int v, int min, int max, int L, int Cdf[])
{
	double temp = ((double)(Cdf[v] - min) / (double)(max - min)) * ((double)(L)-1.0);
	int ans = static_cast<int>(temp);
	return ans;
}

void face_detector::HistogramEqualization(cv::Mat& src, cv::Mat& dst)
{
	int L = 256;
	int Pk[256] = { 0 };
	int n = src.rows * src.cols;
	int Cdf[256] = { 0 };
	int minCdf, maxCdf;
	for (int y = 0; y < src.rows; y++) //计算灰度直方图
	{
		uchar* Psrc = src.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			int k = Psrc[x];   // k灰度级
			Pk[k]++;
		}
	}
	for (int i = 0; i < L; i++)   //计算累计直方图cdf
	{
		if (i == 0)
		{
			Cdf[i] = Pk[i];
			minCdf = Cdf[i];
			maxCdf = Cdf[i];
		}
		else
		{
			Cdf[i] = Cdf[i - 1] + Pk[i];
			if (Cdf[i] < minCdf)
				minCdf = Cdf[i];
			if (Cdf[i] > maxCdf)
				maxCdf = Cdf[i];
		}
	}

	//变换到目标图像
	for (int y = 0; y < src.rows; y++)
	{
		uchar* Psrc = src.ptr<uchar>(y);
		uchar* Pdst = dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			int v = Psrc[x];
			Pdst[x] = newPixelKValue(v, minCdf, maxCdf, L, Cdf);
		}
	}
}

void face_detector::MyBilateralfilter(cv::Mat& src, cv::Mat& dst, int nsize, double sigmaSpace, double sigmaWeight, std::vector<cv::Rect>& faces)
{
	// 定义权重模板
	std::vector<std::vector<double>> templ(nsize, std::vector<double>(nsize));  //二维vector
	int center = (nsize - 1) / 2;			// center coordinate for the template
	double normalisation = 0;

	// 计算距离权重
	for (int i = 0; i < nsize; i++)
		for (int j = 0; j < nsize; j++)
		{
			int isquare = (i - center) * (i - center);				//离中心点y方向距离
			int jsquare = (j - center) * (j - center);				//离中心点x方向距离	
			templ[i][j] = std::exp((double)(-isquare - jsquare) / (2.0 * sigmaSpace * sigmaSpace ));
		}

	//计算输出结果
	int cols = src.cols * src.channels() - 1;
	int offset = src.channels();
	int rows = src.rows;

	// 计算色差权重
	std::vector<double> colorWeight(offset * 256);
	for (int i = 0; i < offset * 256; i++)
	{
		colorWeight[i] = (double)std::exp(-i * i / (2.0 * sigmaWeight * sigmaWeight));
	}


	for (size_t i = 0; i < faces.size(); i++)		// 对每一张检测到的脸做双边滤波算法
	{
		int start_X, end_X, start_Y, end_Y;
		if (faces[i].x * offset < offset * center)  //考虑到超出边界
		{
			start_X = offset * center;
		}
		else
		{
			start_X = faces[i].x * offset;
		}
		if ((faces[i].x + faces[i].width ) * offset > cols - offset * center)
		{
			end_X = cols - offset * center;
		}
		else
		{
			end_X = (faces[i].x + faces[i].width) * offset;
		}
		if (faces[i].y  < center)
		{
			start_Y = center;
		}
		else
		{
			start_Y = faces[i].y;
		}
		if (faces[i].y + faces[i].height > rows - center)
		{
			end_Y = rows - center;
		}
		else
		{
			end_Y = faces[i].y + faces[i].height;
		}

		// 开始进行双边滤波算法
		for (int y = start_Y; y < end_Y; y++)
		{
			uchar* Pdst = dst.ptr<uchar>(y);
			uchar* Psrc_center = src.ptr<uchar>(y);
			for (int x = start_X; x < end_X; x += offset)
			{
				double normalisationSum = 0;
				
				std::vector<double> sum(offset, 0);
				for (int i = 0; i < nsize; i++)		//计算每个像素点乘与模板值
				{
					uchar* Psrc = src.ptr<uchar>(y - center + i);
					for (int j = 0; j < nsize; j++)
					{
						int colorWeight_indicator = 0;
						for (int c = 0; c < offset; c++)
						{
							colorWeight_indicator += std::abs(Psrc[x - offset * (center - j) + c] - Psrc_center[x + c]);
						}
						normalisationSum += templ[i][j] * colorWeight[colorWeight_indicator];
						for (int c = 0; c < offset; c++)
						{
							sum[c] += templ[i][j] * colorWeight[colorWeight_indicator] * Psrc[x - offset * (center - j) + c];
						}
					}
				}
				for (int c = 0; c < offset; c++)
				{
					Pdst[x + c] = sum[c] / normalisationSum;
				}			

			}
		}
	}
}




#pragma once
#include <QtCore/qstring.h>
#include <QtCore/qdir.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define SKELETONIZATION_HPP_INCLUDED


using namespace cv;
using namespace std;

void SaveImage(Mat pImage, QString fileName)
{
#ifdef _DEBUG
#else
		return;
#endif
	QString path = QDir::currentPath().section("/", 0, -2) + fileName;
	std::string file = path.toLocal8Bit().data();
	imwrite(file, pImage);
}

void SaveImage(float* pImage, QString fileName,int w, int h)
{
	if (!pImage)
		pImage = new float[w * h];

	Mat img(Size(w, h), CV_8UC1, Scalar(0));

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			if (isnan(pImage[y * h + x]))
				pImage[y * h + x] = 0;
			img.at<uchar>(y, x) = pImage[y * h + x] * 255;
		}
	}
	QString path = QDir::currentPath().section("/", 0, -2) + fileName;
	std::string file = path.toLocal8Bit().data();
	imwrite(file, img);
}


void convertToImage(Mat opencvImage, float* pImage, int w, int h)
{
	if (!pImage)
		pImage = new float[w * h];

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = opencvImage.at<uchar>(y, x);
			if (int(val) == 0)
				pImage[y * h + x] = 0;
			else
				pImage[y * h + x] = 1;
		}
	}
}

Mat ImageCut(Mat inputImage, int nRange)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	int w = inputImage.rows;
	int h = inputImage.cols;
	memcpy(outputImage.data, inputImage.data, sizeof(unsigned char) * inputImage.rows * inputImage.cols);

	for (int y = h - nRange; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			outputImage.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < nRange; y++)
	{
		for (int x = 0; x < w; x++)
		{
			outputImage.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < h; y++)
	{
		for (int x = w - nRange; x < w; x++)
		{
			outputImage.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < nRange; x++)
		{
			outputImage.at<uchar>(y, x) = 0;
		}
	}

	return outputImage;
}

Mat CreateColorImage(int* pImage,int w, int h)
{
	Mat img(Size(w, h), CV_32FC3, pImage);
	return img;
}

Mat CreateImage(float* pImage, int w, int h, bool bBorder = false ,int nRange = 10,bool btest = true)
{
	float* data = new float[w * h];
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			if (btest)
			{

				if (pImage[y * h + x] > 0)
				{
					data[y * h + x] = 255;
				}
				else
					data[y * h + x] = 0;
			}
			else
			{
				if(pImage[y * h + x] != 0)
					data[y * h + x] = fabs(pImage[y * h + x])*255;
			}
		}
	}
	Mat img(Size(w, h), CV_32FC1, data);
	img.convertTo(img, CV_8UC1);

	if (bBorder)
	{
		img = ImageCut(img, nRange);
		//SaveImage(img, QString("\\Test\\ImageCut.png"));
	}
	delete[] data;
	data = nullptr;
	SaveImage(img, QString("\\Test\\CreateImage.png"));
	return img;
}


void SaveImage(float* pImage, int w, int h, int viewIndex, QString fileName, QString folderName,bool btest = true)
{
	QString strName;
	if (viewIndex == 0)
		strName = "L_" + fileName;
	else
		strName = "R_" + fileName;
	std::stringstream ss;

	auto opencvImage = CreateImage(pImage, w, h, false, 10, btest);
	auto path = QDir::currentPath().section("/", 0, -2) + "\\" + folderName + "\\" + strName + ".bmp";
	std::string file = path.toLocal8Bit().data();
	imwrite(file, opencvImage);

}

Mat Labeling(Mat pImage, int nArea,bool skel=false)
{
	Mat img(pImage.size(), CV_8UC1, Scalar(0));
	//Mat img2(pImage.size(), CV_8UC1, Scalar(0));
	SaveImage(pImage, "\\Test\\Labeling1.png");

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(pImage, labels, stats, centroids);
	if (cnt <= 2)
		return pImage;
	if (skel)
	{
		vector<float> vecArea;
		int sum = 0;
		for (int i = 1; i < cnt; i++)
		{
			int* p = stats.ptr<int>(i);
			vecArea.push_back(p[4]);
			sum += p[4];
		}
		if(nArea > sum * 0.2)
			nArea = sum * 0.2;
	}
	Mat surfSup = stats.col(4) > nArea;

	for (int i = 1; i < cnt; i++)
	{
		if (surfSup.at<uchar>(i, 0))
			img = img | (labels == i);
	}
	SaveImage(img, "\\Test\\Labeling2.png");
	return img;
}

Mat LabelingMax(Mat pImage)
{
	Mat img(pImage.size(), CV_8UC1, Scalar(0));
	//Mat img2(pImage.size(), CV_8UC1, Scalar(0));

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(pImage, labels, stats, centroids);

	float sum = 0;
	vector <int> vecArea;
	for (int i = 1; i < cnt; i++)
		vecArea.push_back(stats.at<int>(i, CC_STAT_AREA));
	sort(vecArea.begin(), vecArea.end());
	auto area = vecArea[vecArea.size() - 1];
	for (int i = 1; i < cnt; i++)
	{
		if (area == stats.at<int>(i, CC_STAT_AREA))
			img = img | (labels == i);
	}

	//img = img | (labels == (vecArea.size()-1));
	//SaveImage(img, "\\Test\\Labeling1.png");
	return img;
}

Mat Skeleton(Mat pImage)
{
	cv::Mat skel(pImage.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::erode(pImage, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(pImage, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(pImage);

		done = (cv::norm(pImage) == 0);
	} while (!done);
	//cv::dilate(skel, skel, element); // temp = open(img)

	SaveImage(skel, "\\Test\\skel2.png");
	return skel;
	//pImage = medial_axis(pImage).astype(uint8) * 255;
}


void thinningIteration(Mat& im, int iter)
{
	Mat marker = Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

Mat thinning(Mat& im)
{
	im /= 255;

	Mat prev = Mat::zeros(im.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (countNonZero(diff) > 0);

	im *= 255;

	return im;
}


Mat skeletonization(Mat inputImage)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	outputImage = thinning(inputImage);
	SaveImage(outputImage, "\\Test\\skel1.png");
	return outputImage;
}


Mat Dilate(Mat inputImage,int N)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	dilate(inputImage, outputImage, Mat::ones(Size(N, N), CV_8UC1), Point(-1, -1));
	return outputImage;
}

Mat Erode(Mat inputImage, int N)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	erode(inputImage, outputImage, Mat::ones(Size(N, N), CV_8UC1), Point(-1, -1));
	return outputImage;
}

Mat BordCut(Mat inputImage, int w, int h, int N)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	inputImage.copyTo(outputImage);
	SaveImage(inputImage, "\\Test\\BordCut1.png");

	Mat img_houghP = outputImage;
	//Canny(outputImage, img_houghP, 50, 255);

	//img_houghP = Dilate((img_houghP), 3);
	//SaveImage(img_houghP, "\\Test\\BordCut2.png");

	//Mat img_houghP = Dilate(skeletonization(inputImage), 3);

	vector<Vec4i> linesP;
	HoughLinesP(img_houghP, linesP, 1, (CV_PI / 180), 150, 50, 5);
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		float2 pt1 = make_float2(l[0], l[1]);
		float2 pt2 = make_float2(l[2], l[3]);

		if (pt1.x < N && pt2.x < N) // #¿Þ
			line(outputImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.y < N && pt2.y < N) // #À§
			line(outputImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.x > w - N && pt2.x > w - N) // #¿À
			line(outputImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.y > h - N && pt2.y > h - N) // # ¾Æ·¡
			line(outputImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
	}
	SaveImage(outputImage, "\\Test\\BordCut3.png");
	return outputImage;
}

float* CreateSkeletonImage(float* pbuffer, int w, int h, bool bBorder)
{
	int N = 10;
	auto opencvImage = CreateImage(pbuffer, w, h, true, N);
	auto SkeletonImage = skeletonization(opencvImage);
	if (bBorder)
	{
		vector<Vec4i> linesP;
		auto labelImage = LabelingMax(SkeletonImage);
		Mat img_houghP = Dilate(labelImage, 3);
		SaveImage(img_houghP, "\\Test\\skel3.png");

		HoughLinesP(img_houghP, linesP, 1, (CV_PI / 180), 100, 150, 5);
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i l = linesP[i];
			float2 pt1 = make_float2(l[0], l[1]);
			float2 pt2 = make_float2(l[2], l[3]);
			if (pt1.x < w / 2 && pt2.x < w / 2) // #¿Þ
				line(labelImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		}
		SaveImage(labelImage, "\\Test\\skel4.png");
		labelImage = Labeling(labelImage, 300,true);
		SaveImage(labelImage, "\\Test\\skel5.png");
		convertToImage(labelImage, pbuffer, w, h);
	}
	else
	{
		auto labelImage = Labeling(SkeletonImage, 300,true);
		convertToImage(labelImage, pbuffer, w, h);
	}

	return pbuffer;
}

float* CreateDilateImage(float* pbuffer,int N, int w, int h)
{
	auto opencvImage = CreateImage(pbuffer, w, h);
	opencvImage = Dilate(opencvImage,N);
	convertToImage(opencvImage, pbuffer, w, h);
	return pbuffer;
}

Mat CreateColorImage(QString filePath)
{
	cv::Mat opencvImage1 = cv::imread(filePath.toLocal8Bit().data(), cv::IMREAD_COLOR);
	return opencvImage1;
}

void DrawCentLine(Mat cvColorImage, vector<float2> pts, int w, int h,int n)
{
	cv::Mat outputImage(cvColorImage.rows, cvColorImage.cols, CV_8UC3);//w,h
	cvColorImage.copyTo(outputImage);

	for (int i = 0; i < pts.size(); i++)
	{
		int x = pts[i].x;
		int y = pts[i].y;
		outputImage.at<cv::Vec3b>(y, x)[0] = 34;
		outputImage.at<cv::Vec3b>(y, x)[1] = 177;
		outputImage.at<cv::Vec3b>(y, x)[2] = 76;
	}

	auto fileName = "\\Test\\Line" + QString::number(n) + ".png";
	SaveImage(outputImage, fileName);
}


void DrawCircle(float*& pbuffer, int2 pos,int size, int w, int h)
{
	cv::Mat inputImage = CreateImage(pbuffer, w, h);
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	outputImage = Mat::zeros(outputImage.size(), CV_8UC1);
	circle(outputImage, Point(pos.x, pos.y), size, Scalar(255, 255, 255), -1, LINE_AA, 0);
	vector<int2> pos1;
	SaveImage(outputImage, QString("\\Test\\Circle.png"));

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = outputImage.at<uchar>(y, x);
			auto val2 = inputImage.at<uchar>(y, x);
			if (int(val) > 200 || int(val2) == 255)
			{
				outputImage.at<uchar>(y, x) = 255;
				inputImage.at<uchar>(y, x) = 255;
				pos1.push_back(make_int2(x, y));
			}
			else
			{
				outputImage.at<uchar>(y, x) = 0;
				inputImage.at<uchar>(y, x) = 0;
			}
		}
	}
	SaveImage(outputImage, QString("\\Test\\Circle1.png"));

	convertToImage(outputImage, pbuffer, w, h);
}

vector<float2> SetFloodFill(float*& pbuffer,vector<float2> data,int w,int h)
{
	cv::Mat inputImage = CreateImage(pbuffer, w, h);
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	outputImage = Mat::zeros(outputImage.size(), CV_8UC1);
	//imshow("222", inputImage);

	for (int i = 0; i < data.size() - 1; i++)
		line(outputImage, Point(data[i].x, data[i].y), Point(data[i + 1].x , data[i + 1].y ), Scalar(255, 255, 255), 1, 8);
	floodFill(outputImage, cv::Point(1, 1), Scalar(255));
	//imshow("dd", outputImage);
	data.clear();
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = outputImage.at<uchar>(y, x);
			if (int(val) == 255)
				outputImage.at<uchar>(y, x) = 0;
			else
			{
				outputImage.at<uchar>(y, x) = 255;
			}

			val = outputImage.at<uchar>(y, x);
			auto val2 = inputImage.at<uchar>(y, x);
			if (int(val) == 255 || int(val2) == 255)
			{
				outputImage.at<uchar>(y, x) = 255;
				inputImage.at<uchar>(y, x) = 255;
				data.push_back(make_float2(x, y));
			}
			else
			{
				inputImage.at<uchar>(y, x) = 0;
			}
		}
	}
	convertToImage(inputImage, pbuffer, w, h);
	return data;
}


vector<float2> LoadImageBrush(Mat img)
{
	vector<float2> pts;
	auto green = Scalar(34, 177, 76);

	Mat img_color;
	inRange(img, Scalar(0, 128, 0), Scalar(100, 255, 100), img_color);

	imwrite("dd.png", img_color);
	auto sum = countNonZero(img_color);
	int h = img.cols;
	int w = img.rows;

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = img_color.at<uchar>(y, x);
			if (int(val) != 0)
			{
				pts.push_back(make_float2(x, y));
			}
		}
	}
	return pts;
}

void convert_image(float* input_image, Mat& output_image, int w, int h, bool bin)
{
	Mat temp(Size(w, h), CV_32FC1);
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			float val = input_image[y * w + x];
			temp.at<float>(y, x) = (bin && val > 0) ? 255.0f : val * 255;
		}
	}
	temp.convertTo(output_image, CV_8UC1);
	SaveImage(output_image, "Test\\convert_image.png");
	SaveImage(output_image, "Test\\convert_image.png");
}

void draw_line(float* arr, const std::vector<float2>& line_c1, const std::vector<float2>& line_c2, const std::vector<float2>& points, const int& w, const int& h, std::string fileName)
{
	Mat cvimage(Size(w, h), CV_32FC3);
	convert_image(arr, cvimage, w, h, false);
	cv::Mat outputImage(cvimage.rows, cvimage.cols, CV_32FC3);//w,h
	cvimage.copyTo(outputImage);
	if (line_c1.size() != 0)
	{
		for (int i = 0; i < line_c1.size() - 1; i++)
			line(outputImage, Point(line_c1[i].x, line_c1[i].y), Point(line_c1[i + 1].x, line_c1[i + 1].y), Scalar(255, 255, 255), 1, 8);
	}
	if (line_c2.size() != 0)
	{
		for (int i = 0; i < line_c2.size() - 1; i++)
			line(outputImage, Point(line_c2[i].x, line_c2[i].y), Point(line_c2[i + 1].x, line_c2[i + 1].y), Scalar(255, 255, 255), 1, 8);
	}

	if (points.size() != 0)
	{
		for (int i = 0; i < points.size() - 1; i++)
			line(outputImage, Point(points[i].x, points[i].y), Point(points[i + 1].x, points[i + 1].y), Scalar(255, 255, 255), 1, 8);
	}

	std::string  save_name = "\\Test\\" + fileName;
	SaveImage(outputImage, save_name.c_str());
}

void LoadImagelebeling(Mat ColorImage, QString filePath, vector<float2> pts,int w, int h)
{
	for (int i = 0; i < pts.size();i++)
	{
		int x = pts[i].x;
		int y = pts[i].y;
		ColorImage.at<cv::Vec3b>(y, x)[0] = 34;
		ColorImage.at<cv::Vec3b>(y, x)[1] = 177;
		ColorImage.at<cv::Vec3b>(y, x)[2] = 76;
	}
	imwrite(filePath.toLocal8Bit().data(), ColorImage);
}

void Imagelebeling(Mat ColorImage,QString filePath,float*& pbuffer, int w, int h)
{
	cv::Mat inputImage = CreateImage(pbuffer, w, h);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = inputImage.at<uchar>(y, x);
			if (int(val) != 0)
			{
				ColorImage.at<cv::Vec3b>(y, x)[0] = 34;
				ColorImage.at<cv::Vec3b>(y, x)[1] = 177;
				ColorImage.at<cv::Vec3b>(y, x)[2] = 76;
			}
		}
	}
	imwrite(filePath.toLocal8Bit().data(), ColorImage);
}


void Testlebeling(Mat ColorImage, QString filePath, vector<int2> left, vector<int2> right,int w, int h)
{
	auto rows = ColorImage.rows;
	auto cols = ColorImage.cols;
	Mat img(ColorImage.size(), CV_8UC1, Scalar(0));

	auto green = Scalar(34, 177, 76);
	

	line(img, Point(left[0].x, left[0].y), Point(right[0].x, right[0].y), Scalar(255, 255, 255), 2, 8);
	line(img, Point(left[left.size()-1].x, left[left.size()-1].y), Point(right[right.size()-1].x, right[right.size()-1].y), Scalar(255, 255, 255), 2, 8);
	for (int i = 0; i < left.size()-1; i++)
	{
		line(img, Point(left[i].x, left[i].y), Point(left[i + 1].x, left[i + 1].y), Scalar(255, 255, 255), 2, 8);
		line(img, Point(right[i].x, right[i].y), Point(right[i+1].x, right[i+1].y), Scalar(255, 255, 255), 2, 8);
	}
	floodFill(img, cv::Point(0, 0), Scalar(255));

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = img.at<uchar>(y, x);
			if (int(val) == 0)
			{
				cv::Vec3b& p1 = ColorImage.at<cv::Vec3b>(y, x);
				ColorImage.at<cv::Vec3b>(y, x)[0] = 34;
				ColorImage.at<cv::Vec3b>(y, x)[1] = 177;
				ColorImage.at<cv::Vec3b>(y, x)[2]= 76;
			}
		}
	}

	line(ColorImage, Point(left[0].x, left[0].y), Point(right[0].x, right[0].y), green, 2, 8);
	line(ColorImage, Point(left[left.size() - 1].x, left[left.size() - 1].y), Point(right[right.size() - 1].x, right[right.size() - 1].y), green, 2, 8);
	std::vector<cv::Point> pts;
	std::vector<cv::Point> pts2;

	for (int i = 0; i < left.size() - 1; i++)
	{
		pts.push_back(cv::Point(left[i].x, left[i].y));
		pts2.push_back(cv::Point(right[i].x, right[i].y));
		line(ColorImage, Point(left[i].x, left[i].y), Point(left[i + 1].x, left[i + 1].y), green, 2, 8);
		line(ColorImage, Point(right[i].x, right[i].y), Point(right[i + 1].x, right[i + 1].y), green, 2, 8);
	}
	//cv::polylines(opencvImage1, pts, true, green, 2);
	//cv::polylines(opencvImage1, pts2, true, green, 2);


	imwrite(filePath.toLocal8Bit().data(), ColorImage);
}

void EndPoints(vector<vector<float2>> lebeling_points ,vector<float2> center_point,int w,int h)
{
	auto calculate_distance = [&](float2 a, float2 b)
	{
		return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
	};

	auto GetDistance2End = [&](float2* vecLinepos, float2 end)
	{
		Object::dm _dm;
		float _min = FLT_MAX;
		int _minId = 0, current = 0;
		float2 pos = make_float2(0, 0);
		for (auto i = 0; i < 100; i++)
		{
			auto f = vecLinepos[i];
			auto _val = (f.x - end.x) * (f.x - end.x) + (f.y - end.y) * (f.y - end.y);
			if (_val < _min)
			{
				_min = _val;
				_minId = current;
				pos.x = f.x;
				pos.y = f.y;
			}
			current++;
		}
		return _minId;
	};

	int nEndId = 30;
	float2 ptEndPos = (center_point[nEndId]);
	//float2 ptEndPos = (pCenterData[40]);

	QString dataFolder = QDir::currentPath().section("/", 0, -2) + "\\data\\EndPoints.dat";

	auto fEndpoint = fopen((dataFolder.toLocal8Bit().data()), "w");
	fprintf(fEndpoint, "%d\n", lebeling_points.size());
	bool bfix = false;
	int nSavefileNumber = 0;
	auto p1 = make_float2(center_point[1].x + (center_point[0].x - center_point[1].x) * 5, center_point[1].y + (center_point[0].y - center_point[1].y) * 5);
	int id_prev = -1;
	for (int i = 0; i < lebeling_points.size(); i++)
	{
		if (nSavefileNumber == 20)
			break;
		auto lebel_image_Data = lebeling_points[i];
		if (lebel_image_Data.size() == 0)
			continue;

		{
			vector<float2> arrPts;
			vector<float> arrdis;
			nSavefileNumber++;
			Mat img_color(Size(w, h), CV_8UC3, Scalar(0, 0, 0));
			Mat img_color2(Size(w, h), CV_8UC3, Scalar(0, 0, 0));
			for (int n = 0; n < lebeling_points[i].size(); n++)
			{
				auto pos = make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y);
				img_color.at<cv::Vec3b>(pos.y, pos.x)[0] = 255;
				img_color.at<cv::Vec3b>(pos.y, pos.x)[1] = 255;
				img_color.at<cv::Vec3b>(pos.y, pos.x)[2] = 255;
				arrPts.push_back(make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y));
				arrdis.push_back(calculate_distance(make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y), p1));
			}
			if (arrdis.size() != 0)
			{
				fprintf(fEndpoint, "%d,", i + 1);
				int minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				int id = GetDistance2End(center_point.data(), arrPts[minid]);

				auto ptOffset = make_float2((arrPts[minid].x - center_point[0].x), (arrPts[minid].y - center_point[0].y));
				vector<float2> newArrPts;
				arrdis.clear();

				for (int n = 0; n < center_point.size()-1; n++)
				{
					line(img_color, Point(center_point[n].x, center_point[n].y), Point(center_point[n + 1].x, center_point[n + 1].y), Scalar(0, 0, 255), 2, 8);
					line(img_color2, Point(center_point[n].x, center_point[n].y), Point(center_point[n + 1].x, center_point[n + 1].y), Scalar(0, 0, 255), 2, 8);
				}
				auto savefile1 = QDir::currentPath().section("/", 0, -2) + "\\data\\test\\" + QString::number(i+1) + ".bmp";
				img_color.at<Vec3b>(arrPts[minid].y, arrPts[minid].x) = Vec3b(0, 255, 255);
				line(img_color, Point(arrPts[minid].x - 3, arrPts[minid].y), Point(arrPts[minid].x + 3, arrPts[minid].y ), Scalar(255, 0, 255), 2, 8);
				line(img_color, Point(arrPts[minid].x, arrPts[minid].y - 3), Point(arrPts[minid].x, arrPts[minid].y + 3), Scalar(255, 0, 255), 2, 8);
				imwrite(savefile1.toLocal8Bit().data(), img_color);


				for (int n = 0; n < arrPts.size(); n++)
				{
					auto pos = make_float2(arrPts[n].x - ptOffset.x, arrPts[n].y - ptOffset.y);
					if (pos.x > 0 && pos.x < w && pos.y > 0 && pos.y < h)
					{
						img_color2.at<cv::Vec3b>(pos.y, pos.x)[0] = 255;
						img_color2.at<cv::Vec3b>(pos.y, pos.x)[1] = 255;
						img_color2.at<cv::Vec3b>(pos.y, pos.x)[2] = 255;
						arrdis.push_back(calculate_distance(make_float2(pos.x, pos.y), ptEndPos));
						newArrPts.push_back(make_float2(pos.x, pos.y));
					}
				}
				minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				id = GetDistance2End(center_point.data(), newArrPts[minid]);

				line(img_color2, Point(newArrPts[minid].x - 3, newArrPts[minid].y), Point(newArrPts[minid].x + 3, newArrPts[minid].y), Scalar(0, 255, 255), 2, 8);
				line(img_color2, Point(newArrPts[minid].x, newArrPts[minid].y - 3), Point(newArrPts[minid].x, newArrPts[minid].y + 3), Scalar(0, 255, 255), 2, 8);
				auto savefile = QDir::currentPath().section("/", 0, -2) + "\\data\\test\\" + QString::number(i + 1) + "_Move.bmp";
				imwrite(savefile.toLocal8Bit().data(), img_color2);
				fprintf(fEndpoint, "%d,%d\n", int(newArrPts[minid].x), int(newArrPts[minid].y));


				if (nEndId + id >= center_point.size())
					bfix = true; 
				if (bfix)
					ptEndPos = center_point[center_point.size() - 1];
				else
					ptEndPos = center_point[id + nEndId];
				id_prev = id;
			}
			else
			{
			}
		}

	}
	fclose(fEndpoint);
}
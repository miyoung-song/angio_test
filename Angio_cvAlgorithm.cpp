#include "Angio_cvAlgorithm.h"


// 두 포인트 사이의 거리 계산 함수
double euclideanDistance(const Point& a, const Point& b) {
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

std::vector<std::string> split(const std::string& str, char sep) {
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(str);

	while (std::getline(tokenStream, token, sep)) {
		tokens.push_back(token);
	}

	return tokens;
}

std::string section(const std::string& str, char sep, int start, int end = -1) {
	std::vector<std::string> tokens = split(str, sep);
	std::string result;

	// Handle negative indices for 'end' value
	if (end < 0) {
		end = tokens.size() + end;
	}

	if (start < 0) {
		start = tokens.size() + start;  // Read from the right if start is negative
	}

	// Ensure 'start' and 'end' are within valid range
	start = std::max(0, start);
	end = std::min(static_cast<int>(tokens.size()) - 1, end);

	for (int i = start; i <= end; ++i) {
		if (i != start) result += sep;  // Add separator back
		result += tokens[i];
	}

	return result;
}

Angio_cvAlgorithm::Angio_cvAlgorithm()
{
	program_path_ = std::filesystem::current_path().string();
	size_t pos = program_path_.find_last_of('\\');
	if (pos != std::string::npos)
		program_path_ = program_path_.substr(0, pos);
}

Angio_cvAlgorithm::~Angio_cvAlgorithm()
{
}

void Angio_cvAlgorithm::draw_line(float* arr, const std::vector<float2>& line_c1, const std::vector<float2>& line_c2, const std::vector<float2>& points, const int& w, const int& h, std::string fileName)
{
}



void Angio_cvAlgorithm::save_image(Mat input_image, std::string file_name, int w, int h)
{
#ifdef SAVE_TEST_IMAGE
#else
	return;
#endif
	imwrite(program_path_ + "\\" + file_name + ".png", input_image);
}

void Angio_cvAlgorithm::save_image(float* input_image, std::string file_name, int w, int h, bool bin)
{
#ifdef SAVE_TEST_IMAGE
#else
	return;
#endif

	Mat img_org(Size(w, h), CV_8UC1, Scalar(0));
	Mat img_bin(Size(w, h), CV_8UC1, Scalar(0));

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			if (input_image[y * h + x] > 0)
				img_bin.at<uchar>(y, x) = 255;
			img_org.at<uchar>(y, x) = input_image[y * h + x] * 255;
		}
	}

	//imwrite(program_path_ + "\\" + file_name + "_org.png", img_org);
	if (bin)
		imwrite(program_path_ + "\\" + file_name + ".png", img_bin);
	else
		imwrite(program_path_ + "\\" + file_name + ".png", img_org);
}

void Angio_cvAlgorithm::convert_image(Mat input_image, float* output_image, int w, int h, bool bin, bool border)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			auto val = input_image.at<uchar>(y, x);
			output_image[y * w + x] = (bin && val > 0) ? 1.0f : static_cast<float>(val);
		}
	}
	//save_image(output_image, "Test\\convert_image");
}

void Angio_cvAlgorithm::convert_image(float* input_image, Mat& output_image, int w, int h, bool bin, bool border)
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
	if (border)
	{
		//테두리 자르기
		Mat borderImage;
		cut_image(temp, borderImage, 10);
		output_image = borderImage.clone();
	}
	else
	{
		temp.convertTo(output_image, CV_8UC1);
	}
	//save_image(output_image, "Test\\convert_image");
}


Mat Angio_cvAlgorithm::dilate_image(Mat inputImage, int N)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	dilate(inputImage, outputImage, Mat::ones(Size(N, N), CV_8UC1), Point(-1, -1));
	return outputImage;
}

Mat Angio_cvAlgorithm::erode_image(Mat inputImage, int N)
{
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);//w,h
	erode(inputImage, outputImage, Mat::ones(Size(N, N), CV_8UC1), Point(-1, -1));
	return outputImage;
}

void Angio_cvAlgorithm::cut_image(Mat input_image, Mat& output_image, int nRange)
{
	int rows = input_image.rows;
	int cols = input_image.cols;
	cv::Mat outputImage(input_image.rows, input_image.cols, CV_8UC1);//w,h
	memcpy(outputImage.data, input_image.data, sizeof(unsigned char) * input_image.rows * input_image.cols);


	// 상단 및 하단 테두리 부분을 0으로 설정
	for (int y = 0; y < nRange; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			output_image.at<uchar>(y, x) = 0; // 상단 테두리
			output_image.at<uchar>(rows - 1 - y, x) = 0; // 하단 테두리
		}
	}

	// 좌측 및 우측 테두리 부분을 0으로 설정
	for (int y = 0; y < rows; y++)
	{
		for (int x = 0; x < nRange; x++)
		{
			output_image.at<uchar>(y, x) = 0; // 좌측 테두리
			output_image.at<uchar>(y, cols - 1 - x) = 0; // 우측 테두리
		}
	}
}

void Angio_cvAlgorithm::detect_filter_edges(Mat input_image, Mat& output_image, int w, int h, int N)
{
	input_image.copyTo(output_image);
	save_image(output_image, "Test\\BordCut1");
	vector<Vec4i> linesP;
	HoughLinesP(output_image, linesP, 1, (CV_PI / 180), 150, 50, 5);
	// 테두리 영역 내에 있는 선만 남기기
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		float2 pt1 = make_float2(l[0], l[1]);
		float2 pt2 = make_float2(l[2], l[3]);

		if (pt1.x < N && pt2.x < N) // #왼
			line(output_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.y < N && pt2.y < N) // #위
			line(output_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.x > w - N && pt2.x > w - N) // #오
			line(output_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		if (pt1.y > h - N && pt2.y > h - N) // # 아래
			line(output_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
	}
	save_image(output_image, "Test\\BordCut3");
}

void Angio_cvAlgorithm::filter_objects(Mat input_image, Mat& output_image, int nArea, bool skel)
{
	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(input_image, labels, stats, centroids);
	if (cnt <= 2)
		input_image.convertTo(output_image, CV_8UC1);
	else
	{
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
			if (nArea > sum * 0.1)
				nArea = sum * 0.1;
		}
		Mat surfSup = stats.col(4) > nArea;
		Mat img(input_image.size(), CV_8UC1, Scalar(0));
		for (int i = 1; i < cnt; i++)
		{
			if (surfSup.at<uchar>(i, 0))
				img = img | (labels == i);
		}
		img.convertTo(output_image, CV_8UC1);
	}
	save_image(output_image, "Test\\Labeling2");
}

Mat Angio_cvAlgorithm::image_cut(Mat input_image, int range)
{
	cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC1);//w,h
	int w = input_image.rows;
	int h = input_image.cols;
	input_image.convertTo(output_image, CV_8UC1);

	// 상, 하, 좌, 우 경계 부분을 0으로 설정
	auto set_border_zero = [&](int x_start, int x_end, int y_start, int y_end)
	{
		for (int y = y_start; y < y_end; ++y) {
			for (int x = x_start; x < x_end; ++x) {
				output_image.at<uchar>(y, x) = 0;
			}
		}
	};

	set_border_zero(0, w, 0, range);           // 상단
	set_border_zero(0, w, h - range, h);       // 하단
	set_border_zero(0, range, 0, h);           // 좌측
	set_border_zero(w - range, w, 0, h);       // 우측
	return output_image;
}

void Angio_cvAlgorithm::create_filter_image(float* input_imge, float* output_imge, int w, int h)
{
	Mat cvimage(Size(w, h), CV_32FC1);
	Mat BordCutImage(Size(w, h), CV_32FC1);
	Mat LeblelingImage(Size(w, h), CV_32FC1);
	convert_image(input_imge, cvimage, w, h);
	detect_filter_edges(cvimage, BordCutImage, w, h, 15);
	filter_objects(BordCutImage, LeblelingImage, 500);
	convert_image(LeblelingImage, output_imge, w, h);
}


void Angio_cvAlgorithm::thinningIteration(Mat& im, int iter)
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

Mat Angio_cvAlgorithm::thinning(Mat& im)
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


Mat Angio_cvAlgorithm::labeling_max(Mat pImage)
{
	Mat img(pImage.size(), CV_8UC1, Scalar(0));
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
	return img;
}



Mat Angio_cvAlgorithm::labeling(Mat pImage, int nArea, bool skel = false)
{
	Mat img(pImage.size(), CV_8UC1, Scalar(0));

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
		if (nArea > sum * 0.1)
			nArea = sum * 0.1;
	}
	Mat surfSup = stats.col(4) > nArea;

	for (int i = 1; i < cnt; i++)
	{
		if (surfSup.at<uchar>(i, 0))
			img = img | (labels == i);
	}
	return img;
}


void Angio_cvAlgorithm::create_skeleton_image(float* input_imge, float* output_imge, int w, int h, bool bBorder)
{
	Mat img(Size(w, h), CV_32FC1);
	convert_image(input_imge, img, w, h);
	save_image(input_imge, "\\test\\skeleton_x");

	auto cut_image = image_cut(img, 10);
	auto skeleton = thinning(cut_image);
	if (bBorder)
	{
		save_image(skeleton, "\\test\\skeleton_B");
		vector<Vec4i> linesP;
		auto labelImage = labeling_max(cut_image);
		Mat img_houghP = dilate_image(labelImage, 3);
		save_image(img_houghP, "\\test\\skeleton_D");

		HoughLinesP(img_houghP, linesP, 1, (CV_PI / 180), 100, 150, 5);
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i l = linesP[i];
			float2 pt1 = make_float2(l[0], l[1]);
			float2 pt2 = make_float2(l[2], l[3]);
			if (pt1.x < w / 2 && pt2.x < w / 2) // #왼
				line(labelImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 0), 2, 8);
		}
		labelImage = labeling(labelImage, 300, true);
		convert_image(labelImage, output_imge, w, h);
	}
	else
	{
		auto labelImage = labeling(skeleton, 300, true);
		convert_image(labelImage, output_imge, w, h);
	}
	save_image(output_imge, "\\test\\skeleton");
}


Mat Angio_cvAlgorithm::draw_line(float* input_image, const std::vector<float2>& points, const int3& color, const int& w, const int& h, std::string fileName)
{
	Mat input_cvimage(Size(w, h), CV_32FC1);
	convert_image(input_image, input_cvimage, w, h, false);

	cv::Mat outputImage(input_cvimage.rows, input_cvimage.cols, CV_32FC3);//w,h
	input_cvimage.copyTo(outputImage);
	for (int i = 0; i < points.size() - 1; i++)
		line(outputImage, Point(points[i].x, points[i].y), Point(points[i + 1].x, points[i + 1].y), Scalar(color.x, color.y, color.z), 1, 8);
	std::string  save_name = "\\Test\\" + fileName;
	save_image(outputImage, save_name);
	return outputImage;
}

Mat Angio_cvAlgorithm::draw_line(Mat input_cvimage, const std::vector<float2>& points, const int3& color, const int& w, const int& h, std::string fileName)
{
	if (points.size() == 0)
		return input_cvimage;

	cv::Mat outputImage(input_cvimage.rows, input_cvimage.cols, CV_32FC3);//w,h
	input_cvimage.copyTo(outputImage);
	for (int i = 0; i < points.size() - 1; i++)
		line(outputImage, Point(points[i].x, points[i].y), Point(points[i + 1].x, points[i + 1].y), Scalar(color.x, color.y, color.z), 1, 8);
	std::string  save_name = "\\Test\\" + fileName;
	save_image(outputImage, save_name);
	return outputImage;
}



vector<result_info::endPoint_result_Instance> Angio_cvAlgorithm::EndPoints(vector<vector<float2>> lebeling_points, vector<float2> center_point, int w, int h)
{
	auto calculate_distance = [&](float2 a, float2 b)
	{
		return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
	};

	auto GetDistance2End = [&](vector<float2> vecLinepos, float2 end)
	{
		float _min = FLT_MAX;
		int _minId = 0, current = 0;
		float2 pos = make_float2(0, 0);
		for (auto i = 0; i < vecLinepos.size(); i++)
		{
			auto f = vecLinepos[i];
			auto _val = (f.x - end.x) * (f.x - end.x) + (f.y - end.y) * (f.y - end.y);
			if (_val < _min)
			{
				_min = _val;
				_minId = current;
			}
			current++;
		}
		return _minId;
	};

	int nEndId = 30;
	float2 ptEndPos = (center_point[nEndId]);
	bool bfix = false;
	int nSavefileNumber = 0;
	auto p1 = make_float2(center_point[1].x + (center_point[0].x - center_point[1].x) * 5, center_point[1].y + (center_point[0].y - center_point[1].y) * 5);
	vector<result_info::endPoint_result_Instance> output_endpoint_points;
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
			for (int n = 0; n < lebeling_points[i].size(); n++)
			{
				auto pos = make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y);
				arrPts.push_back(make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y));
				arrdis.push_back(calculate_distance(make_float2(lebel_image_Data[n].x, lebel_image_Data[n].y), p1));
			}
			if (arrdis.size() != 0)
			{
				int minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				int id = GetDistance2End(center_point, arrPts[minid]);

				auto ptOffset = make_float2((arrPts[minid].x - center_point[0].x), (arrPts[minid].y - center_point[0].y));
				vector<float2> newArrPts;
				arrdis.clear();

				for (int n = 0; n < arrPts.size(); n++)
				{
					auto pos = make_float2(arrPts[n].x - ptOffset.x, arrPts[n].y - ptOffset.y);
					if (pos.x > 0 && pos.x < w && pos.y > 0 && pos.y < h)
					{
						arrdis.push_back(calculate_distance(make_float2(pos.x, pos.y), ptEndPos));
						newArrPts.push_back(make_float2(pos.x, pos.y));
					}
				}
				minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				id = GetDistance2End(center_point, newArrPts[minid]);

				result_info::endPoint_result_Instance end_points(i + 1, id, newArrPts[minid]);
				output_endpoint_points.push_back(end_points);

				if (nEndId + id >= center_point.size())
					bfix = true;
				if (bfix)
					ptEndPos = center_point[center_point.size() - 1];
				else
					ptEndPos = center_point[id + nEndId];

				if (id == center_point.size() - 1)
					break;
			}
		}
	}
	return output_endpoint_points;
}


vector<result_info::endPoint_result_Instance> Angio_cvAlgorithm::image_EndPoints(vector<float2> center_point, std::string path)
{
	auto strFileName = section(path, '/', 0, -2);
	auto strExtension = section(path, '.', -1);
	bool ret = strExtension.find('.') != std::string::npos;
	auto open_filename = section(path,'/', -1);
	open_filename = section(open_filename, '.', 0, -2);
	open_filename = section(open_filename,'-', 0, -2);
	
	auto getBaseFilename = [&](const std::string & filepath)
	{
		std::filesystem::path path(filepath);
		auto str = section(path.stem().string(), '.', -1);
		return str; // Get the filename without extension
	};
	
	// Function to extract the file extension
	auto getFileExtension = [&](const std::string& filepath)
	{
		std::filesystem::path path(filepath);
		auto str = section(path.extension().string(), '.', -1);
		return str; // Get the file extension
	};
	
	std::vector<std::string> vecstr;
	for (const auto& entry : std::filesystem::directory_iterator(strFileName))
	{
		if (entry.is_regular_file()) {
			auto src = entry.path().string();
			auto type = getFileExtension(src);
			auto filename = getBaseFilename(src);
			filename = section(filename, '-', 0, -2);

			// Further process the filename if needed (removing certain parts)
			//auto sections = split(filename, '-'); // Split by '-'
			//if (!sections.empty()) {
			//	filename = sections[0]; // Get the first part (adjust as needed)
			//}
	
			// Check if the file extension and filename match
			if (strExtension == type && open_filename == filename) {
				vecstr.push_back(src); // Store the absolute path
			}
		}
	}
	
	auto strSort = [](const std::string& a, const std::string& b) -> bool {
		return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
			[](char ac, char bc) { return std::tolower(ac) < std::tolower(bc); });
	};
	
	
	auto calculate_distance = [&](float2 a, float2 b)
	{
		return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
	};
	
	auto GetDistance2End = [&](float2 end)
	{
		float _min = FLT_MAX;
		int _minId = 0, current = 0;
		float2 pos = make_float2(0, 0);
		for (auto i = 0; i < 100; i++)
		{
			auto f = center_point[i];
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
	
	std::sort(vecstr.begin(), vecstr.end(), strSort);
	int nEndId = 30;
	float2 ptEndPos = (center_point[30]);
	//float2 ptEndPos = (pCenterData[40]);
	
	bool bfix = false;
	int nSavefileNumber = 0;
	auto p1 = make_float2(center_point[1].x + (center_point[0].x - center_point[1].x) * 5, center_point[1].y + (center_point[0].y - center_point[1].y) * 5);
	std::vector<result_info::endPoint_result_Instance>  output_endpoint_points;
	
	for (int i = 0; i < vecstr.size(); i++)
	{
		if (nSavefileNumber == 20)
			break;
		auto testFileName = section(strFileName, '/', 0, -1) + "\\";
		auto FileName = section(vecstr[i], '.', 0, -2);
		auto FileNameWithoutDir = section(FileName, '\\', -1);
		auto FileNo = section(FileName, '-', -1, -1);
		Mat img;
		Mat img_color = imread(vecstr[i], IMREAD_COLOR);
		inRange(img_color, Scalar(0, 128, 0), Scalar(100, 255, 100), img);
		auto sum = countNonZero(img);
		int h = img.cols;
		int w = img.rows;
		if (sum > 5)
		{
			vector<float2> arrPts;
			vector<float> arrdis;
			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < w; x++)
				{
					auto val = img.at<uchar>(y, x);
					if (int(val) != 0)
					{
						arrPts.push_back(make_float2(x, y));
						arrdis.push_back(calculate_distance(make_float2(x, y), p1));
					}
				}
			}
			if (arrdis.size() != 0)
			{
				int minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				int id = GetDistance2End(arrPts[minid]);
	
				float2 ptOffset = make_float2(0, 0);
				ptOffset = make_float2((arrPts[minid].x - center_point[0].x), (arrPts[minid].y - center_point[0].y));
				testFileName = testFileName + FileNameWithoutDir + ".bmp";
				Mat img2 = imread(testFileName, IMREAD_COLOR);
				vector<float2> newArrPts;
				arrdis.clear();
				//line(img_color, Point(pCenterData[0].x, pCenterData[0].y), Point(pCenterData[n + 1].x, pCenterData[n + 1].y), Scalar(0, 0, 255), 2, 8);
	
				line(img_color, Point(p1.x, p1.y), Point(center_point[0].x, center_point[0].y), Scalar(0, 255, 255), 2, 8);
	
				for (int n = 0; n < 99; n++)
				{
					line(img2, Point(center_point[n].x, center_point[n].y), Point(center_point[n + 1].x, center_point[n + 1].y), Scalar(0, 0, 255), 2, 8);
					line(img_color, Point(center_point[n].x, center_point[n].y), Point(center_point[n + 1].x, center_point[n + 1].y), Scalar(0, 0, 255), 2, 8);
				}
				img_color.at<cv::Vec3b>(arrPts[minid].y, arrPts[minid].x)[0] = 0;
				img_color.at<cv::Vec3b>(arrPts[minid].y, arrPts[minid].x)[1] = 255;
				img_color.at<cv::Vec3b>(arrPts[minid].y, arrPts[minid].x)[2] = 255;
				auto savefile1 = program_path_ + "\\data\\test\\" + FileNo + ".bmp";
				imwrite(savefile1, img_color);
	
				for (int n = 0; n < arrPts.size(); n++)
				{
					auto pos = make_float2(arrPts[n].x - ptOffset.x, arrPts[n].y - ptOffset.y);
					if (pos.x > 0 && pos.x < w && pos.y > 0 && pos.y < h)
					{
						img2.at<cv::Vec3b>(pos.y, pos.x)[0] = 255;
						img2.at<cv::Vec3b>(pos.y, pos.x)[1] = 255;
						img2.at<cv::Vec3b>(pos.y, pos.x)[2] = 255;
						arrdis.push_back(calculate_distance(make_float2(pos.x, pos.y), ptEndPos));
						newArrPts.push_back(make_float2(pos.x, pos.y));
					}
				}
				minid = min_element(arrdis.begin(), arrdis.end()) - arrdis.begin();
				id = GetDistance2End(newArrPts[minid]);
	
				result_info::endPoint_result_Instance end_points(i + 1, id, newArrPts[minid]);
				output_endpoint_points.push_back(end_points);
	
				auto savefile = program_path_ + "\\data\\test\\" + FileNo + "_Move.bmp";
				imwrite(savefile, img2);
	
				if (nEndId + id > 99)
					bfix = true;
				if (bfix)
					ptEndPos = center_point[99];
				else
					ptEndPos = center_point[id + nEndId];
				nSavefileNumber++;
			}
		}
	}
	return output_endpoint_points;
}

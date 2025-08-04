#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <filesystem>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>
#include <Result_Info.h>

using namespace cv;
using namespace std;

#define SAVE_TEST_IMAGE 0

class Angio_cvAlgorithm
{
public:
    struct PointDist {
        Point point;
        double distance;

        bool operator>(const PointDist& other) const {
            return distance > other.distance;
        }
    };

    Angio_cvAlgorithm();
    ~Angio_cvAlgorithm();


    void draw_line(float* arr, const std::vector<float2>& line_c1, const std::vector<float2>& line_c2, const std::vector<float2>& points, const int& w, const int& h, std::string fileName);

    /**
    * @brief 테스트 이미지 저장
    * @details mat opencv 이미지배열 저장
    * @param mat 형 이미지배열
    * @param 저장폴더경로
    */
    void save_image(Mat input_image, std::string file_name, int w = 512, int h = 512);
    /**
     * @brief 테스트 이미지 저장
     * @details float* 이미지배열 저장
     * @param mat 형 이미지배열
     * @param 저장폴더경로
     */
    void save_image(float* input_image, std::string file_name, int w = 512, int h = 512, bool bin = true);


    /**
     * @brief 이미지 변환 함수
     * @details Mat에서 float 으로
     * @param mat 형 이미지배열
     * @param w 가로
     * @param w 세러
     * @param true = 이진영상으로 저장할껀지, false = 원본영상으로 저장할껀지
     * @param true = 테두리 짜를껀지, false = 원본영상으로 냅둘껀지
     */
    void convert_image(Mat input_image, float* output_image, int w, int h, bool bin = true, bool border = false);

    /**
     * @brief 이미지 변환 함수
     * @details  float에서 Mat으로
     * @param mat 형 이미지배열
     * @param w 가로
     * @param w 세러
     * @param true = 이진영상으로 저장할껀지, false = 원본영상으로 저장할껀지
     * @param true = 테두리 짜를껀지, false = 원본영상으로 냅둘껀지
     */
    void convert_image(float* input_image, Mat& output_image, int w, int h, bool bin = true, bool border = false);

    Mat dilate_image(Mat inputImage, int N);

    Mat erode_image(Mat inputImage, int N);

    /**
    * @brief 이미지 자르기 함수
    * @details  float에서 Mat으로
    * @param mat 형 입력한 이미지배열
    * @param mat 형 출력할 이미지배열
    * @param 사이즈
    */
    void cut_image(Mat input_image, Mat& output_image, int nRange = 10);

    /**
    * @brief 선 탐지 및 필터링 함수
    * @details  쭈욱 뻗은 혈관 선 제거
    * @param w 가로
    * @param h 세로
    * @param 간격사이즈
    */
    void detect_filter_edges(Mat input_image, Mat& output_image, int w, int h, int N);

    /**
    * @brief 라벨링 함수
    * @details  설정한 면적미만은 영상에서 제거
    * @param mat 형 입력한 이미지배열
    * @param mat 형 출력할 이미지배열
    * @param 면적
    * @param 스켈레톤...할껀지
    */
    void filter_objects(Mat input_image, Mat& output_image, int nArea, bool skel = false);

    Mat image_cut(Mat input_image, int range);

    void create_filter_image(float* input_imge, float* output_imge, int w, int h);

    void thinningIteration(Mat& im, int iter);

    Mat thinning(Mat& im);

    Mat labeling_max(Mat pImage);

    Mat labeling(Mat pImage, int nArea, bool skel);

    void create_skeleton_image(float* input_imge, float* output_imge, int w, int h, bool bBorder);

    Mat draw_line(float* input_image, const std::vector<float2>& points, const int3& color, const int& w, const int& h, std::string fileName);

    Mat draw_line(Mat input_cvimage, const std::vector<float2>& points, const int3& color, const int& w, const int& h, std::string fileName);

    vector<result_info::endPoint_result_Instance> EndPoints(vector<vector<float2>> lebeling_points, vector<float2> center_point, int w, int h);

    vector<result_info::endPoint_result_Instance> image_EndPoints(vector<float2> center_point, std::string path);

protected:
private:
	std::string program_path_;
};


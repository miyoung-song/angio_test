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
    * @brief �׽�Ʈ �̹��� ����
    * @details mat opencv �̹����迭 ����
    * @param mat �� �̹����迭
    * @param �����������
    */
    void save_image(Mat input_image, std::string file_name, int w = 512, int h = 512);
    /**
     * @brief �׽�Ʈ �̹��� ����
     * @details float* �̹����迭 ����
     * @param mat �� �̹����迭
     * @param �����������
     */
    void save_image(float* input_image, std::string file_name, int w = 512, int h = 512, bool bin = true);


    /**
     * @brief �̹��� ��ȯ �Լ�
     * @details Mat���� float ����
     * @param mat �� �̹����迭
     * @param w ����
     * @param w ����
     * @param true = ������������ �����Ҳ���, false = ������������ �����Ҳ���
     * @param true = �׵θ� ¥������, false = ������������ ���Ѳ���
     */
    void convert_image(Mat input_image, float* output_image, int w, int h, bool bin = true, bool border = false);

    /**
     * @brief �̹��� ��ȯ �Լ�
     * @details  float���� Mat����
     * @param mat �� �̹����迭
     * @param w ����
     * @param w ����
     * @param true = ������������ �����Ҳ���, false = ������������ �����Ҳ���
     * @param true = �׵θ� ¥������, false = ������������ ���Ѳ���
     */
    void convert_image(float* input_image, Mat& output_image, int w, int h, bool bin = true, bool border = false);

    Mat dilate_image(Mat inputImage, int N);

    Mat erode_image(Mat inputImage, int N);

    /**
    * @brief �̹��� �ڸ��� �Լ�
    * @details  float���� Mat����
    * @param mat �� �Է��� �̹����迭
    * @param mat �� ����� �̹����迭
    * @param ������
    */
    void cut_image(Mat input_image, Mat& output_image, int nRange = 10);

    /**
    * @brief �� Ž�� �� ���͸� �Լ�
    * @details  �޿� ���� ���� �� ����
    * @param w ����
    * @param h ����
    * @param ���ݻ�����
    */
    void detect_filter_edges(Mat input_image, Mat& output_image, int w, int h, int N);

    /**
    * @brief �󺧸� �Լ�
    * @details  ������ �����̸��� ���󿡼� ����
    * @param mat �� �Է��� �̹����迭
    * @param mat �� ����� �̹����迭
    * @param ����
    * @param ���̷���...�Ҳ���
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


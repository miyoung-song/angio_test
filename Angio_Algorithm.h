#pragma once

#include"dcmHelper.h"
#include "nifti2_io.h"

#include "Model_Info.h"
#include "Result_Info.h"

#include "cnpy.h"

#include <stack>
#include "Angio_cvAlgorithm.h"

#define ANGIO_MODEL model_info::get_instance()
//#define angio_result_ result_info::get_instance()


class CubicSpline {
public:
    // 주어진 2차원 좌표들로 스플라인 설정
    void set_points(const std::vector<float2>& points) {
        size_t n = points.size();
        h.resize(n - 1);
        a = points;  // a는 원래 값 자체입니다 (좌표들)

        // h 계산
        for (size_t i = 0; i < n - 1; ++i) {
            h[i] = distance(points[i], points[i + 1]);
        }

        // alpha 계산
        std::vector<float2> alpha(n - 1);
        for (size_t i = 1; i < n - 1; ++i) {
            alpha[i].x = (3.0 / h[i]) * (a[i + 1].x - a[i].x) - (3.0 / h[i - 1]) * (a[i].x - a[i - 1].x);
            alpha[i].y = (3.0 / h[i]) * (a[i + 1].y - a[i].y) - (3.0 / h[i - 1]) * (a[i].y - a[i - 1].y);
        }

        // 삼각 방정식 계수들
        std::vector<double> l(n), mu(n);
        std::vector<float2> z(n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = make_float2(0.0f, 0.0f);

        for (size_t i = 1; i < n - 1; ++i) {
            l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i].x = (alpha[i].x - h[i - 1] * z[i - 1].x) / l[i];
            z[i].y = (alpha[i].y - h[i - 1] * z[i - 1].y) / l[i];
        }

        l[n - 1] = 1.0;
        z[n - 1] = make_float2(0.0f, 0.0f);
        c.resize(n);
        b.resize(n);
        d.resize(n);
        c[n - 1] = make_float2(0.0f, 0.0f);

        for (int j = n - 2; j >= 0; --j) {
            c[j].x = z[j].x - mu[j] * c[j + 1].x;
            c[j].y = z[j].y - mu[j] * c[j + 1].y;
            b[j].x = (a[j + 1].x - a[j].x) / h[j] - h[j] * (c[j + 1].x + 2.0 * c[j].x) / 3.0;
            b[j].y = (a[j + 1].y - a[j].y) / h[j] - h[j] * (c[j + 1].y + 2.0 * c[j].y) / 3.0;
            d[j].x = (c[j + 1].x - c[j].x) / (3.0 * h[j]);
            d[j].y = (c[j + 1].y - c[j].y) / (3.0 * h[j]);
        }
    }

    // 주어진 t 값에서 보간된 2D 좌표 계산
    float2 get_point(float t) const {
        size_t i = std::min(static_cast<size_t>(t), a.size() - 2); // 인덱스 결정
        float dt = t - i;
        return make_float2(
            a[i].x + b[i].x * dt + c[i].x * dt * dt + d[i].x * dt * dt * dt,
            a[i].y + b[i].y * dt + c[i].y * dt * dt + d[i].y * dt * dt * dt
        );
    }

private:
    std::vector<float2> a, b, c, d; // 계수들 (2D)
    std::vector<double> h; // 각 구간의 길이 (1D)

    // 두 점 사이의 거리 계산 함수
    double distance(const float2& p1, const float2& p2) const {
        return std::sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
    }
};

template <typename T>
class history_manager
{
private:
    std::stack<T> undo_stack;  // Undo를 위한 스택
    std::stack<T> redo_stack;  // Redo를 위한 스택

public:
    void save_state(const T& state) {
        undo_stack.push(state);
        while (!redo_stack.empty()) {
            redo_stack.pop();  // 새로운 상태가 저장되면 Redo 스택을 초기화
        }
    }
    void clear()
    {
        while (!undo_stack.empty()) {
            undo_stack.pop();
        }
        while (!redo_stack.empty()) {
            redo_stack.pop();
        }
    }
    bool undo(T& current_state) {
        if (!undo_stack.empty() && undo_stack.size() != 1)
        {
            redo_stack.push(current_state); // 현재 상태를 Redo 스택에 저장
            undo_stack.pop();
            current_state = undo_stack.top(); // 이전 상태로 복원
            return true;
        }
        return false;
    }

    bool redo(T& current_state) {
        if (!redo_stack.empty()) {
            undo_stack.push(current_state); // 현재 상태를 Undo 스택에 저장
            current_state = redo_stack.top(); // 다음 상태로 복원
            redo_stack.pop();
            return true;
        }
        return false;
    }
};

class Angio_Algorithm
{
public:
    enum class direction : int
    {
        x_direction = 0,
        y_direction
    };

    struct distance_and_id
    {
        float distance;
        float2 pos;
        int id;
        distance_and_id() {};
        ~distance_and_id() {};
        distance_and_id(const float& dist, const int& ID, const float2& POS) : distance(dist), id(ID), pos(POS) {};
    };
    struct dcm_info
    {
        bool dcm_file = false;
        float primary_angle = 0;
        float secondary_angle = 0;
        float Patient = 0;
        float detector = 0;
    };

    Angio_Algorithm() : width_(0), height_(0) 
    {
        program_path_ = std::filesystem::current_path().string();
        size_t pos = program_path_.find_last_of('\\');
        if (pos != std::string::npos)
            program_path_ = program_path_.substr(0, pos);
    }
    Angio_Algorithm(int w, int h) : width_(w), height_(h) {}
    ~Angio_Algorithm() { delete_buffer(); }




    float get_distance_points(const float2& a, const float2& b);

    float2 get_rotation_point(const float& radian, const float2& input_point_a, const float2& input_point_b);
    /**
    * 평행하거나 교차하지 않는 경우 false를 반환합니다.
    * @brief 두 선분의 교차점을 계산합니다.
    * 이 함수는 두 선분 (AP1, AP2)과 (BP1, BP2)의 교차 여부를 판단하고,
    * 교차하는 경우 교차점을 IP에 저장합니다.
    * @param AP1 첫 번째 선분의 시작 점
    * @param AP2 첫 번째 선분의 끝 점
    * @param BP1 두 번째 선분의 시작 점
    * @param BP2 두 번째 선분의 끝 점
    * @param IP 교차점 저장 변수 (교차하는 경우 유효)
    */
    bool get_intersect_point(float2 AP1, float2 AP2, float2 BP1, float2 BP2, float2& IP);

    void set_width(int val) { width_ = val; };
    void set_height(int val) { height_ = val; };

    int get_width() { return width_; };
    int get_height() { return height_; };


    void create_buffer(int w, int h);
    void delete_buffer();
    float* get_float_buffer(model_info::image_float_type key);

    bool apply_filters(float* buffer, float* buffer1, float* buffer5, float* filter_image);
    void processImage(float* buffer, float* buffer1, float* buffer5);

    /*!
    * @brief 주어진 선 포인트를 기반으로 수직 거리 균등화를 설정하는 함수
    *
    * @param input_line_points 입력 선의 점들
    * @param id_s 시작 인덱스
    * @param id_e 종료 인덱스
    * @param nminIndex 최소 거리 인덱스 (출력 매개변수)
    * @param equal 균등화 여부
    */
    void set_verticality_Distance(vector<float2> input_line_points, int id_s, int id_e, int& nminIndex, bool equal);
    /*!
    * @brief 주어진 선의 포인트들을 기준으로 균등 간격의 새로운 선을 생성하는 함수
    * @param input_line_points 입력 선의 점들
    * @param percent 균등 간격 비율
    * @param nIndex 브랜치 포인트의 인덱스
    * @return 새로운 균등 간격 선의 점들
    */
    vector<float2> get_equilateral_line(vector<float2> input_line_points, int percent, int nIndex);
    /*!
    * @brief 포물선 라인을 추가하는 함수
    *
    * @param Line 결과 선을 저장할 벡터 (float2)
    * @param ParabolaLine 생성된 포물선 데이터를 저장할 벡터 (float2)
    * @param ptStart 시작 점
    * @param ptEnd 끝 점
    * @param LineForward 전방 선 데이터
    * @param LineBackward 후방 선 데이터
    * @param lfDisForward 전방 거리
    * @param lfDisBackward 후방 거리
    */
    float AppendParabolaLine(vector<float2>& Line, vector<float2>& ParabolaLine, float2 ptStart, float2 ptEnd, vector<float2> LineForward, vector<float2> LineBackward, float lfDisForward, float lfDisBackward);
    /*!
     * @brief 포물선 데이터를 생성하는 함수
     *
     * @param input_points 입력 좌표 벡터 (float2)
     * @param output_parabola_line 생성된 포물선 좌표를 저장할 벡터 (float2)
     * @param id_Start 포물선 생성 시작 인덱스
     * @param id_End 포물선 생성 종료 인덱스
     */
    void make_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end);
    /*!
     * @brief 주어진 입력 좌표에 대해 포물선 데이터를 설정하는 함수
     * @param input_points 입력 좌표 벡터 (float2)
     * @param input_line 기준선으로 사용할 입력 좌표 벡터 (float2)
     * @param pick_point 기준점 (float2)
     * @param output_parabola_line 계산된 포물선 좌표를 저장할 벡터 (float2)
     * @param id_Start 포물선 계산을 시작할 좌표 인덱스
     * @param id_End 포물선 계산을 종료할 좌표 인덱스
     * @param inside_range 범위 내에서 결과를 판단할지 여부
     * @param min 출력 좌표의 x, y 값에 대한 최소 허용 범위
     * @param max 출력 좌표의 x, y 값에 대한 최대 허용 범위
     */
    void set_parabola_data(const vector<float2>& input_points, vector<float2>& input_line, const float2 pick_point, vector<float2>& output_parabola_line, int id_Start, int id_End, bool inside_range, int min, int max);
    /*!
    * @brief 입력된 좌표에 대해 포물선을 추가하는 함수
    * @param input_points 입력 좌표 벡터 (float2)
    * @param output_parabola_line 계산된 포물선 좌표를 저장할 벡터 (float2)
    * @param id_Start 포물선 계산을 시작할 좌표 인덱스
    * @param id_End 포물선 계산을 종료할 좌표 인덱스
    * @param min 출력 좌표의 x, y 값에 대한 최소 허용 범위
    * @param max 출력 좌표의 x, y 값에 대한 최대 허용 범위
    */
    void add_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end, int min = 0, int max = 0);
    /*!
    * @brief 주어진 좌표에 대해 포물선을 계산하는 함수
    * @param input_points 입력 좌표 벡터 (float2)
    * @param out_parapoint 계산된 포물선 좌표가 저장될 벡터 (float2)
    * @param point_cnt 입력 좌표의 총 개수
    * @param dir 포물선을 계산할 방향 (x축 또는 y축)
    * @param id_start 계산 시작 인덱스
    * @param id_end 계산 종료 인덱스
    * @param N 행렬의 크기 (포물선 계산에 사용할 점의 개수)
    * @param inside_range 비교 시 기존 값보다 작은 값으로 설정할지 여부
    * @param bCompare 계산된 포물선 좌표와 기존 좌표를 비교할지 여부
    */
    void get_parabola(vector<float2>& input_points, vector<float2>& out_parapoint, int point_cnt, direction dir, int id_start, int id_end, int N = 3, bool inside_range = true, bool bCompare = false);
    void inv_matrix(int n, double** A, double** B); //역행렬
    distance_and_id get_distance_end(const vector<float2>& sources, const float2& end);
    /*!
    * @brief 등간격의 좌표를 계산하는 함수
    * @param input_data 입력 좌표 데이터 (float2 배열)
    * @param output_data 계산된 등간격 좌표가 저장될 벡터 (vector<float2>)
    * @param n 전체 길이를 n으로 나눈 값 (등간격으로 생성할 점의 개수)
    * @param id_start 시작 범위 (input_data의 시작 인덱스)
    * @param id_end 끝 범위 (input_data의 끝 인덱스)
    */
    void equal_spacing(const vector<float2> input_data, vector<float2>& output_data, int n, int id_start, int id_end);

    void set_end_points(model_info::find_endpoint_type type, std::string fileName);

    vector<float> get_minimum_radius();

    vector<float2> modify_line(const bool& release_button, const int& select_line_id, const float2& pick_point, const float2& mouse_point, int move_range);
    bool line_undo();

    //파일 입출력
    void save_result_out(int view_id, std::string strExtension);
    void save_data_out(int view_id);
    void set_center_points_3D(float3 center_point);
    void save_dcm_file(int view_id, dcmHelper* dcm_helper);
    void read_data_in(int view_id, std::string strExtension);

    void FindCenterLine(float* mapp, bool* testFrozen, float2 ptStart, float2 ptEnd, vector<float2>& Start_Line, vector<float2>& End_Line, float& lfDisForward, float& lfDisBackward, int nType);

    //외부프로그램으로 line 구할때
    void segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type, int index);
    void run_AI_segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type);


    //프로그램 내 알고리즘 사용할때
    void segmentation(model_info::segmentation_manual_type type); 
    void manual_segmentation_edgeline(vector<float2>& center_line);
    bool manual_segmentation(bool find_line);

    //외부프로그램돌린 후 출력데이터 읽어오기
    void segmentation_output(model_info::segmentation_exe_type run_type, int index);
    bool get_segmentation_file_niigz(std::string path, result_info::segmentation_instance& segmentation_instance_, int index);

    void set_file_niigz(int view_id, dcmHelper* dcm_helper); //외부프로그램돌리기 전 다이콤 입력데이터
    
    //세그멘테이션 결과 관련
    result_info::segmentation_line2D_instance get_segmentation_line2D_instance() { return angio_result_.get_segmentation_line2D_instance(); };
    void set_segmentation_line2D_instance(const result_info::segmentation_line2D_instance& instance) { angio_result_.set_segmentation_line2D_instance(instance); };

    vector<float2> get_lines2D(result_info::line_position line_type, bool equal = false) { return angio_result_.get_segmentation_line2D_instance().get_line(line_type, equal); };
    void set_lines2D(result_info::line_position line_type, vector<float2> line, bool equal = false) { auto& instance = angio_result_.get_segmentation_line2D_instance();  instance.set_line(line_type, line, equal); };

    result_info::segmentation_instance get_segmentation_Instance() { return angio_result_.get_segmentation_instance(); };
    void set_segmentation_instance(const result_info::segmentation_instance& instance) { angio_result_.set_segmentation_instance(instance); };

    result_info::endPoint3D_result_Instance get_endPoint3D_result_Instance() { return angio_result_.get_endPoint3D_result_Instance(); };
    void set_endPoint3D_result_Instance(const result_info::endPoint3D_result_Instance instance) { angio_result_.set_endPoint3D_result_Instance(instance); };

    result_info::points_instance get_points_instance() { return angio_result_.get_points_instance(); };
    void set_points_instance(const result_info::points_instance& instance) { return angio_result_.set_points_instance(instance); };

    void result_clear() { angio_result_.result_clear(); };

public:
protected:
private:
    int width_;
    int height_;

    dcm_info dcm_info_;
    result_info angio_result_;
    Angio_cvAlgorithm cvAlgorithm_;

    bool* Frozen_ = nullptr;
    map<model_info::image_float_type, vector<float>> image_buffers_;
    history_manager<std::vector<std::vector<float2>>> history_lines_;
    std::string program_path_;
    vector<vector<float2>> inspect_roi_;
};


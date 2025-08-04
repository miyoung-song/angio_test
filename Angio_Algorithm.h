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
    // �־��� 2���� ��ǥ��� ���ö��� ����
    void set_points(const std::vector<float2>& points) {
        size_t n = points.size();
        h.resize(n - 1);
        a = points;  // a�� ���� �� ��ü�Դϴ� (��ǥ��)

        // h ���
        for (size_t i = 0; i < n - 1; ++i) {
            h[i] = distance(points[i], points[i + 1]);
        }

        // alpha ���
        std::vector<float2> alpha(n - 1);
        for (size_t i = 1; i < n - 1; ++i) {
            alpha[i].x = (3.0 / h[i]) * (a[i + 1].x - a[i].x) - (3.0 / h[i - 1]) * (a[i].x - a[i - 1].x);
            alpha[i].y = (3.0 / h[i]) * (a[i + 1].y - a[i].y) - (3.0 / h[i - 1]) * (a[i].y - a[i - 1].y);
        }

        // �ﰢ ������ �����
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

    // �־��� t ������ ������ 2D ��ǥ ���
    float2 get_point(float t) const {
        size_t i = std::min(static_cast<size_t>(t), a.size() - 2); // �ε��� ����
        float dt = t - i;
        return make_float2(
            a[i].x + b[i].x * dt + c[i].x * dt * dt + d[i].x * dt * dt * dt,
            a[i].y + b[i].y * dt + c[i].y * dt * dt + d[i].y * dt * dt * dt
        );
    }

private:
    std::vector<float2> a, b, c, d; // ����� (2D)
    std::vector<double> h; // �� ������ ���� (1D)

    // �� �� ������ �Ÿ� ��� �Լ�
    double distance(const float2& p1, const float2& p2) const {
        return std::sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
    }
};

template <typename T>
class history_manager
{
private:
    std::stack<T> undo_stack;  // Undo�� ���� ����
    std::stack<T> redo_stack;  // Redo�� ���� ����

public:
    void save_state(const T& state) {
        undo_stack.push(state);
        while (!redo_stack.empty()) {
            redo_stack.pop();  // ���ο� ���°� ����Ǹ� Redo ������ �ʱ�ȭ
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
            redo_stack.push(current_state); // ���� ���¸� Redo ���ÿ� ����
            undo_stack.pop();
            current_state = undo_stack.top(); // ���� ���·� ����
            return true;
        }
        return false;
    }

    bool redo(T& current_state) {
        if (!redo_stack.empty()) {
            undo_stack.push(current_state); // ���� ���¸� Undo ���ÿ� ����
            current_state = redo_stack.top(); // ���� ���·� ����
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
    * �����ϰų� �������� �ʴ� ��� false�� ��ȯ�մϴ�.
    * @brief �� ������ �������� ����մϴ�.
    * �� �Լ��� �� ���� (AP1, AP2)�� (BP1, BP2)�� ���� ���θ� �Ǵ��ϰ�,
    * �����ϴ� ��� �������� IP�� �����մϴ�.
    * @param AP1 ù ��° ������ ���� ��
    * @param AP2 ù ��° ������ �� ��
    * @param BP1 �� ��° ������ ���� ��
    * @param BP2 �� ��° ������ �� ��
    * @param IP ������ ���� ���� (�����ϴ� ��� ��ȿ)
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
    * @brief �־��� �� ����Ʈ�� ������� ���� �Ÿ� �յ�ȭ�� �����ϴ� �Լ�
    *
    * @param input_line_points �Է� ���� ����
    * @param id_s ���� �ε���
    * @param id_e ���� �ε���
    * @param nminIndex �ּ� �Ÿ� �ε��� (��� �Ű�����)
    * @param equal �յ�ȭ ����
    */
    void set_verticality_Distance(vector<float2> input_line_points, int id_s, int id_e, int& nminIndex, bool equal);
    /*!
    * @brief �־��� ���� ����Ʈ���� �������� �յ� ������ ���ο� ���� �����ϴ� �Լ�
    * @param input_line_points �Է� ���� ����
    * @param percent �յ� ���� ����
    * @param nIndex �귣ġ ����Ʈ�� �ε���
    * @return ���ο� �յ� ���� ���� ����
    */
    vector<float2> get_equilateral_line(vector<float2> input_line_points, int percent, int nIndex);
    /*!
    * @brief ������ ������ �߰��ϴ� �Լ�
    *
    * @param Line ��� ���� ������ ���� (float2)
    * @param ParabolaLine ������ ������ �����͸� ������ ���� (float2)
    * @param ptStart ���� ��
    * @param ptEnd �� ��
    * @param LineForward ���� �� ������
    * @param LineBackward �Ĺ� �� ������
    * @param lfDisForward ���� �Ÿ�
    * @param lfDisBackward �Ĺ� �Ÿ�
    */
    float AppendParabolaLine(vector<float2>& Line, vector<float2>& ParabolaLine, float2 ptStart, float2 ptEnd, vector<float2> LineForward, vector<float2> LineBackward, float lfDisForward, float lfDisBackward);
    /*!
     * @brief ������ �����͸� �����ϴ� �Լ�
     *
     * @param input_points �Է� ��ǥ ���� (float2)
     * @param output_parabola_line ������ ������ ��ǥ�� ������ ���� (float2)
     * @param id_Start ������ ���� ���� �ε���
     * @param id_End ������ ���� ���� �ε���
     */
    void make_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end);
    /*!
     * @brief �־��� �Է� ��ǥ�� ���� ������ �����͸� �����ϴ� �Լ�
     * @param input_points �Է� ��ǥ ���� (float2)
     * @param input_line ���ؼ����� ����� �Է� ��ǥ ���� (float2)
     * @param pick_point ������ (float2)
     * @param output_parabola_line ���� ������ ��ǥ�� ������ ���� (float2)
     * @param id_Start ������ ����� ������ ��ǥ �ε���
     * @param id_End ������ ����� ������ ��ǥ �ε���
     * @param inside_range ���� ������ ����� �Ǵ����� ����
     * @param min ��� ��ǥ�� x, y ���� ���� �ּ� ��� ����
     * @param max ��� ��ǥ�� x, y ���� ���� �ִ� ��� ����
     */
    void set_parabola_data(const vector<float2>& input_points, vector<float2>& input_line, const float2 pick_point, vector<float2>& output_parabola_line, int id_Start, int id_End, bool inside_range, int min, int max);
    /*!
    * @brief �Էµ� ��ǥ�� ���� �������� �߰��ϴ� �Լ�
    * @param input_points �Է� ��ǥ ���� (float2)
    * @param output_parabola_line ���� ������ ��ǥ�� ������ ���� (float2)
    * @param id_Start ������ ����� ������ ��ǥ �ε���
    * @param id_End ������ ����� ������ ��ǥ �ε���
    * @param min ��� ��ǥ�� x, y ���� ���� �ּ� ��� ����
    * @param max ��� ��ǥ�� x, y ���� ���� �ִ� ��� ����
    */
    void add_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end, int min = 0, int max = 0);
    /*!
    * @brief �־��� ��ǥ�� ���� �������� ����ϴ� �Լ�
    * @param input_points �Է� ��ǥ ���� (float2)
    * @param out_parapoint ���� ������ ��ǥ�� ����� ���� (float2)
    * @param point_cnt �Է� ��ǥ�� �� ����
    * @param dir �������� ����� ���� (x�� �Ǵ� y��)
    * @param id_start ��� ���� �ε���
    * @param id_end ��� ���� �ε���
    * @param N ����� ũ�� (������ ��꿡 ����� ���� ����)
    * @param inside_range �� �� ���� ������ ���� ������ �������� ����
    * @param bCompare ���� ������ ��ǥ�� ���� ��ǥ�� ������ ����
    */
    void get_parabola(vector<float2>& input_points, vector<float2>& out_parapoint, int point_cnt, direction dir, int id_start, int id_end, int N = 3, bool inside_range = true, bool bCompare = false);
    void inv_matrix(int n, double** A, double** B); //�����
    distance_and_id get_distance_end(const vector<float2>& sources, const float2& end);
    /*!
    * @brief ����� ��ǥ�� ����ϴ� �Լ�
    * @param input_data �Է� ��ǥ ������ (float2 �迭)
    * @param output_data ���� ��� ��ǥ�� ����� ���� (vector<float2>)
    * @param n ��ü ���̸� n���� ���� �� (������� ������ ���� ����)
    * @param id_start ���� ���� (input_data�� ���� �ε���)
    * @param id_end �� ���� (input_data�� �� �ε���)
    */
    void equal_spacing(const vector<float2> input_data, vector<float2>& output_data, int n, int id_start, int id_end);

    void set_end_points(model_info::find_endpoint_type type, std::string fileName);

    vector<float> get_minimum_radius();

    vector<float2> modify_line(const bool& release_button, const int& select_line_id, const float2& pick_point, const float2& mouse_point, int move_range);
    bool line_undo();

    //���� �����
    void save_result_out(int view_id, std::string strExtension);
    void save_data_out(int view_id);
    void set_center_points_3D(float3 center_point);
    void save_dcm_file(int view_id, dcmHelper* dcm_helper);
    void read_data_in(int view_id, std::string strExtension);

    void FindCenterLine(float* mapp, bool* testFrozen, float2 ptStart, float2 ptEnd, vector<float2>& Start_Line, vector<float2>& End_Line, float& lfDisForward, float& lfDisBackward, int nType);

    //�ܺ����α׷����� line ���Ҷ�
    void segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type, int index);
    void run_AI_segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type);


    //���α׷� �� �˰��� ����Ҷ�
    void segmentation(model_info::segmentation_manual_type type); 
    void manual_segmentation_edgeline(vector<float2>& center_line);
    bool manual_segmentation(bool find_line);

    //�ܺ����α׷����� �� ��µ����� �о����
    void segmentation_output(model_info::segmentation_exe_type run_type, int index);
    bool get_segmentation_file_niigz(std::string path, result_info::segmentation_instance& segmentation_instance_, int index);

    void set_file_niigz(int view_id, dcmHelper* dcm_helper); //�ܺ����α׷������� �� ������ �Էµ�����
    
    //���׸����̼� ��� ����
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


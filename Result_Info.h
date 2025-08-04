#pragma once

#include <memory>
#include <mutex>

#include <vector>
#include <vector_types.h>
#include <corecrt_math_defines.h>
#include <vector_functions.h>

using namespace std;

class result_info
{
public:
    enum class return_state { success = 0, fail, buffer_null, line_size_zeno };
    enum class line_position { LEFT_LINE = 0, RIGHT_LINE, CENTER_LINE, LINE_SIZE };

    struct points_instance 
    {
        float2 start_point;
        float2 end_point;
        vector<vector<float2>> calibration_point;
        vector<pair<int, float2>> branch_point;
        void clear_points() {
            branch_point.clear();
            start_point = make_float2(-1, -1);
            end_point = make_float2(-1, -1);
        }
        void clear()
        {
            clear_points();
            calibration_point.clear();
        }
    };

    struct endPoint_result_Instance  
    {
        endPoint_result_Instance(int point_id, int center_line_id, float2 point) : frame_end_point(point), frame_id(point_id), end_center_line_id(center_line_id) {}
        float2 frame_end_point;
        int end_center_line_id;
        int frame_id;
    };

    struct segmentation_instance 
    {
        vector<std::pair<int, vector<float2>>> labeling_points;
        vector<endPoint_result_Instance> end_points;
        vector<float2> center_line_points;
        int optimal_image_id = -1; //최적이미지
        std::string bpm;
        void clear()
        {
            labeling_points.clear();  // 모든 labeling_points를 비웁니다.
            end_points.clear();       // 모든 end_points를 비웁니다.
            center_line_points.clear(); // center_line_points를 비웁니다.
            optimal_image_id = -1;    // 변수 초기화
            bpm.clear();             // bpm 문자열을 비웁니다.
        }

        void set_labeling_points(const std::vector<std::pair<int, std::vector<float2>>>& points) { labeling_points = points; };
        void set_labeling_points(const std::pair<int, std::vector<float2>>& points) { labeling_points.push_back(points); };
        void set_end_points(const std::vector<endPoint_result_Instance>& points) { end_points = points; };
        void set_center_line_points(const std::vector<float2>& points) { center_line_points = points; };
        void set_optimal_image_id(const int& id) { optimal_image_id = id; };
        void set_bpm(const std::string& bpm_string) { bpm = bpm_string; };

        const std::vector<std::pair<int, std::vector<float2>>>& get_labeling_points() const { return labeling_points; };
        const std::vector<endPoint_result_Instance>& get_end_points() const { return end_points; };
        const vector<float2>& get_center_line_points() const { return center_line_points; };
        std::pair<int, vector<float2>> get_labeling_point(int index) const
        {
            if (index < 0 || index >= labeling_points.size()) {
                throw std::out_of_range("Index out of range");
            }
            return labeling_points[index];
        }

        endPoint_result_Instance get_end_points(int index) const
        {
            if (index >= end_points.size()) {
                throw std::out_of_range("Index out of range");
            }
            return end_points[index];
        }
    };

    struct segmentation_line2D_instance
    {
        segmentation_line2D_instance() : line2D_points(static_cast<int>(line_position::LINE_SIZE), vector<float2>(0)), equal_interval_line2D_points(static_cast<int>(line_position::LINE_SIZE), vector<float2>(0)) {}
        segmentation_line2D_instance(int rows, int cols) : line2D_points(rows, std::vector<float2>(cols)), equal_interval_line2D_points(rows, std::vector<float2>(cols)) {};
        ~segmentation_line2D_instance() { clear(); };

        bool is_valid_row(int row, vector<vector<float2>> data) const { return row >= 0 && row < data.size(); };
        bool is_valid_col(int col, vector<float2> data) const { return col >= 0 && !data.empty() && col < data.size(); };
        void set_lines(vector<vector<float2>> lines, bool equal = false)
        {
            if (equal)
                equal_interval_line2D_points = lines;
            else
                line2D_points = lines;
        };
        void set_line(line_position line_type, const vector<float2>& line, bool equal = false)
        {
            auto& target_vector = equal ? equal_interval_line2D_points : line2D_points;

            if (!is_valid_row(static_cast<int>(line_type), target_vector))
            {
                if (target_vector.size() <= static_cast<int>(line_type))
                    target_vector.resize(static_cast<int>(line_type) + 1);
                target_vector[static_cast<int>(line_type)] = line;
            }
            else
                target_vector[static_cast<int>(line_type)] = line;
        };
        const vector<float2>& get_line(line_position line_type, bool equal = false) const { return equal ? equal_interval_line2D_points[static_cast<int>(line_type)] : line2D_points[static_cast<int>(line_type)]; };
        const vector <vector<float2>>& get_lines(bool equal = false) const { return equal ? equal_interval_line2D_points : line2D_points; };
        vector<float> get_equal_interval_dis() { return equal_interval_line_dis; };
        void set_equal_interval_dis(vector<float> dis) { equal_interval_line_dis.clear(); equal_interval_line_dis = (dis); };
        void clear(bool equal) { equal ? equal_interval_line2D_points.clear() : line2D_points.clear(); };
        void clear()
        {
            line2D_points.clear();
            equal_interval_line2D_points.clear();
        }
        vector<vector<float2>> line2D_points;
        vector<std::pair<int, float2>> end_points;
        vector<std::pair<int, float>> dis_line2D_points;

        vector<vector<float2>> equal_interval_line2D_points;
        vector<float> equal_interval_line_dis;
    };


    struct end_point_info
    {
        int frame_id;
        int center_line_id;
        float d;
        float3 pos3D;
        end_point_info(const float& D, const int& ID, const int& IDCENTER, const float3& POS) :d(D), frame_id(ID), center_line_id(IDCENTER), pos3D(POS) {};
    };

    struct endPoint3D_result_Instance
    {
        vector<float3> frame_end_point;
        vector<float3> center_line_points;
        
        vector<end_point_info> sort_center_id_point;
        vector<end_point_info> sort_id_point;
        void clear()
        {
            frame_end_point.clear();
            sort_center_id_point.clear();
            sort_id_point.clear();
        }
    };

    struct simulation_3D_instance
    {
        vector<float> distance_vertical; 
        vector<float> distance_from_start;
        vector<float3> center_points_3d;

        vector<std::pair<int, float2>> end_points;

        void clear()
        {
            distance_vertical.clear();
            distance_from_start.clear();
            center_points_3d.clear();
            end_points.clear();
        }
    };

public:
    result_info();
    virtual ~result_info();

    void result_clear();
    void set_points_instance(const points_instance& instance) { points_instance_ = instance; };
    const points_instance& get_points_instance() const { return points_instance_; };

    void set_segmentation_instance(const segmentation_instance& instance) { segmentation_instance_ = instance; };
    const segmentation_instance& get_segmentation_instance() const { return segmentation_instance_; };

    void set_segmentation_line2D_instance(const segmentation_line2D_instance& instance) { segmentation_line2D_instance_ = instance; };
    segmentation_line2D_instance& get_segmentation_line2D_instance() { return segmentation_line2D_instance_; };

    void set_simulation_3D_instance(const simulation_3D_instance& instance) { simulation_3D_instance_ = instance; };
    const simulation_3D_instance& get_simulation_3D_instance() const { return simulation_3D_instance_; };

    endPoint3D_result_Instance& get_endPoint3D_result_Instance() { return endPoint3D_result_Instance_; };
    void set_endPoint3D_result_Instance(const endPoint3D_result_Instance& instance) { endPoint3D_result_Instance_ = instance; };

private:

    points_instance points_instance_;
    segmentation_instance segmentation_instance_;
    segmentation_line2D_instance segmentation_line2D_instance_;
    simulation_3D_instance simulation_3D_instance_;
    endPoint3D_result_Instance endPoint3D_result_Instance_;
};


#pragma once

#include <memory>
#include <mutex>

// enum representing different model types

#define model model_info

class model_info
{
public:
    enum class model_type :int
    {
        lint = 0,
        branch_points,
        segmentation_points,
        epiline_2d,
        ai_segmentation,
        //Logo_file,
        line_2d,
        equilateral_line_2d,
        line_uniform_points,
        simulation_3d,
        simulation_ffr,
        total_modeltype
    };

    enum class segmentation_manual_type : int
    {
        start_end_findline = 0,
        branch_findline,
        find_centerline,
        run_manual_ai,
        load_file_line
    };

    enum class points_type :int
    {
        start_point = 0,
        end_point,
        branch_point,
        none_point,
    };

    enum class folder_type :int
    {
        data_path = 0,
        output_path,
        result_path,
        all_path
    };
    
    enum class clear_type: int
    {
        clear_line = 0,
        clear_calibration_line,
        clear_equilateral_line,
        clear_all 
    };

    enum class find_endpoint_type : int
    {
        image_open = 0,
        AI_lebeling_data,
        AI_endpoints,
    };

    enum class save_file_type : int
    {
        log_file =0,
        bpm_file,
        points_file,
    };
    // enum representing different segmentation model types
    enum class segmentation_model_type : int
    {
        lad = 0,
        lcx,
        rca,
        none
    };

    enum class segmentation_exe_type : int
    {
        run_centerline = 0,
        run_outlines,
        run_endpoints,
    };

    enum class image_float_type : int
    {
        ORIGIN_IMAGE = 0,
        CENTLINE_IMAGE,
        LEBELING_IMAGE,
        SPEED_IMAGE,
        BOUNDARY_IMAGE,
        IMAGE_SIZE
    };

    model_info();
    virtual ~model_info();

    static model_info* get_instance();


    //void set_model_type(model_type type) { cur_model_ = type; };
    //model_type get_model_type() { return cur_model_; };

    //void set_segmentation_model_type(segmentation_model_type type) { cur_segmentation_model_ = type; };
    //segmentation_model_type get_segmentation_model_type() { return cur_segmentation_model_; };


    //파일 경로 프로그램 실행 시 결정.
    void set_segmentation_model_path(const std::string& strpath) { str_segmentation_model_path_ = strpath; };
    std::string get_segmentation_model_path() { return str_segmentation_model_path_; };

    void set_data_path(const std::string& strpath) { str_data_path_ = strpath; };
    std::string get_data_path() { return str_data_path_; };

    void set_output_path(const std::string& strpath) { str_output_path_ = strpath; };
    std::string get_output_path() { return str_output_path_; };

    void set_result_path(const std::string& strpath) { str_result_path_ = strpath; };
    std::string get_result_path() { return str_result_path_; };

    void set_case_path(const std::string& strpath) { str_case_path_ = strpath; };
    std::string get_case_path() { return str_case_path_; };

private:
    static std::unique_ptr<model_info> instance_;
    static std::mutex instance_mutex_;

    static void destroy();

    //model_type cur_model_;
    //segmentation_model_type cur_segmentation_model_;


    std::string str_segmentation_model_path_;
    std::string str_data_path_;
    std::string str_output_path_;
    std::string str_result_path_;
    std::string str_case_path_;
};
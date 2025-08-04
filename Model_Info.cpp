#include "Model_info.h"

std::unique_ptr<model_info> model_info::instance_ = nullptr;
std::mutex model_info::instance_mutex_;

model_info::model_info() {}//:cur_model_(model_type::lint) , cur_segmentation_model_(segmentation_model_type::lad) {}

model_info::~model_info() {}

model_info* model_info::get_instance() 
{
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_ == nullptr) {
        instance_ = std::unique_ptr<model_info>(new model_info());
    }
    return instance_.get();
}

void model_info::destroy()
{
    instance_.reset(nullptr);
}

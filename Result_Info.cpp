#include "Result_Info.h"

result_info::result_info() {}

result_info::~result_info() {}

void result_info::result_clear()
{

	points_instance_.clear();
	segmentation_instance_.clear();
	segmentation_line2D_instance_.clear();
	simulation_3D_instance_.clear();
	endPoint3D_result_Instance_.clear();
}

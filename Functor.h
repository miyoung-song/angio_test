#pragma once
#define _USE_MATH_DEFINES

#include<cassert>
#include<vector_types.h>
//#include<unordered_set>
//#include<math>
#include<math.h>
#include<iostream>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include<cuda_fp16.h>

#include<typeinfo>

#include<numeric>
#include<array>
#include<memory>

#include <algorithm>


#pragma warning(disable:4819)

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#define KERNEL_U8  1<<0
#define KERNEL_F32 1<<1
#define KERNEL_F64 1<<2
#define DEFAULT_VALUE(x) = x

constexpr int THRESHOLD_CPU = 64;

#define CUDA_CALL(call) {\
 cudaError_t e=call;     \
 if(e!=cudaSuccess) {    \
\
 }                                                                 \
}

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err,__FILE__,__LINE__)
inline void __checkCudaErrors(CUresult err, const char* file, const int line) {
    if (CUDA_SUCCESS != err) {
        const char* errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

#define rep(i,n) for(int(i) = 0; i < int(n); i++)

static const type_info& _test0 = typeid(__half*);
static const type_info& _test1 = typeid(int*);
static const type_info& _test2 = typeid(float*);
static const type_info& _test3 = typeid(double*);

void getDeviceCount();
bool __strtob(char*& pen);
inline int __strtol(const char*& pen, int val = 0);
inline float __strtof(const char*& pen);


//Decode
extern"C"
void decode(uchar * inout, const int cols, const int rows, const int bitsStored, const int bitsAllocated, const int highBit, const int samplePerPixel, const char* photomericInterp = nullptr, const int planarConfig = 0);

//convert variable type of data(H2H)
//template<class T, class T2>
//void convertSrc2Dest(T*, T2*, const int, const int);


extern "C"
void convertU8toU16H2H(uchar*, ushort*, const int, const int, const int, const int);
extern "C"
void convertS16toU16H2H(short*, ushort*, const int, const int, const int, const int);

extern "C"
void convertU8toF32H2H(uchar * in, float* out, const int width, const int height, const int maxFrame, const int planarConfig = -1);
extern "C"
void convertF32toU8H2H(float * in, uchar * out, const int width, const int height, const int maxFrame, const int planarConfig = -1);
extern "C"
void convertU16toF32H2H(ushort* in, float* out, const int width, const int height, const int maxFrame, const int planarConfig = -1);
extern "C"
void convertS16toF32H2H(short * in, float* out, const int width, const int height, const int maxFrame, const int planarConfig = -1);
extern "C"
void convertU16toF32(uchar * in, float* out, const int width, const int height, const int perpixel);



//upsampling
void upsampling_half(__half*, __half*, const int&, const int&, const int&, const int&);
void upsampling_int(int*, int*, const int&, const int&, const int&, const int&);
void upsampling_float(float*, float*, const int&, const int&, const int&, const int&);
void upsampling_double(double*, double*, const int&, const int&, const int&, const int&);

extern "C"
void gpu_upsampling_half(__half*, __half*, const int, const int, const int, const int);
extern "C"
void gpu_upsampling_int(int*, int*, const int, const int, const int, const int);
extern "C"
void gpu_upsampling_float(float*, float*, const int, const int, const int, const int);
extern "C"
void gpu_upsampling_double(double*, double*, const int, const int, const int, const int);
    
template<class T>
void cpu_upsampling(T*, T*, const int&, const int&, const int&, const int&);


//convolve1d
void convolve1d_half(__half*, __half*, __half*, const int&, const int&, const int&);
void convolve1d_int(int*, int*, float*, const int&, const int&, const int&);
void convolve1d_float(float*, float*, float*, const int&, const int&, const int&);
void convolve1d_double(double*, double*, double*, const int&, const int&, const int&);


extern "C"
void gpu_convolve1d_half(__half*, __half*, __half*, const int, const int, const int);
extern "C"
void gpu_convolve1d_int(int*, int*, float*, const int, const int, const int);
extern "C"
void gpu_convolve1d_float(float*, float*, float*, const int, const int, const int);
extern "C"
void gpu_convolve1d_double(double*, double*, double*, const int, const int, const int);

template<class T>
void cpu_convolve1d(T*, T*, T*, const int&, const int&, const int&);

//convolve2d
void convolve2d_half(__half*, __half*, __half*, const int&, const int&, const int&, const int&);
void convolve2d_int(int*, int*, float*, const int&, const int&, const int&, const int&);
void convolve2d_float(float*, float*, float*, const int&, const int&, const int&, const int&);
void convolve2d_double(double*, double*, double*, const int&, const int&, const int&, const int&);

extern "C"
void gpu_convolve2d_half(__half*, __half*, float*, const int, const int, const int, const int);
extern "C"
void gpu_convolve2d_int(int*, int*, float*, const int, const int, const int, const int);
extern "C"
void gpu_convolve2d_float(float*, float*, float*, const int, const int, const int, const int);
extern "C"
void gpu_convolve2d_double(double*, double*, double*, const int, const int, const int, const int);
template<class T>
void cpu_convolve2d(T*, T*, float*, const int&, const int&, const int&, const int&);

//derivative2d
void derivative2d_int(int*, int*, int*, const int&, const int&);
void derivative2d_float(float*, float*, float*, const int&, const int&);
void derivative2d_double(double*, double*, double*, const int&, const int&);

extern "C"
void gpu_derivative2d_int(int*, int*, int*, const int, const int);
extern "C"
void gpu_derivative2d_float(float*, float*, float*, const int, const int);
extern "C"
void gpu_derivative2d_double(double*, double*, double*, const int, const int);
template<class T>
void cpu_derivative2d(T*, T*, T*, const int&, const int&);

//diffuse
bool diffusefilt_half(__half*, __half*, const int&, const int&, const int&, const float&, const float&, const float&, const float&, const float&, const int&, const int&);
bool diffusefilt_float(float*, float*, const int&, const int&, const int&, const float&, const float&, const float&, const float&, const float&, const int&, const int&);
bool diffusefilt_double(double*, double*, const int&, const int&, const int&, const float&, const float&, const float&, const float&, const float&, const int&, const int&);

extern"C"
void gpu_diffusefilt_half(__half*, __half*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);
extern"C"
void gpu_diffusefilt_float(float*, float*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);
//void gpu_diffusefilt_float(unsigned char*, float*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);

extern"C"
void gpu_diffusefilt_double(double*, double*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);

//Eigen
bool eigenimg_half(__half*, __half*, __half*, __half*, __half*, const float&, const int&, const int&);
bool eigenimg_float(float*, float*, float*, float*, float*, const float&, const int&, const int&);
bool eigenimg_double(double*, double*, double*, double*, double*, const float&, const int&, const int&);

extern"C"
void gpu_eigenimg_half(__half*, __half*, __half*, __half*, __half*, const float, const int, const int);
extern"C"
void gpu_eigenimg_float(float*, float*, float*, float*, float*, const float, const int, const int);
extern"C"
void gpu_eigenimg_double(double*, double*, double*, double*, double*, const float, const int, const int);


bool frangifilt_half(__half*, __half*, unsigned int*, const int&, const int&, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing);
bool frangifilt_float(float*, float*, unsigned int*, const int&, const int&, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing);
bool frangifilt_double(double*, double*, unsigned int*, const int&, const int&, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing);

extern"C"
void gpu_frangifilt_half(__half*, __half*, unsigned int*, const int, const int, const float, const float, const float, const float, const float, const float,const bool);
extern"C"
void gpu_frangifilt_float(float*, float*, float*, const int, const int, const float, const float, const float, const float, const float, const float,const bool);
extern"C"
void gpu_frangifilt_float2(float*, float*, float*, const int, const int, const float, const float, const float, const float, const float, const float, const bool);
extern"C"
void gpu_frangifilt_double(double*, double*, unsigned int*, const int, const int, const float, const float, const float, const float, const float, const float,const bool);



//extern"C"
//void gpu_speedImage_half(__half*, __half*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);
extern"C"
void gpu_speedImage_float(float*, float*, bool*,const int, const int, const int,const bool, const bool);
//extern"C"
//void gpu_speedImage_double(double*, double*, const int, const int, const int, const float, const float, const float, const float, const float, const int, const int);



extern"C"
void test_gpuBASOC(float* h_lamMap, float2 * h_pnt, float2 * h_medial, float2 * h_left, float2 * h_right, float* h_radius, const int pntnum, const int factor,const int width, const int height);

extern"C"
void test_gpuBASOC2(float* h_lamMap, float* h_x, float* h_y, float* h_xy, float2 * h_pnt, float2 * h_medial, float2 * h_left, float2 * h_right, float* h_radius, const int pntnum, const int factor, const int width, const int height);

extern"C"
float device_bicubic2(const float* __restrict__ src, const float* __restrict__ dx,
    const float* __restrict__ dy, const float* __restrict__ dxy,
    const const float2 & point, const int& width, const int& height);

extern"C"
void test_gpuMSRCR(
    unsigned char* h_in,
    float* h_out,
    const float* h_weights,
    const float* h_sigmas,
    const float gain,
    const float offset,
    const int paramlen,
    const int width,
    const int height);

extern"C"
void test_gpuMSRCR2(
    float* h_in,
    void* h_out,
    const float* h_weights,
    const float* h_sigmas,
    const float gain,
    const float offset,
    const int paramlen,
    const int width,
    const int height);

extern "C"
void get_gpu_8_connective_point_min(float* in, float* out1, float* out2, const int width, const int height);

extern "C"
void grid3d_function(
    const float3 * __In_Vertex,
    const float* __In_Rad1,
    const float* __In_Rad2,
    float3 * __Out_Vertex,
    unsigned int* __Out_Index,
    const int& vertexNumber,//precalculate
    const int& indexNumber,//precalculate
    const int& around,
    const int& pointNumber);

extern "C"
float gpu_otsu_threshold(
    float*,
    const int,
    const int,
    const bool
);


extern "C"
void gpu_CLAHE(unsigned char* in, unsigned char* out, const float clipLimit, const int tileGirdSizeX, const int tileGirdSizeY, const int width,const int heigth);

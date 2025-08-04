#pragma once
#define _USE_MATH_DEFINES

#include<vector>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include<cuda_fp16.h>
#include<memory>
//#include<npp.h>
//#include<npps.h>
//#include<nppi_filtering_functions.h>
#include "Functor.h"
//#include<thrust/host_vector.h>
//#include<thrust/device_ptr.h>
//#include<thrust/sort.h>
//#include<thrust/copy.h>
//#include<thrust/unique.h>
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;


#define _HOSTTEST false

#define WARP_SIZE 32
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 6
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define FULLMASK 0xFFFFFFFF
#define M_PI       3.14159265358979323846   // pi
#define CONVXN 32
#define CONVXM 4
#define CONVYN 4
#define CONVYM 64

//#define max(a,b) (a)>(b)?a:b
//#define min(a,b) (a)<(b)?a:b
const int THREADS_PER_BLOCK = 512;
const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

__constant__ __device__ double fEps = (0.1e-10) / (0.1e-10) * 0.5;
__constant__ __device__ double atolerance = 1e-08;
__constant__ __device__ double eta = 2.2204e-16;
__device__ __constant__ unsigned int SUR_VERTICES = 4;
//__device__ __constant__ int InvAMat[1 << 8] = {
//    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//    -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//    2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
//    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
//    0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
//    0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
//    -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
//    0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
//    9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
//    -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
//    2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
//    0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
//    -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
//    4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1
//};
__device__ __constant__ int InvAMat[1 << 8] = {
    1,  
    1,
    -3, 3, -2, -1,
    2, -2, 1, 1, 
    1, 
    1, 
    -3, 3, -2, -1, 
    2, -2, 1, 1, 
    -3, 3, -2, -1, 
    -3, 3, -2, -1, 
    9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
    -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
    2, -2, 1, 1, 
    2, -2, 1, 1, 
    -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
    4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1
};
//__constant__ __device__ float defGaussKernel_5x5[] = { 6.94444444444444405895033867182065e-03,2.08333333333333321768510160154619e-02,2.77777777777777762358013546872826e-02,2.08333333333333321768510160154619e-02,6.94444444444444405895033867182065e-03,
//2.08333333333333321768510160154619e-02,6.250e-02,8.33333333333333287074040640618477e-02,6.250e-02,2.08333333333333321768510160154619e-02,
//2.77777777777777762358013546872826e-02,8.33333333333333287074040640618477e-02,1.11111111111111104943205418749130e-01,8.33333333333333287074040640618477e-02,2.77777777777777762358013546872826e-02,
//2.08333333333333321768510160154619e-02,6.250e-02,8.33333333333333287074040640618477e-02,6.250e-02,2.08333333333333321768510160154619e-02,
//6.94444444444444405895033867182065e-03,2.08333333333333321768510160154619e-02,2.77777777777777762358013546872826e-02,2.08333333333333321768510160154619e-02,6.94444444444444405895033867182065e-03
//};


constexpr float _def[6] = { 
    1.0 / 144,3.0 / 144,4.0 / 144,9.0 / 144,12.0 / 144,16.0 / 144 
};
constexpr double par[16] = {
    0.00549691757211334, 4.75686155670860e-10, 3.15405721902292e-11, \
    0.00731109320628158, 1.40937549145842e-10, 0.0876322157772825, \
    0.0256808495553998, 5.87110587298283e-11, 0.171008417902939, \
    3.80805359553021e-12, 9.86953381462523e-12, 0.0231020787600445, \
    0.00638922328831119, 0.0350184289706385 , 0
    };

__host__ __device__ __inline__ float3& operator+(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ __inline__ float3& operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ __inline__ float3& operator/=(const float3& a, const float& b) { return b != 0 ? make_float3(a.x / b, a.y / b, a.z / b) : make_float3(NAN, NAN, NAN); }
__host__ __device__ __inline__ float3& operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }


template<class T>
__global__ void device_simpleAdd(T* in1, T* in2, T* out, const int width)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;
    out[gid] = in1[gid] + in2[gid];
}
template<class T>
__global__ void device_simpleSub(T* in1, T* in2, T* out)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * gridDim.x * blockDim.x + tidx;
    out[gid] = in1[gid] - in2[gid];

}
template<class T>
__global__ void device_simpleMul(T* in1, T* in2, T* out, const int width)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;
    out[gid] = in1[gid] * in2[gid];
}
template<class T>
__global__ void device_simpleNorm2(T* in, T* out, const T min, const T max,const T scale = 1,const T threshold = 0)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * gridDim.x + tidx;
    const T buffer = T(in[gid] - min) / (max - min);
    out[gid] = ((buffer < threshold) ? 0 : buffer) * scale;
}
template<class T>
__global__ void device_simpleScale(T* in, T* out, const float scale,const int width)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;
    out[gid] = in[gid] * scale;
}

template<class T>
__global__ void device_simplePower(T* in1, T* out, const int coeff, const int width)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;
    out[gid] = __powf(in1[gid], coeff);
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, float* fdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 32]] ? sdata[tid] : sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 16]] ? sdata[tid] : sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 8]] ? sdata[tid] : sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 4]] ? sdata[tid] : sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 2]] ? sdata[tid] : sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] = fdata[sdata[tid]] < fdata[sdata[tid + 1]] ? sdata[tid] : sdata[tid + 1];
}

template <class T,unsigned int blockSize>
__global__ void reduce(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata_[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata_[tid] = 0;
    while (i < n) { sdata_[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata_[tid] += sdata_[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata_[tid] += sdata_[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata_[tid] += sdata_[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata_[tid] += sdata_[tid + 64]; } __syncthreads(); }
    if (tid < 32)
    {
        //warpReduce(sdata_, g_idata, tid);
        if (blockSize >= 64) sdata_[tid] += sdata_[tid + 32];
        if (blockSize >= 32) sdata_[tid] += sdata_[tid + 16];
        if (blockSize >= 16) sdata_[tid] += sdata_[tid + 8];
        if (blockSize >= 8)  sdata_[tid] += sdata_[tid + 4];
        if (blockSize >= 4)  sdata_[tid] += sdata_[tid + 2];
        if (blockSize >= 2)  sdata_[tid] += sdata_[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata_[0];
}

__device__
inline int lane_id(void) { return threadIdx.x % WARP_SIZE; }
__device__
inline int warp_id(void) { return threadIdx.x / WARP_SIZE; }

template<class T, typename Function>
__device__
inline T warpReduce(T val, const Function& f)
{
#pragma unroll 5
	for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
	{
		const T x_ = __shfl_xor_sync(0xffffffff,val, mask);
		val = f(x_,val);
	}
	return val;
}


template<class T, typename Function>
__inline__ __device__
T blockReduce(T val, const Function& f) {

	static __shared__ T shared[32];
	const int lane = threadIdx.x % WARP_SIZE;
	const int wid = threadIdx.x / WARP_SIZE;

	val = warpReduce(val,f);

	if (lane == 0) shared[wid] = val;

	__syncthreads();
	val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

	if (wid == 0) val = warpReduce(val);

	return val;
}
template<unsigned int blockSize, class T, typename Function>
__global__ void device_reduction(const T* __restrict__ in, T* __restrict__ out, const Function f)
{
    extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
    T* sdata_ = reinterpret_cast<T*>(sdata);
    //extern __shared__ T sdata_[];
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    const unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata_[tid] = in[tid + blockIdx.x * blockDim.x];
    while (i < blockDim.x*gridDim.x)
    {
        sdata_[tid] = f(sdata_[tid], f(in[i], in[i + blockSize]));
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) { sdata_[tid] = f(sdata_[tid],sdata_[tid + 512]); } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata_[tid] = f(sdata_[tid],sdata_[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata_[tid] = f(sdata_[tid],sdata_[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata_[tid] = f(sdata_[tid], sdata_[tid + 64]); } __syncthreads(); }
    if (tid < 32)
    {
        //warpReduce(sdata_, g_idata, tid);
        if (blockSize >= 64) sdata_[tid] = f(sdata_[tid],sdata_[tid + 32]);
        if (blockSize >= 32) sdata_[tid] = f(sdata_[tid],sdata_[tid + 16]);
        if (blockSize >= 16) sdata_[tid] = f(sdata_[tid],sdata_[tid + 8]);
        if (blockSize >= 8)  sdata_[tid] = f(sdata_[tid],sdata_[tid + 4]);
        if (blockSize >= 4)  sdata_[tid] = f(sdata_[tid],sdata_[tid + 2]);
        if (blockSize >= 2)  sdata_[tid] = f(sdata_[tid],sdata_[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata_[0];
 //   T val;
 //   //reduce multiple elements per thread
	//for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	i < n;
	//	i += blockDim.x * gridDim.x) {
	//	val = f(val,in[i]);
	//}
 //   val = blockReduce(val,f);
	//if (threadIdx.x == 0)
	//	out[blockIdx.x] = val;
}


template<class T, typename Function>
__global__ void device_customFunctor_in_2param(T* in, T* out, const Function f, const float param1, const float param2,const int misc)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * gridDim.x * blockDim.x + tidx;

    out[gid] = f(in[gid],param1,param2, misc);
}

template<class T, typename Function>
__global__ void device_customFunctor_in_1param(T* in, T* out, const Function f, const float param1, const int misc)
{
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * gridDim.x * blockDim.x + tidx;

    out[gid] = f(in[gid],param1, misc);
}


template<class T>
__inline__ __device__ T G(T& x, T& y)
{
    return 1.0f - sqrtf((x - y) * (x - y)) / sqrtf(3.0f);
}


template<class T>
__global__ void prescan_large(T* output, T* input, int n, T* sums) {

    extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
    T* temp = reinterpret_cast<T*>(sdata);

    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = input[blockOffset + ai];
    temp[bi + bankOffsetB] = input[blockOffset + bi];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            //temp[bi] = f(temp[bi] , temp[ai] );
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            T _t = temp[ai];
            temp[ai] = temp[bi];
            //temp[bi] = f(temp[bi], _t);// += _t;
            temp[bi] += _t;
        }
    }
    __syncthreads();

    output[blockOffset + ai] = temp[ai + bankOffsetA];
    output[blockOffset + bi] = temp[bi + bankOffsetB];
}

//template<class T, typename Function>
template<class T>
__global__ void prescan_arbitrary(T* output, T* input, int n, int powerOfTwo)
{
    extern __shared__ __align__(sizeof(T)) unsigned char sdata[];
    T* temp = reinterpret_cast<T*>(sdata);
    //extern __shared__ T temp[];// allocated on invocation
    int threadID = threadIdx.x;

    int ai = threadID;
    int bi = threadID + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


    if (threadID < n) {
        temp[ai + bankOffsetA] = input[ai];
        temp[bi + bankOffsetB] = input[bi];
    }
    else {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }


    int offset = 1;
    for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            //temp[bi] = f(temp[bi], temp[ai]);// += temp[ai];
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadID == 0) {
        temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
    }

    for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            T _t = temp[ai];
            temp[ai] = temp[bi];
            //temp[bi] = f(temp[bi], _t);// += _t;
            temp[bi] += _t;
        }
    }
    __syncthreads();

    if (threadID < n) {
        output[ai] = temp[ai + bankOffsetA];
        output[bi] = temp[bi + bankOffsetB];
    }
}

template<class T>
__global__ void add(T* output, int length, T* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}

template<class T>
__global__ void add(T* output, int length, T* n1, T* n2) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

template <class T, bool isBackward>
__global__ void compactData(T* d_out,
    size_t* d_numValidElements,
    const unsigned int* d_indices, // Exclusive Sum-Scan Result
    const unsigned int* d_isValid,
    const T* d_in,
    unsigned int       numElements)
{
    if (threadIdx.x == 0)
    {
        if (isBackward)
            d_numValidElements[0] = d_isValid[0] + d_indices[0];
        else
            d_numValidElements[0] = d_isValid[numElements - 1] + d_indices[numElements - 1];
    }

    // The index of the first element (in a set of eight) that this
    // thread is going to set the flag for. We left shift
    // blockDim.x by 3 since (multiply by 8) since each block of 
    // threads processes eight times the number of threads in that
    // block
    unsigned int iGlobal = blockIdx.x * (blockDim.x << 3) + threadIdx.x;

    // Repeat the following 8 (SCAN_ELTS_PER_THREAD) times
    // 1. Check if data in input array d_in is null
    // 2. If yes do nothing
    // 3. If not write data to output data array d_out in
    //    the position specified by d_isValid
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
}


template<class T>
__global__ void device_convert2F32(T* in, float* out, const int perpixel)
{
    //const int movement = bitalloc - 1;
    if (perpixel == 1)
    {
        
        out[threadIdx.x + blockDim.x * blockIdx.x] = float(in[threadIdx.x + blockDim.x * blockIdx.x]);
    }
    else
    {
        for (int j = 0; j < 3; j++)
            out[3 * threadIdx.x + j + (3 * blockDim.x * blockIdx.x)] = float(in[3 * threadIdx.x + j + (3 * blockDim.x * blockIdx.x)]);
    }
}

template<class T,class T1>
__global__ void device_convertType(T* in, T1* out, const int conf, const float ratio)
{
    /* monochrome 1,2 = -1
     * rgb = 0,1
     * ybr is not defined,yet.
     */
    if (conf == -1)
    {

        out[threadIdx.x + blockDim.x * blockIdx.x] = T1(in[threadIdx.x + blockDim.x * blockIdx.x] * ratio);
    }
    else
    {
        if (conf == 0)
        {

            for (int j = 0; j < 3; j++)
                out[3 * threadIdx.x + j + (3 * blockDim.x * blockIdx.x)] = T1(in[3 * threadIdx.x + j + (3 * blockDim.x * blockIdx.x)] * ratio);
        }
        else
        {

            for (int j = 0; j < 3; j++)
                out[3 * threadIdx.x + j + (3 * blockDim.x * blockIdx.x)] = T1(in[threadIdx.x + blockDim.x * blockIdx.x + j * (blockDim.x * gridDim.x)] * ratio);
        }
    }
}

template<class T>
__inline__ __device__ T device_bilinearInterpolation(const T* __restrict__ src, const float2& v, const int& width, const int & height)
{
    const uint2 q00 = make_uint2(__float2int_rd(v.x), __float2int_rd(v.y));
    const uint2 q11 = make_uint2(fminf(q00.x + 1, width - 1.0f), fminf(q00.y + 1.0f, height - 1.0f));

    const uint2 q01 = make_uint2(q00.x, q11.y);
    const uint2 q10 = make_uint2(q11.x, q00.y);

    const float inv_denom = (q11.x - q00.x) * (q11.y - q00.y);

    if (inv_denom != 0.0f)
    {
        const float2 _a = make_float2(q11.x - v.x, v.x - q00.x);
        const float2 _b = make_float2(q11.y - v.y, v.y - q00.y);

        return ((_a.x * src[q00.x + q00.y * width] + _a.y * src[q10.x + q10.y * width]) * _b.x\
            + (_a.x * src[q01.x + q01.y * width] + _a.y * src[q11.x + q11.y * width]) * _b.y) / inv_denom;
    }
    else
    {
        return src[q00.x + q00.y * width];
    }
}

template<class T, int scale, typename Function>
__global__ void device_morphology(
    const T* __restrict__ in,
    T* __restrict__ out,
    const Function f,
    const int width,
    const int height)
{
   /* extern __shared__ __align__(sizeof(T)) unsigned char _s[];
    int* sdata = reinterpret_cast<int*>(_s);*/

    const int g_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int g_y = threadIdx.y + blockIdx.y * blockDim.y;
    const int halves = scale / 2;
    const int gIdx = g_x + g_y * width;
    
    /*const int sharedWidth = blockDim.x + halves * 2;

    const int s_x = threadIdx.x + halves;
    const int s_y = threadIdx.y + halves;
    const int sIdx = s_x + s_y * sharedWidth;

    sdata[sIdx] = in[gIdx];

    __syncthreads();

    if (threadIdx.y < halves)
    {
        const int _stride = blockDim.x + halves - 1;
        const int _stride2 = blockDim.y + halves - 1;

        const int _left = (g_y < halves) ? 0 : g_y - halves;
        const int _right = (g_y + _stride) >= width ? width - 1 : (g_x + _stride);

        sdata[sIdx - halves] = in[_left + g_y * width];
        sdata[sIdx + _stride] = in[_right + g_y * width];

        const int _upper = (g_y - halves) < 0 ? 0 : g_y - halves;
        const int _bottom = (g_y + _stride2) >= height ? height - 1 : (g_y + _stride2);

        sdata[sIdx - halves - sharedWidth] = in[_left + _upper * width];
        sdata[sIdx + _stride - sharedWidth] = in[_right + _upper * width];

        sdata[sIdx - halves + sharedWidth] = in[_left + _bottom * width];
        sdata[sIdx + _stride + sharedWidth] = in[_right + _bottom * width];

        const int _sIdxTrans = s_y + s_x * sharedWidth;
            
        sdata[s_y + (threadIdx.y) * sharedWidth] = in[g_y + _left * width];
        sdata[s_y + (s_x + blockDim.y) * sharedWidth] = in[g_y + _right * width];

    }
    __syncthreads();*/
    float re = 0;
    for (auto x = -halves; x <= halves; x++)
    {
        const int X = x + g_x;
        for (auto y = -halves; y <= halves; y++)
        {
            const int Y = y + g_y;
            if ((x != 0 && y != 0) && (X > 0 & X <= (width - 1) & Y > 0 & Y <= (height - 1)))
            {
                re = f(in[X + Y * width], re);
            }
        }
    }
    out[gIdx] = re;

}


template<class T>
__host__ __inline__
T* getKernel1d(const float& sigma, int& kernelSize)
{
    int _init = -ceilf(kernelSize / 2.0f);
    const float _sig = (sigma * sigma);
    const int _itermax = int(abs(_init) * 2 + 1);
    //std::unique_ptr<float> H;
    //H = std::make_unique<float>(_itermax);
    T* H = new T[_itermax];
    T sum = 0;
    for (auto i = 0; i < _itermax; i++)
    {
        H[i] = expf(-((_init * _init) / (2.0f * _sig)));
        _init++;
        sum += H[i];
    }
    for (auto i = 0; i < _itermax; i++)
        H[i] /= sum;
    kernelSize = _itermax;
    return H;
}
template<class T>
__host__ __inline__
T* getDerivativeKernel(const char& type, const int& mode = 1)
{
    T* re;
    if (mode == 1)
    {
        re = new T[9];
        memset(re, 0, sizeof(T) * 9);
    }
    if (type == 'x')
    {
        if (mode == 1)
        {

            re[0] = 3.0f / 32.0f;
            re[1] = 10.0f / 32.0f;
            re[2] = 3.0f / 32.0f;
            re[3] = 0;
            re[4] = 0;
            re[5] = 0;
            re[6] = -3.0f / 32.0f;
            re[7] = -10.0f / 32.0f;
            re[8] = -3.0f / 32.0f;

            return re;
        }
        //else if (mode == 2)
        {

        }
    }
    else if (type == 'y')
    {
        if (mode == 1)
        {

            re[0] = 3.0f / 32.0f;
            re[1] = 0;
            re[2] = -3.0f / 32.0f;
            re[3] = 10.0f / 32.0f;
            re[4] = 0;
            re[5] = -10.0f / 32.0f;
            re[6] = 3.0f / 32.0f;
            re[7] = 0;
            re[8] = -3.0f / 32.0f;
            return re;
        }
        //else if (mode == 2)
        {

        }
    }
    return nullptr;
}
template<class T>
__host__ __inline__
T* getKernelOpt(const char* _s)
{
    T* re;
    if (_s[1] == '_')
    {
        re = new T[3 * 3];
        if (_s[0] == 'x')
        {
            re[0] = par[12]; re[1] = par[13]; re[2] = par[12];
            re[3] = par[15]; re[4] = par[15]; re[5] = par[15];
            re[6] = -par[12]; re[7] = -par[13]; re[8] = -par[12];
        }
        else
        {
            re[0] = par[12]; re[1] = par[15]; re[2] = -par[12];
            re[3] = par[13]; re[4] = par[15]; re[5] = -par[13];
            re[6] = par[12]; re[7] = par[15]; re[8] = -par[12];
        }
    }
    else
    {
        re = new T[5 * 5];
        if (_s[1] == 'y')
        {
            re[0] = par[9]; re[1] = par[1]; re[2] = par[15]; re[3] = -par[10]; re[4] = -par[9];
            re[5] = par[10]; re[6] = par[11]; re[7] = par[15]; re[8] = -par[11]; re[9] = -par[10];
            re[10] = par[15]; re[11] = par[15]; re[12] = par[15]; re[13] = par[15]; re[14] = par[15];
            re[15] = -par[10]; re[16] = -par[11]; re[17] = par[15]; re[18] = par[11]; re[19] = par[10];
            re[20] = -par[9]; re[21] = -par[10]; re[22] = par[15]; re[23] = par[10]; re[24] = par[9];
        }
        else
        {
            if (_s[0] == 'x')
            {
                re[0] = par[0]; re[1] = par[1]; re[2] = par[2]; re[3] = par[1]; re[4] = par[0];
                re[5] = par[3]; re[6] = par[4]; re[7] = par[5]; re[8] = par[4]; re[9] = par[3];
                re[10] = -par[6]; re[11] = -par[7]; re[12] = -par[8]; re[13] = -par[7]; re[14] = -par[6];
                re[15] = par[3]; re[16] = par[4]; re[17] = par[5]; re[18] = par[4]; re[19] = par[3];
                re[20] = par[0]; re[21] = par[1]; re[22] = par[2]; re[23] = par[1]; re[24] = par[0];

            }
            else
            {
                re[0] = par[0]; re[1] = par[3]; re[2] = -par[6]; re[3] = par[3]; re[4] = par[0];
                re[5] = par[1]; re[6] = par[4]; re[7] = -par[7]; re[8] = par[4]; re[9] = par[1];
                re[10] = par[2]; re[11] = par[5]; re[12] = -par[8]; re[13] = par[5]; re[14] = par[2];
                re[15] = par[1]; re[16] = par[4]; re[17] = -par[6]; re[18] = par[4]; re[19] = par[1];
                re[20] = par[0]; re[21] = par[3]; re[22] = -par[6]; re[23] = par[3]; re[24] = par[0];
            }
        }
    }
    return re;


}
template<class T>
void getGaussSigma2dField(std::unique_ptr<T[]>& dGxx, std::unique_ptr<T[]>& dGxy, std::unique_ptr<T[]>& dGyy, const float& sigma, int& _kernelHalf,int& _kernSz)
{
    int _init = -roundf(sigma * 3.0f);
    const float sig2 = sigma * sigma;
    const int _itermax = int(abs(_init) * 2 + 1);
    const int _itermax2 = _itermax * _itermax;
    _kernelHalf = int(_itermax / 2);
    std::unique_ptr<float[]> X(new float[_itermax2]);
    std::unique_ptr<float[]> Y(new float[_itermax2]);

    _kernSz = sizeof(T) * _itermax * _itermax;

    dGxx = std::make_unique<T[]>(_kernSz);
    dGxy = std::make_unique<T[]>(_kernSz);
    dGyy = std::make_unique<T[]>(_kernSz);

//#pragma omp parallel for
    for (int _i = 0; _i < _itermax; _i++)
    {
        float _val = _init;
        for (int _j = 0; _j < _itermax; _j++)
        {
            X[int(_j * _itermax + _i)] = _val;
            Y[int(_j + _itermax * _i)] = _val;
            _val += 1.0f;
        }
    }

    const float _bufxx = 1.0f / (2.0f * M_PI * powf(sigma, 4.0f)) * (1.0f / sig2);
    const float _bufxy = 1.0f / (2.0f * M_PI * powf(sigma, 6.0f));


    //const float _bufyy
//#pragma omp parallel for
    for (int _j = 0; _j < _itermax; _j++)
    {
//#pragma omp parallel for
        for (int _i = 0; _i < _itermax; _i++)
        {

            const int _pos = _i + _j * _itermax;
            const int _invpos = _j + _i * _itermax;
            const float& _x = X[_pos];
            const float& _y = Y[_pos];
            const float _inBuf = expf(-(_x * _x + _y * _y) / (2.0f * sig2));
            dGxx[_pos] = _bufxx * (_x * _x - sig2) * _inBuf;
            dGxy[_pos] = _bufxy * _x * _y * _inBuf;
            dGyy[_invpos] = dGxx[_pos];


        }
    }
    
}





template<class T>
void get1Dfrom2D(T* kernel, T* outr, T* outc, const int bound)
{
    for (auto i = 0; i < bound; i++)
    {
        outr[i] = kernel[i];
        outc[i] = kernel[i * bound];

    }
    //rat /= (outr[0]* outc[0]);
}

template<class T>
T compatibilityPyramidDown(T* d_in, T* d_out, T* kernel, const int& in_width, const int& in_height, const int& out_width, const int& out_height, const int& kernel_wh)
{
    /*dim3 grids = dim3(fmaxf(1, in_width / 8), fmaxf(1, in_height / 8));
    dim3 blocks = dim3(8, 8);
    device_convolve2d<T><<<grids,blocks>>>(d_in,d_out)*/
}

template<class T>
std::vector<T> device_percentile(T* d_data, const T& _low, const T& _band1, const T& _band2, const int fullnumber)
{
    /*T* d_ptr;
    CUDA_CALL(cudaMalloc((void**) & d_ptr, fullnumber * sizeof(T)));
    CUDA_CALL(cudaMemcpy(d_ptr, d_data, fullnumber * sizeof(T), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<T> th_dptr(d_ptr);
    thrust::host_vector<T> hv(fullnumber);
    
    thrust::sort(th_dptr, th_dptr + fullnumber);
    thrust::unique(th_dptr, th_dptr + fullnumber);
    thrust::copy(th_dptr, th_dptr + fullnumber, hv.begin());

    int i = 1;
    while (hv[i-1] < hv[i])i++;
    
    std::vector<T> re;
    re.push_back(th_dptr[i * _low]);
    re.push_back(th_dptr[i * _band1]);
    re.push_back(th_dptr[i * _band2]);
    CUDA_CALL(cudaFree(d_ptr));
    return std::move(re);*/
    return std::vector<T>();
}


inline int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;

}
template<class T>
void scanLargeDeviceArray(T* d_out, T* d_in, int length);
template<class T>
void scanSmallDeviceArray(T* d_out, T* d_in, int length);
template<class T>
void scanLargeEvenDeviceArray(T* d_out, T* d_in, int length)
{
    const int blocks = std::fmaxf(1, length / ELEMENTS_PER_BLOCK);
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(T);

    T* d_sums, * d_incr;
    cudaMalloc((void**)&d_sums, blocks * sizeof(T));
    cudaMalloc((void**)&d_incr, blocks * sizeof(T));
    /*auto _fs = [=] __device__(float a, float b) { return a + b; };
    auto _fx = [=] __device__(float a, float b) { return a > b ? a : b; };
    auto _fn = [=] __device__(float a, float b) { return a < b ? a : b; };*/
    prescan_large<T> << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
    cudaDeviceSynchronize();

    const int sumsArrThreadsNeeded = (blocks + 1) / 2;
    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
        // perform a large scan on the sums arr
        scanLargeDeviceArray(d_incr, d_sums, blocks);
    }
    else {
        // only need one block to scan sums arr so can use small scan
        scanSmallDeviceArray(d_incr, d_sums, blocks);
    }

    add<T> << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);

    cudaFree(d_sums);
    cudaFree(d_incr);
}
template<class T>
void scanSmallDeviceArray(T* d_out, T* d_in, int length) {
    int powerOfTwo = nextPowerOfTwo(length);
    /*auto _fx = [=] __device__(float a, float b) { return a > b ? a : b; };
    auto _fn = [=] __device__(float a, float b) { return a < b ? a : b; };
    auto _fs = [=] __device__(float a, float b) { return a + b; };*/
    prescan_arbitrary<T> << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(T) >> > (d_out, d_in, length, powerOfTwo);
    cudaDeviceSynchronize();
}

template<class T>
void scanLargeDeviceArray(T* d_out, T* d_in, int length)
{
    int remainder = length % (ELEMENTS_PER_BLOCK);
    if (remainder == 0)
    {
        scanLargeEvenDeviceArray(d_out, d_in, length);
    }
    else
    {
        // perform a large scan on a compatible multiple of elements
        int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);

        // scan the remaining elements and add the (inclusive) last element of the large scan to this
        T* startOfOutputArray = &(d_out[lengthMultiple]);
        scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

        add<T> << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
    }
}

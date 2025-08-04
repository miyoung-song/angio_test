#pragma once

#include "Functor.h"
#include "pub.cuh"
#include <fstream>
#include <sstream>
#include <unordered_set>

// fp16 is not concerned.

template <class T, class T2>
void arrsum(T* buf, T2 line, const int w = 512,
    const int h = 512)
{
    return;
    int sz = w * h;
    T* h_arr = new T[w * h];
    cudaMemcpy(h_arr, buf, sizeof(T) * w * h, cudaMemcpyDeviceToHost);

    auto ac = std::accumulate(h_arr, h_arr + sz, 0.0f);
 //   qDebug() << line << ac;
}


template <class T, class T2>
void devicePrint(T* arr, T2 line, const int w = 512,
    const int h = 512)
{
    return;

#ifdef _DEBUG
#else
    return;
#endif

    CUDA_CALL(cudaDeviceSynchronize());
    T* h_test;
    std::ofstream out;
    CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(T) * w * h));
    CUDA_CALL(cudaMemcpy(h_test, arr, sizeof(T) * w * h, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    std::stringstream ss;
    
    ss << "D:\\Project\\Test\\bin\\" << line << ".bin";

    out.open(ss.str().c_str(), std::ios::binary | std::ios::out);
    out.write(reinterpret_cast<char*>(h_test), sizeof(T) * w * h);
    out.close();
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w;x++)
        {
            if (h_test[y * h + x] > 0)
            {
                int s = 0;
            }
        }
    }
    CUDA_CALL(cudaFreeHost(h_test));
}


template <class T, class T2>
void deviceArraySumPrint(T* arr, T2 line, const int w = 512,
    const int h = 512) {
    return;
    T* h_arr = new T[w * h];
    cudaMemcpy(h_arr, arr, sizeof(T) * w * h, cudaMemcpyDeviceToHost);

    double tmp = 0;
    double min = 0;
    double max = 0;

    // const size_t len = sizeof(h_arr) / sizeof(h_arr[0]);

    std::unordered_set<float> s(h_arr, h_arr + w * h);

#pragma omp parallel for
    for (int i = 0; i < w * h; i++) {
        tmp += h_arr[i];
        if (min > h_arr[i]) {
            min = h_arr[i];
        }
        else if (max < h_arr[i]) {
            max = h_arr[i];
        }
    }
  //  qDebug() << line << tmp << "|" << min << "|" << max << "||" << s.size();
    delete[] h_arr;

#if false
    {
        CUDA_CALL(cudaDeviceSynchronize());
        T* h_test;
        std::ofstream out;
        CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(T) * w * h));
        CUDA_CALL(
            cudaMemcpy(h_test, arr, sizeof(T) * w * h, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
        std::stringstream ss;
#ifdef _DEBUG
        ss << line << "_d.bin";
#else
        ss << line << "_r.bin";
#endif
        out.open(ss.str().c_str(), std::ios::binary | std::ios::out);
        out.write(reinterpret_cast<char*>(h_test), sizeof(T) * w * h);
        out.close();

        CUDA_CALL(cudaFreeHost(h_test));
    }
#endif // _DEBUG_HOSTTEST
}

// template<class T, class T2>
// void deviceArrayFileOut(T* arr, T2 line, const int w = 512, const int h =
// 512)
//{
// #if _DEBUG_HOSTTEST
//     {
//         float* h_test;
//         std::ofstream out;
//         CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(float) * width *
//         height)); CUDA_CALL(cudaMemcpy(h_test, d_in, sizeof(float) * width *
//         height, cudaMemcpyDeviceToHost));
// #ifdef _DEBUG
//         out.open("d_out_d.bin", std::ios::binary | std::ios::out);
// #else
//         out.open("d_out_r.bin", std::ios::binary | std::ios::out);
// #endif
//         out.write(reinterpret_cast<char*>(h_test), sizeof(float) * width *
//         height); out.close();
//
//         CUDA_CALL(cudaFreeHost(h_test));
//     }
// #endif // _DEBUG_HOSTTEST
// }
//__constant__ int shrdmx1[2];
//__constant__ int shrdmx2[2];

__global__ void device_bucket(const float* __restrict__ in, int* __restrict__ out,
    const int num) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num) {
        auto _val = __float2int_rn(in[tid]);

        atomicAdd(out + _val, 1);
    }
}

__global__ void device_otsu(int* in, float* out, const int _de, const int num) {
    const int id = threadIdx.x + blockDim.x * blockIdx.x;

    double _weight = 0, _mean = 0, _var = 0;
    double _weight2 = 0, _mean2 = 0, _var2 = 0;

    for (auto i = 0; i < _de; i++) {
        if (i <= id) {
            _weight += __int2double_rn(in[i]);
            _mean += __int2double_rn(in[i] * i);
        }
        else {
            _weight2 += __int2double_rn(in[i]);
            _mean2 += __int2double_rn(in[i] * i);
        }
    }

    // float _td = _weight;
    // float _td2 = _weight2;

    _mean = (_weight == 0) ? 0 : __ddiv_rn(_mean, _weight);
    _weight = __ddiv_rn(_weight, double(num));

    _mean2 = (_weight2 == 0) ? 0 : __ddiv_rn(_mean2, _weight2);
    _weight2 = __ddiv_rn(_weight2, double(num));

    for (auto i = 0; i < _de; i++) {
        if (i <= id)
            _var += (pow(__int2double_rn(i) - _mean, 2.0) * __int2double_rn(in[i]));
        else
            _var2 += (pow(__int2double_rn(i) - _mean2, 2.0) * __int2double_rn(in[i]));
    }

    out[id] = __double2float_rn(
        __dadd_rn(__dmul_rn(_weight, _var), __dmul_rn(_weight2, _var2)));
    // out[id] = in[id];
}

template <class T>
__global__ void device_subsampling(const T* __restrict__ src,
    T* __restrict__ dest, const int src_width,
    const int src_height, const int dest_width,
    const int dest_height) {
    // extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    // T* sdata = reinterpret_cast<T*>(_sdata);

    const unsigned int t_x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int t_y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int gid = t_x + t_y * blockDim.x * gridDim.x;

    int dw = src_width / dest_width;
    int dh = src_height / dest_height;

    const unsigned int t_x_ = t_x * dw;
    const unsigned int t_y_ = t_y * dh;
    const unsigned int gid_ = t_x_ + t_y_ * blockDim.x * gridDim.x * dw;

    dest[gid] = src[gid_];
}

template <class T>
__global__ void
device_upsampling(T* src, T* dest, const int iSrc_w, const int iSrc_h,
    const float fStride_w = 2.0f, const float fStride_h = 2.0f) {
    // input (n,m) blocks, grid (W/n,H/m) from dest image size
    const unsigned int t_x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int t_y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int offset = t_x + t_y * blockDim.x * gridDim.x;

    float2 xy = make_float2((t_x) / fStride_w, (t_y) / fStride_h);

    // const uint2 q00 = make_uint2(__float2int_rn(xy.x), __float2int_rn(xy.y));
    // const uint2 q11 = make_uint2(fminf(q00.x + 1, iSrc_w - 1.0f), fminf(q00.y +
    // 1, iSrc_h - 1.0f));
    ////fminf(q11.x, iSrc_w - 1.0f)
    //////int2 border = make_uint2(iSrc_w, iSrc_h)
    ////q11.x = ;
    ////q11.y = fminf(q11.y, iSrc_h - 1.0f);

    // const uint2 q01 = make_uint2(q00.x, q11.y);
    // const uint2 q10 = make_uint2(q11.x, q00.y);

    dest[offset] = device_bilinearInterpolation<T>(src, xy, iSrc_w, iSrc_h);
}

template <class T>
__global__ void device_convolve1d(T* src, T* dest, T* kernel,
    const int boundMax, const int fwh,
    const bool flip = false) {
    const int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int tid_y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int verticIdx = boundMax * (flip ? tid_x : tid_y);
    const unsigned int _Idx = flip ? tid_y : tid_x;
    const unsigned int gIdx = tid_x + tid_y * boundMax; // (flip ? tid_y : tid_x);

    double sum = 0;
    int _ttid = 0;

#pragma unroll
    for (int i = -fwh; i <= fwh; i++) {
        _ttid = _Idx + i;
        if (_ttid < 0)
            _ttid = 0;
        else if (_ttid >= boundMax)
            _ttid = boundMax - 1;
        const int lIdx = flip ? _ttid * boundMax + tid_x : (verticIdx + _ttid);
        sum += (src[lIdx] * kernel[i + fwh]);
    }
    dest[gIdx] = __double2float_rd(sum);
}

template <class T, typename... Args> __device__ T adder(T first, Args... args) {
    return first + adder(args...);
}

template <class T, int i> __device__ T convRow(T* data, T* kernel) {
    return data[i] * kernel[i] + convRow<i - 1>(data, kernel);
}
template <class T> __device__ T convRow<-1>(T* data, T* kernel) { return 0; }

template <class T, int N, int M>
__global__ void device_convolve1d_shared_X(T* src, T* dest, T* kernel,
    const int fwh, const int width,
    const int height) {
    extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    T* sdata = reinterpret_cast<T*>(_sdata);
    const int sWidth = (N + fwh * 2);
    T* skern = sdata + (sWidth * M);

    const int i = threadIdx.x;
    const int j = blockIdx.x * blockDim.y + threadIdx.y;
    const int k = blockIdx.y;
    const int si = i + fwh;
    const int sj = threadIdx.y;

    int gIdx = k * N * M + j * M + i;

    sdata[si + sj * sWidth] = src[gIdx];

    if (threadIdx.x <= fwh * 2) {
        skern[threadIdx.x] = kernel[threadIdx.x];
    }

    __syncthreads();

    if (threadIdx.x < fwh) {
        const int pgx = max(int(blockIdx.x * blockDim.y - fwh), 0);
        const int ngx = min(int(blockIdx.x * blockDim.y + sWidth + fwh), width - 1);
        sdata[si - fwh + sj * sWidth] =
            src[k * N * M + (pgx + threadIdx.y) * M + i];
        sdata[si + N + sj * sWidth] = src[k * N * M + (ngx + threadIdx.y) * M + i];
    }
    __syncthreads();
    T sum = 0;

    for (int _i = -fwh; _i <= fwh; _i++) {
        sum += sdata[si + _i + sj * sWidth] * skern[fwh + _i];
    }

    // dest[gIdx] = convRow<T,fullWidth>(sdata + (threadIdx.x + threadIdx.y *
    // sWidth), skern);
    dest[gIdx] = sum;
}

template <class T, int N, int M>
__global__ void device_convolve1d_shared_Y(T* src, T* dest, T* kernel,
    const int fwh, const int width,
    const int height) {
    extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    T* sdata = reinterpret_cast<T*>(_sdata);
    const int sHeight = (M + fwh * 2);
    T* skern = sdata + (sHeight * N);

    const int gx = threadIdx.x + blockDim.x * blockIdx.x;
    const int gy = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = blockIdx.y;
    const int si = threadIdx.x;

    for (int j = threadIdx.y; j < M; j += blockDim.y) {
        const int gIdx = k * M * N + j * N + gx;
        const int sj = j + fwh;
        sdata[si + sj * N] = src[gIdx];
    }

    const int fullWidth = fwh * 2 + 1;

    if (threadIdx.y <= fwh * 2) {
        skern[threadIdx.y] = kernel[threadIdx.y];
    }

    __syncthreads();
    int sj = threadIdx.y + fwh;

    if (sj < fwh * 2) {
        const int pgy = fmaxf(gy - fwh, 0);
        const int ngy = fminf(gy + sHeight + fwh, height - 1);
        sdata[si + (sj - fwh) * N] = src[si + pgy * height];
        sdata[si + (sj + M) * N] = src[si + ngy * height];
    }

    __syncthreads();

    for (int j = threadIdx.y; j < M; j += blockDim.y) {
        const int gIdx = k * M * N + j * N + gx;
        const int sj = j + fwh;
        sdata[si + sj * N] = src[gIdx];

        T sum = 0;
        for (int i = -fwh; i <= fwh; i++) {
            sum += sdata[si + (sj + i) * N] * skern[fwh + i];
        }
        dest[gIdx] = sum;
    }

    // dest[gIdx] = convRow<T,fullWidth>(sdata + (threadIdx.x + threadIdx.y *
    // sWidth), skern);
}
template <class T>
__global__ void
device_convolve1d_shared(const T* __restrict__ src, T* __restrict__ dest,
    const T* __restrict__ kernel, const int boundMax,
    const int fwh, const float ratio,
    const bool flip = false) {
    extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    T* sdata = reinterpret_cast<T*>(_sdata);

    const int id = flip ? threadIdx.y : threadIdx.x;
    const int vid = flip ? blockIdx.x : blockIdx.y;
    // const int gId = id + vid * boundMax;
    const int gId = id * boundMax + vid;

    sdata[id + fwh] = src[gId];
    /*if (id <= fwh*2)
    {
        skern[id] = kernel[id];
    }*/
    __syncthreads();

    if (id < fwh) {
        sdata[id] = sdata[fwh];
        sdata[id + fwh + boundMax] = sdata[fwh + boundMax - 1];
        // flip ? src[fwh * (fwh - 1) + id] : src[vid + boundMax - 1];
    }
    __syncthreads();

    double sum = 0;
#pragma unroll
    for (int i = 0; i <= fwh * 2; i++)
        sum += (sdata[id + i] * kernel[i]);
    // float sum = convolutionRow<fwh * 2>(sdata + id, kernel, fwh);
    // flip ? convolutionColumn<fwh * 2>(sdata + id, kernel, fwh) :
    dest[gId] = sum;
}

template <int i>
__device__ float convolutionRow(float* data, float* kernel,
    const int kernelHalf) {
    return data[kernelHalf - i] * kernel[i] +
        convolutionRow<i - 1>(data, kernel, kernelHalf);
}

template <>
__device__ float convolutionRow<-1>(float* data, float* kernel,
    const int kernelHalf) {
    return 0;
}

template <int i>
__device__ float convolutionColumn(float* data, float* kernel,
    const int kernelHalf) {
    return data[(kernelHalf - i) * blockDim.y] * kernel[i] +
        convolutionColumn<i - 1>(data, kernel, kernelHalf);
}

template <>
__device__ float convolutionColumn<-1>(float* data, float* kernel,
    const int kernelHalf) {
    return 0;
}

template <class T>
__global__ void device_convolutionRowGPU(T* src, T* out, T* d_Kernel,
    const int width, const int height,
    const int kernelHalf) {
    extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    T* sdata = reinterpret_cast<T*>(_sdata);

    // __shared__ float sdata[ TILE_H * (blockDim.x + kernelHalf * 2) ];

    const int x0 = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
    const int tid_y = threadIdx.y + __mul24(blockDim.y, blockIdx.y);

    // src[tid_y] = 1st column of each rows
    // src[tid_y + width - 1] = last column of each rows

    const int gIdx = x0 + __mul24(tid_y, width);

    int x;

    // const int x0 = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
    const int shift = threadIdx.y * (blockDim.x + kernelHalf * 2);

    x = x0 - kernelHalf;
    sdata[threadIdx.x + shift] = (x < 0) ? src[tid_y] : src[gIdx - kernelHalf];

    x = x0 + kernelHalf;
    sdata[threadIdx.x + blockDim.x + shift] =
        (x >= width) ? src[tid_y + width - 1] : src[gIdx + kernelHalf];

    __syncthreads();

    float sum = 0;
    x = kernelHalf + threadIdx.x;
    /*const int kk = 2 * kernelHalf;
    sum = convolutionRow<kk>(sdata + shift + 2 *
    kernelHalf,d_Kernel,&kernelHalf);*/
    for (int i = -kernelHalf; i <= kernelHalf; i++)
        sum += sdata[x + i + shift] * d_Kernel[kernelHalf + i];

    out[gIdx] = sum;
}

template <class T>
__global__ void device_convolutionColGPU(T* src, T* out, T* d_Kernel,
    const int width, const int height,
    const int kernelHalf) {
    extern __shared__ __align__(sizeof(T)) unsigned char _sdata[];
    T* sdata = reinterpret_cast<T*>(_sdata);
    // __shared__ float sdata[blockDim.x * (TILE_H + kernelHalf * 2)];

    const int tid_x = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
    const int y0 = threadIdx.y + __mul24(blockDim.y, blockIdx.y);

    const int gIdx = tid_x + __mul24(y0, width);

    int y;

    // const int y0 = threadIdx.y + __mul24(blockIdx.y, blockDim.y);
    const int shift = threadIdx.y * (blockDim.x);

    y = y0 - kernelHalf;
    sdata[threadIdx.x + shift] =
        (y < 0) ? 0 : src[gIdx - __mul24(width, kernelHalf)];

    y = y0 + kernelHalf;
    const int shift1 = shift + __mul24(blockDim.y, blockDim.x);

    sdata[threadIdx.x + shift1] =
        (y > height - 1) ? 0 : src[gIdx + __mul24(width, kernelHalf)];

    __syncthreads();

    float sum = 0;
    /*sum = convolutionColumn<2 * kernelHalf>(sdata + threadIdx.x + (threadIdx.y +
     * kernelHalf * 2) * blockDim.x, d_Kernel, kernelHalf);*/
    for (int i = 0; i <= kernelHalf * 2; i++)
        sum += sdata[threadIdx.x + (threadIdx.y + i) * blockDim.x] * d_Kernel[i];

    out[gIdx] = sum;
}

template <class T>
__global__ void device_convolve2d_shared(T* src, T* dest, T* kernel,
    const int iSrc_w, const int iSrc_h,
    const int fwh, const int fhh) {
    extern __shared__ __align__(sizeof(T)) unsigned char s_data[];
    T* sdata = reinterpret_cast<T*>(s_data);

    const int2 sDim = make_int2(fwh * 2 + blockDim.x, fhh * 2 + blockDim.y);

    T* skernel = sdata + (sDim.x * sDim.y);

    const int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    const int i = threadIdx.x + fwh;
    const int j = threadIdx.y + fhh;

    const int kernelwidth = fwh * 2 + 1;
    const int kernelheight = fhh * 2 + 1;

    const int gIdx = iSrc_w * tid_y + tid_x;
    const int sIdx = sDim.x * j + i;

    sdata[sIdx] = src[gIdx];
    int ttidx_l = tid_x;
    int ttidx_r = tid_x;
    int ttidy_u = tid_y;
    int ttidy_d = tid_y;
    if (threadIdx.x < fwh) {
        /*int ttidx_l = tid_x - fwh;
        int ttidx_r = tid_x + blockDim.x;*/

        ttidx_l -= fwh;
        ttidx_r += blockDim.x;

        if (ttidx_l < 0)
            ttidx_l = 0;
        else if (ttidx_r >= iSrc_w)
            ttidx_r = iSrc_w - 1;

        sdata[sIdx - fwh] = src[iSrc_w * tid_y + ttidx_l];
        sdata[sIdx + blockDim.x] = src[iSrc_w * tid_y + ttidx_r];
    }

    if (threadIdx.y < fhh) {
        ttidy_u -= fhh;
        ttidy_d += blockDim.y;

        if (ttidy_u < 0)
            ttidy_u = 0;
        else if (ttidy_d >= iSrc_h)
            ttidy_d = iSrc_h - 1;

        sdata[sDim.x * threadIdx.y + i] = src[iSrc_w * ttidy_u + tid_x];
        sdata[sDim.x * (j + blockDim.y) + i] = src[iSrc_w * ttidy_d + tid_x];

        if (threadIdx.x < fwh) {
            /*int ttidx_l = tid_x - fwh;
            int ttidx_r = tid_x + blockDim.x;


            if (ttidx_l < 0) ttidx_l = 0;
            else if (ttidx_r >= iSrc_w) ttidx_r = iSrc_w - 1;*/

            sdata[sDim.x * threadIdx.y + threadIdx.x] =
                src[iSrc_w * ttidy_u + ttidx_l];
            sdata[sDim.x * (j + blockDim.y) + threadIdx.x] =
                src[iSrc_w * ttidy_d + ttidx_l];
            sdata[sDim.x * threadIdx.y + i + blockDim.x] =
                src[iSrc_w * ttidy_u + ttidx_r];
            sdata[sDim.x * (j + blockDim.y) + i + blockDim.x] =
                src[iSrc_w * ttidy_d + ttidx_r];

            /*skernel[threadIdx.x + threadIdx.y * kernelwidth] = kernel[threadIdx.x +
            threadIdx.y * kernelwidth]; skernel[threadIdx.x + (threadIdx.y + fhh + 1)
            * kernelwidth] = kernel[threadIdx.x + (threadIdx.y + fhh + 1) *
            kernelwidth]; skernel[(threadIdx.x + fwh + 1) + threadIdx.y * kernelwidth]
            = kernel[(threadIdx.x + fwh + 1) + threadIdx.y * kernelwidth];
            skernel[(threadIdx.x + fwh + 1) + (threadIdx.y + fhh + 1) * kernelwidth] =
            kernel[(threadIdx.x + fwh + 1) + (threadIdx.y + fhh + 1) * kernelwidth];


            if (threadIdx.x == 0)
            {
                skernel[kernelwidth - 1 + (threadIdx.y) * kernelwidth] =
            kernel[(threadIdx.x + fwh + 1) + (threadIdx.y + fhh + 1) * kernelwidth];
            }
            if (threadIdx.y == 0)
            {

            }*/
        }
    }
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int _ki = 0; _ki <= kernelwidth; _ki++)
            for (int _kj = 0; _kj <= kernelheight; _kj++) {
                const int _idx = _ki + _kj * kernelwidth;
                skernel[_idx] = kernel[_idx];
            }
    }
    __syncthreads();

    // kernel
    /*if ((threadIdx.x < kernelwidth) && (threadIdx.y < kernelheight))
    {
        const int _bufIdx = threadIdx.x + kernelwidth * threadIdx.y;
        skernel[_bufIdx] = kernel[_bufIdx];
    }
    __syncthreads();*/

    float sum = 0;

    for (int _i = -fwh, ki = 0; _i <= fwh; _i++, ki++) {
        for (int _j = -fhh, kj = 0; _j <= fhh; _j++, kj++) {
            const int _sIdx = sIdx + _i + (_j * sDim.x);
            sum += sdata[_sIdx] * skernel[ki + kj * kernelwidth];
            // sum = __fmaf_rn(sdata[__float2int_rn(__fadd_rn(sIdx, __fmaf_rn(_j,
            // sDim.x, _i)))], kernel[__float2int_rn(__fmaf_rn(kj, kernelwidth, ki))],
            // sum);
        }
    }
    dest[gIdx] = sum;
}

template <class T>
__global__ void device_convolve2d(const T* __restrict__ src,
    T* __restrict__ dest, T* kernel,
    const int iSrc_w, const int iSrc_h,
    const int fwh, const int fhh) {
    const int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    /*const int i = threadIdx.x + fwh;
    const int j = threadIdx.y + fhh;*/

    const int gIdx = iSrc_w * tid_y + tid_x;

    double sum = 0;
    int ttidx = 0, ttidy = 0;

    for (int i = -fwh; i <= fwh; i++) {
        ttidx = tid_x + i;
        if (ttidx < 0)
            ttidx = 0;
        else if (ttidx >= iSrc_w)
            ttidx = iSrc_w - 1;

        for (int j = -fhh; j <= fhh; j++) {
            ttidy = tid_y + j;
            if (ttidy < 0)
                ttidy = 0;
            else if (ttidy >= iSrc_h)
                ttidy = iSrc_h - 1;
            sum = __fma_rd(src[iSrc_w * ttidy + ttidx],
                kernel[(i + fwh) + (j + fhh) * (fwh * 2 + 1)], sum);
        }
    }
    dest[gIdx] = __double2float_rd(sum);
}

template <class T>
__global__ void device_derivativeY(T* in, T* out, const int width,
    const int height) {
    extern __shared__ float s_f[];

    int i = threadIdx.x;
    int k = blockIdx.y;

    int si = i + 1;
    // int sj = threadIdx.y;

    int globIdx = k * width + i;

    const int prev = (si - 1);
    const int cur = (si);
    const int nex = (si + 1);

    s_f[cur] = in[globIdx];

    __syncthreads();

    if (i < 1) {
        s_f[prev] = s_f[cur] - (s_f[nex] - s_f[cur]);
        // s_f[si - 1] = - s_f[si + 1];
        s_f[si + width] =
            s_f[si + width - 1] - (s_f[si + width - 1] - s_f[si + width - 2]);
    }

    __syncthreads();
    if (sizeof(T) == 8) {
        out[globIdx] = __dmul_rd((s_f[nex] - s_f[cur]), fEps) +
            __dmul_rd((s_f[cur] - s_f[prev]), fEps);
    }
    else {
        out[globIdx] = __fmul_rd((s_f[nex] - s_f[cur]), __double2float_rd(fEps)) +
            __fmul_rd((s_f[cur] - s_f[prev]), __double2float_rd(fEps));
    }
    // out[globIdx] = (((s_f[nex] - s_f[cur]) * fEps) + ((s_f[cur] - s_f[prev]) *
    // fEps));
}

template <class T>
__global__ void device_derivativeX(T* in, T* out, const int width,
    const int height) {
    extern __shared__ float s_f[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    int k = blockIdx.y;

    int si = threadIdx.x;
    int sj = j + 1;

    int globIdx = k * width * height + j * width + i; // wrong

    const int prev = (sj - 1) + (si * blockDim.y);
    const int cur = (sj)+(si * blockDim.y);
    const int nex = (sj + 1) + (si * blockDim.y);

    s_f[cur] = in[globIdx];

    __syncthreads();

    if (threadIdx.y < 1) {
        s_f[prev] = s_f[cur] - (s_f[nex] - s_f[cur]);
        s_f[sj + height + si * blockDim.y] =
            s_f[sj + height - 1 + si * blockDim.y] -
            (s_f[sj + height - 1 + si * blockDim.y] -
                s_f[sj + height - 2 + si * blockDim.y]);
    }

    __syncthreads();

    if (sizeof(T) == 8) {
        out[globIdx] = __dmul_rd((s_f[nex] - s_f[cur]), fEps) +
            __dmul_rd((s_f[cur] - s_f[prev]), fEps);
    }
    else {
        out[globIdx] = __fmul_rd((s_f[nex] - s_f[cur]), __double2float_rd(fEps)) +
            __fmul_rd((s_f[cur] - s_f[prev]), __double2float_rd(fEps));
    }
    // out[globIdx] = (((s_f[nex] - s_f[cur]) * fEps) + ((s_f[cur] - s_f[prev]) *
    // fEps));
}

// bs's value must be higher than 8, else case is not intereption.
// this is only operating on 16bits
__global__ void device_decoder(uchar* data, const int bs, const int ba,
    const int hb, const int ib, const int spp,
    const int pc) {
    const int _x = threadIdx.x * spp * bs / 2;
    const int _y = blockDim.x * blockIdx.x * spp * bs / 2;
    const int tid = _x + _y;

    if (spp == 1) {
        uint16_t _tmp = ((*(&data[tid] + 1)) << (bs - 8 - ib)) +
            (data[tid] & (uchar)(powf(2, (bs - 8 + ib) - 1)));
        *(&data[tid] + 1) = _tmp & 0xFF;
        data[tid] = (_tmp & 0xFF00) >> 8;
    }
}

template <class T>
__global__ void device_eigen2Image(T* xx, T* xy, T* yy, T* _lam1, T* _lam2,
    T* _ix, T* _iy, const int width) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;

    // Need to more script for compatibility here
    const T _z = (xx[gid] - yy[gid]) * (xx[gid] - yy[gid]);
    const T _x = xy[gid] * xy[gid];
    const T tmp = __fsqrt_rd(__fmaf_rd(_x, 4.0f, _z));
    T _v2x = 2.0f * xy[gid];
    T _v2y = yy[gid] - xx[gid] + tmp;
    const double _t1 = _v2x * _v2x;
    const double _t2 = _v2y * _v2y;
    const double mag = __dsqrt_rd(_t1 + _t2);

    if (mag != 0) {
        if (fabs(mag) > atolerance) {
            _v2x = __fdiv_rd(_v2x, __double2float_rd(mag));
            _v2y = __fdiv_rd(_v2y, __double2float_rd(mag));
        }
    }

    /*T _v1x = -_v2y;
    T _v1y = _v2x;*/

    T _mu1 = 0.5f * (xx[gid] + yy[gid] + tmp);
    T _mu2 = 0.5f * (xx[gid] + yy[gid] - tmp);

    const bool check = (fabsf(_mu1) > fabsf(_mu2)) ? true : false;
    if (check) {
        _lam1[gid] = (_mu2);
        _lam2[gid] = (_mu1);
        _ix[gid] = (_v2x);
        _iy[gid] = (_v2y);
    }
    else {
        _lam1[gid] = (_mu1);
        _lam2[gid] = (_mu2);
        _ix[gid] = (-_v2y); //_ix[gid] = _v1x;
        _iy[gid] = (_v2x);  // _v1y;
    }
}

template <class T>
__global__ void device_postFrangi(T* lam1, T* lam2, T* Ix, T* Iy, T* out,
    const float beta, const float C,
    const int width) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;

    // const T _angle = atan2f(Ix[gid], Iy[gid]);
    const double _lam1 = (lam1[gid] == 0) ? eta : lam1[gid];
    const double _Rb =
        __dmul_rd(__ddiv_rd(lam2[gid], _lam1), __ddiv_rd(lam2[gid], _lam1));
    // const double S2 = __dadd_rd(pow(_lam1, 2.0), pow(lam2[gid], 2.0));
    const double S2 = __dadd_rd(pow(_lam1, 2.0), pow(double(lam2[gid]), 2.0));
    // const T c = fabsf(sqrtf(S2)) / 2.0;
    T result =
        (_lam1 < 0)
        ? 0
        : __double2float_rd(__dmul_rd(exp(__ddiv_rd(-_Rb, beta)),
            (1.0f - exp(__ddiv_rd(-S2, C)))));
    // T result = (_lam1 < 0) ? 0 : (__expf(-_Rb / beta) * (1.0f - __expf(-S2 /
    // C)));

    // if((out[gid] != 0) && (result != 0))
    out[gid] = fmaxf(out[gid], result);
}

template <class T>
__global__ void device_preGrowingCut(T* in, T* state1, T* state2, const T _low1,
    const T _low2, const T _high,
    const int width) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;

    state1[gid] = 0;
    state2[gid] = 0;

    if (in[gid] <= _low1) {
        state1[gid] = -1;
        state2[gid] = 1;
    }
    else if ((in[gid] >= _low2) && (in[gid] <= _high)) {
        state1[gid] = 1;
        state2[gid] = 1;
    }
}

template <class T>
__global__ void device_diffusionTensor(T* mu1, T* mu2, T* v2x, T* v2y, T* _xx,
    T* _xy, T* _yy, const float alpha,
    const float C, const int width) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * width + tidx;

    const T v1x = -v2y[gid];
    const T v1y = v2x[gid];

    double di = mu1[gid] - mu2[gid];

    if ((di < eta) && (di > -eta))
        di = eta;

    const T _lam1 =
        alpha +
        __fmul_rn((1 - alpha),
            __expf(__fdiv_rd(-C, __double2float_rd(__dmul_rn(di, di)))));
    const T _lam2 = alpha;

    _xx[gid] = (__fmul_rn(_lam1, __fmul_rn(v1x, v1x))) +
        (__fmul_rn(__fmul_rn(_lam2, v2x[gid]), v2x[gid]));
    _xy[gid] = (__fmul_rn(_lam1, __fmul_rn(v1x, v1y))) +
        (__fmul_rn(__fmul_rn(_lam2, v2x[gid]), v2y[gid]));
    _yy[gid] = (__fmul_rn(_lam1, __fmul_rn(v1y, v1y))) +
        (__fmul_rn(__fmul_rn(_lam2, v2y[gid]), v2y[gid]));
}
template <class T>
__global__ void
device_diffuseOptimizationS1(T* d_u, T* d_xx, T* d_xy, T* d_yy, T* d_out, T* mx,
    T* my, const int width, const int height) {

    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    const int gIdx = width * tid_y + tid_x;

    const int fwh = 1, fhh = 1;

    double sum = 0;
    int ttidx = 0, ttidy = 0;

    for (int i = -fwh; i <= fwh; i++) {
        ttidx = tid_x + i;
        if (ttidx < 0)
            ttidx = 0;
        else if (ttidx >= width)
            ttidx = width - 1;

        for (int j = -fhh; j <= fhh; j++) {
            ttidy = tid_y + j;
            if (ttidy < 0)
                ttidy = 0;
            else if (ttidy >= height)
                ttidy = height - 1;
            const int _ind1 = width * ttidy + ttidx;
            const int _ind2 = (i + fwh) + (j + fhh) * (fwh * 2 + 1);
            //sum += __fmul_rn(__fmul_rn(d_u[_ind1], mx[_ind2]), (__fmul_rn(d_xx[_ind1], mx[_ind2]) + __fmul_rn(d_xy[_ind1], my[_ind2]))) \
                  //    + __fmul_rn(__fmul_rn(d_u[_ind1], my[_ind2]), (__fmul_rn(d_xy[_ind1], mx[_ind2]) + __fmul_rn(d_yy[_ind1], my[_ind2])));

            const double t1 = __dmul_rn(d_u[_ind1], mx[_ind2]);
            const double t2 = __dmul_rn(d_xx[_ind1], mx[_ind2]);
            const double t3 = __dmul_rn(d_xy[_ind1], my[_ind2]);

            const double t4 = __dmul_rn(d_u[_ind1], my[_ind2]);
            const double t5 = __dmul_rn(d_xy[_ind1], mx[_ind2]);
            const double t6 = __dmul_rn(d_yy[_ind1], my[_ind2]);

            sum += __dmul_rn(t1, (t2 + t3)) + __dmul_rn(t4, (t5 + t6));
        }
    }

    d_out[gIdx] = __double2float_rd(sum);
}

template <class T>
__global__ void device_diffuseOptimizationS2(
    const T* __restrict__ d_u, T* __restrict__ d_xx, T* __restrict__ d_xy,
    T* __restrict__ d_yy, T* __restrict__ d_u1, T* __restrict__ d_out,
    const T* mxx, const T* mxy, const T* myy, const float dt, const int width,
    const int height) {
    // const int tid_x = ;
    // const int tid_y = ;

    const int gIdx = width * (threadIdx.y + blockDim.y * blockIdx.y) +
        threadIdx.x + blockDim.x * blockIdx.x;

    // const int fwh = 2, fhh = 2;

    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

#pragma unroll
    for (int i = -2, ttidx = 0, tid_x = threadIdx.x + blockDim.x * blockIdx.x,
        tid_y = threadIdx.y + blockDim.y * blockIdx.y;
        i <= 2; i++) {
        ttidx = tid_x + i;
        if (ttidx < 0)
            ttidx = 0;
        else if (ttidx >= width)
            ttidx = width - 1;

        for (int j = -2, ttidy = 0; j <= 2; j++) {
            ttidy = tid_y + j;
            if (ttidy < 0)
                ttidy = 0;
            else if (ttidy >= height)
                ttidy = height - 1;
            const int _ind1 = width * ttidy + ttidx;
            const int _ind2 = (i + 2) + (j + 2) * 5;
            sum0 = __fma_rd(d_u[_ind1], mxx[_ind2], sum0);
            sum1 = __fma_rd(d_u[_ind1], mxy[_ind2], sum1);
            sum2 = __fma_rd(d_u[_ind1], myy[_ind2], sum2);
        }
    }

    T t1 = __fmul_rd(__double2float_rd(sum0), d_xx[gIdx]);
    T t2 = __fmul_rd(__double2float_rd(sum1), d_xy[gIdx]);
    T t3 = __fadd_rd(t1, t2);
    t1 = __fmul_rd(__double2float_rd(sum2), d_yy[gIdx]);
    t2 = __fadd_rd(t3, t1);
    t1 = __fmul_rd(dt, t2);
    t2 = __fmul_rd(t1, d_u1[gIdx]);
    t3 = __fadd_rd(d_u[gIdx], t2);

    /*d_out[gIdx] = d_u[gIdx] + dt * (sum * d_xx[gIdx] + sum1 * d_xy[gIdx] + sum2
     * * d_yy[gIdx]) * d_u1[gIdx];*/

    if (!isfinite(t3))
        t3 = 0;
    d_out[gIdx] = (t3);
}

template <class T>
__global__ void device_diffuseRotationS1(T* d_xx, T* d_xy, T* d_yy, T* d_ux,
    T* d_uy, T* d_out1, T* d_out2,
    const int width, const int height) {
    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    const int gIdx = width * tid_y + tid_x;

    d_out1[gIdx] = d_xx[gIdx] * d_ux[gIdx] + d_xy[gIdx] * d_uy[gIdx];
    d_out2[gIdx] = d_xy[gIdx] * d_ux[gIdx] + d_yy[gIdx] * d_uy[gIdx];

    if ((tid_x == 0) || (tid_x == width - 1) || (tid_y == 0) ||
        (tid_y == height - 1)) {
        d_out1[gIdx] = 0;
        d_out2[gIdx] = 0;
    }
}
template <class T>
__global__ void
device_diffuseRotationS2(T* d_in, T* d_j1, T* d_j2, T* d_out, T* kernel_x,
    T* kernel_y, const int fwh, const int fhh,
    const float dt, const int width, const int height) {
    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    int tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    const int gIdx = width * tid_y + tid_x;

    float sum = 0;
    int ttidx = 0, ttidy = 0;

    for (int i = -fwh; i <= fwh; i++) {
        ttidx = tid_x + i;
        if (ttidx < 0)
            ttidx = 0;
        else if (ttidx >= width)
            ttidx = width - 1;

        for (int j = -fhh; j <= fhh; j++) {
            ttidy = tid_y + j;
            if (ttidy < 0)
                ttidy = 0;
            else if (ttidy >= height)
                ttidy = height - 1;
            const int _ind1 = width * ttidy + ttidx;
            const int _ind2 = (i + fwh) + (j + fhh) * (fwh * 2 + 1);
            sum += (d_j1[_ind1] * kernel_x[_ind2]) + (d_j2[_ind1] * kernel_y[_ind2]);
        }
    }
    d_out[gIdx] = d_in[gIdx] + sum * dt;
}
// float* kernel_x, float* kernel_y, const int fwh, const int fhh, const float
// dt, const int width, const int height
template <class T>
__device__ __inline__ void
alloc_Ext_shared(T* d_or, T* d_xx, T* d_xy, T* d_yy, T* s_or, T* s_xx, T* s_xy,
    T* s_yy, const int& globIdx, const int& cur, const int& _up,
    const int& _down, const int& _left, const int& _right,
    const int& _ul, const int& _ur, const int& _dl, const int& _dr,
    const int& width, const int& height) {
    s_or[cur] = d_or[globIdx];
    s_xx[cur] = d_xx[globIdx];
    s_xy[cur] = d_xy[globIdx];
    s_yy[cur] = d_yy[globIdx];

    if (blockIdx.y != 0) {
        s_or[_up] = d_or[globIdx - width];
        s_xx[_up] = d_xx[globIdx - width];
        s_xy[_up] = d_xy[globIdx - width];
        s_yy[_up] = d_yy[globIdx - width];
    }
    if (blockIdx.y != (height - 1)) {
        s_or[_down] = d_or[globIdx + width];
        s_xx[_down] = d_xx[globIdx + width];
        s_xy[_down] = d_xy[globIdx + width];
        s_yy[_down] = d_yy[globIdx + width];
    }
    __syncthreads();

    if (blockIdx.y == 0) {
        s_or[_up] = s_or[cur];
        s_xx[_up] = s_xx[cur];
        s_xy[_up] = s_xy[cur];
        s_yy[_up] = s_yy[cur];
    }
    if (blockIdx.y == (height - 1)) {
        s_or[_down] = s_or[cur];
        s_xx[_down] = s_xx[cur];
        s_xy[_down] = s_xy[cur];
        s_yy[_down] = s_yy[cur];
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        s_or[_left] = s_or[cur];
        s_xx[_left] = s_xx[cur];
        s_xy[_left] = s_xy[cur];
        s_yy[_left] = s_yy[cur];

        s_or[_ul] = s_or[_up];
        s_xx[_ul] = s_xx[_up];
        s_xy[_ul] = s_xy[_up];
        s_yy[_ul] = s_yy[_up];

        s_or[_dl] = s_or[_down];
        s_xx[_dl] = s_xx[_down];
        s_xy[_dl] = s_xy[_down];
        s_yy[_dl] = s_yy[_down];
    }
    if (blockIdx.x == (width - 1)) {
        s_or[_right] = s_or[cur];
        s_xx[_right] = s_xx[cur];
        s_xy[_right] = s_xy[cur];
        s_yy[_right] = s_yy[cur];

        s_or[_ur] = s_or[_up];
        s_xx[_ur] = s_xx[_up];
        s_xy[_ur] = s_xy[_up];
        s_yy[_ur] = s_yy[_up];

        s_or[_dr] = s_or[_down];
        s_xx[_dr] = s_xx[_down];
        s_xy[_dr] = s_xy[_down];
        s_yy[_dr] = s_yy[_down];
    }
    __syncthreads();
}
template <class T>
__global__ void device_diffuseNoneNegative(T* ori, T* Dxx, T* Dxy, T* Dyy,
    T* d_out, const float dt,
    const int width, const int height) {
    // extern __shared__ T shrd[];
    extern __shared__ __align__(sizeof(T)) unsigned char _shrd[];
    T* shrd = reinterpret_cast<T*>(_shrd);
    const int mx = blockDim.x;
    const int shardDim = mx + 2;

    T* s_or = shrd + (shardDim * 3 * 0);
    T* s_xx = shrd + (shardDim * 3 * 1);
    T* s_xy = shrd + (shardDim * 3 * 2);
    T* s_yy = shrd + (shardDim * 3 * 3);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = threadIdx.y;

    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;

    const int globIdx = j * mx + i;

    const int cur = (si)+(sj * shardDim);
    const int _up = (si)+((sj + 1) * shardDim);
    const int _down = (si)+((sj - 1) * shardDim);
    const int _left = (si - 1) + (sj * shardDim);
    const int _right = (si + 1) + (sj * shardDim);
    const int _ul = (si - 1) + ((sj - 1) * shardDim);
    const int _ur = (si + 1) + ((sj - 1) * shardDim);
    const int _dl = (si - 1) + ((sj + 1) * shardDim);
    const int _dr = (si + 1) + ((sj + 1) * shardDim);

    alloc_Ext_shared(ori, Dxx, Dxy, Dyy, s_or, s_xx, s_xy, s_yy, globIdx, cur,
        _up, _down, _left, _right, _ul, _ur, _dl, _dr, width,
        height);
    const T wbR1 = (0.25) * ((fabsf(s_xy[_ur]) - s_xy[_ur]) +
        (fabsf(s_xy[cur]) - s_xy[cur]));
    const T wbL3 = (0.25) * ((fabsf(s_xy[_dr]) + s_xy[_dr]) +
        (fabsf(s_xy[cur]) + s_xy[cur]));
    const T wtM2 = (0.5) * ((s_yy[_right] + s_yy[cur]) -
        (fabsf(s_xy[_right]) + fabsf(s_xy[cur])));
    const T wmR4 =
        (0.5) * ((s_xx[_up] + s_xx[cur]) - (fabsf(s_xy[_up]) + fabsf(s_xy[cur])));
    const T wmL6 = (0.5) * ((s_xx[_down] + s_xx[cur]) -
        (fabsf(s_xy[_down]) + fabsf(s_xy[cur])));
    const T wmB8 = (0.5) * ((s_yy[_left] + s_yy[cur]) -
        (fabsf(s_xy[_left]) + fabsf(s_xy[cur])));
    const T wtR7 = (0.25) * ((fabsf(s_xy[_ul]) + s_xy[_ul]) +
        (fabsf(s_xy[cur]) + s_xy[cur]));
    const T wtL9 = (0.25) * ((fabsf(s_xy[_dl]) - s_xy[_dl]) +
        (fabsf(s_xy[cur]) - s_xy[cur]));

    d_out[globIdx] =
        s_or[cur] +
        dt * (wbR1 * (s_or[_ur] - s_or[cur]) + wtM2 * (s_or[_right] - s_or[cur]) +
            wbL3 * (s_or[_dr] - s_or[cur]) + wmR4 * (s_or[_up] - s_or[cur]) +
            wmL6 * (s_or[_down] - s_or[cur]) + wtR7 * (s_or[_ul] - s_or[cur]) +
            wmB8 * (s_or[_left] - s_or[cur]) + wtL9 * (s_or[_dl] - s_or[cur]));
}

template <class T>
__global__ void device_diffuseStandard(T* ori, T* Dxx, T* Dxy, T* Dyy, T* d_out,
    const float dt, const int width,
    const int height) {
    extern __shared__ __align__(sizeof(T)) unsigned char _shrd[];
    T* shrd = reinterpret_cast<T*>(_shrd);

    const int mx = blockDim.x;
    const int shardDim = mx + 2;

    T* s_or = shrd + (shardDim * 3 * 0);
    T* s_xx = shrd + (shardDim * 3 * 1);
    T* s_xy = shrd + (shardDim * 3 * 2);
    T* s_yy = shrd + (shardDim * 3 * 3);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = threadIdx.y;

    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;

    const int globIdx = j * mx + i;

    const int cur = (si)+(sj * shardDim);
    const int _up = (si)+((sj + 1) * shardDim);
    const int _down = (si)+((sj - 1) * shardDim);
    const int _left = (si - 1) + (sj * shardDim);
    const int _right = (si + 1) + (sj * shardDim);
    const int _ul = (si - 1) + ((sj - 1) * shardDim);
    const int _ur = (si + 1) + ((sj - 1) * shardDim);
    const int _dl = (si - 1) + ((sj + 1) * shardDim);
    const int _dr = (si + 1) + ((sj + 1) * shardDim);

    alloc_Ext_shared(ori, Dxx, Dxy, Dyy, s_or, s_xx, s_xy, s_yy, globIdx, cur,
        _up, _down, _left, _right, _ul, _ur, _dl, _dr, width,
        height);

    d_out[globIdx] =
        s_or[cur] +
        dt * ((0.25) * (s_xy[_up] - s_xy[_right]) * (s_or[_ur] - s_or[cur]) +
            (0.5) * (s_yy[_right] + s_yy[cur]) * (s_or[_right] - s_or[cur]) +
            (0.25) * (s_xy[_down] + s_xy[_right]) * (s_or[_dr] - s_or[cur]) +
            (0.5) * (s_xx[_up] + s_xx[cur]) * (s_or[_up] - s_or[cur]) +
            (0.5) * (s_xx[_down] + s_xx[cur]) * (s_or[_down] - s_or[cur]) +
            (0.25) * (s_xy[_up] + s_xy[_left]) * (s_or[_ul] - s_or[cur]) +
            (0.5) * (s_yy[_left] + s_yy[cur]) * (s_or[_left] - s_or[cur]) +
            (0.25) * (s_xy[_down] - s_xy[_left]) * (s_or[_dl] - s_or[cur]));
}

template <class T>
__global__ void
device_growingCut(const T* __restrict__ Image, const T* __restrict__ state1,
    const T* __restrict__ state2, float* __restrict__ change,
    T* __restrict__ t0, T* __restrict__ t1,
    const unsigned int windowSz, const int width,
    const int height) {
    __shared__ float smem[4800];
    const int t_x = threadIdx.x + blockDim.x * blockIdx.x;
    const int t_y = threadIdx.y + blockDim.y * blockIdx.y;
    const int offset = t_x + t_y * blockDim.x * gridDim.x;

    const T C_p = Image[offset];
    const T S_p2 = state2[offset];
    const T rr = __frsqrt_rn(3.0f);
    // #pragma unroll
    for (int jj = fmaxf(0, t_x - windowSz); jj < fminf(t_x + windowSz + 1, width);
        jj++) {
        // #pragma unroll
        for (int ii = fmaxf(0, t_y - windowSz);
            ii < fminf(t_y + windowSz + 1, height); ii++) {
            const int _offset = jj + ii * blockDim.x * gridDim.x;
            const T C_q = Image[_offset];
            const T S_q1 = state1[_offset];
            const T S_q2 = state2[_offset];

            const double dgct = __dsub_rd(C_q, C_p);
            const T gc =
                1.0f - __double2float_rd(__dsqrt_rd(__dmul_rd(dgct, dgct))) * rr;
            const T r = __fmul_rd(gc, S_q2);
            if (r > S_p2) {
                t0[offset] = S_q1;
                t1[offset] = r;

                change[offset] = 1;
                break;
            }
        }
    }
}
void decode(uchar* inout, const int cols, const int rows, const int bitsStored,
    const int bitsAllocated, const int highBit,
    const int samplePerPixel, const char* photomericInterp,
    const int planarConfig) {
    uchar* d_result;
    int _tot = int(cols * rows * bitsAllocated * samplePerPixel / 8);
    CUDA_CALL(cudaMalloc((void**)&d_result, _tot));
    CUDA_CALL(cudaMemcpy(d_result, inout, _tot, cudaMemcpyHostToDevice));
    const int initbit = highBit - bitsAllocated + 1;
    int pctype = 0; // defulat monotype
    if (photomericInterp != 0) {
    }

    device_decoder << <rows, cols >> > (d_result, bitsStored, bitsAllocated, highBit,
        initbit, samplePerPixel, pctype);
}

template <class T>
__global__ void device_1dRowCloset_Ex(T* in, int* out, const int width,
    const int yIdx) {
    extern __shared__ __align__(sizeof(int)) unsigned char sd[];
    int* sdata = reinterpret_cast<int*>(sd);
    int* LN = sdata + width;
    int* RN = sdata + width * 2;
    const int tid_x = threadIdx.x + blockDim.x * blockIdx.x;

    const int gIdx = tid_x + width * yIdx;
    int tid_x1 = tid_x + 1;
    if (tid_x1 >= width)
        tid_x1 = width - 1;
    // int gIdx1 = yIdx * width + tid_x1;

    int x = -1;
    if (in[gIdx] != 1)
        x = tid_x;

    const int wid = warp_id();
    const int lane = lane_id();
    unsigned int voted_x = __ballot_sync(FULLMASK, x > -1);
    unsigned int masked_x = ((FULLMASK << (lane + 1)) ^ FULLMASK) & voted_x;
    int count_zeros = __clz(masked_x);
    int closet_index = -1;

    int closet_p = 0;
    if (count_zeros < WARP_SIZE)
        closet_p = WARP_SIZE - count_zeros - 1;

    closet_index = __shfl_sync(FULLMASK, x, closet_p);

    if (lane == WARP_SIZE - 1)
        sdata[wid] = closet_index;

    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        x = sdata[threadIdx.x];
        voted_x = __ballot_sync(FULLMASK, x > -1);
        masked_x = ((FULLMASK << (lane + 1)) ^ FULLMASK) & voted_x;
        count_zeros = __clz(masked_x);
        if (count_zeros < WARP_SIZE)
            closet_p = WARP_SIZE - count_zeros - 1;
        sdata[threadIdx.x] = __shfl_sync(FULLMASK, x, closet_p);
    }
    __syncthreads();

    if ((wid > 0) && (closet_index == -1))
        closet_index = sdata[wid - 1];

    if ((closet_index >= 0) && (closet_index < width))
        LN[tid_x1] = closet_index;
    else
        LN[tid_x1] = -1;

    if (tid_x == 0)
        LN[tid_x] = -1;

    __syncthreads();

    if (in[gIdx] == 0)
        RN[LN[tid_x]] = tid_x;
    __syncthreads();

    if (tid_x == 0) {
        int _buf = 0;
        while ((LN[_buf] == -1) && (_buf < width))
            _buf++;
        if (_buf == width)
            _buf = -1;
        RN[0] = _buf;
    }
    __syncthreads();

    if (in[gIdx] != 1)
        out[gIdx] = 0;
    else {
        const int _LN = (tid_x == 0) ? 1 : LN[tid_x];
        const int lb = fabsf(tid_x - _LN);
        const int rb = fabsf(tid_x - RN[_LN]);
        const int _buf = (lb > rb) ? RN[_LN] : _LN;
        out[gIdx] = (tid_x - _buf) * (tid_x - _buf);
    }
}

template <class T>
__global__ void device_1dRowCloset_Ex2(T* src, float* LN, int* CS,
    const int width, const int yIdx) {
    extern __shared__ __align__(sizeof(int)) unsigned char sd[];
    int* RN = reinterpret_cast<int*>(sd);

    int tid_x = threadIdx.x + blockDim.x * blockIdx.x;

    const int Y = width * yIdx;
    int gIdx = tid_x + Y;

    int _ln = int(LN[gIdx]);
    if (src[gIdx] == 0)
        RN[_ln] = tid_x;

    if (tid_x == 0) {
        int _buf = 0;
        while ((LN[_buf + Y] == -1) && (_buf < width))
            _buf++;
        if (_buf == width)
            _buf = -1;
        RN[0] = _buf;
    }
    __syncthreads();

    if (src[gIdx] == 0)
        CS[gIdx] = 0;
    else {
        const int _LN = (tid_x == 0) ? width : int(LN[gIdx]);
        const int lb = fabsf(tid_x - _LN);
        const int rb = fabsf(tid_x - RN[_LN]);
        const int _buf = (lb > rb) ? RN[_LN] : _LN;
        CS[gIdx] = (tid_x - _buf) * (tid_x - _buf);
    }
}

template <class T>
__global__ void device_1dColCloset_Ex(int* fmap, T* out, const int width,
    const int xIdx) {
    extern __shared__ __align__(sizeof(int)) unsigned char sd[];
    int* f = reinterpret_cast<int*>(sd);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const int gIdx = xIdx + width * tid;

    f[tid] = fmap[gIdx];

    __syncthreads();

    // for (int row = tid; row < width; row += blockDim.x)
    if (f[tid] != 0) {
        bool bchange = false;
        float value = f[tid];
        for (int row_i = 1, d = 1; row_i < width - tid; row_i++) {
            value = fminf(value, f[tid + row_i] + d);
            d += (1 + 2 * row_i);
        }
        for (int row_i = 1, d = 1; row_i < tid; row_i++) {
            value = fminf(value, f[tid - row_i] + d);
            d += (1 + 2 * row_i);
        }
        out[gIdx] = value;
    }
    else
        out[gIdx] = 0;
}

//
//__inline__ __device__ Sint16 GetSint16(const Uint16& MSB, const  Float64& RS,
//const Float64& RI, const Uint16& BS)
//{
//
//    //Uint16 tmp = ((*LSB & 0xFF) << (HighBit - 8)) + (*MSB & (HighBit - 8));
//    //return (((*LSB & 0xFF) << ((HighBit - 8))) + (*MSB & ((HighBit -
//    8))))*RS + RI;
//    /*INT16 tmp = ((*LSB & 0xFF) << (HighBit - 8)) + (*MSB & (HighBit - 8));*/
//    Float64 tmp = (MSB & (0xFF) << (BS - 8)) +
//        (((MSB & 0xFF00) >> 8) & (uchar)((powf(2, BS - 8) - 1)));
//    Float64 result = (tmp)*RS + RI;
//    if (result > INT16_MAX - 1)
//        return INT16_MAX - 1;
//    else
//        return (Sint16)result;
//}
// template<typename T>
//__global__ void CalculateMONO2(Uint16* d_buffer, T* output, bool* masking,
//const int x, const int y, const int z, const Float64 WC, const Float64 WW,
//const Float64 RS, const Float64 RI, const Uint16 BS, const Uint16 PVMin, const
//Uint16 PVMax, const Plane plane)
//{
//    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    //int pixel_x_index = tid % blockDim.x;	// x
//    //int pixel_y_index = tid / blockDim.x;	// y
//
//    {
//        const Float32 wMin = WC - 0.5 - (WW - 1) / 2;
//        const Float32 wMax = WC - 0.5 + (WW - 1) / 2;
//        if (tid < x * y)
//        {
//            int pnt = getPoint(plane, threadIdx.x, blockIdx.x, x, y, z);
//
//            if (masking != nullptr && masking[pnt])
//            {
//                output[tid] = UINT16_MAX - 1;
//            }
//            else
//            {
//                Sint16 temp = 0;
//                if (BS != 16)
//                    temp = GetSint16(d_buffer[pnt], RS, RI, BS);
//                else
//                    temp = d_buffer[pnt];
//
//                //Float32 WC = 300;// 300 : window center WL
//                //Float32 WW = 870;// 870 : window width WW
//
//                if (temp <= wMin) output[tid] = PVMin;
//                else if (temp > wMax) output[tid] = PVMax;
//                else output[tid] = ((temp - (WC - 0.5)) / (WW - 1) + 0.5) *
//                (PVMax == 0 ? 1 : PVMax);
//            }
//        }
//        __syncthreads();
//    }
//}

void convertU8toU16H2H(uchar* src, ushort* dest, const int iWidth,
    const int iHeight, const int maxFrame,
    const int planarConfig = -1) {
    uchar* d_in;
    ushort* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = iWidth * iHeight * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(uchar) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(ushort) * mat));

    const float _r = float(USHRT_MAX) / UCHAR_MAX;
    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, src + mat * itt, sizeof(uchar) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<uchar, ushort>
            << <iHeight, iWidth >> > (d_in, d_out, planarConfig, _r);

        CUDA_CALL(cudaMemcpyAsync(dest + mat * itt, d_out, sizeof(ushort) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void convertS16toU16H2H(short* src, ushort* dest, const int iWidth,
    const int iHeight, const int maxFrame,
    const int planarConfig = -1) {
    short* d_in;
    ushort* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = iWidth * iHeight * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(short) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(ushort) * mat));

    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, src + mat * itt, sizeof(short) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<short, ushort>
            << <iHeight, iWidth >> > (d_in, d_out, planarConfig, 1);

        CUDA_CALL(cudaMemcpyAsync(dest + mat * itt, d_out, sizeof(ushort) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}
void convertU8toF32H2H(uchar* in, float* out, const int width, const int height,
    const int maxFrame, const int planarConfig) {
    uchar* d_in;
    float* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = width * height * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(uchar) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(float) * mat));
    const float _r = 1.0f / float(UCHAR_MAX);
    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, in + mat * itt, sizeof(uchar) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<uchar, float>
            << <height, width >> > (d_in, d_out, planarConfig, _r);

        CUDA_CALL(cudaMemcpyAsync(out + mat * itt, d_out, sizeof(float) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void convertF32toU8H2H(float* in, uchar* out, const int width, const int height,
    const int maxFrame, const int planarConfig) {
    float* d_in;
    uchar* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = width * height * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(float) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(uchar) * mat));
    const float _r = UCHAR_MAX;
    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, in + mat * itt, sizeof(float) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<float, uchar>
            << <height, width >> > (d_in, d_out, planarConfig, _r);

        CUDA_CALL(cudaMemcpyAsync(out + mat * itt, d_out, sizeof(uchar) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void convertU16toF32H2H(ushort* in, float* out, const int width,
    const int height, const int maxFrame,
    const int planarConfig) {
    ushort* d_in;
    float* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = width * height * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(ushort) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(float) * mat));
    const float _r = 1.0f / float(USHRT_MAX);
    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, in + mat * itt, sizeof(ushort) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<ushort, float>
            << <height, width >> > (d_in, d_out, planarConfig, _r);

        CUDA_CALL(cudaMemcpyAsync(out + mat * itt, d_out, sizeof(float) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}
void convertS16toF32H2H(short* in, float* out, const int width,
    const int height, const int maxFrame,
    const int planarConfig) {
    short* d_in;
    float* d_out;
    int _color = (planarConfig == -1) ? 1 : 3;
    const int mat = width * height * _color;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(short) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(float) * mat));
    const float _r = 1.0f / float(SHRT_MAX);
    for (auto itt = 0; itt < maxFrame; itt++) {
        CUDA_CALL(cudaMemcpyAsync(d_in, in + mat * itt, sizeof(short) * mat,
            cudaMemcpyHostToDevice));

        // For 32-bits per threads
        device_convertType<short, float>
            << <height, width >> > (d_in, d_out, planarConfig, _r);

        CUDA_CALL(cudaMemcpyAsync(out + mat * itt, d_out, sizeof(float) * mat,
            cudaMemcpyDeviceToHost));
    }
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void convertU16toF32(uchar* in, float* out, const int width, const int height,
    const int perpixel) {
    ushort* d_in;
    float* d_out;
    const int mat = width * height * perpixel;
    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(ushort) * mat));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(float) * mat));

    CUDA_CALL(cudaMemcpyAsync(d_in, in, sizeof(uchar) * mat * 2,
        cudaMemcpyHostToDevice));

    // For 32-bits per threads
    device_convert2F32<ushort> << <height, width >> > (d_in, d_out, perpixel);

    CUDA_CALL(
        cudaMemcpyAsync(out, d_out, sizeof(float) * mat, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

template <typename T>
void upsampling(T* a, T* b, const int iSrcWidth, const int iSrcHeight,
    const int iDestWidth, const int iDestHeight) {
    /* const type_info& _test0 = typeid(a);
     const type_info& _test1 = typeid(int*);
     const type_info& _test2 = typeid(float*);
     const type_info& _test3 = typeid(double*);

     if (_test0.hash_code() == _test1.hash_code())
         upsampling_int(a, b, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
     else if (_test0.hash_code() == _test2.hash_code())
         upsampling_float(a, b, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
     else if (_test0.hash_code() == _test3.hash_code())
         upsampling_double(a, b, iSrcWidth, iSrcHeight, iDestWidth,
     iDestHeight);*/
}
void gpu_upsampling_half(__half* h_in, __half* h_out, const int iSrcWidth,
    const int iSrcHeight, const int iDestWidth,
    const int iDestHeight) {
    /*const float fStride_w = std::ceilf(float(iDestWidth) / iSrcWidth);
    const float fStride_h = std::ceilf(float(iDestHeight) / iSrcHeight);
    int* d_in, * d_out;

    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(int) * (iSrcWidth * iSrcHeight)));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(int) * (iDestWidth *
    iDestHeight)));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, sizeof(int) * (iSrcWidth * iSrcHeight),
    cudaMemcpyHostToDevice));

    dim3 blocks = dim3(8, 8);
    dim3 grids = dim3(iDestWidth / blocks.x, iDestHeight / blocks.y);

    device_upsampling<int> << < grids, blocks >> > (d_in, d_out, iSrcWidth,
    iSrcHeight, fStride_w, fStride_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, sizeof(int) * (iDestWidth *
    iDestHeight), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));*/
}
void gpu_upsampling_int(int* h_in, int* h_out, const int iSrcWidth,
    const int iSrcHeight, const int iDestWidth,
    const int iDestHeight) {
    const float fStride_w = std::ceilf(float(iDestWidth) / iSrcWidth);
    const float fStride_h = std::ceilf(float(iDestHeight) / iSrcHeight);
    int* d_in, * d_out;

    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(int) * (iSrcWidth * iSrcHeight)));
    CUDA_CALL(
        cudaMalloc((void**)&d_out, sizeof(int) * (iDestWidth * iDestHeight)));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, sizeof(int) * (iSrcWidth * iSrcHeight),
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(8, 8);
    dim3 grids = dim3(iDestWidth / blocks.x, iDestHeight / blocks.y);

    device_upsampling<int> << <grids, blocks >> > (d_in, d_out, iSrcWidth, iSrcHeight,
        fStride_w, fStride_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out,
        sizeof(int) * (iDestWidth * iDestHeight),
        cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}
void gpu_upsampling_float(float* h_in, float* h_out, const int iSrcWidth,
    const int iSrcHeight, const int iDestWidth,
    const int iDestHeight) {
    const float fStride_w = std::ceilf(float(iDestWidth) / iSrcWidth);
    const float fStride_h = std::ceilf(float(iDestHeight) / iSrcHeight);
    float* d_in, * d_out;

    CUDA_CALL(
        cudaMalloc((void**)&d_in, sizeof(float) * (iSrcWidth * iSrcHeight)));
    CUDA_CALL(
        cudaMalloc((void**)&d_out, sizeof(float) * (iDestWidth * iDestHeight)));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in,
        sizeof(float) * (iSrcWidth * iSrcHeight),
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(8, 8);
    dim3 grids = dim3(iDestWidth / blocks.x, iDestHeight / blocks.y);

    device_upsampling<float> << <grids, blocks >> > (d_in, d_out, iSrcWidth,
        iSrcHeight, fStride_w, fStride_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out,
        sizeof(float) * (iDestWidth * iDestHeight),
        cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}
void gpu_upsampling_double(double* h_in, double* h_out, const int iSrcWidth,
    const int iSrcHeight, const int iDestWidth,
    const int iDestHeight) {
    const float fStride_w = std::ceilf(float(iDestWidth) / iSrcWidth);
    const float fStride_h = std::ceilf(float(iDestHeight) / iSrcHeight);
    double* d_in, * d_out;

    CUDA_CALL(
        cudaMalloc((void**)&d_in, sizeof(double) * (iSrcWidth * iSrcHeight)));
    CUDA_CALL(
        cudaMalloc((void**)&d_out, sizeof(double) * (iDestWidth * iDestHeight)));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in,
        sizeof(double) * (iSrcWidth * iSrcHeight),
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(8, 8);
    dim3 grids = dim3(iDestWidth / blocks.x, iDestHeight / blocks.y);

    device_upsampling<double> << <grids, blocks >> > (
        d_in, d_out, iSrcWidth, iSrcHeight, fStride_w, fStride_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out,
        sizeof(double) * (iDestWidth * iDestHeight),
        cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
}

void gpu_convolve1d_half(__half* h_in, __half* h_out, __half* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelSize) {}
void gpu_convolve1d_int(int* h_in, int* h_out, float* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelSize) {
    /*int* d_in, * d_out;
    float* d_kernel;
    const int iTotalbyte = sizeof(int) * (iImgWidth * iImgHeight);
    const int iKernelHalf = int(iKernelSize / 2);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * iKernelSize));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel, sizeof(float) * iKernelSize,
    cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    device_convolve1d<int> << < grids, blocks >> > (d_in, d_out,
    d_kernel,iImgWidth, iKernelHalf);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_kernel));*/
}
void gpu_convolve1d_float(float* h_in, float* h_out, float* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelSize) {
    float* d_in, * d_out;
    float* d_kernel;
    const int iTotalbyte = sizeof(float) * (iImgWidth * iImgHeight);
    const int iKernelHalf = int(iKernelSize / 2);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * iKernelSize));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel, sizeof(float) * iKernelSize,
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    device_convolve1d<float>
        << <grids, blocks >> > (d_in, d_out, d_kernel, iImgWidth, iKernelHalf);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_kernel));
}
void gpu_convolve1d_double(double* h_in, double* h_out, double* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelSize) {
    double* d_in, * d_out;
    double* d_kernel;
    const int iTotalbyte = sizeof(double) * (iImgWidth * iImgHeight);
    const int iKernelHalf = int(iKernelSize / 2);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(double) * iKernelSize));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel, sizeof(double) * iKernelSize,
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    device_convolve1d<double>
        << <grids, blocks >> > (d_in, d_out, d_kernel, iImgWidth, iKernelHalf);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_kernel));
}

void gpu_convolve2d_half(__half* h_in, __half* h_out, float* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelWidth, const int iKernelHeight) {
    //__half* d_in, * d_out;
    // float* d_kernel;
    // const int iTotalbyte = sizeof(__half) * (iImgWidth * iImgHeight);
    // const int iKernelHalf_w = int(iKernelWidth / 2);
    // const int iKernelHalf_h = int(iKernelHeight / 2);

    // CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    // CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    // CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * iKernelWidth *
    // iKernelHeight));

    // CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    // CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel, sizeof(float) * iKernelWidth *
    // iKernelHeight, cudaMemcpyHostToDevice));

    // dim3 blocks = dim3(32, 32);
    // dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    ////device_convolve2d<__half> << < grids, blocks >> > (d_in, d_out, d_kernel,
    ///iImgWidth, iImgHeight, iKernelHalf_w, iKernelHalf_h);

    // CUDA_CALL(cudaDeviceSynchronize());

    // CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte,
    // cudaMemcpyDeviceToHost));

    // CUDA_CALL(cudaFree(d_in));
    // CUDA_CALL(cudaFree(d_out));
    // CUDA_CALL(cudaFree(d_kernel));
}

void gpu_convolve2d_int(int* h_in, int* h_out, float* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelWidth, const int iKernelHeight) {
    /* int* d_in, * d_out;
     float* d_kernel;
     const int iTotalbyte = sizeof(int) * (iImgWidth * iImgHeight);
     const int iKernelHalf_w = int(iKernelWidth / 2);
     const int iKernelHalf_h = int(iKernelHeight / 2);

     CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
     CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
     CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * iKernelWidth*
     iKernelHeight));

     CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel, sizeof(float) * iKernelWidth *
     iKernelHeight, cudaMemcpyHostToDevice));

     dim3 blocks = dim3(32, 32);
     dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

     device_convolve2d<int> << < grids, blocks >> > (d_in, d_out, d_kernel,
     iImgWidth, iImgHeight, iKernelHalf_w, iKernelHalf_h);

     CUDA_CALL(cudaDeviceSynchronize());

     CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

     CUDA_CALL(cudaFree(d_in));
     CUDA_CALL(cudaFree(d_out));
     CUDA_CALL(cudaFree(d_kernel));*/
}

void gpu_convolve2d_float(float* h_in, float* h_out, float* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelWidth, const int iKernelHeight) {
    float* d_in, * d_out;
    float* d_kernel;
    const int iTotalbyte = sizeof(float) * (iImgWidth * iImgHeight);
    const int iKernelHalf_w = int(iKernelWidth / 2);
    const int iKernelHalf_h = int(iKernelHeight / 2);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_kernel,
        sizeof(float) * iKernelWidth * iKernelHeight));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel,
        sizeof(float) * iKernelWidth * iKernelHeight,
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    device_convolve2d<float> << <grids, blocks >> > (d_in, d_out, d_kernel, iImgWidth,
        iImgHeight, iKernelHalf_w,
        iKernelHalf_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_kernel));
}
void gpu_convolve2d_double(double* h_in, double* h_out, double* kernel,
    const int iImgWidth, const int iImgHeight,
    const int iKernelWidth, const int iKernelHeight) {
    double* d_in, * d_out;
    double* d_kernel;
    const int iTotalbyte = sizeof(double) * (iImgWidth * iImgHeight);
    const int iKernelHalf_w = int(iKernelWidth / 2);
    const int iKernelHalf_h = int(iKernelHeight / 2);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_out, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_kernel,
        sizeof(double) * iKernelWidth * iKernelHeight));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyAsync(d_kernel, kernel,
        sizeof(double) * iKernelWidth * iKernelHeight,
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(iImgWidth / blocks.x, iImgHeight / blocks.y);

    device_convolve2d<double> << <grids, blocks >> > (d_in, d_out, d_kernel, iImgWidth,
        iImgHeight, iKernelHalf_w,
        iKernelHalf_h);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, iTotalbyte, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_kernel));
}

void gpu_derivative2d_int(int* h_in, int* h_outx, int* h_outy, const int iWidth,
    const int iHeight) {
    int* d_in, * d_outx, * d_outy;
    const int iTotalbyte = sizeof(int) * (iWidth * iHeight);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outx, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outy, iTotalbyte));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));

    cudaStream_t cs[2];

    CUDA_CALL(cudaStreamCreate(&cs[0]));
    CUDA_CALL(cudaStreamCreate(&cs[1]));

    dim3 blocks1 = dim3(1, iHeight);
    dim3 grids1 = dim3(iWidth, 1);

    dim3 blocks2 = dim3(iWidth, 1);
    dim3 grids2 = dim3(1, iHeight);

    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            /*int _shrdDim1[2] = { 1,iHeight + 2 };
            CUDA_CALL(cudaMemcpyToSymbol(&shrdmx1, &_shrdDim1,
            sizeof(_shrdDim1)*2));*/

            device_derivativeX<int>
                << <grids1, blocks1, sizeof(float)* (iHeight + 2), cs[i] >> > (
                    d_in, d_outx, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outx, d_outx, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
        else {
            /*int _shrdDim1[2] = { iWidth + 2,1 };
            CUDA_CALL(cudaMemcpyToSymbol(&shrdmx2, &_shrdDim1, sizeof(_shrdDim1) *
            2));*/

            device_derivativeY<int>
                << <grids2, blocks2, sizeof(float)* (iWidth + 2), cs[i] >> > (
                    d_in, d_outy, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outy, d_outy, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
    }

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_outx));
    CUDA_CALL(cudaFree(d_outy));

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
}

void gpu_derivative2d_float(float* h_in, float* h_outx, float* h_outy,
    const int iWidth, const int iHeight) {
    float* d_in, * d_outx, * d_outy;
    const int iTotalbyte = sizeof(float) * (iWidth * iHeight);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outx, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outy, iTotalbyte));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));

    cudaStream_t cs[2];

    CUDA_CALL(cudaStreamCreate(&cs[0]));
    CUDA_CALL(cudaStreamCreate(&cs[1]));

    dim3 grids1 = dim3(1, iHeight);
    dim3 blocks1 = dim3(iWidth, 1);

    dim3 grids2 = dim3(iWidth, 1);
    dim3 blocks2 = dim3(1, iHeight);
    // Need to more script for compatibility to shrd mem
    for (auto i = 0; i < 2; i++) {
        if (i == 0) {
            device_derivativeX<float>
                << <grids1, blocks1, sizeof(float)* (iHeight + 2), cs[i] >> > (
                    d_in, d_outx, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outx, d_outx, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
        else {
            device_derivativeY<float>
                << <grids2, blocks2, sizeof(float)* (iWidth + 2), cs[i] >> > (
                    d_in, d_outy, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outy, d_outy, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_outx));
    CUDA_CALL(cudaFree(d_outy));

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
}

void gpu_derivative2d_double(double* h_in, double* h_outx, double* h_outy,
    const int iWidth, const int iHeight) {
    double* d_in, * d_outx, * d_outy;
    const int iTotalbyte = sizeof(double) * (iWidth * iHeight);

    CUDA_CALL(cudaMalloc((void**)&d_in, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outx, iTotalbyte));
    CUDA_CALL(cudaMalloc((void**)&d_outy, iTotalbyte));

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, iTotalbyte, cudaMemcpyHostToDevice));

    cudaStream_t cs[2];

    CUDA_CALL(cudaStreamCreate(&cs[0]));
    CUDA_CALL(cudaStreamCreate(&cs[1]));

    dim3 blocks1 = dim3(1, iHeight);
    dim3 grids1 = dim3(iWidth, 1);

    dim3 blocks2 = dim3(iWidth, 1);
    dim3 grids2 = dim3(1, iHeight);

    for (auto i = 0; i < 2; i++) {
        if (i == 0) {
            device_derivativeX<double>
                << <grids1, blocks1, sizeof(double)* (iHeight + 2), cs[i] >> > (
                    d_in, d_outx, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outx, d_outx, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
        else {
            device_derivativeY<double>
                << <grids2, blocks2, sizeof(double)* (iWidth + 2), cs[i] >> > (
                    d_in, d_outy, iWidth, iHeight);
            CUDA_CALL(cudaMemcpyAsync(h_outy, d_outy, iTotalbyte,
                cudaMemcpyDeviceToHost, cs[i]));
        }
    }

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_outx));
    CUDA_CALL(cudaFree(d_outy));

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
}

void gpu_diffusefilt_half(__half* h_in, __half* h_out, const int width,
    const int height, const int iter, const float dt,
    const float rho, const float sigma, const float alpha,
    const float C, const int mode, const int dftype) {
    // gpu_diffusefilt(in, out, width, height, iter, dt, rho, sigma, alpha, C,
    // mode, dftype,16);
}

void gpu_diffusefilt_float(float* h_in, float* h_out, const int width,
    const int height, const int iter, const float dt,
    const float rho, const float sigma,
    const float alpha, const float C, const int mode,
    const int dftype) {

    /*float* h_test;
    CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(float) * width * height));*/

    // in&out
    float* d_in, * d_out;
    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, totalByte, cudaMemcpyHostToDevice));

    // fp32 buffer
    const int bufferByte = sizeof(float) * width * height;
    float* d_usigma;
    float* d_ux, * d_uy;
    float* d_xx, * d_xy, * d_yy;
    float* d_mu1, * d_mu2, * d_ix, * d_iy;

    float* d_kernel, * d_kernel2;
    float* d_kernelx, * d_kernely;
    float* d_kernelMxx, * d_kernelMxy, * d_kernelMyy, * d_kernelMx, * d_kernelMy;

    int kernelSize = sigma * 4;
    int kernelSize2 = rho * 6;

    // blocks and threads

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(width / blocks.x, height / blocks.y);

    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    const int numStream = (dftype > 1) ? 4 : 3;

    cudaStream_t* cs = nullptr;
    cs = new cudaStream_t[numStream];
    float* h_kernel = getKernel1d<float>(sigma, kernelSize);
    float* h_kernel2 = getKernel1d<float>(rho, kernelSize2);
    float* h_kernelx, * h_kernely;
    float* h_kernelMxx, * h_kernelMxy, * h_kernelMyy, * h_kernelMx, * h_kernelMy;

    int fwh, fhh;

    const size_t sharedMemSize = width * sizeof(float);

    // Allocation
    CUDA_CALL(cudaMalloc((void**)&d_usigma, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_ux, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_uy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_xx, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_xy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_yy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_mu1, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_mu2, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_ix, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_iy, bufferByte));

    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * kernelSize));
    CUDA_CALL(cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernelSize,
        cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void**)&d_kernel2, sizeof(float) * kernelSize2));
    CUDA_CALL(cudaMemcpy(d_kernel2, h_kernel2, sizeof(float) * kernelSize2,
        cudaMemcpyHostToDevice));

    for (int i = 0; i < numStream; i++) {
        CUDA_CALL(cudaStreamCreate(&cs[i]));
    }

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    // Normalize
    float _min = 0.0f, _max = 0.0f;

    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[1] >> > (d_in, d_ux, mn);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[1] >> > (d_ux, d_xx, mn);   
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[2] >> > (d_in, d_uy, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[2] >> > (d_uy, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));

    device_simpleNorm2<float> << <width, height >> > (d_in, d_in, _min, _max);
    CUDA_CALL(cudaDeviceSynchronize());
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    // Kernel setting
    if (dftype > 1) {
        h_kernelx = getDerivativeKernel<float>('x', mode);
        h_kernely = getDerivativeKernel<float>('y', mode);

        // mode 1 case
        fwh = 1, fhh = 1;
        CUDA_CALL(cudaMalloc((void**)&d_kernelx, sizeof(float) * 9));
        CUDA_CALL(cudaMalloc((void**)&d_kernely, sizeof(float) * 9));
        CUDA_CALL(cudaMemcpyAsync(d_kernelx, h_kernelx, sizeof(float) * 9,
            cudaMemcpyHostToDevice, cs[0]));
        CUDA_CALL(cudaMemcpyAsync(d_kernely, h_kernely, sizeof(float) * 9,
            cudaMemcpyHostToDevice, cs[1]));
        CUDA_CALL(cudaStreamSynchronize(cs[0]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));

        if (dftype == 3) {
            h_kernelMxx = getKernelOpt<float>("xx");
            h_kernelMxy = getKernelOpt<float>("xy");
            h_kernelMyy = getKernelOpt<float>("yy");
            h_kernelMx = getKernelOpt<float>("x_");
            h_kernelMy = getKernelOpt<float>("y_");

            CUDA_CALL(cudaMalloc((void**)&d_kernelMxx, sizeof(float) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMxy, sizeof(float) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMyy, sizeof(float) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMx, sizeof(float) * 9));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMy, sizeof(float) * 9));

            // for (int i = 0; i <25; i++) {

            //    qDebug() << h_kernelMxx[i]<<'\t'<< h_kernelMxy[i] << '\t'<<
            //    h_kernelMyy[i] << '\t';
            //}
            CUDA_CALL(cudaMemcpyAsync(d_kernelMxx, h_kernelMxx, sizeof(float) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMxy, h_kernelMxy, sizeof(float) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMyy, h_kernelMyy, sizeof(float) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMx, h_kernelMx, sizeof(float) * 9,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMy, h_kernelMy, sizeof(float) * 9,
                cudaMemcpyHostToDevice));
        }
    }

    CUDA_CALL(cudaDeviceSynchronize());
    const int halfkernel = int(kernelSize / 2);
    // iter start
    for (int _iter = 0; _iter < iter; _iter++) {
        // device_convolve1d<float> << < grids, blocks >> > (d_in, d_out, d_kernel,
        // width, int(kernelSize / 2), false);
        CUDA_CALL(cudaDeviceSynchronize());
        //// 1D gaussian kernel
        device_convolve1d_shared<float>
            << <grids1, blocks1, sizeof(float)* (width + halfkernel * 2) >> > (
                d_in, d_out, d_kernel, width, halfkernel, 1, false);
        // device_convolve1d_shared_X<float, CONVXN, CONVXM> << <dim3(width /
        // CONVXN, height/ CONVXM), dim3(CONVXN, CONVXM), sizeof(float)* ((CONVXM +
        // halfkernel * 2) * CONVXN + kernelSize) >> > (d_in, d_out, d_kernel,
        // halfkernel, width, height);
        CUDA_CALL(cudaDeviceSynchronize());
        // dim3 g1 = dim3(width / CONVYN, height / CONVYM);
        // dim3 b1 = dim3(CONVYN, CONVYM);
        // device_convolve1d_shared_Y<float, CONVYN, CONVYM> << <g1,b1,
        // sizeof(float)* ((CONVYN + halfkernel * 2)* CONVYM + kernelSize) >> >
        // (d_out, d_usigma, d_kernel, halfkernel, width, height);
        device_convolve1d_shared<float>
            << <grids2, blocks2, sizeof(float)* (width + halfkernel * 2) >> > (
                d_out, d_usigma, d_kernel, width, halfkernel, 1, true);
        // CUDA_CALL(cudaDeviceSynchronize());

        // 1D gaussian kernel

        // device_convolve1d<float> << < grids, blocks >> > (d_out, d_usigma,
        // d_kernel, width, int(kernelSize / 2), true);
        CUDA_CALL(cudaDeviceSynchronize());
        if (_iter == 0) {
            // deviceArraySumPrint(d_usigma, __LINE__, width, height);
        }

        /*device_convolutionRowGPU<float><<< grids, blocks,sizeof(float)*(blocks.y *
        (blocks.x + kernelSize)) >> > (d_in,d_out, d_kernel, width,height,
        int(kernelSize / 2)); device_convolutionColGPU<float> << < grids, blocks,
        sizeof(float)* (blocks.x* (blocks.y + kernelSize)) >> > (d_out, d_usigma,
        d_kernel, width, height, int(kernelSize / 2));
        CUDA_CALL(cudaDeviceSynchronize());*/
        if (dftype > 1) {
            if (mode == 1) {
                device_convolve2d<float> << <grids, blocks, 0, cs[0] >> > (
                    d_usigma, d_ux, d_kernelx, width, height, fwh, fhh);
                device_convolve2d<float> << <grids, blocks, 0, cs[1] >> > (
                    d_usigma, d_uy, d_kernely, width, height, fwh, fhh);
            }
        }
        else {

            for (auto i = 0; i < 2; i++) {
                if (i == 0) {
                    device_derivativeX<float>
                        << <grids2, blocks2, sizeof(float)* (height + 2), cs[i] >> > (
                            d_usigma, d_ux, width, height);
                }
                else {
                    device_derivativeY<float>
                        << <grids1, blocks1, sizeof(float)* (width + 2), cs[i] >> > (
                            d_usigma, d_uy, width, height);
                }
            }
        }
        CUDA_CALL(cudaStreamSynchronize(cs[0]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));
        if (_iter == 0) {
            // deviceArraySumPrint(d_ux, __LINE__, width, height);
            // deviceArraySumPrint(d_uy, __LINE__, width, height);
        }

        const int halfkernel2 = int(kernelSize2 / 2);

        device_simpleMul<float>
            << <grids, blocks, 0, cs[0] >> > (d_ux, d_ux, d_xx, width);
        // using buffer to d_mu1
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[0] >> > (
                d_xx, d_mu1, d_kernel2, width, halfkernel2, false);
        // CUDA_CALL(cudaStreamSynchronize(cs[0]));
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[0] >> > (
                d_mu1, d_xx, d_kernel2, height, halfkernel2, true);

        device_simpleMul<float>
            << <grids, blocks, 0, cs[1] >> > (d_ux, d_uy, d_xy, width);
        // using buffer to d_mu2
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[1] >> > (
                d_xy, d_mu2, d_kernel2, width, halfkernel2, false);
        // CUDA_CALL(cudaStreamSynchronize(cs[1]));
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[1] >> > (
                d_mu2, d_xy, d_kernel2, height, halfkernel2, true);

        device_simpleMul<float>
            << <grids, blocks, 0, cs[2] >> > (d_uy, d_uy, d_yy, width);
        // using buffer to d_ix
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[2] >> > (
                d_yy, d_ix, d_kernel2, width, halfkernel2, false);
        // CUDA_CALL(cudaStreamSynchronize(cs[2]));
        device_convolve1d_shared<float>
            << <grids, blocks, sizeof(float)* (width + halfkernel2 * 2), cs[2] >> > (
                d_ix, d_yy, d_kernel2, height, halfkernel2, true);

        CUDA_CALL(cudaDeviceSynchronize());
        // if (_iter == 0) {
        //     //deviceArraySumPrint(d_xx, __LINE__, width, height);
        //     //deviceArraySumPrint(d_xy, __LINE__, width, height);
        //     //deviceArraySumPrint(d_yy, __LINE__, width, height);
        // }
        device_eigen2Image<float>
            << <grids, blocks >> > (d_xx, d_xy, d_yy, d_mu1, d_mu2, d_ix, d_iy, width);
        CUDA_CALL(cudaDeviceSynchronize());
        // if (_iter == 0) {
        //     //deviceArraySumPrint(d_mu1, __LINE__, width, height);
        //     //deviceArraySumPrint(d_mu2, __LINE__, width, height);
        //     //deviceArraySumPrint(d_ix, __LINE__, width, height);
        //     //deviceArraySumPrint(d_iy, __LINE__, width, height);
        // }
        device_diffusionTensor<float> << <grids, blocks >> > (
            d_mu1, d_mu2, d_ix, d_iy, d_xx, d_xy, d_yy, alpha, C, width);
        CUDA_CALL(cudaDeviceSynchronize());
        // if (_iter == 0) {
        //     //deviceArraySumPrint(d_xx, __LINE__, width, height);
        //     //deviceArraySumPrint(d_xy, __LINE__, width, height);
        //     //deviceArraySumPrint(d_yy, __LINE__, width, height);
        // }
        switch (dftype) {
        case 3: // opt
        {
            device_diffuseOptimizationS1<float> << <grids, blocks >> > (
                d_in, d_xx, d_xy, d_yy, d_mu1, d_kernelMx, d_kernelMy, width, height);
            CUDA_CALL(cudaDeviceSynchronize());
            if (_iter == 0) {
                // deviceArraySumPrint(d_xx, __LINE__, width, height);
                // deviceArraySumPrint(d_xy, __LINE__, width, height);
                // deviceArraySumPrint(d_yy, __LINE__, width, height);
                // deviceArraySumPrint(d_mu1, __LINE__, width, height);
            }
            device_diffuseOptimizationS2<float>
                << <grids, blocks >> > (d_in, d_xx, d_xy, d_yy, d_mu1, d_out, d_kernelMxx,
                    d_kernelMxy, d_kernelMyy, dt, width, height);
            CUDA_CALL(cudaDeviceSynchronize());
            if (_iter == 0) {
                // deviceArraySumPrint(d_in, __LINE__, width, height);
                // deviceArraySumPrint(d_xx, __LINE__, width, height);
                // deviceArraySumPrint(d_xy, __LINE__, width, height);
                // deviceArraySumPrint(d_yy, __LINE__, width, height);
                // deviceArraySumPrint(d_mu1, __LINE__, width, height);
                // deviceArraySumPrint(d_out, __LINE__, width, height);
            }
        } break;
        case 2: // rotation
        {

            device_convolve2d<float> << <grids, blocks, 0, cs[0] >> > (
                d_in, d_ux, d_kernelx, width, height, fwh, fhh);
            device_convolve2d<float> << <grids, blocks, 0, cs[1] >> > (
                d_in, d_uy, d_kernely, width, height, fwh, fhh);
            CUDA_CALL(cudaDeviceSynchronize());

            device_diffuseRotationS1<float> << <grids, blocks >> > (
                d_xx, d_xy, d_yy, d_ux, d_uy, d_mu1, d_mu2, width, height);
            CUDA_CALL(cudaDeviceSynchronize());
            device_diffuseRotationS2<float>
                << <grids, blocks >> > (d_in, d_mu1, d_mu2, d_out, d_kernelx, d_kernely,
                    fwh, fhh, dt, width, height);

        } break;
        case 1: // NoneNegative
            device_diffuseNoneNegative<float>
                << <grids1, blocks1, sizeof(float)* (width + 2) * 3 * 4 >> > (
                    d_in, d_xx, d_xy, d_yy, d_out, dt, width, height);
            break;
        case 0: // standard
        default:
            device_diffuseStandard<float>
                << <grids1, blocks1, sizeof(float)* (width + 2) * 3 * 4 >> > (
                    d_in, d_xx, d_xy, d_yy, d_out, dt, width, height);
            break;
        }

        CUDA_CALL(cudaDeviceSynchronize());

        if (_iter != iter - 1)
            //{
            //    CUDA_CALL(cudaMemcpyAsync(d_out, d_out, totalByte,
            //    cudaMemcpyDeviceToDevice));
            //
            //}
            // else
        {
            CUDA_CALL(
                cudaMemcpyAsync(d_in, d_out, totalByte, cudaMemcpyDeviceToDevice));
        }
        CUDA_CALL(cudaDeviceSynchronize());
    }

    // device_reduction<512, float> << <width, height, sharedMemSize, cs[1] >> >
    // (d_out, d_ux, mn); device_reduction<1, float> << <1, width, sharedMemSize,
    // cs[1] >> > (d_ux, d_xx, mn); device_reduction<512, float> << <width,
    // height, sharedMemSize, cs[2] >> > (d_out, d_uy, mx); device_reduction<1,
    // float> << <1, width, sharedMemSize, cs[2] >> > (d_uy, d_yy, mx);
    //
    // CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
    // cudaMemcpyDeviceToHost, cs[1])); CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0],
    // sizeof(float), cudaMemcpyDeviceToHost, cs[2]));
    // CUDA_CALL(cudaStreamSynchronize(cs[1]));
    // CUDA_CALL(cudaStreamSynchronize(cs[2]));
    //
    ////device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min,
    ///_max, USHRT_MAX);
    // device_simpleScale<float> << <width, height >> > (d_out, d_out,
    // USHRT_MAX,width); CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto i = 0; i < numStream; i++)
        (cuStreamDestroy_v2(cs[i]));

    if (dftype > 1) {
        CUDA_CALL(cudaFree(d_kernelx));
        CUDA_CALL(cudaFree(d_kernely));
        delete[] h_kernelx;
        delete[] h_kernely;

        if (dftype == 3) {
            CUDA_CALL(cudaFree(d_kernelMxx));
            CUDA_CALL(cudaFree(d_kernelMxy));
            CUDA_CALL(cudaFree(d_kernelMyy));
            CUDA_CALL(cudaFree(d_kernelMx));
            CUDA_CALL(cudaFree(d_kernelMy));

            delete[] h_kernelMxx;
            delete[] h_kernelMxy;
            delete[] h_kernelMyy;
            delete[] h_kernelMx;
            delete[] h_kernelMy;
        }
    }

    delete[] cs;
    CUDA_CALL(cudaFree(d_kernel2));
    CUDA_CALL(cudaFree(d_kernel));
    CUDA_CALL(cudaFree(d_usigma));
    CUDA_CALL(cudaFree(d_ux));
    CUDA_CALL(cudaFree(d_uy));
    CUDA_CALL(cudaFree(d_xx));
    CUDA_CALL(cudaFree(d_xy));
    CUDA_CALL(cudaFree(d_yy));
    CUDA_CALL(cudaFree(d_mu1));
    CUDA_CALL(cudaFree(d_mu2));
    CUDA_CALL(cudaFree(d_ix));
    CUDA_CALL(cudaFree(d_iy));
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    delete[] h_kernel;
    delete[] h_kernel2;

    // CUDA_CALL(cudaFreeHost(h_test));
}

void gpu_diffusefilt_double(double* h_in, double* h_out, const int width,
    const int height, const int iter, const float dt,
    const float rho, const float sigma,
    const float alpha, const float C, const int mode,
    const int dftype) {

    CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    /*double* h_test;
    CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(double) * width * height));*/

    // in&out
    double* d_in, * d_out;
    const int totalByte = sizeof(double) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, totalByte, cudaMemcpyHostToDevice));

    // fp64 buffer
    const int bufferByte = sizeof(double) * width * height;
    double* d_usigma;
    double* d_ux, * d_uy;
    double* d_xx, * d_xy, * d_yy;
    double* d_mu1, * d_mu2, * d_ix, * d_iy;

    double* d_kernel, * d_kernel2;
    double* d_kernelx, * d_kernely;
    double* d_kernelMxx, * d_kernelMxy, * d_kernelMyy, * d_kernelMx, * d_kernelMy;

    int kernelSize = sigma * 4;
    int kernelSize2 = rho * 6;

    // blocks and threads

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(width / blocks.x, height / blocks.y);

    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    const int numStream = (dftype > 1) ? 4 : 3;

    cudaStream_t* cs = nullptr;
    cs = new cudaStream_t[numStream];
    double* h_kernel = getKernel1d<double>(sigma, kernelSize);
    double* h_kernel2 = getKernel1d<double>(rho, kernelSize2);
    double* h_kernelx, * h_kernely;
    double* h_kernelMxx, * h_kernelMxy, * h_kernelMyy, * h_kernelMx, * h_kernelMy;

    int fwh, fhh;

    const size_t sharedMemSize = width * sizeof(double);

    // Allocation
    CUDA_CALL(cudaMalloc((void**)&d_usigma, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_ux, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_uy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_xx, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_xy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_yy, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_mu1, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_mu2, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_ix, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_iy, bufferByte));

    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(double) * kernelSize));
    CUDA_CALL(cudaMemcpy(d_kernel, h_kernel, sizeof(double) * kernelSize,
        cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void**)&d_kernel2, sizeof(double) * kernelSize2));
    CUDA_CALL(cudaMemcpy(d_kernel2, h_kernel2, sizeof(double) * kernelSize2,
        cudaMemcpyHostToDevice));

    for (int i = 0; i < numStream; i++) {
        CUDA_CALL(cudaStreamCreate(&cs[i]));
    }

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    // Normalize
    double _min = 0.0, _max = 0.0;

    device_reduction<512, double>
        << <width, height, sharedMemSize, cs[1] >> > (d_in, d_ux, mn);
    device_reduction<1, double>
        << <1, width, sharedMemSize, cs[1] >> > (d_ux, d_xx, mn);
    device_reduction<512, double>
        << <width, height, sharedMemSize, cs[2] >> > (d_in, d_uy, mx);
    device_reduction<1, double>
        << <1, width, sharedMemSize, cs[2] >> > (d_uy, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(double),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(double),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));

    device_simpleNorm2<double> << <width, height >> > (d_in, d_in, _min, _max);
    CUDA_CALL(cudaDeviceSynchronize());

    // Kernel setting
    if (dftype > 1) {
        h_kernelx = getDerivativeKernel<double>('x', mode);
        h_kernely = getDerivativeKernel<double>('y', mode);

        // mode 1 case
        fwh = 1, fhh = 1;
        CUDA_CALL(cudaMalloc((void**)&d_kernelx, sizeof(double) * 9));
        CUDA_CALL(cudaMalloc((void**)&d_kernely, sizeof(double) * 9));
        CUDA_CALL(cudaMemcpyAsync(d_kernelx, h_kernelx, sizeof(double) * 9,
            cudaMemcpyHostToDevice, cs[0]));
        CUDA_CALL(cudaMemcpyAsync(d_kernely, h_kernely, sizeof(double) * 9,
            cudaMemcpyHostToDevice, cs[1]));
        CUDA_CALL(cudaStreamSynchronize(cs[0]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));

        if (dftype == 3) {
            h_kernelMxx = getKernelOpt<double>("xx");
            h_kernelMxy = getKernelOpt<double>("xy");
            h_kernelMyy = getKernelOpt<double>("yy");
            h_kernelMx = getKernelOpt<double>("x_");
            h_kernelMy = getKernelOpt<double>("y_");

            CUDA_CALL(cudaMalloc((void**)&d_kernelMxx, sizeof(double) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMxy, sizeof(double) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMyy, sizeof(double) * 25));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMx, sizeof(double) * 9));
            CUDA_CALL(cudaMalloc((void**)&d_kernelMy, sizeof(double) * 9));

            CUDA_CALL(cudaMemcpyAsync(d_kernelMxx, h_kernelMxx, sizeof(double) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMxy, h_kernelMxy, sizeof(double) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMyy, h_kernelMyy, sizeof(double) * 25,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMx, h_kernelMx, sizeof(double) * 9,
                cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(d_kernelMy, h_kernelMy, sizeof(double) * 9,
                cudaMemcpyHostToDevice));
        }
    }

    // iter start
    for (auto _iter = 0; _iter < iter; _iter++) {
        // 1D gaussian kernel
        device_convolve1d<double> << <grids, blocks >> > (d_in, d_out, d_kernel, width,
            int(kernelSize / 2), false);
        CUDA_CALL(cudaDeviceSynchronize());
        device_convolve1d<double> << <grids, blocks >> > (
            d_out, d_usigma, d_kernel, width, int(kernelSize / 2), true);
        CUDA_CALL(cudaDeviceSynchronize());

        if (dftype > 1) {
            if (mode == 1) {
                device_convolve2d<double> << <grids, blocks, 0, cs[0] >> > (
                    d_usigma, d_ux, d_kernelx, width, height, fwh, fhh);
                device_convolve2d<double> << <grids, blocks, 0, cs[1] >> > (
                    d_usigma, d_uy, d_kernely, width, height, fwh, fhh);
            }
        }
        else {
            for (auto i = 0; i < 2; i++) {
                if (i == 0) {
                    device_derivativeX<double>
                        << <grids2, blocks2, sizeof(double)* (height + 2), cs[i] >> > (
                            d_usigma, d_ux, width, height);
                }
                else {
                    device_derivativeY<double>
                        << <grids1, blocks1, sizeof(double)* (width + 2), cs[i] >> > (
                            d_usigma, d_uy, width, height);
                }
            }
        }
        CUDA_CALL(cudaStreamSynchronize(cs[0]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));

        device_simpleMul<double>
            << <grids, blocks, 0, cs[0] >> > (d_ux, d_ux, d_xx, width);
        // using buffer to d_mu1
        device_convolve1d<double> << <grids, blocks, 0, cs[0] >> > (
            d_xx, d_mu1, d_kernel2, width, int(kernelSize2 / 2), false);
        // CUDA_CALL(cudaStreamSynchronize(cs[0]));
        device_convolve1d<double> << <grids, blocks, 0, cs[0] >> > (
            d_mu1, d_xx, d_kernel2, height, int(kernelSize2 / 2), true);

        device_simpleMul<double>
            << <grids, blocks, 0, cs[1] >> > (d_ux, d_uy, d_xy, width);
        // using buffer to d_mu2
        device_convolve1d<double> << <grids, blocks, 0, cs[1] >> > (
            d_xy, d_mu2, d_kernel2, width, int(kernelSize2 / 2), false);
        // CUDA_CALL(cudaStreamSynchronize(cs[1]));
        device_convolve1d<double> << <grids, blocks, 0, cs[1] >> > (
            d_mu2, d_xy, d_kernel2, height, int(kernelSize2 / 2), true);

        device_simpleMul<double>
            << <grids, blocks, 0, cs[2] >> > (d_uy, d_uy, d_yy, width);
        // using buffer to d_ix
        device_convolve1d<double> << <grids, blocks, 0, cs[2] >> > (
            d_yy, d_ix, d_kernel2, width, int(kernelSize2 / 2), false);
        // CUDA_CALL(cudaStreamSynchronize(cs[2]));
        device_convolve1d<double> << <grids, blocks, 0, cs[2] >> > (
            d_ix, d_yy, d_kernel2, height, int(kernelSize2 / 2), true);

        CUDA_CALL(cudaDeviceSynchronize());

        device_eigen2Image<double>
            << <grids, blocks >> > (d_xx, d_xy, d_yy, d_mu1, d_mu2, d_ix, d_iy, width);
        CUDA_CALL(cudaDeviceSynchronize());

        device_diffusionTensor<double> << <grids, blocks >> > (
            d_mu1, d_mu2, d_ix, d_iy, d_xx, d_xy, d_yy, alpha, C, width);
        CUDA_CALL(cudaDeviceSynchronize());

        switch (dftype) {
        case 3: // opt
        {
            device_diffuseOptimizationS1<double> << <grids, blocks >> > (
                d_in, d_xx, d_xy, d_yy, d_mu1, d_kernelMx, d_kernelMy, width, height);
            CUDA_CALL(cudaDeviceSynchronize());
            device_diffuseOptimizationS2<double>
                << <grids, blocks >> > (d_in, d_xx, d_xy, d_yy, d_mu1, d_out, d_kernelMxx,
                    d_kernelMxy, d_kernelMyy, dt, width, height);
        } break;
        case 2: // rotation
        {

            device_convolve2d<double> << <grids, blocks, 0, cs[0] >> > (
                d_in, d_ux, d_kernelx, width, height, fwh, fhh);
            device_convolve2d<double> << <grids, blocks, 0, cs[1] >> > (
                d_in, d_uy, d_kernely, width, height, fwh, fhh);
            CUDA_CALL(cudaDeviceSynchronize());

            device_diffuseRotationS1<double> << <grids, blocks >> > (
                d_xx, d_xy, d_yy, d_ux, d_uy, d_mu1, d_mu2, width, height);
            CUDA_CALL(cudaDeviceSynchronize());
            device_diffuseRotationS2<double>
                << <grids, blocks >> > (d_in, d_mu1, d_mu2, d_out, d_kernelx, d_kernely,
                    fwh, fhh, dt, width, height);

        } break;
        case 1: // NoneNegative
            device_diffuseNoneNegative<double>
                << <grids1, blocks1, sizeof(double)* (width + 2) * 3 * 4 >> > (
                    d_in, d_xx, d_xy, d_yy, d_out, dt, width, height);
            break;
        case 0: // standard
        default:
            device_diffuseStandard<double>
                << <grids1, blocks1, sizeof(double)* (width + 2) * 3 * 4 >> > (
                    d_in, d_xx, d_xy, d_yy, d_out, dt, width, height);
            break;
        }

        CUDA_CALL(cudaDeviceSynchronize());

        if (_iter == iter - 1) {
            CUDA_CALL(
                cudaMemcpyAsync(d_out, d_out, totalByte, cudaMemcpyDeviceToDevice));

        }
        else {
            CUDA_CALL(
                cudaMemcpyAsync(d_in, d_out, totalByte, cudaMemcpyDeviceToDevice));
        }
        CUDA_CALL(cudaDeviceSynchronize());
    }

    device_reduction<512, double>
        << <width, height, sharedMemSize, cs[1] >> > (d_out, d_ux, mn);
    device_reduction<1, double>
        << <1, width, sharedMemSize, cs[1] >> > (d_ux, d_xx, mn);
    device_reduction<512, double>
        << <width, height, sharedMemSize, cs[2] >> > (d_out, d_uy, mx);
    device_reduction<1, double>
        << <1, width, sharedMemSize, cs[2] >> > (d_uy, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(double),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(double),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));

    // device_simpleNorm2<double> << <width, height >> > (d_out, d_out, _min,
    // _max, USHRT_MAX);
    device_simpleScale<double> << <width, height >> > (d_out, d_out, USHRT_MAX, width);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto i = 0; i < numStream; i++)
        (cuStreamDestroy_v2(cs[i]));

    if (dftype > 1) {
        CUDA_CALL(cudaFree(d_kernelx));
        CUDA_CALL(cudaFree(d_kernely));
        delete[] h_kernelx;
        delete[] h_kernely;

        if (dftype == 3) {
            CUDA_CALL(cudaFree(d_kernelMxx));
            CUDA_CALL(cudaFree(d_kernelMxy));
            CUDA_CALL(cudaFree(d_kernelMyy));
            CUDA_CALL(cudaFree(d_kernelMx));
            CUDA_CALL(cudaFree(d_kernelMy));

            delete[] h_kernelMxx;
            delete[] h_kernelMxy;
            delete[] h_kernelMyy;
            delete[] h_kernelMx;
            delete[] h_kernelMy;
        }
    }

    delete[] cs;
    CUDA_CALL(cudaFree(d_kernel2));
    CUDA_CALL(cudaFree(d_kernel));
    CUDA_CALL(cudaFree(d_usigma));
    CUDA_CALL(cudaFree(d_ux));
    CUDA_CALL(cudaFree(d_uy));
    CUDA_CALL(cudaFree(d_xx));
    CUDA_CALL(cudaFree(d_xy));
    CUDA_CALL(cudaFree(d_yy));
    CUDA_CALL(cudaFree(d_mu1));
    CUDA_CALL(cudaFree(d_mu2));
    CUDA_CALL(cudaFree(d_ix));
    CUDA_CALL(cudaFree(d_iy));
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    delete[] h_kernel;
    delete[] h_kernel2;

    // CUDA_CALL(cudaFreeHost(h_test));
    CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault));
}

std::vector<float> getSigmaDistribution(const float& start, const float& end,
    const float& step) {
    std::vector<float> re;
    if (step > 1) // exponential distribution
    {
        const float _ip = end - start;
        const float _x = logf(end);
        const float _n = logf(start);
        for (int _s = 0; _s < step; _s++) {
            re.push_back(-(_x + (-(_x - _n) / (step - 1) * _s) - end - start));
        }
    }
    else // follow step
    {
        for (auto _s = end; _s >= start; _s -= step) {
            re.push_back(_s);
        }

        if (re.back() != start) {
            re.push_back(start);
        }
    }
    return std::move(re);
}

void gpu_eigenimg_half(__half* in, __half* outlam1, __half* outlam2,
    __half* outIx, __half* outIy, const float sigma,
    const int width, const int height) {}

void gpu_eigenimg_float(float* in, float* outlam1, float* outlam2, float* outIx,
    float* outIy, const float sigma, const int width,
    const int height) {
    // in&out
    float* d_in;
    float* d_lam1, * d_lam2, * d_Ix, * d_Iy;
    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMemcpyAsync(d_in, in, totalByte, cudaMemcpyHostToDevice));

    // buffer
    float* d_kernelxx, * d_kernelxy, * d_kernelyy;
    float* d_xx, * d_xy, * d_yy;
    int _kernelSize = 0, _kernelSizeHalf = 0;
    const float sig2 = sigma * sigma;

    CUDA_CALL(cudaMalloc((void**)&d_xx, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_xy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_yy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam1, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam2, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Ix, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Iy, totalByte));

    std::unique_ptr<float[]> dGxx;
    std::unique_ptr<float[]> dGxy;
    std::unique_ptr<float[]> dGyy;
    getGaussSigma2dField<float>(dGxx, dGxy, dGyy, sigma, _kernelSizeHalf,
        _kernelSize);
    CUDA_CALL(cudaMalloc((void**)&d_kernelxx, _kernelSize));
    CUDA_CALL(cudaMalloc((void**)&d_kernelxy, _kernelSize));
    CUDA_CALL(cudaMalloc((void**)&d_kernelyy, _kernelSize));

    // config
    cudaStream_t cs[4];

    CUDA_CALL(cudaStreamCreate(&cs[0]));
    CUDA_CALL(cudaStreamCreate(&cs[1]));
    CUDA_CALL(cudaStreamCreate(&cs[2]));
    CUDA_CALL(cudaStreamCreate(&cs[3]));
    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(width / blocks.x, height / blocks.y);

    // device_simpleScale << <512, 512 >> > (d_in, d_in, (1.0f / float(0xFFFF)),
    // width);

    int smemSize =
        sizeof(float) *
        ((blocks.x + _kernelSizeHalf * 2) * (blocks.y + _kernelSizeHalf * 2) +
            ((_kernelSizeHalf * 2 + 1) * (_kernelSizeHalf * 2 + 1)));

    CUDA_CALL(cudaMemcpyAsync(d_kernelxx, dGxx.get(), _kernelSize,
        cudaMemcpyHostToDevice, cs[0]));
    device_convolve2d<float> << <grids, blocks, 0, cs[0] >> > (
        d_in, d_xx, d_kernelxx, width, height, _kernelSizeHalf, _kernelSizeHalf);
    device_simpleScale<float>
        << <grids, blocks, 0, cs[0] >> > (d_xx, d_xx, sig2, width);

    CUDA_CALL(cudaMemcpyAsync(d_kernelxy, dGxy.get(), _kernelSize,
        cudaMemcpyHostToDevice, cs[1]));
    device_convolve2d<float> << <grids, blocks, 0, cs[1] >> > (
        d_in, d_xy, d_kernelxy, width, height, _kernelSizeHalf, _kernelSizeHalf);
    device_simpleScale<float>
        << <grids, blocks, 0, cs[1] >> > (d_xy, d_xy, sig2, width);

    CUDA_CALL(cudaMemcpyAsync(d_kernelyy, dGyy.get(), _kernelSize,
        cudaMemcpyHostToDevice, cs[2]));
    device_convolve2d<float> << <grids, blocks, 0, cs[2] >> > (
        d_in, d_yy, d_kernelyy, width, height, _kernelSizeHalf, _kernelSizeHalf);
    device_simpleScale<float>
        << <grids, blocks, 0, cs[2] >> > (d_yy, d_yy, sig2, width);
    CUDA_CALL(cudaDeviceSynchronize());

    // Multiple each kernel to power of sigma

    device_eigen2Image<float>
        << <grids, blocks >> > (d_xx, d_xy, d_yy, d_lam2, d_lam1, d_Ix, d_Iy, width);
    CUDA_CALL(cudaDeviceSynchronize());

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    ////Normalize
    // float _min = 0.0f, _max = 0.0f;
    // const size_t sharedMemSize = width * sizeof(float);

    // device_reduction<512, float> << <width, height, sharedMemSize, cs[1] >> >
    // (d_lam1, d_xx, mn); device_reduction<1, float> << <1, width, sharedMemSize,
    // cs[1] >> > (d_xx, d_xx, mn); device_reduction<512, float> << <width,
    // height, sharedMemSize, cs[2] >> > (d_lam1, d_yy, mx); device_reduction<1,
    // float> << <1, width, sharedMemSize, cs[2] >> > (d_yy, d_yy, mx);

    // CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
    // cudaMemcpyDeviceToHost, cs[1])); CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0],
    // sizeof(float), cudaMemcpyDeviceToHost, cs[2]));
    // CUDA_CALL(cudaStreamSynchronize(cs[1]));
    // CUDA_CALL(cudaStreamSynchronize(cs[2]));

    // device_simpleNorm2<float> << <width, height >> > (d_lam1, d_lam1, _min,
    // _max); CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpyAsync(outlam1, d_lam1, totalByte, cudaMemcpyDeviceToHost,
        cs[0]));
    CUDA_CALL(cudaMemcpyAsync(outlam2, d_lam2, totalByte, cudaMemcpyDeviceToHost,
        cs[1]));
    CUDA_CALL(
        cudaMemcpyAsync(outIx, d_Ix, totalByte, cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(
        cudaMemcpyAsync(outIy, d_Iy, totalByte, cudaMemcpyDeviceToHost, cs[3]));
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
    CUDA_CALL(cudaStreamDestroy(cs[2]));
    CUDA_CALL(cudaStreamDestroy(cs[3]));
    CUDA_CALL(cudaFree(d_kernelxx));
    CUDA_CALL(cudaFree(d_kernelxy));
    CUDA_CALL(cudaFree(d_kernelyy));
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_xx));
    CUDA_CALL(cudaFree(d_xy));
    CUDA_CALL(cudaFree(d_yy));
    CUDA_CALL(cudaFree(d_lam1));
    CUDA_CALL(cudaFree(d_lam2));
    CUDA_CALL(cudaFree(d_Ix));
    CUDA_CALL(cudaFree(d_Iy));
}

void gpu_eigenimg_double(double* in, double* outlam1, double* outlam2,
    double* outIx, double* outIy, const float sigma,
    const int width, const int height) {}

void gpu_frangifilt_half(__half* h_in, __half* h_out, unsigned int* h_B,
    const int width, const int height, const float sigMn,
    const float sigMx, const float sigStep,
    const float beta, const float gamma,
    const float threshold, const bool bRegionGrowing) {}


void gpu_frangifilt_float2(float* h_in, float* h_out, float* h_frangi,
    const int width, const int height, const float sigMn,
    const float sigMx, const float sigStep,
    const float beta, const float gamma,
    const float threshold, const bool bRegionGrowing)
{
    float* d_in, * d_out, * d_dil, * d_dil2;
    // unsigned int* d_B;

    float* d_xx, * d_xy, * d_yy;
    float* d_lam1, * d_lam2, * d_Ix, * d_Iy;
    float* d_kernelxx = nullptr, * d_kernelxy = nullptr, * d_kernelyy = nullptr;

    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    CUDA_CALL(cudaMemcpy(d_in, h_in, totalByte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_out, 0, totalByte));
    CUDA_CALL(cudaMemset(d_dil, 0, totalByte));
    CUDA_CALL(cudaMemset(d_dil2, 0, totalByte));

    // if (bRegionGrowing)
    //{
    //     //CUDA_CALL(cudaMalloc((void**)&d_B, width * height * sizeof(unsigned
    //     int)));
    // }

    std::vector<float> h_sigs = getSigmaDistribution(sigMn, sigMx, sigStep);

    CUDA_CALL(cudaMalloc((void**)&d_xx, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_xy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_yy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam1, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam2, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Ix, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Iy, totalByte));

    const float Beta = 2 * beta * beta;
    const float C = 2 * gamma * gamma;

    const size_t sharedMemSize = width * sizeof(float);

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(width / blocks.x, height / blocks.y);

    cudaStream_t cs[3];

    CUDA_CALL(cudaStreamCreateWithFlags(&cs[0], cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&cs[1], cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&cs[2], cudaStreamNonBlocking));

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    // Normalize
    float _min = 0.0f, _max = 0.0f;

    // floating precision
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[1] >> > (d_in, d_lam1, mn);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[2] >> > (d_in, d_lam2, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    device_simpleNorm2<float> << <width, height >> > (d_in, d_in, _min, _max, 255.0f);
    CUDA_CALL(cudaDeviceSynchronize());
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    for (auto& sigma : h_sigs) {

        int _kernelSize = 0;
        int _kernelSizeHalf = 0;

        // Hessian matrix
        std::unique_ptr<float[]> dGxx;
        std::unique_ptr<float[]> dGxy;
        std::unique_ptr<float[]> dGyy;
        getGaussSigma2dField<float>(dGxx, dGxy, dGyy, sigma, _kernelSizeHalf,
            _kernelSize);
        CUDA_CALL(cudaMalloc((void**)&d_kernelxx, _kernelSize));
        CUDA_CALL(cudaMalloc((void**)&d_kernelxy, _kernelSize));
        CUDA_CALL(cudaMalloc((void**)&d_kernelyy, _kernelSize));
        const float sig2 = sigma * sigma;
        int smemSize =
            sizeof(float) *
            ((blocks.x + _kernelSizeHalf * 2) * (blocks.y + _kernelSizeHalf * 2) +
                ((_kernelSizeHalf * 2 + 1) * (_kernelSizeHalf * 2 + 1)));

        CUDA_CALL(cudaMemcpyAsync(d_kernelxx, dGxx.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[0]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[0] >> >
        // (d_in, d_xx, d_kernelxx, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[0] >> > (d_in, d_xx, d_kernelxx, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[0] >> > (d_xx, d_xx, sig2, width);

        CUDA_CALL(cudaMemcpyAsync(d_kernelxy, dGxy.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[1]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[1] >> >
        // (d_in, d_xy, d_kernelxy, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[1] >> > (d_in, d_xy, d_kernelxy, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[1] >> > (d_xy, d_xy, sig2, width);

        CUDA_CALL(cudaMemcpyAsync(d_kernelyy, dGyy.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[2]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[2] >> >
        // (d_in, d_yy, d_kernelyy, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[2] >> > (d_in, d_yy, d_kernelyy, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[2] >> > (d_yy, d_yy, sig2, width);

        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
        // //arrsum(d_xy, __LINE__);
        // //arrsum(d_yy, __LINE__);
        ////deviceArraySumPrint(d_xx, __LINE__);
        ////deviceArraySumPrint(d_xy, __LINE__);
        ////deviceArraySumPrint(d_yy, __LINE__);

        device_eigen2Image<float> << <grids, blocks >> > (d_xx, d_xy, d_yy, d_lam2,
            d_lam1, d_Ix, d_Iy, width);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_lam1, __LINE__);
        // //arrsum(d_lam2, __LINE__);
        // //arrsum(d_Ix, __LINE__);
        // //arrsum(d_Iy, __LINE__);
        ////deviceArraySumPrint(d_lam1, __LINE__);
        ////deviceArraySumPrint(d_lam2, __LINE__);
        ////deviceArraySumPrint(d_Ix, __LINE__);
        ////deviceArraySumPrint(d_Iy, __LINE__);

        device_postFrangi<float>
            << <grids, blocks >> > (d_lam1, d_lam2, d_Ix, d_Iy, d_out, Beta, C, width);
        CUDA_CALL(cudaDeviceSynchronize());
        ////deviceArraySumPrint(d_out, __LINE__);
        // //arrsum(d_out, __LINE__);
        if (threshold != 0) {
            device_reduction<512, float>
                << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
            device_reduction<1, float>
                << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
            device_reduction<512, float>
                << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
            device_reduction<1, float>
                << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

            CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
                cudaMemcpyDeviceToHost, cs[1]));
            CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
                cudaMemcpyDeviceToHost, cs[2]));
            CUDA_CALL(cudaStreamSynchronize(cs[1]));
            CUDA_CALL(cudaStreamSynchronize(cs[2]));
            // qDebug() << __LINE__ << _min << _max;
            device_simpleNorm2<float>
                << <width, height >> > (d_out, d_out, _min, _max, 1, threshold);
            CUDA_CALL(cudaDeviceSynchronize());
            ////deviceArraySumPrint(d_out, __LINE__);
        }

        CUDA_CALL(cudaFree(d_kernelxx));
        CUDA_CALL(cudaFree(d_kernelxy));
        CUDA_CALL(cudaFree(d_kernelyy));
        // dGxx.release();
        // dGxy.release();
        // dGyy.release();

        if (h_frangi != nullptr) {
            CUDA_CALL(cudaMemcpy(h_frangi, d_out, totalByte, cudaMemcpyDeviceToHost));
        }
    }


    if (bRegionGrowing)
    {
        device_reduction<512, float>
            << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
        device_reduction<1, float>
            << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
        device_reduction<512, float>
            << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
        device_reduction<1, float>
            << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

        CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
            cudaMemcpyDeviceToHost, cs[1]));
        CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
            cudaMemcpyDeviceToHost, cs[2]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));
        CUDA_CALL(cudaStreamSynchronize(cs[2]));
        // qDebug() << __LINE__ << _min << _max;
        device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min, _max);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_out, __LINE__);
        deviceArraySumPrint(d_out, __LINE__);

        float kernel[] = { _def[0], _def[1], _def[2], _def[1], _def[0],
                          _def[1], _def[3], _def[4], _def[3], _def[1],
                          _def[2], _def[4], _def[5], _def[4], _def[2],
                          _def[1], _def[3], _def[4], _def[3], _def[1],
                          _def[0], _def[1], _def[2], _def[1], _def[0] };
        float* d_kernel_t;

        CUDA_CALL(cudaMalloc((void**)&d_kernel_t, sizeof(float) * 25));
        // CUDA_CALL(cudaMalloc((void**)&d_checker, sizeof(int)* width* height));
        CUDA_CALL(cudaMemset(d_Ix, 0, sizeof(float) * width * height));
        CUDA_CALL(cudaMemcpy(d_kernel_t, kernel, sizeof(float) * 25,
            cudaMemcpyHostToDevice));

        device_convolve2d<float>
            << <grids, blocks >> > (d_out, d_xx, d_kernel_t, width, height, 2, 2);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
        deviceArraySumPrint(d_xx, __LINE__);
        device_simpleNorm2<float> << <width, height >> > (d_xx, d_xx, 0, 1, 1);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
        deviceArraySumPrint(d_xx, __LINE__);
        auto aa = gpu_otsu_threshold(d_xx, width, height, false);
        // std::vector<float>ths = device_percentile<float>(d_xx, 0.06f, 0.90f,
        // 0.96f, width * height);
        std::vector<float> ths;
        ths.push_back(aa);
        auto s = 0.66;
        ths.push_back(0.67);
        ths.push_back(0.91);
        /*0.09387954, 0.9132153, 0.9813198*/


        device_preGrowingCut<float> << <width, height >> > (d_xx, d_lam1, d_lam2, ths[0],
            ths[1], ths[2], width);
        devicePrint(d_xx, __LINE__);
        devicePrint(d_lam1, __LINE__);
        devicePrint(d_lam2, __LINE__);

        CUDA_CALL(cudaDeviceSynchronize());

        dim3 grid_t = dim3(width / 8, height / 8);
        dim3 block_t = dim3(8, 8);

        float* t0, * t1;
        cudaMalloc((void**)&t0, totalByte);
        cudaMalloc((void**)&t1, totalByte);

        for (int i = 0; i < 500; i++)
        {
            cudaMemcpyAsync(t0, d_lam1, totalByte, cudaMemcpyDeviceToDevice, cs[0]);
            cudaMemcpyAsync(t1, d_lam2, totalByte, cudaMemcpyDeviceToDevice, cs[1]);
            cudaStreamSynchronize(cs[0]);
            cudaStreamSynchronize(cs[1]);
            // cudaMemcpyAsync(t2, d_Ix, totalByte, cudaMemcpyDeviceToDevice,cs[2]);

            device_growingCut<float> << <grid_t, block_t >> > (d_xx, d_lam1, d_lam2, d_Ix,
                t0, t1, 2, width, height);
            CUDA_CALL(cudaDeviceSynchronize());

            device_reduction<512, float>
                << <width, height, sharedMemSize >> > (d_Ix, d_Iy, mx);
            device_reduction<1, float> << <1, width, sharedMemSize >> > (d_Iy, d_yy, mx);
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(&_max, &d_yy[0], sizeof(float), cudaMemcpyDeviceToHost));

            if (_max == 0)
            {
                //CUDA_CALL(cudaMemcpyAsync(d_out, d_lam1, sizeof(float) * width * height, cudaMemcpyDeviceToDevice));
                break;
            }
            else {
                cudaMemcpyAsync(d_lam1, t0, totalByte, cudaMemcpyDeviceToDevice, cs[0]);
                cudaMemcpyAsync(d_lam2, t1, totalByte, cudaMemcpyDeviceToDevice, cs[1]);
                cudaStreamSynchronize(cs[0]);
                cudaStreamSynchronize(cs[1]);

                CUDA_CALL(cudaMemset(d_Ix, 0, sizeof(float) * width * height));
            }
            CUDA_CALL(cudaDeviceSynchronize());
        }
        cudaFree(t0);
        cudaFree(t1);
        CUDA_CALL(cudaFree(d_kernel_t));
    }

    // floating precision
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));
    // qDebug() << __LINE__ << _min << _max;

    // dummy border check
    int brdstep = 0;
    {
        auto targ = std::make_unique<int[]>(8);

        for (; brdstep < width / 2; brdstep++) {
            targ[0] = brdstep + brdstep * height;
            targ[1] = (width / 2 - 1) + brdstep * height;
            targ[2] = (width - brdstep - 1) + brdstep * height;
            targ[3] = brdstep + (height / 2 - 1) * height;
            targ[4] = (width - brdstep - 1) + (height / 2 - 1) * height;
            targ[5] = brdstep + (height - brdstep - 1) * height;
            targ[6] = (width / 2 - 1) + (height - brdstep - 1) * height;
            targ[7] = (width - brdstep - 1) + (height - brdstep - 1) * height;

            if (h_in[targ[0]] == h_in[targ[1]] == h_in[targ[2]] == h_in[targ[3]] ==
                h_in[targ[4]] == h_in[targ[5]] == h_in[targ[6]] == h_in[targ[7]])
                continue;
            break;
        }
    }

    auto func = [=] __device__(const float& val, const float mnn,
        const float& mxx, const int& brd) -> float {
        const int tidx = threadIdx.x; // +blockDim.x * blockIdx.x;
        const int tidy = blockIdx.x;  // .y + blockDim.y * blockIdx.y;
        // const int gid = tidy * gridDim.x + tidx;

        if (tidx < brd || tidx >= (blockDim.x - brd) || tidy < brd ||
            tidy >= (gridDim.x - brd))
            return 0;
        else
            return (val - mnn) / (mxx - mnn);
    };
    device_customFunctor_in_2param<float>
        << <width, height >> > (d_out, d_out, func, _min, _max, brdstep);
    // device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min,
    // _max);
    CUDA_CALL(cudaDeviceSynchronize());
    // deviceArraySumPrint(d_out, __LINE__);

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));

    auto dd = gpu_otsu_threshold(d_out, width, height, false);


    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
    CUDA_CALL(cudaStreamDestroy(cs[2]));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_xx));
    CUDA_CALL(cudaFree(d_xy));
    CUDA_CALL(cudaFree(d_yy));
    CUDA_CALL(cudaFree(d_lam1));
    CUDA_CALL(cudaFree(d_lam2));
    CUDA_CALL(cudaFree(d_Ix));
    CUDA_CALL(cudaFree(d_Iy));
}

void gpu_frangifilt_float(float* h_in, float* h_out, float* h_frangi,
    const int width, const int height, const float sigMn,
    const float sigMx, const float sigStep,
    const float beta, const float gamma,
    const float threshold, const bool bRegionGrowing) {
    // std::vector<float> hv(262144);

    // in&out
    float* d_in, * d_out, *d_dil ,*d_dil2;
    // unsigned int* d_B;

    float* d_xx, * d_xy, * d_yy;
    float* d_lam1, * d_lam2, * d_Ix, * d_Iy;
    float* d_kernelxx = nullptr, * d_kernelxy = nullptr, * d_kernelyy = nullptr;

    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    CUDA_CALL(cudaMemcpy(d_in, h_in, totalByte, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_out, 0, totalByte));
    CUDA_CALL(cudaMemset(d_dil, 0, totalByte));
    CUDA_CALL(cudaMemset(d_dil2, 0, totalByte));

    // if (bRegionGrowing)
    //{
    //     //CUDA_CALL(cudaMalloc((void**)&d_B, width * height * sizeof(unsigned
    //     int)));
    // }

    std::vector<float> h_sigs = getSigmaDistribution(sigMn, sigMx, sigStep);

    CUDA_CALL(cudaMalloc((void**)&d_xx, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_xy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_yy, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam1, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_lam2, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Ix, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_Iy, totalByte));

    const float Beta = 2 * beta * beta;
    const float C = 2 * gamma * gamma;

    const size_t sharedMemSize = width * sizeof(float);

    dim3 blocks = dim3(32, 32);
    dim3 grids = dim3(width / blocks.x, height / blocks.y);

    cudaStream_t cs[3];

    CUDA_CALL(cudaStreamCreateWithFlags(&cs[0], cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&cs[1], cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&cs[2], cudaStreamNonBlocking));

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    // Normalize
    float _min = 0.0f, _max = 0.0f;

    // floating precision
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[1] >> > (d_in, d_lam1, mn);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[2] >> > (d_in, d_lam2, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    device_simpleNorm2<float> << <width, height >> > (d_in, d_in, _min, _max, 255.0f);
    CUDA_CALL(cudaDeviceSynchronize());
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    for (auto& sigma : h_sigs) 
    {
        int _kernelSize = 0;
        int _kernelSizeHalf = 0;

        // Hessian matrix
        std::unique_ptr<float[]> dGxx;
        std::unique_ptr<float[]> dGxy;
        std::unique_ptr<float[]> dGyy;
        getGaussSigma2dField<float>(dGxx, dGxy, dGyy, sigma, _kernelSizeHalf,
            _kernelSize);
        CUDA_CALL(cudaMalloc((void**)&d_kernelxx, _kernelSize));
        CUDA_CALL(cudaMalloc((void**)&d_kernelxy, _kernelSize));
        CUDA_CALL(cudaMalloc((void**)&d_kernelyy, _kernelSize));
        const float sig2 = sigma * sigma;
        int smemSize =
            sizeof(float) *
            ((blocks.x + _kernelSizeHalf * 2) * (blocks.y + _kernelSizeHalf * 2) +
                ((_kernelSizeHalf * 2 + 1) * (_kernelSizeHalf * 2 + 1)));

        CUDA_CALL(cudaMemcpyAsync(d_kernelxx, dGxx.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[0]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[0] >> >
        // (d_in, d_xx, d_kernelxx, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[0] >> > (d_in, d_xx, d_kernelxx, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[0] >> > (d_xx, d_xx, sig2, width);

        CUDA_CALL(cudaMemcpyAsync(d_kernelxy, dGxy.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[1]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[1] >> >
        // (d_in, d_xy, d_kernelxy, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[1] >> > (d_in, d_xy, d_kernelxy, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[1] >> > (d_xy, d_xy, sig2, width);

        CUDA_CALL(cudaMemcpyAsync(d_kernelyy, dGyy.get(), _kernelSize,
            cudaMemcpyHostToDevice, cs[2]));
        // device_convolve2d_shared<float> << < grids, blocks, smemSize, cs[2] >> >
        // (d_in, d_yy, d_kernelyy, width, height, _kernelHalf, _kernelHalf);
        device_convolve2d<float>
            << <grids, blocks, 0, cs[2] >> > (d_in, d_yy, d_kernelyy, width, height,
                _kernelSizeHalf, _kernelSizeHalf);
        device_simpleScale<float>
            << <grids, blocks, 0, cs[2] >> > (d_yy, d_yy, sig2, width);

        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
        // //arrsum(d_xy, __LINE__);
        // //arrsum(d_yy, __LINE__);
        ////deviceArraySumPrint(d_xx, __LINE__);
        ////deviceArraySumPrint(d_xy, __LINE__);
        ////deviceArraySumPrint(d_yy, __LINE__);

        device_eigen2Image<float> << <grids, blocks >> > (d_xx, d_xy, d_yy, d_lam2,
            d_lam1, d_Ix, d_Iy, width);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_lam1, __LINE__);
        // //arrsum(d_lam2, __LINE__);
        // //arrsum(d_Ix, __LINE__);
        // //arrsum(d_Iy, __LINE__);
        ////deviceArraySumPrint(d_lam1, __LINE__);
        ////deviceArraySumPrint(d_lam2, __LINE__);
        ////deviceArraySumPrint(d_Ix, __LINE__);
        ////deviceArraySumPrint(d_Iy, __LINE__);

        device_postFrangi<float>
            << <grids, blocks >> > (d_lam1, d_lam2, d_Ix, d_Iy, d_out, Beta, C, width);
        CUDA_CALL(cudaDeviceSynchronize());
        ////deviceArraySumPrint(d_out, __LINE__);
        // //arrsum(d_out, __LINE__);
        if (threshold != 0) {
            device_reduction<512, float>
                << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
            device_reduction<1, float>
                << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
            device_reduction<512, float>
                << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
            device_reduction<1, float>
                << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

            CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
                cudaMemcpyDeviceToHost, cs[1]));
            CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
                cudaMemcpyDeviceToHost, cs[2]));
            CUDA_CALL(cudaStreamSynchronize(cs[1]));
            CUDA_CALL(cudaStreamSynchronize(cs[2]));
            // qDebug() << __LINE__ << _min << _max;
            device_simpleNorm2<float>
                << <width, height >> > (d_out, d_out, _min, _max, 1, threshold);
            CUDA_CALL(cudaDeviceSynchronize());
            ////deviceArraySumPrint(d_out, __LINE__);
        }

        CUDA_CALL(cudaFree(d_kernelxx));
        CUDA_CALL(cudaFree(d_kernelxy));
        CUDA_CALL(cudaFree(d_kernelyy));
        // dGxx.release();
        // dGxy.release();
        // dGyy.release();

        if (h_frangi != nullptr) {
            CUDA_CALL(cudaMemcpy(h_frangi, d_out, totalByte, cudaMemcpyDeviceToHost));
        }
        devicePrint(d_out, __LINE__);
    }




    if (bRegionGrowing) 
    {
        device_reduction<512, float>
            << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
        device_reduction<1, float>
            << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
        device_reduction<512, float>
            << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
        device_reduction<1, float>
            << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

        CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
            cudaMemcpyDeviceToHost, cs[1]));
        CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
            cudaMemcpyDeviceToHost, cs[2]));
        CUDA_CALL(cudaStreamSynchronize(cs[1]));
        CUDA_CALL(cudaStreamSynchronize(cs[2]));
        // qDebug() << __LINE__ << _min << _max;
        device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min, _max);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_out, __LINE__);
         deviceArraySumPrint(d_out, __LINE__);

        float kernel[] = { _def[0], _def[1], _def[2], _def[1], _def[0],
                          _def[1], _def[3], _def[4], _def[3], _def[1],
                          _def[2], _def[4], _def[5], _def[4], _def[2],
                          _def[1], _def[3], _def[4], _def[3], _def[1],
                          _def[0], _def[1], _def[2], _def[1], _def[0] };
        float* d_kernel_t;

        CUDA_CALL(cudaMalloc((void**)&d_kernel_t, sizeof(float) * 25));
        // CUDA_CALL(cudaMalloc((void**)&d_checker, sizeof(int)* width* height));
        CUDA_CALL(cudaMemset(d_Ix, 0, sizeof(float) * width * height));
        CUDA_CALL(cudaMemcpy(d_kernel_t, kernel, sizeof(float) * 25,
            cudaMemcpyHostToDevice));

        device_convolve2d<float>
            << <grids, blocks >> > (d_out, d_xx, d_kernel_t, width, height, 2, 2);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
         deviceArraySumPrint(d_xx, __LINE__);
        device_simpleNorm2<float> << <width, height >> > (d_xx, d_xx, 0, 1, 1);
        CUDA_CALL(cudaDeviceSynchronize());
        // //arrsum(d_xx, __LINE__);
        deviceArraySumPrint(d_xx, __LINE__);
        auto aa = gpu_otsu_threshold(d_xx, width, height, false);
        // std::vector<float>ths = device_percentile<float>(d_xx, 0.06f, 0.90f,
        // 0.96f, width * height);
        std::vector<float> ths;
        ths.push_back(aa);
        auto s = aa + 0.4;
        if (s > 0.87)
            s = aa;
        ths.push_back(aa + 0.4);
        ths.push_back(0.87);
        /*0.09387954, 0.9132153, 0.9813198*/
   

        device_preGrowingCut<float> << <width, height >> > (d_xx, d_lam1, d_lam2, ths[0],
            ths[1], ths[2], width);
        devicePrint(d_xx, __LINE__);
        devicePrint(d_lam1, __LINE__);
        devicePrint(d_lam2, __LINE__);

        CUDA_CALL(cudaDeviceSynchronize());

        dim3 grid_t = dim3(width / 8, height / 8);
        dim3 block_t = dim3(8, 8);

        float* t0, * t1;
        cudaMalloc((void**)&t0, totalByte);
        cudaMalloc((void**)&t1, totalByte);

        for (int i = 0; i < 500; i++) 
        {
            cudaMemcpyAsync(t0, d_lam1, totalByte, cudaMemcpyDeviceToDevice, cs[0]);
            cudaMemcpyAsync(t1, d_lam2, totalByte, cudaMemcpyDeviceToDevice, cs[1]);
            cudaStreamSynchronize(cs[0]);
            cudaStreamSynchronize(cs[1]);
            // cudaMemcpyAsync(t2, d_Ix, totalByte, cudaMemcpyDeviceToDevice,cs[2]);

            device_growingCut<float> << <grid_t, block_t >> > (d_xx, d_lam1, d_lam2, d_Ix,
                t0, t1, 2, width, height);
            CUDA_CALL(cudaDeviceSynchronize());

            device_reduction<512, float>
                << <width, height, sharedMemSize >> > (d_Ix, d_Iy, mx);
            device_reduction<1, float> << <1, width, sharedMemSize >> > (d_Iy, d_yy, mx);
            CUDA_CALL(cudaDeviceSynchronize());
            CUDA_CALL(cudaMemcpy(&_max, &d_yy[0], sizeof(float), cudaMemcpyDeviceToHost));

            if (_max == 0) 
            {
                CUDA_CALL(cudaMemcpyAsync(d_out, d_lam1, sizeof(float) * width * height, cudaMemcpyDeviceToDevice));
                break;
            }
            else {
                cudaMemcpyAsync(d_lam1, t0, totalByte, cudaMemcpyDeviceToDevice, cs[0]);
                cudaMemcpyAsync(d_lam2, t1, totalByte, cudaMemcpyDeviceToDevice, cs[1]);
                cudaStreamSynchronize(cs[0]);
                cudaStreamSynchronize(cs[1]);

                CUDA_CALL(cudaMemset(d_Ix, 0, sizeof(float) * width * height));
            }
            CUDA_CALL(cudaDeviceSynchronize());
        }
        cudaFree(t0);
        cudaFree(t1);
        CUDA_CALL(cudaFree(d_kernel_t));
    }

    // floating precision
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[1] >> > (d_out, d_lam1, mn);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[1] >> > (d_lam1, d_xx, mn);
    device_reduction<512, float>
        << <width, height, sharedMemSize, cs[2] >> > (d_out, d_lam2, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize, cs[2] >> > (d_lam2, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[1]));
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0], sizeof(float),
        cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));
    // qDebug() << __LINE__ << _min << _max;

    // dummy border check
    int brdstep = 0;
    {
        auto targ = std::make_unique<int[]>(8);

        for (; brdstep < width / 2; brdstep++) {
            targ[0] = brdstep + brdstep * height;
            targ[1] = (width / 2 - 1) + brdstep * height;
            targ[2] = (width - brdstep - 1) + brdstep * height;
            targ[3] = brdstep + (height / 2 - 1) * height;
            targ[4] = (width - brdstep - 1) + (height / 2 - 1) * height;
            targ[5] = brdstep + (height - brdstep - 1) * height;
            targ[6] = (width / 2 - 1) + (height - brdstep - 1) * height;
            targ[7] = (width - brdstep - 1) + (height - brdstep - 1) * height;

            if (h_in[targ[0]] == h_in[targ[1]] == h_in[targ[2]] == h_in[targ[3]] ==
                h_in[targ[4]] == h_in[targ[5]] == h_in[targ[6]] == h_in[targ[7]])
                continue;
            break;
        }
    }

    auto func = [=] __device__(const float& val, const float mnn,
        const float& mxx, const int& brd) -> float {
        const int tidx = threadIdx.x; // +blockDim.x * blockIdx.x;
        const int tidy = blockIdx.x;  // .y + blockDim.y * blockIdx.y;
        // const int gid = tidy * gridDim.x + tidx;

        if (tidx < brd || tidx >= (blockDim.x - brd) || tidy < brd ||
            tidy >= (gridDim.x - brd))
            return 0;
        else
            return (val - mnn) / (mxx - mnn);
    };
    device_customFunctor_in_2param<float>
        << <width, height >> > (d_out, d_out, func, _min, _max, brdstep);
    // device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min,
    // _max);
    CUDA_CALL(cudaDeviceSynchronize());
    // deviceArraySumPrint(d_out, __LINE__);

    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));


    auto dilmax = [=] __device__(const float& v, const float& re) -> float {
        return fmax(v, re);
    };
    auto dilmin = [=] __device__(const float& v, const float& re) -> float {
        return fmin(v, re);
    };

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaStreamDestroy(cs[0]));
    CUDA_CALL(cudaStreamDestroy(cs[1]));
    CUDA_CALL(cudaStreamDestroy(cs[2]));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_xx));
    CUDA_CALL(cudaFree(d_xy));
    CUDA_CALL(cudaFree(d_yy));
    CUDA_CALL(cudaFree(d_lam1));
    CUDA_CALL(cudaFree(d_lam2));
    CUDA_CALL(cudaFree(d_Ix));
    CUDA_CALL(cudaFree(d_Iy));
}

void gpu_frangifilt_double(double* h_in, double* h_out, unsigned int* h_B,
    const int width, const int height, const float sigMn,
    const float sigMx, const float sigStep,
    const float beta, const float gamma,
    const float threshold, const bool bRegionGrowing) {
    // later
}

template <typename... Args> void gpu_msfm2d_float(Args &...arg) {
    // something..
    // Initialize start point.
    // device_distanceTransform()---------deprecate
    // device_speedImageProcess()
    //  D(x) = eps - (abs(I) - T)
    //  C(x,t) = ∇ · ∇φ(x,t??t)
    //             | ∇φ(x, t??t) |
    //  F = alpha * C(x,t) + (1 - alpha) * D(x)
    // activation seed (maximal sector), narrow: _n
    // get seed coordinates
    // mask seed
    // compact seed_n?seed?
    // Parameters
    //  :T():-1,  Y():0,  F():0
    // H = (1 + (Zx * *2)) * Zyy + (1 + (Zy * *2)) * Zxx - 2 * Zx * Zy * Zxy
    // H  = ((1 + (Zx **2)) * Zyy + (1 + (Zy **2)) * Zxx - 2 * Zx * Zy * Zxy)\
          // ((2 * (1 + (Zx **2) + (Zy **2))) **1.5)
    // Loop
    // Integer seed points: SP
    //  F(SP): 1, T(SP): 0,
    //  T(SP_n): 1/max(F(SP_n), eps)

    // Loop
    //
    // premapping (CPU?)
    //  to Device

    // 1st/2nd order ffm(narrow band)
}

void gpu_speedImage_float(float* h_in, float* h_out, bool* h_boundary,
    const int width, const int height, const int coeff,
    const bool bBinary, const bool bScaleUS16) {
    // auto arrsum = [](auto s, auto line) {
    //   // float hv[262144]={0,};
    //   float *hv = new float[262144];
    //   // cudaPointerAttributes attrib;
    //   // cudaPointerGetAttributes(&attrib, s);

    //  // if (attrib.type == cudaMemoryTypeDevice)
    //  cudaMemcpy(hv, (float *)s, sizeof(float) * 262144,
    //             cudaMemcpyDeviceToHost);
    //  // else
    //  //   std::copy_n(s, s + 262144, hv);

    //  int cntMark = 0;
    //  double val = 0.0;
    //  double2 _ = {0, 0};
    //  int2 mx = {0, 0};
    //  int2 mn = {INT_MAX, INT_MAX};

    //  for (int i = 0; i < 262144; i++) {
    //    if (hv[i] != 0) {
    //      cntMark++;
    //      val += hv[i];
    //      int2 xyz = {i % 512, i / 512 % 512};
    //      _.x += xyz.x;
    //      _.y += xyz.y;
    //      mx.x = fmaxf(mx.x, xyz.x);
    //      mx.y = fmaxf(mx.y, xyz.y);
    //      mn.x = fminf(mn.x, xyz.x);
    //      mn.y = fminf(mn.y, xyz.y);
    //    }
    //  }
    //  // qDebug() << std::fixed;
    //  qDebug() << "sum(" << line << ") : " << cntMark << "(" << _.x / cntMark
    //           << ", " << _.y / cntMark << ") " << val;
    //  delete[] hv;
    //};
    // in&out
    float* d_in, * d_out;
    // bool* d_boundary;
    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    // CUDA_CALL(cudaMalloc((void**)&d_boundary, sizeof(bool) * width * height));
    CUDA_CALL(cudaMemcpyAsync(d_in, h_in, totalByte, cudaMemcpyHostToDevice));

    // NppiSize _nppisz;
    //_nppisz.width = width;
    //_nppisz.height = height;
    //
    // NppiPoint _nppipnt;
    //_nppipnt.x = 0;
    //_nppipnt.y = 0;
    //
    // nppiDilate3x3Border_32f_C1R(h_in, sizeof(Npp32f) * width, _nppisz,
    // _nppipnt, h_out, sizeof(Npp32f) * width, _nppisz, NPP_BORDER_REPLICATE);

    // buffer
    const int bufferByte = sizeof(int) * width * height;
    int* d_cs;
    float* d_reduce;
    // float* d_buffer0, * d_buffer1;
    CUDA_CALL(cudaMalloc((void**)&d_cs, bufferByte));
    CUDA_CALL(cudaMalloc((void**)&d_reduce, totalByte));
    /*CUDA_CALL(cudaMalloc((void**)&d_buffer0, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_buffer1, totalByte));*/

    // configuation
    dim3 blocks1 = dim3(width, 1);
    dim3 blocks2 = dim3(height, 1);
    dim3 blocks3 = dim3(32, 32);
    dim3 grids = dim3(1, 1);
    dim3 grids2 = dim3(width / 32, height / 32);
    constexpr size_t streamCount = 4;
    cudaStream_t cs[streamCount];
    for (auto& _cs : cs)
        CUDA_CALL(cudaStreamCreate(&_cs));

    const size_t sharedMemSize1 = width * sizeof(int) * 3;
    const size_t sharedMemSize2 = height * sizeof(int);
    const size_t sharedMemSize3 = width * sizeof(float);

    // CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    // CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));
    //
    // CUDA_CALL(cudaMemcpy(d_in, h_in,totalByte,cudaMemcpyHostToDevice));

    auto dil = [=] __device__(const float& v, const float& re) -> float {
        return fmax(v, re);
    };
    //arrsum(d_in, __LINE__);
    for (auto i = 0; i < 2; i++) {
        device_morphology<float, 3>
            << <grids2, blocks3 >> > (d_in, d_out, dil, width, height);
        CUDA_CALL(cudaMemcpy(d_in, d_out, totalByte, cudaMemcpyDeviceToDevice));
    }
    devicePrint(d_out, __LINE__);

    // //arrsum(d_out, __LINE__);
    // deviceArraySumPrint(d_in, __LINE__, width, height);
    // deviceArraySumPrint(d_out, __LINE__, width, height);

    // Normalize
    float _min = 0.0f, _max = 0.0f;
    auto mx = [=] __device__(const auto & a, const auto & b)
    {
        return a > b ? a : b;
    };


    device_reduction<512, float>
        << <width, height, sharedMemSize3 >> > (d_out, d_reduce, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize3 >> > (d_reduce, d_reduce, mx);
    CUDA_CALL(
        cudaMemcpy(&_max, &d_reduce[0], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    // floating precision
    /*device_reduction<512, float> << <width, height, sharedMemSize, cs[1] >> >
    (d_in, d_buffer0, mn); device_reduction<1, float> << <1, width, sharedMemSize,
    cs[1] >> > (d_buffer0, d_xx, mn); device_reduction<512, float> << <width,
    height, sharedMemSize, cs[2] >> > (d_in, d_uy, mx); device_reduction<1, float>
    << <1, width, sharedMemSize, cs[2] >> > (d_uy, d_yy, mx);

    CUDA_CALL(cudaMemcpyAsync(&_min, &d_xx[0], sizeof(float),
    cudaMemcpyDeviceToHost, cs[1])); CUDA_CALL(cudaMemcpyAsync(&_max, &d_yy[0],
    sizeof(float), cudaMemcpyDeviceToHost, cs[2]));
    CUDA_CALL(cudaStreamSynchronize(cs[1]));
    CUDA_CALL(cudaStreamSynchronize(cs[2]));*/

    /*device_simpleScale<float> << <width, height >> > (d_in, d_in, 1.0f /
    USHRT_MAX, width); CUDA_CALL(cudaDeviceSynchronize());*/
    //arrsum(d_in, __LINE__);
    for (auto hs = 0; hs < height; hs++) {
        // for (auto s = 0; s < streamCount; s++)
        {
            // const int y = s + hs * streamCount;
            // CUDA_CALL(cudaStreamSynchronize(cs[s]));
            device_1dRowCloset_Ex<float>

                << <grids, blocks1, sharedMemSize1 >> > (d_in, d_cs, width, hs);
            CUDA_CALL(cudaDeviceSynchronize());
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    //arrsum(d_cs, __LINE__);
    // deviceArraySumPrint(d_cs, __LINE__, width, height);
    for (auto ws = 0; ws < (width / streamCount); ws++) {
        for (auto s = 0; s < streamCount; s++) {
            const int x = s + ws * streamCount;
            // CUDA_CALL(cudaStreamSynchronize(cs[s]));
            device_1dColCloset_Ex << <grids, blocks2, sharedMemSize2, cs[s] >> > (
                d_cs, d_out, width, x);
        }
    }

    //arrsum(d_out, __LINE__);

    // deviceArraySumPrint(d_out, __LINE__, width, height);
    CUDA_CALL(cudaMemcpyAsync(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());
    std::transform(h_out, h_out + width * height, h_boundary,
        [](const float& val) { return val == 0 ? true : false; });

    device_reduction<512, float>
        << <width, height, sharedMemSize3 >> > (d_out, d_reduce, mx);
    device_reduction<1, float>
        << <1, width, sharedMemSize3 >> > (d_reduce, d_reduce, mx);
    CUDA_CALL(
        cudaMemcpy(&_max, &d_reduce[0], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    auto cf = [=] __device__(const float& a, const float& _deno,
        const float& _coeff, const int& misc) {
        return (a != 0) ? (powf(sqrtf(__fdiv_rd(a, _deno)), _coeff) * (1)) : misc;
    };

    /*device_simpleScale<float> << <grids2, blocks2 >> > (d_out, d_out, 1.0 /
    _max, width); device_simplePower<float> << <grids2, blocks2 >> > (d_out,
    d_out, coeff, width);*/
    device_customFunctor_in_2param<float>
        << <grids2, blocks3 >> > (d_out, d_out, cf, _max, coeff, 0);
    CUDA_CALL(cudaDeviceSynchronize());

    // deviceArraySumPrint(d_out, __LINE__, width, height);
    CUDA_CALL(cudaMemcpy(h_out, d_out, totalByte, cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(h_boundary, d_boundary, sizeof(bool) * 512 * 512,
    // cudaMemcpyDeviceToHost));

    for (auto& _cs : cs)
        CUDA_CALL(cudaStreamDestroy(_cs));

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    // CUDA_CALL(cudaFree(d_boundary));
    CUDA_CALL(cudaFree(d_cs));
    CUDA_CALL(cudaFree(d_reduce));
    /*CUDA_CALL(cudaFree(d_buffer0));
    CUDA_CALL(cudaFree(d_buffer1));*/
}

__inline__ __device__ float2 device_vecvec(const float2* __restrict__ in,
    const int& id, const int& max) {
    float2 dt;
    if (id == max - 1) {
        dt.x = in[id].x - in[id - 1].x;
        dt.y = in[id].y - in[id - 1].y;
    }
    else {
        dt.x = in[id + 1].x - in[id].x;
        dt.y = in[id + 1].y - in[id].y;
    }
    //dt.x = __fmul_rn(dt.x, 1000.0f);
    //dt.y = __fmul_rn(dt.y, 1000.0f);
    const float dfn = hypotf(dt.x, dt.y);
    return make_float2(__fdiv_rn(-dt.y, dfn), __fdiv_rn(dt.x, dfn));
}

__inline__ __device__ float2 device_2DboundaryClamp(const float2& v,
    const int& width,
    const int& height,
    bool& b) {
    float2 result;
    if (v.x < 1) {
        result.x = 1;
        b = false;
    }
    else if (v.x >= width) {
        result.x = width - 1;
        b = false;
    }
    else
        result.x = v.x;

    if (v.y < 1) {
        result.y = 1;
        b = false;
    }
    else if (v.y >= height) {
        result.y = height - 1;
        b = false;
    }
    else
        result.y = v.y;

    return result;
}

__inline__ __device__ float
device_bicubic(const float* __restrict__ src, const float* __restrict__ dx,
    const float* __restrict__ dy, const float* __restrict__ dxy,
    const const float2& point, const int& width, const int& height) {
    const uint2 q00 =
        make_uint2(__float2int_rd(point.x), __float2int_rd(point.y));
    const uint2 q11 = make_uint2(fminf(q00.x + 1.0f, width - 1.0f),
        fminf(q00.y + 1.0f, height - 1.0f));

    const float2 _xy = make_float2(point.x - q00.x, point.y - q00.y);
    const float2 _xy2 = make_float2(_xy.x * _xy.x, _xy.y * _xy.y);
    const float2 _xy3 = make_float2(_xy2.x * _xy.x, _xy2.y * _xy.y);

    const uint f00 = q00.x + q00.y * width;
    const uint f01 = q00.x + q11.y * width;
    const uint f10 = q11.x + q00.y * width;
    const uint f11 = q11.x + q11.y * width;

    const double X[] = { src[f00], src[f10], src[f01], src[f11], dx[f00], dx[f10],
                        dx[f01],  dx[f11],  dy[f00],  dy[f10],  dy[f01], dy[f11],
                        dxy[f00], dxy[f10], dxy[f01], dxy[f11] };

    const double a[] = {
        InvAMat[0] * X[0],
        InvAMat[1] * X[1],
        InvAMat[2] * X[2] + InvAMat[3] * X[2] + InvAMat[4] * X[2] +
            InvAMat[5] * X[2],
        InvAMat[6] * X[3] + InvAMat[7] * X[3] + InvAMat[8] * X[3] +
            InvAMat[9] * X[3],
        InvAMat[10] * X[4],
        InvAMat[11] * X[5],
        InvAMat[12] * X[6] + InvAMat[13] * X[6] + InvAMat[14] * X[6] +
            InvAMat[15] * X[6],
        InvAMat[16] * X[7] + InvAMat[17] * X[7] + InvAMat[18] * X[7] +
            InvAMat[19] * X[7],
        InvAMat[20] * X[8] + InvAMat[21] * X[8] + InvAMat[22] * X[8] +
            InvAMat[23] * X[8],
        InvAMat[24] * X[9] + InvAMat[25] * X[9] + InvAMat[26] * X[9] +
            InvAMat[27] * X[9],

        InvAMat[28] * X[10] + InvAMat[29] * X[10] + InvAMat[30] * X[10] +
            InvAMat[31] * X[10] + InvAMat[32] * X[10] + InvAMat[33] * X[10] +
            InvAMat[34] * X[10] + InvAMat[35] * X[10] + InvAMat[36] * X[10] +
            InvAMat[37] * X[10] + InvAMat[38] * X[10] + InvAMat[39] * X[10] +
            InvAMat[40] * X[10] + InvAMat[41] * X[10] + InvAMat[42] * X[10] +
            InvAMat[43] * X[10],

        InvAMat[44] * X[11] + InvAMat[45] * X[11] + InvAMat[46] * X[11] +
            InvAMat[47] * X[11] + InvAMat[48] * X[11] + InvAMat[49] * X[11] +
            InvAMat[50] * X[11] + InvAMat[51] * X[11] + InvAMat[52] * X[11] +
            InvAMat[53] * X[11] + InvAMat[54] * X[11] + InvAMat[55] * X[11] +
            InvAMat[56] * X[11] + InvAMat[57] * X[11] + InvAMat[58] * X[11] +
            InvAMat[59] * X[11],

        InvAMat[60] * X[12] + InvAMat[61] * X[12] + InvAMat[62] * X[12] +
            InvAMat[63] * X[12],

        InvAMat[64] * X[13] + InvAMat[65] * X[13] + InvAMat[66] * X[13] +
            InvAMat[67] * X[13],

        InvAMat[68] * X[14] + InvAMat[69] * X[14] + InvAMat[70] * X[14] +
            InvAMat[71] * X[14] + InvAMat[72] * X[14] + InvAMat[73] * X[14] +
            InvAMat[74] * X[14] + InvAMat[75] * X[14] + InvAMat[76] * X[14] +
            InvAMat[77] * X[14] + InvAMat[78] * X[14] + InvAMat[79] * X[14] +
            InvAMat[80] * X[14] + InvAMat[81] * X[14] + InvAMat[82] * X[14] +
            InvAMat[83] * X[14],

        InvAMat[84] * X[15] + InvAMat[85] * X[15] + InvAMat[86] * X[15] +
            InvAMat[87] * X[15] + InvAMat[88] * X[15] + InvAMat[89] * X[15] +
            InvAMat[90] * X[15] + InvAMat[91] * X[15] + InvAMat[92] * X[15] +
            InvAMat[93] * X[15] + InvAMat[94] * X[15] + InvAMat[95] * X[15] +
            InvAMat[96] * X[15] + InvAMat[97] * X[15] + InvAMat[98] * X[15] +
            InvAMat[99] * X[15] };


    //return a[0] + a[4] * _xy.y + a[8] * _xy2.y + a[16] * _xy3.y\
          + a[1] * _xy.x + a[5] * _xy.x * _xy.y + a[9] * _xy.x * _xy2.y + a[13] * _xy.x * _xy3.y\
          + a[2] * _xy2.x + a[6] * _xy2.x * _xy.y + a[10] * _xy2.x * _xy2.y + a[14] * _xy2.x * _xy3.y\
          + a[3] * _xy3.x + a[7] * _xy3.x * _xy.y + a[11] * _xy3.x * _xy2.y + a[15] * _xy3.x * _xy3.y;
    return __double2float_rn(a[0] + __dmul_rn(a[4], _xy.y) + __dmul_rn(a[8], _xy2.y) +
        __dmul_rn(a[15], _xy3.y) + __dmul_rn(a[1], _xy.x) +
        __dmul_rn(__dmul_rn(a[5], _xy.x), _xy.y) +
        __dmul_rn(__dmul_rn(a[9], _xy.x), _xy2.y) +
        __dmul_rn(__dmul_rn(a[12], _xy.x), _xy3.y) + __dmul_rn(a[2], _xy2.x) +
        __dmul_rn(__dmul_rn(a[6], _xy2.x), _xy.y) +
        __dmul_rn(__dmul_rn(a[10], _xy2.x), _xy2.y) +
        __dmul_rn(__dmul_rn(a[13], _xy2.x), _xy3.y) + __dmul_rn(a[3], _xy3.x) +
        __dmul_rn(__dmul_rn(a[7], _xy3.x), _xy.y) +
        __dmul_rn(__dmul_rn(a[11], _xy3.x), _xy2.y) +
        __dmul_rn(__dmul_rn(a[14], _xy3.x), _xy3.y));
}


float device_bicubic2(const float* __restrict__ src, const float* __restrict__ dx,
    const float* __restrict__ dy, const float* __restrict__ dxy,
    const const float2& point, const int& width, const int& height)
{

    int InvAMat[1 << 8] = {
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
    const int q00_x = static_cast<int>(floor(point.x));
    const int q00_y = static_cast<int>(floor(point.y));
    const int q11_x = std::min(q00_x + 1, width - 1);
    const int q11_y = std::min(q00_y + 1, height - 1);

    const float _xy_x = point.x - q00_x;
    const float _xy_y = point.y - q00_y;
    const float _xy2_x = _xy_x * _xy_x;
    const float _xy2_y = _xy_y * _xy_y;
    const float _xy3_x = _xy2_x * _xy_x;
    const float _xy3_y = _xy2_y * _xy_y;

    const int f00 = q00_x + q00_y * width;
    const int f01 = q00_x + q11_y * width;
    const int f10 = q11_x + q00_y * width;
    const int f11 = q11_x + q11_y * width;

    const double X[] = {
        src[f00], src[f10], src[f01], src[f11],
        dx[f00], dx[f10], dx[f01], dx[f11],
        dy[f00], dy[f10], dy[f01], dy[f11],
        dxy[f00], dxy[f10], dxy[f01], dxy[f11]
    };

    const double a[] = {
        InvAMat[0] * X[0],
        InvAMat[1] * X[1],
        InvAMat[2] * X[2] + InvAMat[3] * X[2] + InvAMat[4] * X[2] + InvAMat[5] * X[2],
        InvAMat[6] * X[3] + InvAMat[7] * X[3] + InvAMat[8] * X[3] + InvAMat[9] * X[3],
        InvAMat[10] * X[4],
        InvAMat[11] * X[5],
        InvAMat[12] * X[6] + InvAMat[13] * X[6] + InvAMat[14] * X[6] + InvAMat[15] * X[6],
        InvAMat[16] * X[7] + InvAMat[17] * X[7] + InvAMat[18] * X[7] + InvAMat[19] * X[7],
        InvAMat[20] * X[8] + InvAMat[21] * X[8] + InvAMat[22] * X[8] + InvAMat[23] * X[8],
        InvAMat[24] * X[9] + InvAMat[25] * X[9] + InvAMat[26] * X[9] + InvAMat[27] * X[9],

        InvAMat[28] * X[10] + InvAMat[29] * X[10] + InvAMat[30] * X[10] + InvAMat[31] * X[10] +
        InvAMat[32] * X[10] + InvAMat[33] * X[10] + InvAMat[34] * X[10] + InvAMat[35] * X[10] +
        InvAMat[36] * X[10] + InvAMat[37] * X[10] + InvAMat[38] * X[10] + InvAMat[39] * X[10] +
        InvAMat[40] * X[10] + InvAMat[41] * X[10] + InvAMat[42] * X[10] + InvAMat[43] * X[10],

        InvAMat[44] * X[11] + InvAMat[45] * X[11] + InvAMat[46] * X[11] + InvAMat[47] * X[11] +
        InvAMat[48] * X[11] + InvAMat[49] * X[11] + InvAMat[50] * X[11] + InvAMat[51] * X[11] +
        InvAMat[52] * X[11] + InvAMat[53] * X[11] + InvAMat[54] * X[11] + InvAMat[55] * X[11] +
        InvAMat[56] * X[11] + InvAMat[57] * X[11] + InvAMat[58] * X[11] + InvAMat[59] * X[11],

        InvAMat[60] * X[12] + InvAMat[61] * X[12] + InvAMat[62] * X[12] + InvAMat[63] * X[12],
        InvAMat[64] * X[13] + InvAMat[65] * X[13] + InvAMat[66] * X[13] + InvAMat[67] * X[13],
        InvAMat[68] * X[14] + InvAMat[69] * X[14] + InvAMat[70] * X[14] + InvAMat[71] * X[14] +
        InvAMat[72] * X[14] + InvAMat[73] * X[14] + InvAMat[74] * X[14] + InvAMat[75] * X[14] +
        InvAMat[76] * X[14] + InvAMat[77] * X[14] + InvAMat[78] * X[14] + InvAMat[79] * X[14] +
        InvAMat[80] * X[14] + InvAMat[81] * X[14] + InvAMat[82] * X[14] + InvAMat[83] * X[14],

        InvAMat[84] * X[15] + InvAMat[85] * X[15] + InvAMat[86] * X[15] + InvAMat[87] * X[15] +
        InvAMat[88] * X[15] + InvAMat[89] * X[15] + InvAMat[90] * X[15] + InvAMat[91] * X[15] +
        InvAMat[92] * X[15] + InvAMat[93] * X[15] + InvAMat[94] * X[15] + InvAMat[95] * X[15] +
        InvAMat[96] * X[15] + InvAMat[97] * X[15] + InvAMat[98] * X[15] + InvAMat[99] * X[15]
    };



    return static_cast<float>(
        a[0] + a[4] * _xy_y + a[8] * _xy2_y + a[15] * _xy3_y +
        a[1] * _xy_x +
        a[5] * _xy_x * _xy_y +
        a[9] * _xy_x * _xy2_y +
        a[12] * _xy_x * _xy3_y +
        a[2] * _xy2_x +
        a[6] * _xy2_x * _xy_y +
        a[10] * _xy2_x * _xy2_y +
        a[13] * _xy2_x * _xy3_y +
        a[3] * _xy3_x +
        a[7] * _xy3_x * _xy_y +
        a[11] * _xy3_x * _xy2_y +
        a[14] * _xy3_x * _xy3_y
        );
}

__global__ void device_BASOC(const float* __restrict__ src,
    const float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ dxy, float2* d_in,
    float2* d_medial, float2* d_left, float2* d_right,
    float* d_radius, const float widely,
    const float scale, const int itor_max,
    const int num, const int width, const int height) 
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        float2 vv = device_vecvec(d_in, tid, num);
        float2 v = make_float2(__fmaf_rn(vv.x, widely, d_in[tid].x),
            __fmaf_rn(vv.y, widely, d_in[tid].y));
        vv.x = __fmul_rn(vv.x, scale);
        vv.y = __fmul_rn(vv.y, scale);

        float _buffer = 512.0f;
        int _id = -1;
        constexpr float initInterv = 0.5f;
        const float intvDivScale = __fdiv_rn(initInterv, scale);
        float R = initInterv - scale;
        float _R = R;
        int tol = 0;
        for (int i = 0; i < itor_max; i++)
        {
            if (i != 0)
            {
                v.x -= vv.x;
                v.y -= vv.y;
            }

            if ((v.x < 0) || (v.y < 0) || (v.x >= width) || (v.y >= height))
                continue;

            float temp = device_bicubic(src, dx, dy, dxy, v, width, height);
            if (temp < _buffer) {
                _buffer = temp;
                _id = i;
                tol = 0;
            }
            else if (temp == _buffer)
                tol++;
        }

        const float _val = (_id + (tol / 2)) * scale;
        d_medial[tid].x = d_in[tid].x - vv.x * _val;
        d_medial[tid].y = d_in[tid].y - vv.y * _val;

        float2 C = make_float2(d_medial[tid].x - vv.x * intvDivScale,
            d_medial[tid].y - vv.y * intvDivScale);
        float p1 = device_bicubic(src, dx, dy, dxy, C, width, height);
        while ((p1 < 0) /* && (ptest-->0)*/) {
            C.x -= vv.x;
            C.y -= vv.y;
            if ((C.x < 0) || (C.y < 0) || (C.x >= width) || (C.y >= height))
                break;
            p1 = device_bicubic(src, dx, dy, dxy, C, width, height);
            R += scale;
        }

        float2 C2 = make_float2(d_medial[tid].x + vv.x * intvDivScale, d_medial[tid].y + vv.y * intvDivScale);
        float p2 = device_bicubic(src, dx, dy, dxy, C2, width, height);

        while (p2 < 0) {
            C2.x += vv.x;
            C2.y += vv.y;
            if (C2.x < 0 || C2.y < 0 || C2.x >= width || C2.y >= height) break;
            p2 = device_bicubic(src, dx, dy, dxy, C2, width, height);
            _R += scale;
        }

        d_left[tid] = C;
        d_right[tid] = C2;
        d_radius[tid] = R;
    }
}


__global__ void device_MVA1d(float2* in, float2* out, const int factor,
    const int num) {
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const int half = factor / 2;
    double2 sum = make_double2(0, 0);
    int imin = 0, imax = 0;
    if ((tid > 0) && (tid < num - 1)) {

        if (tid < half) {
            imin = 0;
            imax = tid + half;
        }
        else if (tid >= (num - 1 - half)) {
            imin = tid - half;
            imax = num - 1;
        }
        else {
            imin = tid - half;
            imax = tid + half;
        }
        for (auto i = imin; i <= imax; i++) {
            sum.x = __dadd_rd(in[i].x, sum.x);
            sum.y = __dadd_rd(in[i].y, sum.y);
        }

        out[tid].x = __double2float_rd(__ddiv_rd(sum.x, (imax - imin + 1)));
        out[tid].y = __double2float_rd(__ddiv_rd(sum.y, (imax - imin + 1)));
    }
    else if ((tid == 0) || (tid == num - 1)) {
        out[tid].x = in[tid].x;
        out[tid].y = in[tid].y;
    }
}


void test_gpuBASOC2(float* h_lamMap,float* h_x, float* h_y, float* h_xy, float2* h_pnt, float2* h_medial,
    float2* h_left, float2* h_right, float* h_radius,
    const int pntnum, const int factor, const int width,
    const int height)
{
    float* d_lamMap;

    float* d_x, * d_y, * d_xy;
    float2* d_pnt;

    float2* d_medial, * d_left, * d_right;
    float* d_radius;

    float2* d_medial2, * d_left2, * d_right2;

    constexpr float widely = 5.0f;
    constexpr float scale = 0.01f;

    constexpr int itor_max = (widely / ((scale != 0) ? scale : 1)) * 2 + 1;

    CUDA_CALL(cudaMalloc((void**)&d_lamMap, sizeof(float) * width * height));

    CUDA_CALL(cudaMalloc((void**)&d_pnt, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_medial, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_left, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_right, sizeof(float2) * pntnum));

    CUDA_CALL(cudaMalloc((void**)&d_medial2, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_left2, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_right2, sizeof(float2) * pntnum));

    CUDA_CALL(cudaMalloc((void**)&d_x, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_y, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_xy, sizeof(float) * width * height));



    CUDA_CALL(cudaMemcpy(d_lamMap, h_lamMap, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_x, h_x, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_y, h_y, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_xy, h_xy, sizeof(float) * width * height, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pnt, h_pnt, sizeof(float2) * pntnum, cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 1);
    dim3 grids = dim3(ceilf(float(pntnum) / blocks.x), 1);

    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    device_MVA1d << <grids, blocks >> > (d_pnt, d_medial2, factor, pntnum);


    device_BASOC << <grids, blocks >> > (d_lamMap, d_x, d_y, d_xy, d_medial2, d_medial, d_left, d_right, d_radius, widely, scale, itor_max, pntnum, width, height);

    CUDA_CALL(cudaDeviceSynchronize());
    //arrsum(d_medial, __LINE__);
    //arrsum(d_right, __LINE__);
    //arrsum(d_left, __LINE__);
    // device_MVA1d << <grids, blocks >> > (d_medial, d_medial2, factor, pntnum);
    device_MVA1d << <grids, blocks >> > (d_left, d_left2, factor, pntnum);
    device_MVA1d << <grids, blocks >> > (d_right, d_right2, factor, pntnum);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_medial, d_medial, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_left, d_left2, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_right, d_right2, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_radius, d_radius, sizeof(float) * pntnum,
        cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_lamMap));
    CUDA_CALL(cudaFree(d_pnt));
    CUDA_CALL(cudaFree(d_medial));
    CUDA_CALL(cudaFree(d_left));
    CUDA_CALL(cudaFree(d_right));
    CUDA_CALL(cudaFree(d_medial2));
    CUDA_CALL(cudaFree(d_left2));
    CUDA_CALL(cudaFree(d_right2));

    CUDA_CALL(cudaFree(d_x));
    CUDA_CALL(cudaFree(d_y));
    CUDA_CALL(cudaFree(d_xy));

    CUDA_CALL(cudaFree(d_radius));

}

void test_gpuBASOC(float* h_lamMap, float2* h_pnt, float2* h_medial,
    float2* h_left, float2* h_right, float* h_radius,
    const int pntnum, const int factor, const int width,
    const int height) {
    float* d_lamMap;
    float2* d_pnt;

    float2* d_medial, * d_left, * d_right;
    float* d_radius;

    float2* d_medial2, * d_left2, * d_right2;

    float* d_x, * d_y, * d_xy;

    constexpr float widely = 15.0f;
    constexpr float scale = 0.01f;

    constexpr int itor_max = (widely / ((scale != 0) ? scale : 1)) * 2 + 1;

    CUDA_CALL(cudaMalloc((void**)&d_lamMap, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_pnt, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_medial, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_left, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_right, sizeof(float2) * pntnum));

    CUDA_CALL(cudaMalloc((void**)&d_medial2, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_left2, sizeof(float2) * pntnum));
    CUDA_CALL(cudaMalloc((void**)&d_right2, sizeof(float2) * pntnum));

    CUDA_CALL(cudaMalloc((void**)&d_radius, sizeof(float) * pntnum));

    CUDA_CALL(cudaMalloc((void**)&d_x, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_y, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_xy, sizeof(float) * width * height));

    CUDA_CALL(cudaMemcpy(d_lamMap, h_lamMap, sizeof(float) * width * height,
        cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pnt, h_pnt, sizeof(float2) * pntnum,
        cudaMemcpyHostToDevice));

    dim3 blocks = dim3(32, 1);
    dim3 grids = dim3(ceilf(float(pntnum) / blocks.x), 1);

    dim3 blocks1 = dim3(width, 1);
    dim3 grids1 = dim3(1, height);

    dim3 blocks2 = dim3(1, height);
    dim3 grids2 = dim3(width, 1);

    device_derivativeX<float> << <grids2, blocks2, sizeof(float)* (height + 2) >> > (
        d_lamMap, d_x, width, height);
    device_derivativeY<float> << <grids1, blocks1, sizeof(float)* (width + 2) >> > (
        d_lamMap, d_y, width, height);
    CUDA_CALL(cudaDeviceSynchronize());
    device_simpleMul<float> << <height, width >> > (d_x, d_y, d_xy, width);
    CUDA_CALL(cudaDeviceSynchronize());
    ////arrsum(d_lamMap, __LINE__);
    ////arrsum(d_x, __LINE__);
    ////arrsum(d_y, __LINE__);
    ////arrsum(d_xy, __LINE__);
    deviceArraySumPrint(d_lamMap, __LINE__, width, height);
    deviceArraySumPrint(d_x, __LINE__, width, height);
    deviceArraySumPrint(d_y, __LINE__, width, height);
    deviceArraySumPrint(d_xy, __LINE__, width, height);
    device_MVA1d << <grids, blocks >> > (d_pnt, d_medial2, factor, pntnum);
    //qDebug() << pntnum;
    //arrsum(d_medial2, __LINE__);
    // deviceArraySumPrint(d_xy, __LINE__, width, height);
    device_BASOC << <grids, blocks >> > (d_lamMap, d_x, d_y, d_xy, d_medial2, d_medial,
        d_left, d_right, d_radius, widely, scale,
        itor_max, pntnum, width, height);

    CUDA_CALL(cudaDeviceSynchronize());
    //arrsum(d_medial, __LINE__);
    //arrsum(d_right, __LINE__);
    //arrsum(d_left, __LINE__);
    // device_MVA1d << <grids, blocks >> > (d_medial, d_medial2, factor, pntnum);
    device_MVA1d << <grids, blocks >> > (d_left, d_left2, factor, pntnum);
    device_MVA1d << <grids, blocks >> > (d_right, d_right2, factor, pntnum);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_medial, d_medial, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_left, d_left2, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_right, d_right2, sizeof(float2) * pntnum,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_radius, d_radius, sizeof(float) * pntnum,
        cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_lamMap));
    CUDA_CALL(cudaFree(d_pnt));
    CUDA_CALL(cudaFree(d_medial));
    CUDA_CALL(cudaFree(d_left));
    CUDA_CALL(cudaFree(d_right));
    CUDA_CALL(cudaFree(d_medial2));
    CUDA_CALL(cudaFree(d_left2));
    CUDA_CALL(cudaFree(d_right2));

    CUDA_CALL(cudaFree(d_x));
    CUDA_CALL(cudaFree(d_y));
    CUDA_CALL(cudaFree(d_xy));

    CUDA_CALL(cudaFree(d_radius));
}

void fastfilter(float** const d_I, float** const d_O, float** const d_kernel,
    float sigma, const int& _width, const int& _height) {
    if (sigma > 300)
        sigma = 300;
    // qDebug() <<"sigma" << sigma;
    int filter_size = int(floorf(sigma * 6) / 2) * 2 + 1;

    if (filter_size < 3) {
        CUDA_CALL(cudaMemcpy(*d_O, *d_I, sizeof(float) * _width * _height,
            cudaMemcpyDeviceToDevice));
        return;
    }

    float* d_buf;

    if (filter_size < 10) {

        int _fsize = filter_size;
        float* h_k = getKernel1d<float>(sigma, _fsize);
        float* d_k;

        CUDA_CALL(cudaMalloc((void**)&d_buf, sizeof(float) * _width * _height));

        CUDA_CALL(cudaMalloc((void**)&d_k, sizeof(float) * _fsize));
        CUDA_CALL(
            cudaMemcpy(d_k, h_k, sizeof(float) * _fsize, cudaMemcpyHostToDevice));

        device_convolve1d<float> << <dim3(_width / 2, _height / 2), dim3(2, 2) >> > (
            *d_I, d_buf, d_k, _width, int(_fsize / 2), false);
        // CUDA_CALL(cudaStreamSynchronize(cs[2]));

        device_convolve1d<float> << <dim3(_width / 2, _height / 2), dim3(2, 2) >> > (
            d_buf, *d_O, d_k, _height, int(_fsize / 2), true);
        CUDA_CALL(cudaDeviceSynchronize());

        ////deviceArraySumPrint(*d_O, __LINE__, _width , _height);

        CUDA_CALL(cudaFree(d_k));
        delete[] h_k;

    }
    else {
        if ((_width < 2) || (_height < 2)) {
            CUDA_CALL(cudaMemcpy(*d_O, *d_I, sizeof(float) * _width * _height,
                cudaMemcpyDeviceToDevice));
            return;
        }
        float* d_buf2;
        float* d_buf3;

        // grids.x = _width / 2;
        // grids.y = _height / 2;

        const float _sig2 = sigma / 2.0f;
        const int _width2 = _width / 2;
        const int _height2 = _height / 2;

        CUDA_CALL(cudaMalloc((void**)&d_buf, sizeof(float) * _width * _height));
        CUDA_CALL(cudaMalloc((void**)&d_buf2, sizeof(float) * _width2 * _height2));
        CUDA_CALL(cudaMalloc((void**)&d_buf3, sizeof(float) * _width2 * _height2));
        CUDA_CALL(cudaMemset(d_buf, 0, sizeof(float) * _width * _height));
        CUDA_CALL(cudaMemset(d_buf2, 0, sizeof(float) * _width2 * _height2));
        CUDA_CALL(cudaMemset(d_buf3, 0, sizeof(float) * _width2 * _height2));
        device_convolve2d<float> << <dim3(_width / 4, _height / 4), dim3(4, 4) >> > (
            *d_I, d_buf, *d_kernel, _width, _height, 2, 2);
        CUDA_CALL(cudaDeviceSynchronize());
        // if (sigma == 15) {

        //    //deviceArraySumPrint(*d_I, __LINE__, _width, _height);
        //    //deviceArraySumPrint(d_buf, __LINE__, _width, _height);
        //}

        // grids.x = _width2 / 2;
        // grids.y = _height2 / 2;
        device_subsampling<float> << <dim3(_width2 / 2, _height2 / 2), dim3(2, 2) >> > (
            d_buf, d_buf2, _width, _height, _width2, _height2);
        CUDA_CALL(cudaDeviceSynchronize());
        // if (sigma == 15) {
        //     //deviceArraySumPrint(d_buf2, __LINE__, _width2, _height2);
        // }

        /////
        fastfilter(&d_buf2, &d_buf3, d_kernel, _sig2, _width2, _height2);

        // #if _DEBUG_HOSTTEST
        //         {
        //             float* h_test;
        //             std::ofstream out;
        //             CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(float) *
        //             _width2 * _height2)); CUDA_CALL(cudaMemcpy(h_test, d_buf3,
        //             sizeof(float) * _width2 * _height2, cudaMemcpyDeviceToHost));
        //
        //             out.open("ttt.bin", std::ios::binary | std::ios::out);
        //             out.write(reinterpret_cast<char*>(h_test), sizeof(float) *
        //             _width2 * _width2); out.close();
        //
        //             CUDA_CALL(cudaFreeHost(h_test));
        //     }
        // #endif // _DEBUG_HOSTTEST
        ///
        device_upsampling<float> << <dim3(_width / 2, _height / 2), dim3(2, 2) >> > (
            d_buf3, *d_O, _width2, _height2);
        CUDA_CALL(cudaDeviceSynchronize());
        ////deviceArraySumPrint(*d_O, __LINE__, _width, _height);

        CUDA_CALL(cudaFree(d_buf));
        CUDA_CALL(cudaFree(d_buf2));


        CUDA_CALL(cudaFree(d_buf3));
    }
};

void test_gpuMSRCR(unsigned char* h_in, float* h_out, const float* h_weights,
    const float* h_sigmas, const float gain, const float offset,
    const int paramlen, const int width, const int height) {

    float* d_in, * d_out;
    float* d_fA, * d_fB; // , * d_fC;

    // float* d_weights, * d_sigmas;
    // float* d_sigmas;
    float* d_kernel;
    float kernel[] = { _def[0], _def[1], _def[2], _def[1], _def[0],
                      _def[1], _def[3], _def[4], _def[3], _def[1],
                      _def[2], _def[4], _def[5], _def[4], _def[2],
                      _def[1], _def[3], _def[4], _def[3], _def[1],
                      _def[0], _def[1], _def[2], _def[1], _def[0] };

    dim3 blocks = dim3(8, 8);
    dim3 grids = dim3(width / 8, height / 8);

    const int totalByte = sizeof(float) * width * height;

    CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_fA, totalByte));
    CUDA_CALL(cudaMalloc((void**)&d_fB, totalByte));
    // CUDA_CALL(cudaMalloc((void**)&d_fC, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));

    // CUDA_CALL(cudaMalloc((void**)&d_weights, sizeof(float) * paramlen));
    // CUDA_CALL(cudaMalloc((void**)&d_sigmas, sizeof(float) * paramlen));
    CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * _countof(kernel)));

    // CUDA_CALL(cudaMemcpy(d_in, h_in, sizeof(float) * width * height,
    // cudaMemcpyHostToDevice)); CUDA_CALL(cudaMemcpy(d_weights, h_weights,
    // sizeof(float) * paramlen, cudaMemcpyHostToDevice));
    // CUDA_CALL(cudaMemcpy(d_sigmas, h_sigmas, sizeof(float) * paramlen,
    // cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kernel, kernel, sizeof(float) * _countof(kernel),
        cudaMemcpyHostToDevice));

    float weight = 0;
    for (auto i = 0; i < paramlen; i++)
        weight += h_weights[i];

    auto init_fB = [=] __device__(const float& I, const float& param1,
        const int& misc) {
        return ((I < 0.1e-10) ? (46.0f * __logf(fabsf(125.0f * 0.1e-10)))
            : (46.0f * __logf(fabsf(125.0f * I)))) *
            param1;
    };
    auto cont_fB = [=] __device__(const float& I, const float& param1,
        const int& misc) {
        return ((I == 0) ? -700.0f : (46.0f * __logf(fabsf(125.0f * I)))) * param1;
    };
    auto extras1 = [=] __device__(const float& I, const float& param1,
        const float& param2, const int& misc) {
        return fmaf(I, param1, param2);

        // auto temp = fmaf(I, param1, param2);
        // return temp < 0 ? 0 : temp;
    };

    auto extras2 = [=] __device__(const float& I, const float& param1,
        const int& misc) {
        return fmaxf(I, param1);
    };
    float _min = 0.0f, _max = 0.0f;
    const size_t sharedMemSize = width * sizeof(float);

    auto mn = [=] __device__(const auto & a, const auto & b) {
        return a < b ? a : b;
    };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        return a > b ? a : b;
    };

    auto h_in_transform = std::make_unique<float[]>(width * height);
    std::transform(
        h_in, h_in + width * height, h_in_transform.get(),
        [=](const unsigned char& val) { return (val * 1.0f) / 255.0f; });

    auto aa = gpu_otsu_threshold(h_in_transform.get(), width, height, false);

    CUDA_CALL(cudaMemcpyAsync(d_in, h_in_transform.get(), totalByte,
        cudaMemcpyHostToDevice));

    ////deviceArraySumPrint(d_in,__LINE__);

    // #if _DEBUG_HOSTTEST
    //     {
    //         float* h_test;
    //         std::ofstream out;
    //         CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(float) * width *
    //         height)); CUDA_CALL(cudaMemcpy(h_test, d_in, sizeof(float) * width
    //         * height, cudaMemcpyDeviceToHost));
    //
    //         out.open("d_in.bin", std::ios::binary | std::ios::out);
    //         out.write(reinterpret_cast<char*>(h_test), sizeof(float) * width *
    //         height); out.close();
    //
    //         CUDA_CALL(cudaFreeHost(h_test));
    //     }
    // #endif // _DEBUG_HOSTTEST
    //




    device_customFunctor_in_1param<float>
        << <grids, blocks >> > (d_in, d_out, init_fB, weight, 0);

    CUDA_CALL(cudaDeviceSynchronize());
    ////deviceArraySumPrint(d_out, __LINE__);

    for (auto i = 0; i < paramlen; i++) {
        fastfilter(&d_in, &d_fA, &d_kernel, h_sigmas[i], width, height);

        device_customFunctor_in_1param<float>
            << <grids, blocks >> > (d_fA, d_fB, cont_fB, h_weights[i], 0);
        ////deviceArraySumPrint(d_fA, __LINE__);
        ////deviceArraySumPrint(d_fB, __LINE__);
        ////deviceArraySumPrint(d_out, __LINE__);
        cudaMemcpy(d_fA, d_out, totalByte, cudaMemcpyDeviceToDevice);
        device_simpleSub<float> << <grids, blocks >> > (d_fA, d_fB, d_out);
        ////deviceArraySumPrint(d_out, __LINE__);
    }

    device_customFunctor_in_2param<float>
        << <grids, blocks >> > (d_out, d_fA, extras1, gain, offset, 0);
    ////deviceArraySumPrint(d_fA, __LINE__);
    // device_reduction<512, float> << <width, height, sharedMemSize >> > (d_fA,
    // d_fA, mn); device_reduction<1, float> << <1, width, sharedMemSize >> >
    // (d_fA, d_fB, mn); CUDA_CALL(cudaMemcpyAsync(&_min, &d_fB[0], sizeof(float),
    // cudaMemcpyDeviceToHost));

    device_reduction<512, float>
        << <dim3(width), dim3(height), sharedMemSize >> > (d_fA, d_out, mx);
    CUDA_CALL(cudaDeviceSynchronize());
    device_reduction<1, float> << <1, width, sharedMemSize >> > (d_out, d_out, mx);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(
        cudaMemcpy(&_max, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost));

    // qDebug() << __LINE__ << _max;
    device_customFunctor_in_1param<float>
        << <grids, blocks >> > (d_fA, d_out, extras2, -1.0f * _max, 0);
    ////deviceArraySumPrint(d_out, __LINE__);

    // pp
    {

        // auto extra3 = [=] __device__(const float& I, const float& param1) {

        //    return I < param1 ? param1 : I;
        //};

        //_max = 0.0f;
        ////max cut
        //{

        //    device_reduction<512, float> << <width, height, sharedMemSize >> >
        //    (d_out, d_fA, mx); device_reduction<1, float> << <1, width,
        //    sharedMemSize >> > (d_fA, d_fB, mx); CUDA_CALL(cudaMemcpyAsync(&_max,
        //    &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

        //    device_customFunctor_in_1param<float> << <width, height>> > (d_out,
        //    d_out, extra3, -_max, width);
        //}

        // percentile
        const int _count = width * height;
        float* h_b0; // , * h_b1;
        CUDA_CALL(cudaMallocHost((void**)&h_b0, sizeof(float) * _count));
        // CUDA_CALL(cudaMalloc((void**)&h_b1, sizeof(float)* _count));

        CUDA_CALL(cudaMemcpy(h_b0, d_out, sizeof(float) * _count,
            cudaMemcpyDeviceToHost));

        std::vector<float> _v(h_b0, h_b0 + _count);

        std::sort(_v.begin(), _v.end());
        _v.erase(std::unique(_v.begin(), _v.end()), _v.end());

        auto xperVal = _v.at(int(_v.size() * 0.99f));
        auto nperVal = _v.at(int(_v.size() * 0.01f));

        auto extra4 = [=] __device__(const float& I, const float& param1,
            const float& param2, const int& misc) {
            if (I < param1)
                return param1;
            else if (I > param2)
                return param2;
            else
                return I;
        };

        device_customFunctor_in_2param<float>
            << <width, height >> > (d_out, d_out, extra4, nperVal, xperVal, 0);

        // CUDA_CALL(cudaMemcpyAsync(d_out,h_b0, sizeof(float)* _count,
        // cudaMemcpyHostToDevice));

        CUDA_CALL(cudaFreeHost(h_b0));
        // CUDA_CALL(cudaFreeHost(h_b1));
    }

    // Normalize
    _min = 0.0f, _max = 0.0f;
    arrsum(d_out, __LINE__);

    device_reduction<512, float>
        << <width, height, sharedMemSize >> > (d_out, d_fA, mn);

    arrsum(d_fA, __LINE__);

    device_reduction<1, float> << <1, width, sharedMemSize >> > (d_fA, d_fB, mn);

    arrsum(d_fB, __LINE__);

    CUDA_CALL(
        cudaMemcpyAsync(&_min, &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

    device_reduction<512, float>
        << <width, height, sharedMemSize >> > (d_out, d_fA, mx);



    device_reduction<1, float> << <1, width, sharedMemSize >> > (d_fA, d_fB, mx);
    CUDA_CALL(
        cudaMemcpyAsync(&_max, &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

    // CUDA_CALL(cudaStreamSynchronize(cs[1]));
    // CUDA_CALL(cudaStreamSynchronize(cs[2]));


    device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min, _max);


    ////deviceArraySumPrint(d_out, __LINE__);
    device_convolve2d<float>
        << <grids, blocks >> > (d_out, d_fA, d_kernel, width, height, 2, 2);
    arrsum(d_out, __LINE__);

    ////deviceArraySumPrint(d_fA, __LINE__);
    CUDA_CALL(cudaMemcpy(h_out, d_fA, sizeof(float) * width * height,
        cudaMemcpyDeviceToHost));

    arrsum(d_out, __LINE__);

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_fA));
    CUDA_CALL(cudaFree(d_fB));
    // CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_kernel));
}


 void test_gpuMSRCR2(
    float* h_in,
    void* h_out,
    const float* h_weights,
    const float* h_sigmas,
    const float gain,
    const float offset,
    const int paramlen,
    const int width,
    const int height)
{
     float* d_in, * d_out;
     float* d_fA, * d_fB; // , * d_fC;

     // float* d_weights, * d_sigmas;
     // float* d_sigmas;
     float* d_kernel;
     float kernel[] = { _def[0], _def[1], _def[2], _def[1], _def[0],
                       _def[1], _def[3], _def[4], _def[3], _def[1],
                       _def[2], _def[4], _def[5], _def[4], _def[2],
                       _def[1], _def[3], _def[4], _def[3], _def[1],
                       _def[0], _def[1], _def[2], _def[1], _def[0] };

     dim3 blocks = dim3(8, 8);
     dim3 grids = dim3(width / 8, height / 8);

     const int totalByte = sizeof(float) * width * height;

     CUDA_CALL(cudaMalloc((void**)&d_in, totalByte));
     CUDA_CALL(cudaMalloc((void**)&d_fA, totalByte));
     CUDA_CALL(cudaMalloc((void**)&d_fB, totalByte));
     // CUDA_CALL(cudaMalloc((void**)&d_fC, sizeof(float) * width * height));
     CUDA_CALL(cudaMalloc((void**)&d_out, totalByte));

     // CUDA_CALL(cudaMalloc((void**)&d_weights, sizeof(float) * paramlen));
     // CUDA_CALL(cudaMalloc((void**)&d_sigmas, sizeof(float) * paramlen));
     CUDA_CALL(cudaMalloc((void**)&d_kernel, sizeof(float) * _countof(kernel)));

     // CUDA_CALL(cudaMemcpy(d_in, h_in, sizeof(float) * width * height,
     // cudaMemcpyHostToDevice)); CUDA_CALL(cudaMemcpy(d_weights, h_weights,
     // sizeof(float) * paramlen, cudaMemcpyHostToDevice));
     // CUDA_CALL(cudaMemcpy(d_sigmas, h_sigmas, sizeof(float) * paramlen,
     // cudaMemcpyHostToDevice));
     CUDA_CALL(cudaMemcpy(d_kernel, kernel, sizeof(float) * _countof(kernel),
         cudaMemcpyHostToDevice));

     float weight = 0;
     for (auto i = 0; i < paramlen; i++)
         weight += h_weights[i];

     auto init_fB = [=] __device__(const float& I, const float& param1,
         const int& misc) {
         return ((I < 0.1e-10) ? (46.0f * __logf(fabsf(125.0f * 0.1e-10)))
             : (46.0f * __logf(fabsf(125.0f * I)))) *
             param1;
     };
     auto cont_fB = [=] __device__(const float& I, const float& param1,
         const int& misc) {
         return ((I == 0) ? -700.0f : (46.0f * __logf(fabsf(125.0f * I)))) * param1;
     };
     auto extras1 = [=] __device__(const float& I, const float& param1,
         const float& param2, const int& misc) {
         return fmaf(I, param1, param2);

         // auto temp = fmaf(I, param1, param2);
         // return temp < 0 ? 0 : temp;
     };

     auto extras2 = [=] __device__(const float& I, const float& param1,
         const int& misc) {
         return fmaxf(I, param1);
     };
     float _min = 0.0f, _max = 0.0f;
     const size_t sharedMemSize = width * sizeof(float);

     auto mn = [=] __device__(const auto & a, const auto & b) {
         return a < b ? a : b;
     };
     auto mx = [=] __device__(const auto & a, const auto & b) {
         return a > b ? a : b;
     };

     auto h_in_transform = std::make_unique<float[]>(width * height);
     std::transform(
         h_in, h_in + width * height, h_in_transform.get(),
         [=](const float& val) { return (val * 1.0f); });

     CUDA_CALL(cudaMemcpyAsync(d_in, h_in_transform.get(), totalByte,
         cudaMemcpyHostToDevice));

     ////deviceArraySumPrint(d_in,__LINE__);

     // #if _DEBUG_HOSTTEST
     //     {
     //         float* h_test;
     //         std::ofstream out;
     //         CUDA_CALL(cudaMallocHost((void**)&h_test, sizeof(float) * width *
     //         height)); CUDA_CALL(cudaMemcpy(h_test, d_in, sizeof(float) * width
     //         * height, cudaMemcpyDeviceToHost));
     //
     //         out.open("d_in.bin", std::ios::binary | std::ios::out);
     //         out.write(reinterpret_cast<char*>(h_test), sizeof(float) * width *
     //         height); out.close();
     //
     //         CUDA_CALL(cudaFreeHost(h_test));
     //     }
     // #endif // _DEBUG_HOSTTEST
     //




     device_customFunctor_in_1param<float>
         << <grids, blocks >> > (d_in, d_out, init_fB, weight, 0);

     CUDA_CALL(cudaDeviceSynchronize());
     ////deviceArraySumPrint(d_out, __LINE__);

     for (auto i = 0; i < paramlen; i++) {
         fastfilter(&d_in, &d_fA, &d_kernel, h_sigmas[i], width, height);

         device_customFunctor_in_1param<float>
             << <grids, blocks >> > (d_fA, d_fB, cont_fB, h_weights[i], 0);
         ////deviceArraySumPrint(d_fA, __LINE__);
         ////deviceArraySumPrint(d_fB, __LINE__);
         ////deviceArraySumPrint(d_out, __LINE__);
         cudaMemcpy(d_fA, d_out, totalByte, cudaMemcpyDeviceToDevice);
         device_simpleSub<float> << <grids, blocks >> > (d_fA, d_fB, d_out);
         ////deviceArraySumPrint(d_out, __LINE__);
     }

     device_customFunctor_in_2param<float>
         << <grids, blocks >> > (d_out, d_fA, extras1, gain, offset, 0);
     ////deviceArraySumPrint(d_fA, __LINE__);
     // device_reduction<512, float> << <width, height, sharedMemSize >> > (d_fA,
     // d_fA, mn); device_reduction<1, float> << <1, width, sharedMemSize >> >
     // (d_fA, d_fB, mn); CUDA_CALL(cudaMemcpyAsync(&_min, &d_fB[0], sizeof(float),
     // cudaMemcpyDeviceToHost));

     device_reduction<512, float>
         << <dim3(width), dim3(height), sharedMemSize >> > (d_fA, d_out, mx);
     CUDA_CALL(cudaDeviceSynchronize());
     device_reduction<1, float> << <1, width, sharedMemSize >> > (d_out, d_out, mx);
     CUDA_CALL(cudaDeviceSynchronize());
     CUDA_CALL(
         cudaMemcpy(&_max, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost));

     // qDebug() << __LINE__ << _max;
     device_customFunctor_in_1param<float>
         << <grids, blocks >> > (d_fA, d_out, extras2, -1.0f * _max, 0);
     ////deviceArraySumPrint(d_out, __LINE__);

     // pp
     {

         // auto extra3 = [=] __device__(const float& I, const float& param1) {

         //    return I < param1 ? param1 : I;
         //};

         //_max = 0.0f;
         ////max cut
         //{

         //    device_reduction<512, float> << <width, height, sharedMemSize >> >
         //    (d_out, d_fA, mx); device_reduction<1, float> << <1, width,
         //    sharedMemSize >> > (d_fA, d_fB, mx); CUDA_CALL(cudaMemcpyAsync(&_max,
         //    &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

         //    device_customFunctor_in_1param<float> << <width, height>> > (d_out,
         //    d_out, extra3, -_max, width);
         //}

         // percentile
         const int _count = width * height;
         float* h_b0; // , * h_b1;
         CUDA_CALL(cudaMallocHost((void**)&h_b0, sizeof(float) * _count));
         // CUDA_CALL(cudaMalloc((void**)&h_b1, sizeof(float)* _count));

         CUDA_CALL(cudaMemcpy(h_b0, d_out, sizeof(float) * _count,
             cudaMemcpyDeviceToHost));

         std::vector<float> _v(h_b0, h_b0 + _count);

         std::sort(_v.begin(), _v.end());
         _v.erase(std::unique(_v.begin(), _v.end()), _v.end());

         auto xperVal = _v.at(int(_v.size() * 0.99f));
         auto nperVal = _v.at(int(_v.size() * 0.01f));

         auto extra4 = [=] __device__(const float& I, const float& param1,
             const float& param2, const int& misc) {
             if (I < param1)
                 return param1;
             else if (I > param2)
                 return param2;
             else
                 return I;
         };

         device_customFunctor_in_2param<float>
             << <width, height >> > (d_out, d_out, extra4, nperVal, xperVal, 0);

         // CUDA_CALL(cudaMemcpyAsync(d_out,h_b0, sizeof(float)* _count,
         // cudaMemcpyHostToDevice));

         CUDA_CALL(cudaFreeHost(h_b0));
         // CUDA_CALL(cudaFreeHost(h_b1));
     }

     // Normalize
     _min = 0.0f, _max = 0.0f;
     arrsum(d_out, __LINE__);

     device_reduction<512, float>
         << <width, height, sharedMemSize >> > (d_out, d_fA, mn);

     arrsum(d_fA, __LINE__);

     device_reduction<1, float> << <1, width, sharedMemSize >> > (d_fA, d_fB, mn);

     arrsum(d_fB, __LINE__);

     CUDA_CALL(
         cudaMemcpyAsync(&_min, &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

     device_reduction<512, float>
         << <width, height, sharedMemSize >> > (d_out, d_fA, mx);



     device_reduction<1, float> << <1, width, sharedMemSize >> > (d_fA, d_fB, mx);
     CUDA_CALL(
         cudaMemcpyAsync(&_max, &d_fB[0], sizeof(float), cudaMemcpyDeviceToHost));

     // CUDA_CALL(cudaStreamSynchronize(cs[1]));
     // CUDA_CALL(cudaStreamSynchronize(cs[2]));


     device_simpleNorm2<float> << <width, height >> > (d_out, d_out, _min, _max);


     ////deviceArraySumPrint(d_out, __LINE__);
     device_convolve2d<float>
         << <grids, blocks >> > (d_out, d_fA, d_kernel, width, height, 2, 2);
     arrsum(d_out, __LINE__);

     ////deviceArraySumPrint(d_fA, __LINE__);
     CUDA_CALL(cudaMemcpy(h_out, d_fA, sizeof(float) * width * height,
         cudaMemcpyDeviceToHost));

     arrsum(d_out, __LINE__);

     CUDA_CALL(cudaFree(d_in));
     CUDA_CALL(cudaFree(d_out));
     CUDA_CALL(cudaFree(d_fA));
     CUDA_CALL(cudaFree(d_fB));
     // CUDA_CALL(cudaFree(d_in));
     CUDA_CALL(cudaFree(d_kernel));
}

__global__ void
device_8_connective(float* __restrict__ in, float* __restrict__ out1,
    float* __restrict__ out2, const int2* wv,
    const float mx) //,const int _width, const int _height)
{
    const int tid_x = threadIdx.x; // +blockDim.x * blockIdx.x;
    const int tid_y = blockIdx.x;  // threadIdx.y + blockDim.y * blockIdx.y;

    const int gid = tid_x + tid_y * gridDim.x;

    auto norm2 = [] __device__(const float& a, const float& b) -> float {
        return sqrtf(a * a + b * b);
    };

    float current = in[gid];
    float2 D = make_float2(0, 0);
    out1[gid] = 0;
    out2[gid] = 0;

    for (auto i = 0; i < 8; i++) {
        int x = tid_x + wv[i].x;
        int y = tid_y + wv[i].y;

        float pivot = mx;

        if (!((x < 0) || (x >= (blockDim.x)) || (y < 0) || (y >= (gridDim.x))))
            //    current = mx;
            // else
        {
            pivot = in[x + y * gridDim.x];
        }

        if (pivot < current) {
            current = pivot;
            auto dn = norm2(wv[i].x, wv[i].y);
            out1[gid] = wv[i].x / dn;
            out2[gid] = wv[i].y / dn;
        }
    }
    auto temp = out1[gid];
    out1[gid] = -out2[gid];
    out2[gid] = -temp;
}
void get_gpu_8_connective_point_min(float* h_in, float* h_out1, float* h_out2,
    const int width, const int height) {
    float* d_in, * d_out1, * d_out2;

    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_out1, sizeof(float) * width * height));
    CUDA_CALL(cudaMalloc((void**)&d_out2, sizeof(float) * width * height));

    CUDA_CALL(cudaMemcpy(d_in, h_in, sizeof(float) * width * height,
        cudaMemcpyHostToDevice));

    int2 wv[] = { make_int2(-1, -1), make_int2(0, -1), make_int2(1, -1),
                 make_int2(-1, 0),  make_int2(1, 0),  make_int2(-1, 1),
                 make_int2(0, 1),   make_int2(1, 1) };

    int2* d_wv;
    CUDA_CALL(cudaMalloc((void**)&d_wv, sizeof(int2) * 8));
    CUDA_CALL(cudaMemcpy(d_wv, wv, sizeof(int2) * 8, cudaMemcpyHostToDevice));

    // cudaStream_t cu[8];
    // for (auto i = 0; i < 8; i++) {
    //     CUDA_CALL(cudaStreamCreate(&cu[i]));
    // }

    // Normalize
    float _max = 0.0f;
    const size_t sharedMemSize = width * sizeof(float);

    // auto mn = [=] __device__(const float& a, const float& b) { return a < b ? a
    // : b; };
    auto mx = [=] __device__(const auto & a, const auto & b) {
        if (!(isnan(a) & isnan(b)))
            return a > b ? a : b;
        else
            return FLT_MIN;
    };

    device_reduction<512, float>
        << <width, height, sharedMemSize >> > (d_in, d_out1, mx);
    device_reduction<1, float> << <1, width, sharedMemSize >> > (d_out1, d_out2, mx);
    CUDA_CALL(cudaMemcpyAsync(&_max, &d_out2[0], sizeof(float),
        cudaMemcpyDeviceToHost));

    device_8_connective << <width, height >> > (d_in, d_out1, d_out2, d_wv,
        _max); // , width, height - 1);
// for (auto i = 0; i < 8; i++)
//{
// }

    cudaDeviceSynchronize();

    CUDA_CALL(cudaMemcpy(h_out1, d_out1, sizeof(float) * width * height,
        cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_out2, d_out2, sizeof(float) * width * height,
        cudaMemcpyDeviceToHost));

    // for (auto i = 0; i < 8; i++) {
    //     CUDA_CALL(cudaStreamDestroy(cu[i]));
    // }
    CUDA_CALL(cudaFree(d_wv));
    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out1));
    CUDA_CALL(cudaFree(d_out2));
}
void convertU8toF16(uchar*, float*, const int, const int, cudaStream_t) {}

void convertU16toF16(uchar*, float*, const int, const int, cudaStream_t) {}

void convertS16toF16(uchar*, float*, const int, const int, cudaStream_t) {}

void convertF16toU8(uchar*, float*, const int, const int, cudaStream_t) {}

void convertF16toU16(uchar*, float*, const int, const int, cudaStream_t) {}

void convertF16toS16(uchar*, float*, const int, const int, cudaStream_t) {}

__device__ __inline__ void
device_grid3d_function(float3* __restrict__ out,
    unsigned int* __restrict__ outIdx, const float3& r1,
    const float3& r2, const float3& r3, const float3& center,
    const float& radius, const int& around, const int& id) {
    const float pi = 4.0f * atanf(1.0f);
    for (int i = 0; i < around; i++) {
        const float _angle = 2.0 * pi / float(around) * i;
        const float _x = radius * __cosf(_angle);
        const float _y = radius * __sinf(_angle);
        const float _z = 0.0f;

        out[i].x = (r1.x * _x + r1.y * _y + r1.z * _z) + center.x;
        out[i].y = (r2.x * _x + r2.y * _y + r2.z * _z) + center.y;
        out[i].z = (r3.x * _x + r3.y * _y + r3.z * _z) + center.z;

        if (i != around - 1) {
            outIdx[i * SUR_VERTICES + 0] = i + 1 + id * around;
            outIdx[i * SUR_VERTICES + 1] = i + 1 + id * around + 1;
            outIdx[i * SUR_VERTICES + 2] = i + 1 + id * around + 21;
            outIdx[i * SUR_VERTICES + 3] = i + 1 + id * around + 20;
        }
        else {
            outIdx[i * SUR_VERTICES + 0] = i + 1 + id * around;
            outIdx[i * SUR_VERTICES + 1] = i + 1 + id * around - 19;
            outIdx[i * SUR_VERTICES + 2] = i + 1 + id * around + 1;
            outIdx[i * SUR_VERTICES + 3] = i + 1 + id * around + 20;
        }
    }
}
__global__ void device_grid3d(const float3* __restrict__ in,
    float3* __restrict__ outVtx,
    unsigned int* __restrict__ outIdx,
    const float* __restrict__ radius,
    const int around, const int num) {

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < num) {
        auto len3 = [=](const float3& a) -> float {
            return __fsqrt_rn(a.x * a.x + a.y * a.y + a.z * a.z);
        };

        const float pi = 4.0f * atanf(1.0f);
        const float3 unit_ = make_float3(0.0f, 0.0f, 1.0f);

        float3 unit = (id + 1) == num ? in[id] - in[id - 1] : in[id + 1] - in[id];
        {
            auto am = len3(unit);
            unit /= am;
        }
        float3 unit_tr = make_float3(unit.y * unit_.z - unit.z * unit_.y,
            unit.z * unit_.x - unit.x * unit_.z,
            unit.x * unit_.y - unit.y * unit_.x);
        {
            auto am = len3(unit_tr);
            unit_tr /= am;
        }

        const float _theta = -acosf(unit.z);

        const float _ct = __cosf(_theta);
        const float _st = __sinf(_theta);
        const float _rev_theta = 1.0f - __cosf(_theta);

        const float3 _r1 =
            make_float3(unit_tr.x * unit_tr.x * _rev_theta + _ct,
                unit_tr.y * unit_tr.x * _rev_theta - unit_tr.z * _st,
                unit_tr.z * unit_tr.x * _rev_theta + unit_tr.y * _st);
        const float3 _r2 =
            make_float3(unit_tr.x * unit_tr.y * _rev_theta + unit_tr.z * _st,
                unit_tr.y * unit_tr.y * _rev_theta + _ct,
                unit_tr.z * unit_tr.y * _rev_theta - unit_tr.x * _st);
        const float3 _r3 =
            make_float3(unit_tr.x * unit_tr.z * _rev_theta - unit_tr.y * _st,
                unit_tr.y * unit_tr.z * _rev_theta + unit_tr.z * _st,
                unit_tr.z * unit_tr.z * _rev_theta + _ct);

        device_grid3d_function(outVtx + (id * around),
            outIdx + (id * around * SUR_VERTICES), _r1, _r2, _r3,
            in[id], radius[id], around, id);
    }
}

void grid3d_function(const float3* __In_Vertex, const float* __In_Rad1,
    const float* __In_Rad2, float3* __Out_Vertex,
    unsigned int* __Out_Index,
    const int& vertexNumber, // precalculate
    const int& indexNumber,  // precalculate
    const int& around, const int& pointNumber) {

    float3* d_in;
    float* d_rad;

    float3* d_outVtx;
    unsigned int* d_outIdx;

    cudaMalloc((void**)&d_in, sizeof(float3) * pointNumber);
    cudaMalloc((void**)&d_rad, sizeof(float) * pointNumber);

    cudaMalloc((void**)&d_outVtx, sizeof(float3) * vertexNumber);
    cudaMalloc((void**)&d_outIdx, sizeof(unsigned int) * indexNumber);

    // memcpy
    {
        cudaMemcpy(d_in, __In_Vertex, sizeof(float3) * pointNumber,
            cudaMemcpyHostToDevice);
        auto _r = std::make_unique<float[]>(100);
        for (int i = 0; i < pointNumber; i++)
            _r.get()[i] = fminf(__In_Rad1[i], __In_Rad2[i]);
        cudaMemcpy(d_rad, _r.get(), sizeof(float) * pointNumber,
            cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    device_grid3d << <(pointNumber / 512) + 1, pointNumber >> > (
        d_in, d_outVtx, d_outIdx, d_rad, around, pointNumber);

    // memcpy
    {
        cudaMemcpy(__Out_Vertex, d_outVtx, sizeof(float3) * vertexNumber,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(__Out_Index, d_outIdx, sizeof(unsigned int) * indexNumber,
            cudaMemcpyDeviceToHost);
    }

    cudaFree(d_in);
    cudaFree(d_rad);

    cudaFree(d_outVtx);
    cudaFree(d_outIdx);
}

float gpu_otsu_threshold(float* data, const int width, const int height,
    const bool bNorm) {
    // auto otsu_threshold = [&](float* d_data, const int width, const int
    // height)->float

    cudaPointerAttributes attrib;
    cudaPointerGetAttributes(&attrib, data);

    float* d_data;
    float* d_buffer1 = nullptr, * d_buffer2 = nullptr;
    const int streamSize = width * height;
    float _max = 0.0f;

    CUDA_CALL(cudaMalloc((void**)&d_data, streamSize * sizeof(float)));

    if (attrib.type == cudaMemoryTypeDevice) {
        CUDA_CALL(cudaMemcpyAsync(d_data, data, streamSize * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }
    else {
        CUDA_CALL(cudaMemcpyAsync(d_data, data, streamSize * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    // Normalize
    // if(bNorm)
    {

        CUDA_CALL(cudaMalloc((void**)&d_buffer1, streamSize * sizeof(float)));
        CUDA_CALL(cudaMalloc((void**)&d_buffer2, streamSize * sizeof(float)));

        const size_t sharedMemSize = width * sizeof(float);

        // cudaStream_t cs[2];

        // CUDA_CALL(cudaStreamCreate(&cs[0]));
        // CUDA_CALL(cudaStreamCreate(&cs[1]));

        // auto mn = [=] __device__(const float& a, const float& b) { return a < b ?
        // a : b; };
        auto mx = [=] __device__(const auto & a, const auto & b) {
            return a > b ? a : b;
        };

        // float _min = 0.0f;

        // floating precision
        // device_reduction<512, float> << <width, height, sharedMemSize, cs[1] >> >
        // (d_data, //d_buffer1, /mn); device_reduction<1, float> << <1, width,
        // sharedMemSize, cs[1] >> > (d_buffer1, d_xx, mn);
        device_reduction<512, float>
            << <width, height, sharedMemSize >> > (d_data, d_buffer1, mx);
        device_reduction<1, float>
            << <1, width, sharedMemSize >> > (d_buffer1, d_buffer2, mx);

        CUDA_CALL(cudaMemcpy(&_max, &d_buffer2[0], sizeof(float),
            cudaMemcpyDeviceToHost));
        // CUDA_CALL(cudaStreamSynchronize(cs[1]));
        // CUDA_CALL(cudaStreamSynchronize(cs[2]));

        device_simpleNorm2<float>
            << <width, height >> > (d_data, d_buffer1, 0, _max, 1000.0f);
        CUDA_CALL(cudaDeviceSynchronize());
        // CUDA_CALL(cudaStreamDestroy(cs[0]));
        // CUDA_CALL(cudaStreamDestroy(cs[1]));
    }
    // else
    //{
    // CUDA_CALL(cudaMemcpyAsync(d_buffer1, d_data, streamSize * sizeof(float),
    // cudaMemcpyDeviceToDevice));
    //}

    int* d_bucket;//, *h_bucket;
    float* d_result, * h_result;

    CUDA_CALL(cudaMalloc((void**)&d_bucket, 1001 * sizeof(int)));
    //CUDA_CALL(cudaMallocHost((void **)&h_bucket, 1001 * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_result, 1001 * sizeof(float)));
    CUDA_CALL(cudaMallocHost((void**)&h_result, 1001 * sizeof(float)));

    CUDA_CALL(cudaMemset(d_bucket, 0, 1001 * sizeof(int)));

    device_bucket << <width, height >> > (d_buffer1, d_bucket, streamSize);
    CUDA_CALL(cudaDeviceSynchronize());
    // //arrsum3(d_bucket, __LINE__);
    device_otsu << <1001, 1 >> > (d_bucket, d_result, 1001, streamSize);

    CUDA_CALL(cudaMemcpyAsync(h_result, d_result, 1001 * sizeof(float),
        cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpyAsync(h_bucket, d_bucket, 101 * sizeof(int),
    // cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaDeviceSynchronize());
    // //arrsum2(d_result, __LINE__);

    // std::vector<float> weight;
    // std::vector<float> var;
    // float minWCV = -100;
    // int minWCVid = -1;
    //
    // weight.reserve(101);
    // var.reserve(101);
    //
    //
    // for (auto i = 0; i < 101; i++)
    //{
    //     qDebug() << h_result[i];
    //     //(std::accumulate(&h_bucket[0], &h_bucket[i], 0) /streamSize)*
    //     //std::accumulate(&kernel[0], &kernel[5 * 5], 0);
    // }

    auto argmin = [](const float* arr, const int& n) -> unsigned int {
        return std::distance(arr, std::min_element(arr, arr + n));
    };

    auto argmax= [](const float* arr, const int& n) -> unsigned int {
        return std::distance(arr, std::max_element(arr, arr + n));
    };


    auto result = argmin(h_result, 1001);
    auto result2 = argmax(h_result, 1001);

    auto ss = (bNorm ? 1 : _max) * result2 * 0.001f;
    CUDA_CALL(cudaFree(d_buffer1));
    CUDA_CALL(cudaFree(d_buffer2));

    CUDA_CALL(cudaFree(d_bucket));
    CUDA_CALL(cudaFree(d_result));

    CUDA_CALL(cudaFree(d_data));

    //CUDA_CALL(cudaFreeHost(h_bucket));
    CUDA_CALL(cudaFreeHost(h_result));

    return (bNorm ? 1 : _max) * result * 0.001f;
}

template <class T> __device__ unsigned char saturate(const T& val) {
    return fmaxf(0, fminf(255, val));
}

__global__ void device_CalcHistogram(const unsigned char* __restrict__ img,
    int* __restrict__ hist, const int histSize,
    const int tilesSizeX, const int tilesSizeY,
    const int tilesX) {
    const unsigned int gidx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int gidy = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int gid = gidy * blockDim.x + gidx;

    // const unsigned int sidx = threadIdx.x + blockDim.x * blockIdx.x;
    // const unsigned int sidy = threadIdx.y + blockDim.y * blockIdx.y;

    const int tx = gidx / tilesSizeX;
    const int ty = gidy / tilesSizeY;

    const int tid = tx + ty * tilesX;

    auto tileHist = hist + tid * histSize;

    atomicAdd(&tileHist[img[gid]], 1);
}
__global__ void device_ClipHistogram(int* __restrict__ hist,
    int* __restrict__ clipped, const int pitch,
    const float clipLimit) {
    // extern __shared__ int sclip[];
    const unsigned int id = threadIdx.x;

    const unsigned int tileid = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int hid = tileid;

    auto tileHist = hist + hid * pitch;
    auto tileClipped = clipped + tileid * pitch;

    // sclip[id] = tileClipped[id];

    if (clipLimit > 0) {
        if (tileHist[id] > clipLimit) {
            tileClipped[id] = tileHist[id] - clipLimit;
            tileHist[id] = clipLimit;
        }
        else
            tileClipped[id] = 0;
        __syncthreads();
        int value = tileClipped[id];

        value += __shfl_xor_sync(0xffffffff, value, 1);
        value += __shfl_xor_sync(0xffffffff, value, 2);
        value += __shfl_xor_sync(0xffffffff, value, 4);
        value += __shfl_xor_sync(0xffffffff, value, 8);
        value += __shfl_xor_sync(0xffffffff, value, 16);

        tileClipped[id] = value;
    }
}

__global__ void device_ClipHistogram2(int* __restrict__ hist,
    int* __restrict__ clipped,
    const int histSize) {
    extern __shared__ int sclip[];
    const int id = threadIdx.x;

    const unsigned int tileid = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned int hid = tileid;

    int* tileHist = hist + hid * histSize;
    int* tileClipped = clipped + tileid * histSize;

    const int laneId = id % 0x20;

    sclip[id] = tileClipped[id];

    __syncthreads();

    int clipSum = 0;

    for (int i = laneId; i < histSize; i += 0x20)
        clipSum += sclip[i];
    __syncthreads();

    int redistBatch = clipSum / histSize;
    int residual = clipSum - redistBatch * histSize;

    tileHist[id] += redistBatch;

    if (residual != 0) {
        int residualStep = (histSize < residual) ? 1 : (histSize / residual);

        if ((id % residualStep == 0) && (residual * residualStep - id) > 0)
            tileHist[id]++;
    }
}

__global__ void device_CalcLut(const int* __restrict__ hist,
    unsigned char* __restrict__ lut,
    const float lutScale) {
    extern __shared__ int sums[];
    int tid = threadIdx.x;

    int laneId = tid % 0x20;
    int warpId = threadIdx.x / 0x20;

    const unsigned int hid = blockIdx.y * gridDim.x + blockIdx.x;
    auto tileHist = hist + hid * blockDim.x;
    auto tileLut = lut + hid * blockDim.x;

    int val = tileHist[tid];

    for (int i = 1; i <= blockDim.x; i *= 2) {
        unsigned int mask = 0xffffffff;
        int n = __shfl_up_sync(mask, val, i, blockDim.x);

        if (laneId >= i)
            val += n;
    }

    if (threadIdx.x % 0x20 == 0x1f) {
        sums[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0 && laneId < (blockDim.x / 0x1f)) {
        int warpSum = sums[laneId];

        int mask = (1 << (blockDim.x / 0x20)) - 1;
        for (int i = 1; i <= blockDim.x / 0x20; i *= 2) {
            int n = __shfl_up_sync(mask, warpSum, i, (blockDim.x / 0x20));

            if (laneId >= i)
                warpSum += n;
        }

        sums[laneId] = warpSum;
    }

    __syncthreads();

    int blockSum = 0;
    if (warpId > 0) {
        blockSum = sums[warpId - 1];
    }

    val += blockSum;

    tileLut[tid] = saturate<float>(val * lutScale);
}

__global__ void device_CLAHEInterpolation(
    const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
    /*unsigned int* __restrict__ out,*/
    const unsigned char* __restrict__ lut, const int* __restrict__ ind1_p,
    const int* __restrict__ ind2_p, const float* __restrict__ xa_p,
    const float* __restrict__ xa1_p, const int tileSizeY, const int tilesX,
    const int tilesY, const int tileStep) {
    const int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    const int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    const int gid = tidy * blockDim.x + tidx;

    const float inv_th = 1.0f / tileSizeY;

    const float tyf = tidy * inv_th - 0.5f;

    int ty1 = floorf(tyf);
    int ty2 = ty1 + 1;

    const float ya = tyf - ty1;
    const float ya1 = 1.0f - ya;

    ty1 = fmaxf(ty1, 0);
    ty2 = fminf(ty2, tilesY - 1);

    const unsigned char* lutPlane1 = lut + ty1 * tileStep * tilesX;
    const unsigned char* lutPlane2 = lut + ty2 * tileStep * tilesX;

    const int ind1 = ind1_p[tidx] + in[gid];
    const int ind2 = ind2_p[tidx] + in[gid];

    out[gid] = saturate<float>(
        (lutPlane1[ind1] * xa1_p[tidx] + lutPlane1[ind2] * xa_p[tidx]) * ya1 +
        (lutPlane2[ind1] * xa1_p[tidx] + lutPlane2[ind2] * xa_p[tidx]) * ya);
}

void gpu_CLAHE(unsigned char* in, unsigned char* out, const float clipLimit_,
    const int tilesX, const int tilesY, const int width,
    const int height) {
    int histSize = 256;

    const int tileSizeX = width / tilesX;
    const int tileSizeY = width / tilesY;

    const int tileSizeTotal = tileSizeX * tileSizeY;
    const int gridArea = tilesX * tilesY;
    const int inArea = width * height;
    const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

    int clipLimit = 0;
    if (clipLimit_ > 0.0f) {
        clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
        clipLimit = std::max(clipLimit, 1);
    }

    unsigned char* h_out;
    unsigned char* d_in, * d_out;

    int* d_hist;
    int* d_clipped;

    unsigned char* d_lut;

    CUDA_CALL(cudaMalloc((void**)&d_in, sizeof(unsigned char) * inArea));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(unsigned char) * inArea));
    CUDA_CALL(cudaMalloc((void**)&d_hist, sizeof(int) * gridArea * histSize));
    CUDA_CALL(cudaMalloc((void**)&d_clipped, sizeof(int) * gridArea * histSize));
    CUDA_CALL(
        cudaMalloc((void**)&d_lut, sizeof(unsigned char) * gridArea * histSize));

    CUDA_CALL(cudaMemset(d_hist, 0, sizeof(int) * gridArea * histSize));
    CUDA_CALL(cudaMemset(d_clipped, 0, sizeof(int) * gridArea * histSize));
    CUDA_CALL(cudaMemcpy(d_in, in, sizeof(unsigned char) * inArea,
        cudaMemcpyHostToDevice));

    dim3 grids(tilesX, tilesX);
    // dim3 blocks(tileSizeX, tileSizeY);

    const size_t sharedMemSize = histSize * sizeof(int);

    {
        dim3 grids_(1, height);
        dim3 blocks_(width, 1);
        device_CalcHistogram << <grids_, blocks_ >> > (d_in, d_hist, histSize, tileSizeX,
            tileSizeY, tilesX);
    }
    device_ClipHistogram << <grids, histSize, sharedMemSize >> > (d_hist, d_clipped,
        histSize, clipLimit);
    device_ClipHistogram2 << <grids, histSize, sharedMemSize >> > (d_hist, d_clipped,
        histSize);
    device_CalcLut << <grids, histSize, sharedMemSize >> > (d_hist, d_lut, lutScale);

    {
        auto buf = std::make_unique<int[]>(width << 2);
        int* ind1_p = buf.get();
        int* ind2_p = ind1_p + width;
        float* xa_p = (float*)(ind2_p + width);
        float* xa1_p = xa_p + width;

        int lut_step = histSize / sizeof(unsigned char);
        float inv_tw = 1.0f / tileSizeX;

#pragma omp parallel for
        for (int x = 0; x < width; x++) {
            float txf = x * inv_tw - 0.5f;

            int tx1 = floorf(txf);
            int tx2 = tx1 + 1;

            xa_p[x] = txf - tx1;
            xa1_p[x] = 1.0f - xa_p[x];

            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX - 1);

            ind1_p[x] = tx1 * lut_step;
            ind2_p[x] = tx2 * lut_step;
        }

        int* d_ind1_p, * d_ind2_p;
        float* d_xa_p, * d_xa1_p;

        CUDA_CALL(cudaMalloc((void**)&d_ind1_p, sizeof(int) * width));
        CUDA_CALL(cudaMalloc((void**)&d_ind2_p, sizeof(int) * width));
        CUDA_CALL(cudaMalloc((void**)&d_xa_p, sizeof(float) * width));
        CUDA_CALL(cudaMalloc((void**)&d_xa1_p, sizeof(float) * width));

        CUDA_CALL(cudaMemcpy(d_ind1_p, ind1_p, sizeof(int) * width,
            cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_ind2_p, ind2_p, sizeof(int) * width,
            cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_xa_p, xa_p, sizeof(float) * width,
            cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_xa1_p, xa1_p, sizeof(float) * width,
            cudaMemcpyHostToDevice));

        dim3 grids_(1, height);
        dim3 blocks_(width, 1);

        device_CLAHEInterpolation << <grids_, blocks_ >> > (
            d_in, d_out, d_lut, d_ind1_p, d_ind2_p, d_xa_p, d_xa1_p, tileSizeY,
            tilesX, tilesY, histSize / sizeof(unsigned char));

        h_out = out;
        if (h_out == nullptr) {
            // cudaMallocHost((void**)&h_out, sizeof(unsigned char) * inArea);
            h_out = new unsigned char[inArea];
        }

        CUDA_CALL(cudaMemcpy(h_out, d_out, sizeof(unsigned char) * inArea,
            cudaMemcpyDeviceToHost));

        CUDA_CALL(cudaFree(d_ind1_p));
        CUDA_CALL(cudaFree(d_ind2_p));
        CUDA_CALL(cudaFree(d_xa_p));
        CUDA_CALL(cudaFree(d_xa1_p));
        // cudaFreeHost(h_out);
    }

    CUDA_CALL(cudaFree(d_in));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_hist));
    CUDA_CALL(cudaFree(d_clipped));
    CUDA_CALL(cudaFree(d_lut));
}
#include "Functor.h"

//namespace Functor

void getDeviceCount()
{
    //CUdevice dev;
    int deviceCount = 0;

    cuInit(0);
    cuDeviceGetCount(&deviceCount);

    if (deviceCount <= 0)
    {
        static constexpr bool useCUDA = false;

    }
    else
    {
        static constexpr bool useCUDA = true;
#ifndef useCUDA
#define useCUDA
#endif // !1

    }
    

}
bool __strtob(char*& pen)
{
    if (*pen == '1') return true;
    else return false;
}
inline int __strtol(const char*& pen, int val) {
    for (char c; (c = *pen ^ '0') <= 9; ++pen) val = val * 10 + c;
    return val;
}
inline float __strtof(const char*& pen) {
    static float const exp_table[]
        = { 1e5f, 1e4f, 1e3f, 1e2f, 10.f, 1.f, 0.1f, 1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-7f, 1e-8f, 1e-9f, 1e-10f, 1e-11f, 1e-12f, 1e-13f, 1e-14f, 1e-15f, 1e-16f, 1e-17 },
        * exp_lookup = &exp_table[5];
    bool isNeg = false;

    while (iswspace(*pen) | isspace(*pen))
    {
        *pen++;
    }
    if (*pen == '-')
    {
        isNeg = true;
        pen++;
    }
    int val = __strtol(pen);
    int neg_exp = 0;
    if (*pen == '.') {
        char const* fracs = ++pen;
        val = __strtol(pen, val);
        neg_exp = pen - fracs;
    }
    if ((*pen | ('E' ^ 'e')) == 'e') {
        neg_exp += *++pen == '-' ? __strtol(++pen) : -__strtol(++pen);
    }
    if (isNeg)
        return -1 * val * exp_lookup[neg_exp];
    else
        return val * exp_lookup[neg_exp];
}
void upsampling_half(__half* in, __half* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
#ifdef useCUDA
    if (iSrcWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_upsampling<__half>(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#ifdef useCUDA
    else
        gpu_upsampling_half(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#endif // useCUDA
}

void upsampling_int(int* in, int* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
#ifdef useCUDA
    if(iSrcWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_upsampling<int>(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#ifdef useCUDA
    else
        gpu_upsampling_int(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#endif // useCUDA
}
void upsampling_float(float* in, float* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
#ifdef useCUDA
    if (iSrcWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_upsampling<float>(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#ifdef useCUDA
    else
        gpu_upsampling_float(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#endif // useCUDA
}
void upsampling_double(double* in, double* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
#ifdef useCUDA
    if (iSrcWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_upsampling<double>(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#ifdef useCUDA
    else
        gpu_upsampling_double(in, out, iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
#endif // useCUDA
}






void convolve1d_half(__half* in, __half* out, __half* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelSize)
{
}

void convolve1d_int(int* in, int* out, float* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelSize)
{
//#ifdef useCUDA
//    if(iImgWidth<128)
//#endif // useCUDA
//        cpu_convolve1d<int>(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
//#ifdef useCUDA
//    else
//        gpu_convolve1d_int(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
//#endif // useCUDA
}
void convolve1d_float(float* in, float* out, float* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelSize)
{
#ifdef useCUDA
    if (iImgWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_convolve1d<float>(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
#ifdef useCUDA
    else
        gpu_convolve1d_float(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
#endif // useCUDA
}
void convolve1d_double(double* in, double* out, double* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelSize)
{
#ifdef useCUDA
    if (iImgWidth < THRESHOLD_CPU)
#endif // useCUDA
        cpu_convolve1d<double>(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
#ifdef useCUDA
    else
        gpu_convolve1d_double(in, out, kernel, iImgWidth, iImgHeight, iKernelSize);
#endif // useCUDA
}



//void convolve2d_half(__half* in, __half* out, __half* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
//{
////#ifndef useCUDA
////    cpu_convolve2d<__half>(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
////#else
////    gpu_convolve2d_half(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
////#endif // useCUDA
//}
void convolve2d_half(__half* in, __half* out, __half* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
{
//#ifndef useCUDA
//    cpu_convolve2d<int>(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
//#else
//    gpu_convolve2d_int(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
//#endif // useCUDA
}
void convolve2d_int(int* in, int* out, float* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
{
#ifndef useCUDA
    cpu_convolve2d<int>(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
#else
    gpu_convolve2d_int(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
#endif // useCUDA
}
void convolve2d_float(float* in, float* out, float* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
{
#ifndef useCUDA
    cpu_convolve2d<float>(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
#else
    gpu_convolve2d_float(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
#endif // useCUDA
}
void convolve2d_double(double* in, double* out, double* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
{
#ifndef useCUDA
    cpu_convolve2d<double>(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth, iKernelHeight);
#else
    gpu_convolve2d_double(in, out, kernel, iImgWidth, iImgHeight, iKernelWidth,iKernelHeight);
#endif // useCUDA
}

void derivative2d_int(int* in, int* outx, int* outy, const int& iWidth, const int& iHeight)
{
#ifndef useCUDA
    cpu_derivative2d<int>(in, outx, outy, iWidth, iHeight);
#else
    gpu_derivative2d_int(in, outx, outy, iWidth, iHeight);
#endif // useCUDA
}

void derivative2d_float(float* in, float* outx, float* outy, const int& iWidth, const int& iHeight)
{
#ifndef useCUDA
    cpu_derivative2d<float>(in, outx, outy, iWidth, iHeight);
#else
    gpu_derivative2d_float(in, outx, outy, iWidth, iHeight);
#endif // useCUDA
}

void derivative2d_double(double* in, double* outx, double* outy, const int& iWidth, const int& iHeight)
{
#ifndef useCUDA
    cpu_derivative2d<double>(in, outx, outy, iWidth, iHeight);
#else
    gpu_derivative2d_double(in, outx, outy, iWidth, iHeight);
#endif // useCUDA
}

bool diffusefilt_half(__half* in, __half* out, const int& width, const int& height, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype)
{
#ifndef useCUDA
    return false;
#else
    gpu_diffusefilt_half(in, out, width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
#endif
    return true;
}

bool diffusefilt_float(float* in, float* out, const int& width, const int& height, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype)
{
#ifndef useCUDA
    return false;
#else
    gpu_diffusefilt_float(in, out, width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
#endif
    return true;
}

bool diffusefilt_double(double* in, double* out, const int& width, const int& height, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype)
{
#ifndef useCUDA
    return false;
#else
    gpu_diffusefilt_double(in, out, width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
#endif
    return true;
}


bool eigenimg_half(__half* in, __half* outlam1, __half* outlam2, __half* outIx, __half* outIy, const float& sigma, const int& width, const int& height)
{
#ifndef useCUDA
    return false;
#else
    gpu_eigenimg_half(in, outlam1, outlam2, outIx, outIy, sigma, width, height);
#endif
    return true;
}

bool eigenimg_float(float* in, float* outlam1, float* outlam2, float* outIx, float* outIy, const float& sigma, const int& width, const int& height)
{
#ifndef useCUDA
    return false;
#else
    gpu_eigenimg_float(in, outlam1, outlam2, outIx, outIy, sigma, width, height);
#endif
    return true;
}

bool eigenimg_double(double* in, double* outlam1, double* outlam2, double* outIx, double* outIy, const float& sigma, const int& width, const int& height)
{
#ifndef useCUDA
    return false;
#else
    gpu_eigenimg_double(in, outlam1, outlam2, outIx, outIy, sigma, width, height);
#endif
    return true;
}












bool frangifilt_half(__half* in, __half* out, unsigned int* _B, const int& width, const int& height, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing)
{
#ifndef useCUDA
    return false;
#else
    gpu_frangifilt_half(in, out, _B, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
#endif
    return true;
}

bool frangifilt_float(float* in, float* out, unsigned int* _B, const int& width, const int& height, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing)
{
#ifndef useCUDA
    return false;
#else
    //gpu_frangifilt_float(in, out, _B, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
#endif
    return true;
}

bool frangifilt_double(double* in, double* out, unsigned int* _B, const int& width, const int& height, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing)
{
#ifndef useCUDA
    return false;
#else
    gpu_frangifilt_double(in, out, _B, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
#endif
    return true;
}


template<class T>
void cpu_upsampling(T* in, T* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
    const int iStrideW = ceilf(iDestWidth / iSrcWidth);
    const int iStrideH = ceilf(iDestHeight / iSrcHeight);

    float _denominator = 0.0f, _a[2] = { .0f, }, _b[2] = { .0f, };
    float _x = 0.0f, _y = 0.0f;
    int _x00 = -1, _x11 = -1, _y00 = -1, _y11 = -1;
    int _idx = 0, _col = -1;
    for (int i = 0; i < iDestWidth; i++)
    {
        _x = i / iStrideW;
        _x00 = int(_x);
        _x11 = _x00 + 1;
        
        int _col = i * iDestHeight;

        for (int j = 0; j < iDestHeight; j++)
        {
            _idx = j + _col;
            _y = j / iStrideH;
            _y00 = int(_y);
            _y11 = _y00 + 1;

            _denominator = (_x11 - _x00) * (_y11 - _y00);

            if (_denominator != 0)
            {
                _a[0] = _x11 - _x;
                _a[1] = _x - _x00;
                _b[0] = _y11 - _y;
                _b[1] = _y - _y00;

                out[_idx] = (((_a[0] * in[_x00 + _y00 * iSrcWidth]) + (_a[1] * in[_x11 + _y00 * iSrcWidth])) * _b[0] + \
                    ((_a[0] * in[_x00 + _y11 * iSrcWidth]) + (_a[1] * in[_x11 + _y11 * iSrcWidth])) * _b[1]) / _denominator;
            }
            else
            {
                out[_idx] = in[_x00 + _y00 * iSrcWidth];
            }
        }
    }
}

template<class T>
void cpu_convolve1d(T* in, T* out, T* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelSize)
//Max cycle case is 2e+12 x kernelsize
{
    const int halfIndex = ceilf(iKernelSize / 2) - 1;
    int _ix = iImgWidth - 1;
    rep(j, iImgHeight)
    {
        int _j = j * (iImgWidth);
        rep(i, iImgWidth)
        {
            rep(k, iKernelSize)
            {
                int _ij = std::clamp(i + (k - halfIndex), 0, _ix);
                out[i + _j] += in[_ij + _j] * kernel[k];
            }
        }
    }
    _ix = iImgHeight - 1;
    if (iImgHeight>1)
    {
        rep(j, iImgWidth)
        {
            int _j = j * iImgHeight;
            rep(i, iImgHeight)
            {
                rep(k, iKernelSize)
                {
                    int _ij = std::clamp(j + (k - halfIndex), 0, _ix);
                    out[i + _j] += in[_ij + _j] * kernel[k];
                }
            }
        }
    }
}



template<class T>
void cpu_convolve2d(T* in, T* out, float* kernel, const int& iImgWidth, const int& iImgHeight, const int& iKernelWidth, const int& iKernelHeight)
{

}
template<class T>
void cpu_derivative2d(T* in, T* outx, T* outy, const int& iWidth, const int& iHeight)
{

}

template<class T>
void imageGaussFilt(T* image, const int& iWidth, const int& iHeight, const float& sigma, int kernelSize)
{
    const type_info& _test = typeid(image);

    if (kernelSize == 0)
        kernelSize = sigma * 6;
    int _init = -ceilf(kernelSize / 2.0f);
    const float _sig = (sigma * sigma);
    const int _itermax = int(_init * 2 + 1);
    std::shared_ptr<T[]> H(new T[_itermax]);

    float sum = 0;
    rep(i, _itermax)
    {
        H[i] = expf(-((_init * _init) / (2.0f * _sig)));
        _init++;
        sum += H[i];
    }

    rep(i, _itermax)
        H[i] /= sum;

    auto _out = std::make_unique(sizeof(T) * iWidth * iHeight);
    if (_test.hash_code() == _test0.hash_code())
        convolve1d_half((__half*)(image), (__half*)(_out), H.get(), iWidth, iH.get()eight, kernelSize);
    else if (_test.hash_code() == _test1.hash_code())
        convolve1d_int((int*)(image), (int*)(_out), H.get(), iWidth, iH.get()eight, kernelSize);
    else if (_test.hash_code() == _test2.hash_code())
        convolve1d_float((float*)(image), (float*)(_out), H.get(), iWidth, iH.get()eight, kernelSize);
    else if (_test.hash_code() == _test3.hash_code())
        convolve1d_double((double*)(image), (double*)(_out), H.get(), iWidth, iH.get()eight, kernelSize);
    CUDA_CALL(cudaMemcpyAsync(image, _out, sizeof(T) * iWidth * iHeight, cudaMemcpyHostToHost));
}

//
//template<typename T, typename T2>
//void convertSrc2Dest(T* a, T2* b, const int width, const int height)
//{
//    if (type_info(a) == uchar)
//    {
//        if (type_info(b) == float)
//            convertU8toF16(a, b, width, height);
//    }
//    else if (type_info(a) == ushort)
//    {
//        if (type_info(b) == float)
//            convertU16toF16(a, b, width, height);
//    }
//    else if (type_info(a) == short)
//    {
//        if (type_info(b) == float)
//            convertS16toF16(a, b, width, height);
//    }
//    else if (type_info(a) == uint)
//    {
//        if (type_info(b) == float)
//            convertU32toF16(a, b, width, height);
//    }
//    else if (type_info(a) == int)
//    {
//        if (type_info(b) == float)
//            convertS32toF16(a, b, width, height);
//    }
//    else if (type_info(a) == float)
//    {
//        if (type_info(b) == uchar)
//            convertF16toU8(a, b, width, height);
//        else if (type_info(b) == ushort)
//            convertF16toU16(a, b, width, height);
//        else if (type_info(b) == short)
//            convertF16toS16(a, b, width, height);
//        else if (type_info(b) == uint)
//            convertF16toU32(a, b, width, height);
//        else if (type_info(b) == int)
//            convertF16toS32(a, b, width, height);
//    }   
//}
//

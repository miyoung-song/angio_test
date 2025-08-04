#pragma once

#include"functor.h"
#include<qimage.h>

#include<vector>
#include<string.h>


class Model
{
	
public:
	enum dataType : int {
		GRAYSCALE8 = 0,//8
		RGBA8  = 1,//2,2,2,2
		RGBA16 = 2,//4,4,4,4
		RGB24  = 3,//8,8,8
		RGBA32 =4,//8,8,8,8
		GRAYSCALE16 = 5,//8,8,8,8
		None = -1
	};//
	
public:	
	Model();
	Model(int, int, int dataType = RGBA32);
	Model(int, int, QImage::Format);
	Model(uchar*, int, int, int dataType = RGBA32);
	Model(uchar*, int, int, int, int dataType = RGBA32);
	Model(uchar*, int, int, QImage::Format);
	Model(uchar*, int, int, int, QImage::Format);
	
	Model(const Model& src);
	Model& operator=(const Model&);
	uchar* operator[] (const int& idx);

	const void* operator[] (const int& idx) const;
	

	friend bool operator==(const Model& src, const Model& target);
	friend bool operator==(const Model&& src, const Model&& target);
	friend bool operator!=(const Model& src, const Model& target);
	friend bool operator!=(const Model&& src, const Model&& target);

	
	float& at(const int& _y, const int& _x);

	~Model();

	void create(int dataType = None);

	bool setChanelConfig(int& dataType);
	int getChannelNumber() const; 
	int getChannelBits() const; 
	
	//const uchar* getPageable 

	void release();
	
	
	bool empty();
	uchar* begin();
	uchar* end();

	bool copy(uchar* _begin, int);
	bool copy(uchar* _begin, uchar* _end);

	void convert2Gray() const;
	void convert2RGB() const;

	bool Convolve2d(Model*, float*, const int&, const int&);
	bool Derivative2d(Model*, Model*);
	bool DiffuseFilter(Model*, const int& iter = 1, const float& dt = 0.15, const float& rho = 1, const float& sigma = 1, const float& alpha = 0.001, const float& C = 1e-10, const int& mode = 1, const int& dftype = 0);
	bool Eigen2Image(Model*, Model*, Model*, Model*,const float& sigma = 1);
	bool FrangiFilter(Model*, unsigned int* B = nullptr, const float& sigMn = 0.2, const float& sigMx = 2, const float& sigStep = 0.1, const float& beta = 0.5, const float& gamma = 15, const float& threshold = 0, const bool& bRegionGrowing = false);
	bool Hessian2d(Model*, Model*, Model*, const float&);
	bool ImageGaussianFilter(Model*,const float&,const int&);
	bool Upsampling(Model*);

	QImage::Format Qformat() const;

	bool testPathFinder(std::vector<QPointF>& out, const QPointF& p1, const QPointF& p2, const int boundaryFactor = 2);
	
	float* test();
	
private:
	bool allocate();
	
	void setStep();
	void setChannelNumber(int);
	void setChannelBits(int);
	
	template<class T>
	bool _tConvolve1d(T* in, T* out, float* kernel, const int& inCol, const int& inRow, const int& kernelSize);
	template<class T>
	bool _tConvolve2d(T* in, T* out, T* kernel, const int& inCol, const int& inRow, const int& kernelWidth, const int& kernelHeight);
	template<class T>
	bool _tDerivative2d(T* in, T* outX, T* outY, const int& inCol, const int& inRow);
	template<class T>
	bool _tDiffuseFilter(T* in, T* out, const int& width, const int& height, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype);
	template<class T>
	bool _tEigen2Image(T* in,T* outlam1, T* outlam2, T* outIx, T* outIy, const float& sigma, const int& inCol, const int& inRow);
	template<class T>
	bool _tHessian2d(T* in, T* xx, T* xy, T* yy, const float& sigma, const int& inCol, const int& inRow);
	template<class T>
	bool _tFrangiFilter(T* in, T* out, unsigned int* B, const int& width, const int& height, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing);
	template<class T>
	bool _tImageGaussianFilter2d(T* in, T* out, const float& sigma, const int& inCol, const int& inRow, int kernelSize);
	template<class T>
	bool _tUpsampling(T* in, T* out, const int& inCol, const int& inRow, const int& outCol, const int& outRow);

	bool custom_tracker(const int& _i, const int& _j, std::vector<QPointF>& prev, float* speedImage, std::unique_ptr<bool[]>& activemap, std::vector<int>& snake, std::unique_ptr<float[]>& boundary, std::unique_ptr<float[]>& boxbuffer, std::unique_ptr<float[]>& valbuffer, std::unique_ptr<int[]>& _tempbuffer, std::unique_ptr<int[]>& ssign, const QPointF& p1, const QPointF& p2, const int& bF, const int& width, const int& height, const int& snakeLength, const int& bandWidth);

public:
	uchar* data;
	int rows, cols;
	int dims;
	int step;//A row's number of byte
	
	//No flags

private:
	float* h_data;
	int mVertexNumber, mIndexNumber;
	int mChannelNumber;
	int mChannelBits;

	QImage::Format mQFormat;
};

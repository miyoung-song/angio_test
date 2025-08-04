#pragma once
#include"Model.h"
#include<fstream>

Model::Model() : rows(-1),cols(-1), dims(0), step(0)
{
	
	this->data = nullptr;
	this->h_data = nullptr;

	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = QImage::Format::Format_Invalid;
}

Model::Model(int width, int height, int dataType) :rows(height), cols(width)
{
	this->data = nullptr;
	this->h_data = nullptr;

	this->step = 0;
	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = QImage::Format::Format_Invalid;

	this->create(dataType);
}

Model::Model(int width, int height, QImage::Format fmt) :rows(height), cols(width)
{
	this->data = nullptr;
	this->h_data = nullptr;

	this->step = 0;
	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = fmt;

	this->create(-1);
}



Model::Model(uchar* _bits, int width, int height, int dataType) :rows(height), cols(width)
{
	this->data = _bits;
	this->h_data = nullptr;
	
	this->step = 0;
	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = QImage::Format::Format_Invalid;

	this->create(dataType);
}


Model::Model(uchar* _bits, int width, int height, int step, int dataType) :rows(height), cols(width), step(step)
{
	this->data = _bits;
	this->h_data = nullptr;

	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = QImage::Format::Format_Invalid;

	this->create(dataType);
		
}

Model::Model(uchar* _bits, int width, int height, QImage::Format qformat) : rows(height), cols(width), step(0)
{
	this->data = _bits;
	this->h_data = nullptr;
	
	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = qformat;

	this->create(-1);
}

Model::Model(uchar* _bits, int width, int height, int step, QImage::Format qformat) : rows(height), cols(width), step(step)
{
	this->data = _bits;
	this->h_data = nullptr;

	this->mVertexNumber = 0;
	this->mIndexNumber = 0;

	this->mChannelNumber = 0;
	this->mChannelBits = 0;

	this->mQFormat = qformat;

	this->create(-1);
}


Model::Model(const Model& src)
{
	
	this->data = src.data;
	this->h_data = src.h_data;

	this->cols = src.cols;
	this->rows = src.rows;
	this->dims = src.dims;
	this->step = src.step;

	this->mVertexNumber = src.mVertexNumber;
	this->mIndexNumber = src.mIndexNumber;
		
	this->setChannelBits(src.mChannelBits);
	this->setChannelNumber(src.mChannelNumber);
	this->mQFormat = src.mQFormat;



}

Model& Model::operator=(const Model&)
{
	// TODO: 여기에 return 문을 삽입합니다.
	return *this;
}


uchar* Model::operator[](const int& idx)
{
	if (idx < 0 || idx >= this->cols)
	{
		return nullptr;
	}

	if (data != nullptr)
	{
		return &this->data[idx * step];
	}
	else
		return nullptr;
}

const void* Model::operator[](const int& idx) const
{
	return &this->h_data[idx * cols];
}

bool operator==(const Model& src, const Model& target)
{
	return (src.dims == target.dims) && (src.cols == target.cols) && (src.rows == target.rows) && (src.getChannelBits() == target.getChannelBits()) && (src.getChannelNumber() == target.getChannelNumber());
}
bool operator==(const Model&& src, const Model&& target)
{
	return (src.dims == target.dims) && (src.cols == target.cols) && (src.rows == target.rows) && (src.getChannelBits() == target.getChannelBits()) && (src.getChannelNumber() == target.getChannelNumber());
}

bool operator!=(const Model& src, const Model& target)
{
	return !(src == target);
}
bool operator!=(const Model&& src, const Model&& target)
{
	return !(src == target);
}
//bool operator==(Model& src, Model& target)
//{
//	if (src.dims != target.dims)
//		return false;
//	if (src.cols != target.cols)
//		return false;
//	if (src.rows != target.rows)
//		return false;
//	if (src.getChannelBits() != target.getChannelBits())
//		return false;
//	if (src.getChannelNumber() != target.getChannelNumber())
//		return false;
//	return true;
//}

Model::~Model()
{
	release();
}



void Model::create(int dataType)
{
	bool result = true;

	result = this->setChanelConfig(dataType);
	if (!result)
	{
		perror("Error: Invalid channel\n");
		exit(-1);
	}

	result = !(cols< 0) || (rows < 0);
	if (!result)
	{
		perror("Error: Invalid size\n");
		exit(-1);
	}
	
	result = this->allocate();
	if (!result)
	{
		perror("Error: Failed allocate, check a dimension\n");
		exit(-1);
	}
}


bool Model::empty()
{
	return (this->data == nullptr) ? true : false;
}


uchar* Model::begin()
{
	return &this->data[0];
}


uchar* Model::end()
{
	return &this->data[this->step*cols];
}



bool Model::allocate()
{
	// This is USHRT_MAXt allowed 3 dimension yet
	if (rows == 1)
		dims = 1;
	else
		dims = 2;

	if (this->dims != 0)
	{
		/*if (step != 0)
		{*/
			CUDA_CALL(cudaMallocHost((void**)&h_data, sizeof(float) * cols * rows));
			if (this->data)
				convertU16toF32(this->data, this->h_data, this->cols, this->rows, this->getChannelNumber());
			/*
			_tUpsampling(this->h_data, this->h_data, this->cols, this->rows, this->cols, this->rows);*/
			
		//}
		//else
		//{
		//	perror("Error: Need to more script for compatibility\n");
		//	return false;
		//	//this->data = new uchar[this->rows * this->cols];
		//}
	}
	else
	{
		perror("Error: Invalid dims\n");
		return false;
	}
	return true;
}


void Model::setStep()
{
	this->step = mChannelNumber * sizeof(uchar) * rows;
}


void Model::setChannelNumber(int v)
{
	this->mChannelNumber = v;
}


void Model::setChannelBits(int v)
{
	this->mChannelBits = v;
}

bool Model::custom_tracker(const int& _i, const int& _j, std::vector<QPointF>& prev, float* speedImage, std::unique_ptr<bool[]>& activemap, std::vector<int>& snake, std::unique_ptr<float[]>& boundary, std::unique_ptr<float[]>& boxbuffer, std::unique_ptr<float[]>& valbuffer, std::unique_ptr<int[]>& _tempbuffer, std::unique_ptr<int[]>& ssign, const QPointF& p1, const QPointF& p2, const int& bF, const int& width, const int& height, const int& snakeLength, const int& bandWidth)
{
	if ((abs(p2.x() - _i) <= bandWidth) && (abs(p2.y() - _j) <= bandWidth))
		return true;
	const int wxmn = int(fmaxf(0, _i - bF));
	const int wxmx = int(fminf(width - 1, _i + 1 + bF));
	const int wymn = int(fmaxf(0, _j - bF));
	const int wymx = int(fminf(height - 1, _j + 1 + bF));
	

	if (((wxmx - wxmn) != bandWidth) || ((wymx - wymn) != bandWidth))
		return false;
	//std::unique_ptr<float[]> buffer(new float[bandWidth * bandWidth]);

	float _curval = 0;
	for (auto j = wymn, ij = 0; j < wymx; j++, ij++)
	{
		for (auto i = wxmn, ii = 0; i < wxmx; i++, ii++)
		{
			if ((ij == bF) && (ii == bF))
				_curval = speedImage[i + j * width];

			int locInd = (ii + ij * bandWidth);

			if (boundary[locInd] == 1 && !activemap.get()[i + j * width])
				boxbuffer.get()[locInd] = speedImage[i + j * width];
			else
				boxbuffer.get()[locInd] = 0;

		}
	}


	/*std::unique_ptr<float[]> valbuffer(new float[snakeLength]);
	std::unique_ptr<int[]> _temp(new int[snakeLength]);
	std::unique_ptr<int[]> ssign(new int[snakeLength]);*/

	for (auto i = 0; i < snakeLength; i++)
		valbuffer.get()[i] = boxbuffer.get()[snake[i]];

	for (auto i = 0; i < snakeLength; i++)
	{
		float check = 0;
		if (i != snakeLength - 1)
			check = valbuffer.get()[i + 1] - valbuffer.get()[i];
		else
			check = valbuffer.get()[0] - valbuffer.get()[i];
		if (check == 0)
			_tempbuffer.get()[i] = 0;
		else if (check > 0)
			_tempbuffer.get()[i] = 1;
		else
			_tempbuffer.get()[i] = -1;
	}


	//std::vector<int> ssign; ssign.reserve(snakeLength);
	//std::vector<int> temp; temp.reserve(snakeLength);
	for (auto i = 0; i < snakeLength; i++)
	{
		if (i == 0)
			ssign.get()[i] = _tempbuffer.get()[snakeLength - 1] * _tempbuffer.get()[i];
		else
			ssign.get()[i] = _tempbuffer.get()[i - 1] * _tempbuffer.get()[i];
	}

	std::vector<int> bval;
	for (int i = 0; i < snakeLength; i++)
	{
		/*int temp = 0;
		if (i == 0)
		{
			temp = _temp.get()[snakeLength - 1] * _temp.get()[0];
		}
		else
			temp = _temp.get()[i - 1] * _temp.get()[i];*/

		if (ssign.get()[i] != 1)
		{
			int nxi = ((i + 1) >= snakeLength) ? 0 : i + 1;
			int pvi = ((i - 1) < 0) ? snakeLength - 1 : i - 1;
			if ((valbuffer.get()[nxi] != valbuffer.get()[i]) &
				(0 != valbuffer.get()[i]) &
				(0 != valbuffer.get()[pvi]) &
				(0 != valbuffer.get()[nxi]) &
				//((valbuffer.get()[pvi] < valbuffer.get()[i]) || (valbuffer.get()[nxi] < valbuffer.get()[i])))
				((valbuffer.get()[pvi] < valbuffer.get()[i]) || (valbuffer.get()[nxi] < valbuffer.get()[i])) &
				(_curval * 0.1 < valbuffer.get()[i]))
				bval.push_back(snake[i]);
		}
	}

	const int bvsz = bval.size();

	if (bvsz == 0)
		return false;

	//std::vector<int2> locpnts; locpnts.reserve(bval.size());
	//std::vector<int> locpntsy; locpntsy.reserve(bval.size());
	std::unique_ptr<QPointF[]> locpnts(new QPointF[bvsz]);
	for (auto v = 0; v < bvsz; v++)
	{
		locpnts.get()[v] = QPointF(wxmn + (bval[v] % bandWidth), wymn + (bval[v] / bandWidth));
		//locpntsy.push_back();
	}

	bval.clear();

	std::unique_ptr<std::pair<bool, int>[]> mapping(new std::pair<bool, int>[bandWidth * bandWidth]);
	for (auto j = wymn, ij = 0; j < wymx; j++, ij++)
	{
		for (auto i = wxmn, ii = 0; i < wxmx; i++, ii++)
		{
			int locidx = ii + ij * bandWidth;
			int globidx = i + j * width;
			if (!activemap.get()[globidx])
				mapping.get()[locidx] = std::pair<bool, int>(true, globidx);
			else
				mapping.get()[locidx] = std::pair<bool, int>(false, globidx);
		}
	}

	//for (auto li = 0; li < locpntsx.size(); li++)
	for (auto li = locpnts.get(); li != locpnts.get() + bvsz; li++)
	{
		prev.push_back(*li);
		for (int i = 0; i < bandWidth * bandWidth; i++)
		{
			if (mapping.get()[i].first)
				activemap.get()[mapping.get()[i].second] = true;
		}

		if (!(custom_tracker(li->x(), li->y(), prev, speedImage, (activemap), snake, (boundary), (boxbuffer), (valbuffer), (_tempbuffer), (ssign), p1, p2, bF, width, height, snakeLength, bandWidth)))
		{
			for (int i = 0; i < bandWidth * bandWidth; i++)
			{
				if (mapping.get()[i].first)
					activemap.get()[mapping.get()[i].second] = false;
			}
			prev.pop_back();
		}
		else
		{
			return true;
		}

	}
	return false;
}


float& Model::at(const int& _y, const int& _x)
{
	return this->h_data[_x + int(this->cols * _y)];
}

template<class T>
bool Model::_tHessian2d(T* in, T* outxx, T* outxy, T* outyy, const float& sigma, const int& inCol, const int& inRow)
{
	const type_info& _test = typeid(in);

	int _init = -roundf(sigma * 3.0f);
	const float sig2 = sigma * sigma;
	const int _itermax = int(abs(_init) * 2 + 1);
	const int _itermax2 = _itermax * _itermax;
	
	std::unique_ptr<T[]> X(new T[_itermax2]);
	std::unique_ptr<T[]> Y(new T[_itermax2]);

	std::unique_ptr<T[]> dGxx(new T[_itermax2]);
	std::unique_ptr<T[]> dGxy(new T[_itermax2]);
	std::unique_ptr<T[]> dGyy(new T[_itermax2]);

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
	
	const float _bufxx = 1.0f / (2.0 * M_PI * powf(sigma, 4.0f)) * (1.0f / (sig2 - 1.0f));
	const float _bufxy = 1.0f / (2.0 * M_PI * powf(sigma, 6.0f));
	//const float _bufyy
	for (int _j = 0; _j < _itermax; _j++)
	{
		for (int _i = 0; _i < _itermax; _i++)
		{
			const int _pos = _i + _j * _itermax;
			const int _invpos = _j + _i * _itermax;
			const T& _x = X[_pos];
			const T& _y = Y[_pos];
			const T _inBuf = expf(-(_x * _x + _y * _y) / (2.0f * sig2));
			dGxx[_pos] = _bufxx * (_x * _x) * _inBuf;
			dGxy[_pos] = _bufxy * _x * _y * _inBuf;
			dGyy[_invpos] = dGxx[_pos];
		}
	}

	if (_test.hash_code() == _test0.hash_code())
	{
		convolve2d_half((__half*)(in), (__half*)(outxx), (__half*)dGxx.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_half((__half*)(in), (__half*)(outxy), (__half*)dGxy.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_half((__half*)(in), (__half*)(outyy), (__half*)dGyy.get(), inCol, inRow, _itermax, _itermax);
	}
	else if (_test.hash_code() == _test1.hash_code())
	{
		/*convolve2d_int((int*)(in), (int*)(outxx), dGxx.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_int((int*)(in), (int*)(outxy), dGxy.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_int((int*)(in), (int*)(outyy), dGyy.get(), inCol, inRow, _itermax, _itermax);*/
	}
	else if (_test.hash_code() == _test2.hash_code())
	{
		convolve2d_float((float*)(in), (float*)(outxx), (float*)dGxx.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_float((float*)(in), (float*)(outxy), (float*)dGxy.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_float((float*)(in), (float*)(outyy), (float*)dGyy.get(), inCol, inRow, _itermax, _itermax);
	}
	else if (_test.hash_code() == _test3.hash_code())
	{
		convolve2d_double((double*)(in), (double*)(outxx), (double*)dGxx.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_double((double*)(in), (double*)(outxy), (double*)dGxy.get(), inCol, inRow, _itermax, _itermax);
		convolve2d_double((double*)(in), (double*)(outyy), (double*)dGyy.get(), inCol, inRow, _itermax, _itermax);
	}
	else
		return false;

	return true;
}

template<class T>
bool Model::_tFrangiFilter(T* in, T* out,unsigned int* outB, const int& width, const int& height, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold,const bool& bRegionGrowing)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		return frangifilt_half((__half*)(in), (__half*)(out), outB, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
	else if (_test.hash_code() == _test2.hash_code())
		return frangifilt_float((float*)(in), (float*)(out), outB, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
	else if (_test.hash_code() == _test3.hash_code())
		return frangifilt_double((double*)(in), (double*)(out), outB, width, height, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
	else
		return false;
	return false;
}

template<class T>
bool Model::_tUpsampling(T* in, T* out, const int& iSrcWidth, const int& iSrcHeight, const int& iDestWidth, const int& iDestHeight)
{
	const type_info& _test = typeid(in);

	if (_test.hash_code() == _test0.hash_code())
		upsampling_half((__half*)(in), (__half*)(out), iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
	else if (_test.hash_code() == _test1.hash_code())
		upsampling_int((int*)(in), (int*)(out), iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
	else if (_test.hash_code() == _test2.hash_code())
		upsampling_float((float*)(in), (float*)(out), iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
	else if (_test.hash_code() == _test3.hash_code())
		upsampling_double((double*)(in), (double*)(out), iSrcWidth, iSrcHeight, iDestWidth, iDestHeight);
	else
		return false;
	return true;
}

template<class T>
bool Model::_tConvolve1d(T* in, T* out, float* kernel, const int& inCol, const int& inRow, const int& kernelSize)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		convolve1d_half((__half*)(in), (__half*)(out), kernel, inCol, inRow, kernelSize);
	else if (_test.hash_code() == _test1.hash_code())
		convolve1d_int((int*)(in), (int*)(out), kernel, inCol, inRow, kernelSize);
	else if (_test.hash_code() == _test2.hash_code())
		convolve1d_float((float*)(in), (float*)(out), kernel, inCol, inRow, kernelSize);
	else if (_test.hash_code() == _test3.hash_code())
		convolve1d_double((double*)(in), (double*)(out), kernel, inCol, inRow, kernelSize);
	else
		return false;
	return true;
}

template<class T>
bool Model::_tConvolve2d(T* in, T* out, T* kernel, const int& inCol, const int& inRow, const int& kernelWidth, const int& kernelHeight)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		convolve2d_half((__half*)(in), (__half*)(out), (__half*)kernel,inCol,inRow,kernelWidth,kernelHeight);
	/*else if (_test.hash_code() == _test1.hash_code())
		convolve2d_int((int*)(in), (int*)(out), kernel, inCol, inRow, kernelWidth, kernelHeight);*/
	else if (_test.hash_code() == _test2.hash_code())
		convolve2d_float((float*)(in), (float*)(out), (float*)kernel, inCol, inRow, kernelWidth, kernelHeight);
	else if (_test.hash_code() == _test3.hash_code())
		convolve2d_double((double*)(in), (double*)(out), (double*)kernel, inCol, inRow, kernelWidth, kernelHeight);
	else
		return false;
	return true;
}

template<class T>
bool Model::_tDerivative2d(T* in, T* outX, T* outY, const int& inCol, const int& inRow)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		derivative2d_int((int*)(in), (int*)(outX), (int*)(outY), inCol,inRow);
	else if (_test.hash_code() == _test1.hash_code())
		derivative2d_int((int*)(in), (int*)(outX), (int*)(outY), inCol, inRow);
	else if (_test.hash_code() == _test2.hash_code())
		derivative2d_float((float*)(in), (float*)(outX), (float*)(outY), inCol, inRow);
	else if (_test.hash_code() == _test3.hash_code())
		derivative2d_double((double*)(in), (double*)(outX), (double*)(outY), inCol, inRow);
	else
		return false;
	return true;
}

template<class T>
bool Model::_tDiffuseFilter(T* in, T* out, const int& width, const int& height, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		return diffusefilt_half((__half*)(in), (__half*)(out), width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
	else if (_test.hash_code() == _test2.hash_code())
		return diffusefilt_float((float*)(in), (float*)(out), width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
	else if (_test.hash_code() == _test3.hash_code())
		return diffusefilt_double((double*)(in), (double*)(out), width, height, iter, dt, rho, sigma, alpha, C, mode, dftype);
	else
		return false;
	return false;
}

template<class T>
bool Model::_tEigen2Image(T* in, T* outlam1, T* outlam2, T* outIx, T* outIy, const float& sigma, const int& width, const int& height)
{
	const type_info& _test = typeid(in);
	if (_test.hash_code() == _test0.hash_code())
		return eigenimg_half((__half*)(in), (__half*)(outlam1), (__half*)(outlam1), (__half*)(outlam1), (__half*)(outlam1), sigma, width, height);
	else if (_test.hash_code() == _test2.hash_code())
		return eigenimg_float((float*)(in), (float*)(outlam1), (float*)(outlam1), (float*)(outlam1), (float*)(outlam1), sigma, width, height);
	else if (_test.hash_code() == _test3.hash_code())
		return eigenimg_double((double*)(in), (double*)(outlam1), (double*)(outlam1), (double*)(outlam1), (double*)(outlam1), sigma, width, height);
	else
		return false;
	return false;
}

template<class T>
bool Model::_tImageGaussianFilter2d(T* in, T* out, const float& sigma, const int& inCol, const int& inRow, int kernelSize)
{
	const type_info& _test = typeid(in);
	if (kernelSize == 0)
		kernelSize = sigma * 6;
	int _init = -ceilf(kernelSize / 2.0f);
	const float _sig = (sigma * sigma);
	const int _itermax = int(abs(_init) * 2 + 1);
	std::shared_ptr<float[]> H(new float[_itermax]);
	
	float sum = 0;
	for (auto i = 0; i < _itermax; i++)
	{
		H[i] = expf(-((_init * _init) / (2.0f * _sig)));
		_init++;
		sum += H[i];
	}

	for (auto i = 0; i < _itermax; i++)
		H[i] /= sum;

	if (_test.hash_code() == _test0.hash_code())
		convolve1d_half((__half*)(in), (__half*)(out), (__half*)H.get(), inCol, inRow, kernelSize);
	/*else if (_test.hash_code() == _test1.hash_code())
		convolve1d_int((int*)(in), (int*)(out), H.get(), inCol, inRow, kernelSize);*/
	else if (_test.hash_code() == _test2.hash_code())
		convolve1d_float((float*)(in), (float*)(out), (float*)H.get(), inCol, inRow, kernelSize);
	else if (_test.hash_code() == _test3.hash_code())
		convolve1d_double((double*)(in), (double*)(out), (double*)H.get(), inCol, inRow, kernelSize);
	else
		return false;
	return true;
}


bool Model::copy(uchar* _begin, int number)
{
	return this->copy(_begin, _begin + number);
}


bool Model::copy(uchar* _begin, uchar* _end)
{
	int targSize = (_end - _begin);
	if ((targSize) != (this->end() - this->begin()))
	{
		perror("Error: invoke new size before copying target\n");
		return false;
	}
	std::copy(_begin, _begin + sizeof(uchar) * targSize, this->data);
	//memcpy_s(this->data, sizeof(uchar) * targSize, _begin, sizeof(uchar) * targSize);
	return true;
}

void Model::convert2Gray() const
{
}

void Model::convert2RGB() const
{
}

bool Model::Hessian2d(Model* outxx, Model* outxy, Model* outyy, const float& sigma)
{
	if((*this!=*outxx) || (*this != *outxy) || (*this != *outyy))
		return false;

	return this->_tHessian2d(this->h_data, outxx->h_data, outxy->h_data, outyy->h_data, sigma, this->cols, this->rows);
}

bool Model::Eigen2Image(Model* outlam1, Model* outlam2, Model* outIx, Model* outIy,const float& sigma)
{
	if ((*this != *outlam1) || (*this != *outlam2) || (*this != *outIx) || (*this != *outIy))
		return false;

	return this->_tEigen2Image(this->h_data, outlam1->h_data, outlam2->h_data, outIx->h_data, outIy->h_data, sigma, this->cols, this->rows);
}

bool Model::FrangiFilter(Model* out, unsigned int* B, const float& sigMn, const float& sigMx, const float& sigStep, const float& beta, const float& gamma, const float& threshold, const bool& bRegionGrowing)
{
	if (*this != *out)
		return false;
	if (bRegionGrowing)
	{
		if (B == nullptr)
		{
			B = new unsigned int[sizeof(unsigned int) * this->cols * this->rows];
			//memset(B, 0, sizeof(unsigned int) * this->cols * this->rows);
		}
	}
	return this->_tFrangiFilter(this->h_data, out->h_data, B, this->cols, this->rows, sigMn, sigMx, sigStep, beta, gamma, threshold, bRegionGrowing);
}

bool Model::ImageGaussianFilter(Model* out = nullptr, const float& sigma = 3, const int& kernelSize = 0)
{
	if (out == nullptr)
	{
		Model* _tmp = this;
		if (this->_tImageGaussianFilter2d(this->h_data, _tmp->h_data, sigma, this->cols, this->rows, kernelSize))
		{
			CUDA_CALL(cudaMemcpyAsync(this->h_data, _tmp->h_data, sizeof(this->h_data[0]) * this->cols * this->rows, cudaMemcpyHostToHost));
			return true;
		}
	}
	else
	{
		if (*this != *out)
			return false;
		return this->_tImageGaussianFilter2d(this->h_data, out->h_data, sigma, this->cols, this->rows, kernelSize);
	}
	return false;
}

bool Model::Convolve2d(Model* out, float* kernel, const int& kw, const int& kh)
{
	if (*this != *out)
		return false;
	return this->_tConvolve2d(this->h_data, out->h_data, kernel, this->cols, this->rows, kw,kh);
}

bool Model::Upsampling(Model* out)
{
	return this->_tUpsampling(this->h_data, out->h_data, this->cols, this->rows, out->cols, out->rows);
}

bool Model::Derivative2d(Model* outX, Model* outY)
{
	if(*this != *outX)
		return false;
	if (*this != *outY)
		return false;
	return this->_tDerivative2d(this->h_data, outX->h_data, outY->h_data, this->cols, this->rows);
}

bool Model::DiffuseFilter(Model* out, const int& iter, const float& dt, const float& rho, const float& sigma, const float& alpha, const float& C, const int& mode, const int& dftype)
{
	if (*this != *out)
		return false;
	if (iter < 1)
		return false;
	if ((rho < 0) || (sigma < 0) || (alpha < 0) || (C < 0))
		return false;
	if ((mode != 1) && (mode != 2))
		return false;
	if ((dftype != 0) && (dftype != 1) && (dftype != 2) && (dftype != 3))
		return false;
	
	return this->_tDiffuseFilter(this->h_data, out->h_data, this->cols, this->rows, iter, dt, rho, sigma, alpha, C, mode, dftype);
}

QImage::Format Model::Qformat() const
{
	return mQFormat;
}

bool Model::testPathFinder(std::vector<QPointF>& out, const QPointF& p1, const QPointF& p2, const int bF)
{
	const int bandWidth = bF * 2 + 1;
	//float* boundary = new float[bandWidth * bandWidth];
	int snakeCnt = 0;
	const int width = cols;
	const int height = rows;
	std::vector<int> snake,_bv,_bv2,counters;
	//bool* activemap = new bool[int(width * height)];
	std::unique_ptr<float[]> boundary(new float[bandWidth * bandWidth]);
	std::unique_ptr<bool[]> activemap(new bool[int(width * height)]);
	memset(activemap.get(), 0, sizeof(bool) * width * height);
	
	for (auto i = 0; i < (bF * 2 + 1); i++)
	{
		for (auto j = 0; j < (bF * 2 + 1); j++)
		{
			int cal = j + i * (bF * 2 + 1);
			if ((i == 0) || (j == 0) || (i == bF * 2) || (j == bF * 2))
			{
				boundary.get()[cal] = 1;
				if (i != 0)
				{
					if (i != bF * 2)
					{
						if (j==0)
							_bv2.push_back(cal);
						else
						{
							snake.push_back(cal);
						}
					}
					else
					{
						_bv.push_back(cal);
					}
				}
				else
				{
					snake.push_back(cal);
				}
				counters.push_back(cal);
			}
			else
				boundary.get()[cal] = 0;
		}
	}
	for (auto v = _bv.rbegin(); v != _bv.rend(); v++)
		snake.push_back(*v);
	for (auto v = _bv2.rbegin(); v != _bv2.rend(); v++)
		snake.push_back(*v);
	const int snakeLength = snake.size();
	std::unique_ptr<float[]> boxbuffer(new float[bandWidth * bandWidth]);
	std::unique_ptr<float[]> valbuffer(new float[snakeLength]);
	std::unique_ptr<int[]> _tempbuffer(new int[snakeLength]);
	std::unique_ptr<int[]> ssign(new int[snakeLength]);
	out.push_back(p1);

	//std::ofstream s;
	//s.open("buffer.bin", std::ios::binary | std::ios::out);
	//s.write(reinterpret_cast<char*>(boundary.get()), sizeof(float) * 512 * 512);
	//s.close();

	if (custom_tracker(p1.x(), p1.y(), out, this->h_data, (activemap), snake, (boundary), (boxbuffer), (valbuffer), (_tempbuffer), (ssign), p1, p2, bF, width, height, snakeLength, bandWidth))
	{
		return true;
	}
	return false;
}

float* Model::test()
{
	return this->h_data;
}

bool Model::setChanelConfig(int& dataType)
{
	if (dataType == -1)
	{
		if(mQFormat == QImage::Format::Format_Invalid)
			return false;
		switch (this->mQFormat)
		{
		case QImage::Format::Format_Grayscale8:
			setChannelNumber(1); setChannelBits(8);
			break;
		case QImage::Format::Format_Grayscale16:
			setChannelNumber(1); setChannelBits(16);
			break;
		case QImage::Format::Format_RGB888:
			setChannelNumber(3); setChannelBits(8);
			break;
		case QImage::Format::Format_RGBX64:
			setChannelNumber(4); setChannelBits(16);
			break;
		default:
			break;
		}
	}
	else
	{
		switch (dataType)
		{
		case 0:
			setChannelNumber(1); setChannelBits(8); mQFormat = QImage::Format::Format_Grayscale8; break;
		case 1:
			setChannelNumber(4); setChannelBits(2); mQFormat = QImage::Format::Format_Invalid; break;
		case 2:
			setChannelNumber(4); setChannelBits(4); mQFormat = QImage::Format::Format_ARGB4444_Premultiplied; break;
		case 3:
			setChannelNumber(3); setChannelBits(8); mQFormat = QImage::Format::Format_RGB888; break;
		case 4://Not happen
			setChannelNumber(4); setChannelBits(8); mQFormat = QImage::Format::Format_ARGB32; break;
		case 5:
			setChannelNumber(1); setChannelBits(16); mQFormat = QImage::Format::Format_Grayscale16; break;
		default:
			setChannelNumber(0); setChannelBits(0); mQFormat = QImage::Format::Format_Invalid; return false;
		}
	}
	return true;
}

int Model::getChannelNumber() const
{
	return mChannelNumber;
}

int Model::getChannelBits() const
{
	return mChannelBits;
}


void Model::release()
{
	if(!this->empty())
	{
		data = nullptr;
	}

	if (!(this->h_data == nullptr))
	{
		CUDA_CALL(cudaFreeHost(h_data));
		h_data = nullptr;
	}
}


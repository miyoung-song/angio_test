
#include <stack>
#include <string>
#include <iostream>
#include <windows.h>

#include "Angio_Algorithm.h"
#include <filesystem>

template<typename T>
bool validate(T value, T min, T max) { return value >= min && value <= max; };

vector<float2> ShortestPath(float* T, const float2& startPoint, const vector<float2> sources, const float& stepsize, const int& width, const int& height, bool ismove = false)
{
	auto StartPoint = startPoint;

	struct dm
	{
		float d;
		int id;
		dm() {};
		dm(const float& D, const int& ID) :d(D), id(ID) {};
	};
	auto GetDistance2End = [&](std::vector<float2> vecLinepos, float2 end)->dm
	{
		dm _dm;
		float _min = FLT_MAX;
		int _minId = 0, current = 0;
		float2 pos;
		for (auto i = 0; i < vecLinepos.size(); i++)
		{
			auto f = vecLinepos[i];
			auto _val = (f.x - end.x) * (f.x - end.x) + (f.y - end.y) * (f.y - end.y);
			if (_val < _min)
			{
				_min = _val;
				_minId = current;
				pos.x = f.x;
				pos.y = f.y;
			}
			current++;
		}
		return dm(sqrtf(_min), _minId);
	};

	auto Fy = make_unique<float[]>(width * height);
	auto Fx = make_unique<float[]>(width * height);

	get_gpu_8_connective_point_min(T, Fy.get(), Fx.get(), width, height);

	auto gradX = Fx.get();
	auto gradY = Fy.get();


	if (ismove)
	{
		//float* x = new float[width * height];
		//float* y = new float[width * height];
		std::vector<float2> vecpos1;
		std::vector<float2> vecpos2;
		for (int rep = 0; rep < height * width; rep++)
		{
			auto pos = make_float2(rep % (int)height, (rep / (int)width));
			if (startPoint.x == pos.x && startPoint.y == pos.y)
				continue;
			if (fabs(Fx.get()[rep]) > 0)
			{
				//x[rep] = 1;
				vecpos1.push_back(pos);
			}

			if (fabs(Fy.get()[rep]) > 0)
			{
				//y[rep] = 1;
				vecpos2.push_back(pos);
			}
		}

#ifdef _DEBUG
		//std::ofstream out;
		//out.open("Fx.bin", std::ios::binary | std::ios::out);
		//out.write(reinterpret_cast<char*>(x), sizeof(float)* width* height);
		//out.close();
		//
		//out.open("Fy.bin", std::ios::binary | std::ios::out);
		//out.write(reinterpret_cast<char*>(y), sizeof(float)* width* height);
		//out.close();
#endif
		//delete[] x;
		//delete[] y;

		if (vecpos1.size() != 0)
		{
			auto id = GetDistance2End(vecpos1, startPoint).id;
			auto newpos = vecpos1[id];
			StartPoint = make_float2(newpos.x, StartPoint.y);
		}

		if (vecpos2.size() != 0)
		{
			auto id = GetDistance2End(vecpos2, StartPoint).id;
			auto newpos = vecpos2[id];
			StartPoint = make_float2(newpos.x, newpos.y);
		}
	}



	const int width_ = width - 1;
	const int height_ = height - 1;

	auto norm2 = [=](const float2& a)->float {return sqrtf(a.x * a.x + a.y * a.y); };

	auto intergrad = [&](float* gradX, float* gradY, const float2& point)->float2
	{
		float2 _loc = make_float2(floorf(point.x), floorf(point.y));
		int2 xy0 = make_int2(_loc.x, _loc.y);
		int2 xy1 = make_int2(xy0.x + 1, xy0.y + 1);

		float2 xyC = make_float2(point.x - _loc.x, point.y - _loc.y);
		float2 xyCi = make_float2(1 - xyC.x, 1 - xyC.y);

		float prec[] = { xyCi.x * xyCi.y, xyCi.x * xyC.y, xyC.x * xyCi.y, xyC.x * xyC.y };

		if (xy0.x < 0)
		{
			xy0.x = 0; if (xy1.x < 0)xy1.x = 0;
		}
		else if (xy0.x > width_)
		{
			xy0.x = width_; if (xy1.x > width_)xy1.x = width_;
		}

		if (xy0.y < 0)
		{
			xy0.y = 0; if (xy1.y < 0)xy1.y = 0;
		}
		else if (xy0.y > height_)
		{
			xy0.y = height_; if (xy1.y > height_)xy1.y = height_;
		}
		auto pos = make_float2(gradX[xy0.x + xy0.y * width] * prec[0] +
			gradX[xy0.x + xy1.y * width] * prec[1] +
			gradX[xy1.x + xy0.y * width] * prec[2] +
			gradX[xy1.x + xy1.y * width] * prec[3],
			gradY[xy0.x + xy0.y * width] * prec[0] +
			gradY[xy0.x + xy1.y * width] * prec[1] +
			gradY[xy1.x + xy0.y * width] * prec[2] +
			gradY[xy1.x + xy1.y * width] * prec[3]);
		return pos;
	};



	auto RK4 = [&](const float2 start)->float2
	{
		float2 temp = make_float2(0.0f, 0.0f);
		float2 next = make_float2(0.0f, 0.0f);

		float2 k[4];

		k[0] = intergrad(gradX, gradY, start);
		//k[0].y = intergrad(gradY, start);
		float tempnorm = norm2(k[0]);
		if (tempnorm != 0)
		{
			k[0].x = k[0].x * stepsize / tempnorm;
			k[0].y = k[0].y * stepsize / tempnorm;
		}
		temp.x = start.x - k[0].x * 0.5f;
		temp.y = start.y - k[0].y * 0.5f;

		k[1] = intergrad(gradX, gradY, temp);
		//k[1].y = intergrad(gradY, temp);
		tempnorm = norm2(k[1]);
		if (tempnorm != 0)
		{
			k[1].x = k[1].x * stepsize / tempnorm;
			k[1].y = k[1].y * stepsize / tempnorm;
		}
		temp.x = start.x - k[1].x * 0.5f;
		temp.y = start.y - k[1].y * 0.5f;

		k[2] = intergrad(gradX, gradY, temp);
		//k[2].y = intergrad(gradY, temp);
		tempnorm = norm2(k[2]);
		if (tempnorm != 0)
		{
			k[2].x = k[2].x * stepsize / tempnorm;
			k[2].y = k[2].y * stepsize / tempnorm;
		}
		temp.x = start.x - k[2].x;
		temp.y = start.y - k[2].y;

		k[3] = intergrad(gradX, gradY, temp);
		//k[3].y = intergrad(gradY, temp);
		tempnorm = norm2(k[3]);
		if (tempnorm != 0)
		{
			k[3].x = k[3].x * stepsize / tempnorm;
			k[3].y = k[3].y * stepsize / tempnorm;
		}
		next.x = start.x - (k[0].x + k[1].x * 2.0f + k[2].x * 2.0f + k[3].x) / 6.0f;
		next.y = start.y - (k[0].y + k[1].y * 2.0f + k[2].y * 2.0f + k[3].y) / 6.0f;
		return next;
	};

	auto argmin = [](const float* arr, const int& n)->unsigned int {
		return std::min_element(arr, arr + n) - arr;
		//return std::distance(arr, std::min_element(arr, arr + n)); 
	};




	auto distance2end = [&](const float2& end)->dm
	{
		float _min = FLT_MAX;
		int _minId = 0, current = 0;
		for (auto& f : sources)
		{
			auto _val = (f.x - end.x) * (f.x - end.x) + (f.y - end.y) * (f.y - end.y);
			if (_val < _min)
			{
				_min = _val;
				_minId = current;
			}
			current++;
		}
		return dm(sqrtf(_min), _minId);
	};


	auto dis = [=](const float2& a, const float2& b)->float {return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)); };


	int i = -1;
	int ifree = 10000;
	float movement = 0.0f;

	vector<float2> shortestline;
	shortestline.reserve(1 << 10);
	const int _tol = 25;

	while (true)
	{
		auto end = RK4(StartPoint);
		{
			if (i >= _tol)
				movement = sqrtf((end.x - shortestline.at(i - _tol).x) * (end.x - shortestline.at(i - _tol).x) + \
					(end.y - shortestline.at(i - _tol).y) * (end.y - shortestline.at(i - _tol).y));
			else
				movement += 1.0f;

			if ((end.x == 0) || (movement < stepsize) || (isnan(end.x) || isnan(end.y)))
				break;
		}

		i++;

		auto D = dis(StartPoint, end);
		shortestline.push_back(end);

		auto _dm = distance2end(end);

		if (_dm.d < stepsize)
		{
			i++;
			//shortestline.push_back(sources[_dm.id]);
			break;
		}
		StartPoint = end;
	}

	if (shortestline.size() == 0)
		return shortestline;

	int x = 0;
	int y = 0;
	vector<float2> _shortestline;
	for (int i = 0; i < shortestline.size(); i++)
	{
		if (i == 0)
			_shortestline.push_back(shortestline[i]);
		else
		{
			auto D = dis(shortestline[i], shortestline[i - 1]);
			if (D > 0.1)
			{
				_shortestline.push_back(shortestline[i]);
			}
		}


	}

	if (_shortestline.size() < 2)
	{
		int cnt = _shortestline.size();
		for (int i = 0; i < cnt; i++)
			_shortestline.erase(_shortestline.begin());
		vector<float2>().swap(_shortestline);
		_shortestline.clear();
	}
	else
	{
		for (int i = 0; i < _shortestline.size() - 1; i++)
		{
			if (_shortestline[i].x == _shortestline[i + 1].x && _shortestline[i].x < 10)
			{
				x++;
			}
			if (_shortestline[i].x == _shortestline[i + 1].y && _shortestline[i].y < 10)
			{
				y++;
			}
			if (x > 10 || y > 10)
			{
				_shortestline.clear();
				break;
			}
			if (_shortestline[i].x < 0 || _shortestline[i].y < 0)
			{
				_shortestline.clear();
				break;

			}
		}
	}
	return _shortestline;
}

template<class D>
float CalculateDistance(const D* T, const float& fij, const int2& ij, bool useSec, bool useCross, const bool* const Frozen, const uint2& wh)
{
	//auto T = _T.get();
	//auto Frozen = _Frozen.get();

	constexpr float _eps = 2.2204e-16;
	auto pos = [&](const int2& pnt)->unsigned int {return pnt.x + pnt.y * wh.x; };
	auto isInside = [&](const int2& pnt)->bool { return (pnt.x >= 0) && (pnt.y >= 0) && (pnt.x < wh.x) && (pnt.y < wh.y); };
	auto roots = [](const float coeff[])->float {
		float d = coeff[1] * coeff[1] - 4.0f * coeff[0] * coeff[2];
		if (d < 0)
			d = 0;
		else
			d = sqrtf(d);

		float2 f;
		if (coeff[0] != 0.0f)
		{
			f = make_float2((-coeff[1] - d) / (2.0f * coeff[0]), (-coeff[1] + d) / (2.0f * coeff[0]));

		}
		else
		{
			f = make_float2(coeff[2] * 2.0f / (-coeff[1] - d), coeff[2] * 2.0f / (-coeff[1] + d));
		}
		return fmaxf(f.x, f.y);
	};


	//_Tpatch = np.array([[np.inf]* 5] * 5)
	float Tpatch[] = { INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY ,INFINITY };
	//memset(Tpatch, INFINITY, sizeof(float) * 25);



	//int2 = (useSec | useCross) ? make_int2(-2, 3), make_int2(-1, 2);
	const int bnd = (useSec | useCross) ? 2 : 1;

	//#pragma omp parallel for
	for (auto nx = -bnd; nx <= bnd; nx++)
		for (auto ny = -bnd; ny <= bnd; ny++)
		{
			auto xy = make_int2(ij.x + nx, ij.y + ny);
			if (isInside(xy))
			{
				auto coord = pos(xy);
				if (Frozen[coord])
					Tpatch[nx + 2 + (ny + 2) * 5] = T[coord];
			}

		}

	float order[4] = { 0.0f, };
	float tm[4] = { 0.0f, };
	float tm2[4] = { 0.0f, };
	float fij2 = fij * fij;

	vector<float> direc;
	float direcMin = INFINITY;

	tm[1] = fminf(Tpatch[11], Tpatch[13]);
	if (isfinite(tm[1]))
	{
		order[1] = 1;
		direc.push_back(tm[1]);
		direcMin = tm[1];
		if (direcMin > tm[1])
			direcMin = tm[1];
	}
	tm[0] = fminf(Tpatch[7], Tpatch[17]);
	if (isfinite(tm[0]))
	{
		order[0] = 1;
		direc.push_back(tm[0]);
		if (direcMin > tm[0])
			direcMin = tm[0];
	}

	if (useCross)
	{
		tm[2] = fminf(Tpatch[6], Tpatch[18]);
		if (isfinite(tm[2])) {
			order[2] = 1;
			direc.push_back(tm[2]);
			if (direcMin > tm[2])
				direcMin = tm[2];
		}
		tm[3] = fminf(Tpatch[16], Tpatch[8]);
		if (isfinite(tm[3]))
		{
			order[3] = 1;
			direc.push_back(tm[3]);
			if (direcMin > tm[3])
				direcMin = tm[3];
		}
	}

	if (useSec)
	{
		bool ch1 = (Tpatch[10] < Tpatch[11]) & isfinite(Tpatch[11]);
		bool ch2 = (Tpatch[14] < Tpatch[13]) & isfinite(Tpatch[13]);
		if (ch1 & ch2)
		{
			tm2[0] = fminf((4 * Tpatch[11] - Tpatch[10]) / 3.0f, (4 * Tpatch[13] - Tpatch[14]) / 3.0f);
			order[0] = 2;
		}
		else if (ch1)
		{
			tm2[0] = (4 * Tpatch[11] - Tpatch[10]) / 3.0f;
			order[0] = 2;
		}
		else if (ch2)
		{
			tm2[0] = (4 * Tpatch[13] - Tpatch[14]) / 3.0f;
			order[0] = 2;
		}

		ch1 = (Tpatch[2] < Tpatch[7]) & isfinite(Tpatch[7]);
		ch2 = (Tpatch[22] < Tpatch[17]) & isfinite(Tpatch[17]);
		if (ch1 & ch2)
		{
			tm2[1] = fminf((4 * Tpatch[7] - Tpatch[2]) / 3.0f, (4 * Tpatch[17] - Tpatch[22]) / 3.0f);
			order[1] = 2;
		}
		else if (ch1)
		{
			tm2[1] = (4 * Tpatch[7] - Tpatch[2]) / 3.0f;
			order[1] = 2;
		}
		else if (ch2)
		{
			tm2[1] = (4 * Tpatch[17] - Tpatch[22]) / 3.0f;
			order[1] = 2;
		}

		if (useCross)
		{
			ch1 = (Tpatch[0] < Tpatch[6]) & isfinite(Tpatch[6]);
			ch2 = (Tpatch[24] < Tpatch[18]) & isfinite(Tpatch[18]);
			if (ch1 & ch2)
			{
				tm2[2] = fminf((4 * Tpatch[6] - Tpatch[0]) / 3.0f, (4 * Tpatch[18] - Tpatch[24]) / 3.0f);
				order[2] = 2;
			}
			else if (ch1)
			{
				tm2[2] = (4 * Tpatch[6] - Tpatch[0]) / 3.0f;
				order[2] = 2;
			}
			else if (ch2)
			{
				tm2[2] = (4 * Tpatch[18] - Tpatch[24]) / 3.0f;
				order[2] = 2;
			}

			ch1 = (Tpatch[20] < Tpatch[16]) & isfinite(Tpatch[16]);
			ch2 = (Tpatch[4] < Tpatch[8]) & isfinite(Tpatch[8]);
			if (ch1 & ch2)
			{
				tm2[2] = fminf((4 * Tpatch[16] - Tpatch[20]) / 3.0f, (4 * Tpatch[8] - Tpatch[4]) / 3.0f);
				order[2] = 2;
			}
			else if (ch1)
			{
				tm2[2] = (4 * Tpatch[16] - Tpatch[20]) / 3.0f;
				order[2] = 2;
			}
			else if (ch2)
			{
				tm2[2] = (4 * Tpatch[8] - Tpatch[4]) / 3.0f;
				order[2] = 2;
			}

		}

	}

	float coeff[3] = { 0.0f,0.0f,-1.0f / fmaxf(fij2,_eps) };

	for (auto i = 0; i < 2; i++)
	{
		if (order[i] == 1)
		{
			coeff[0] += 1.0f;
			coeff[1] += (-2.0f * tm[i]);
			coeff[2] += (tm[i] * tm[i]);
		}
		else if (order[i] == 2)
		{
			coeff[0] += 2.25f;
			coeff[1] += (-4.5f * tm2[i]);
			coeff[2] += (2.25f * tm2[i] * tm2[i]);
		}
	}
	float tt = roots(coeff);

	if (useCross)
	{
		coeff[2] += (-1.0f / (fmaxf(fij2, _eps)));
		for (auto i = 2; i < 4; i++)
		{
			if (order[i] == 1)
			{
				coeff[0] += 0.5f;
				coeff[1] -= tm[i];
				coeff[2] += (0.5f * tm[i] * tm[i]);
			}
			else if (order[i] == 2)
			{
				coeff[0] += 1.125f;
				coeff[1] += (-2.25f * tm2[i]);
				coeff[2] += (1.125f * tm2[i] * tm2[i]);
			}
		}
		float tt2 = roots(coeff);
		tt = fminf(tt, tt2);
	}

	for (auto& dr : direc)
	{
		if (dr > tt)
		{
			tt = direcMin + (1.0f / fmaxf(fij, _eps));
			break;
		}
	}
	return tt;
}

void custom_tracker2(float* T, int* Y, float* F, const bool* _Frz, const vector<float2>& _srcPnt, const float2& end, const uint& iterMax, const uint2& wh, const bool& useSec = true, const bool& useCross = true)
{
	const uint2 wh_ = make_uint2(wh.x - 1, wh.y - 1);
	//functor

	//auto pos = [&](const uint2& pnt)->unsigned int {return pnt.x + pnt.y * wh.x; };
	auto pos = [&](const int2& pnt)->unsigned int {return pnt.x + pnt.y * wh.x; };
	auto isInside = [&](const int2& pnt)->bool { return (pnt.x >= 0) & (pnt.y >= 0) & (pnt.x < wh_.x) & (pnt.y < wh_.y); };
	auto argmin = [](const float* arr, const int& n)->unsigned int {
		return std::min_element(arr, arr + n) - arr;
		//return std::distance(arr, std::min_element(arr, arr + n)); 
	};


	//update start point?
	//vector<uint2>
	constexpr float _eps = 2.2204e-16f;
	unsigned int stride = wh.x * wh.y;
	unsigned int neg_free = 65536 * 8;
	int neg_pos = -1;

	//auto F = std::make_unique<float[]>(stride);
	//std::copy(initF, initF + stride, F);


	auto Frozen = std::make_unique<bool[]>(stride);
	memcpy(Frozen.get(), _Frz, stride * sizeof(bool));

	auto _n0 = std::make_unique<float[]>(neg_free);
	auto _n12 = std::make_unique<int2[]>(neg_free);
	//auto _n2 = std::make_unique<int[]>(neg_free);
	auto _n3 = std::make_unique<float[]>(neg_free);




	constexpr int2 ne[4] = { {0, -1},{0, 1},{-1, 0},{1, 0} };


	vector<int2> srcPnt;
	srcPnt.reserve(_srcPnt.size());


	for (auto& src : _srcPnt)
	{
		srcPnt.push_back(make_int2(src.x, src.y));
		auto _src = srcPnt.back();
		Frozen.get()[pos(_src)] = true;
		T[pos(_src)] = 1;
	}

	for (auto& src : srcPnt)
	{
		for (auto& ij : ne)
		{
			int2 xy = make_int2(src.x + ij.x, src.y + ij.y);
			auto coord = pos(xy);
			if (isInside(xy))
			{
				if (!Frozen.get()[coord])
				{
					const float Tt = 1.0f / fmaxf(F[coord], _eps);

					int _pivot = T[coord];

					if (_pivot > 0)
					{
						if (_n0.get()[_pivot] > Tt)
							_n0.get()[_pivot] = Tt;
						if (_n3.get()[_pivot] > 1.0f)
							_n3.get()[_pivot] = 1.0f;
					}
					else
					{
						neg_pos++;
						//never happen
						//if (neg_pos > neg_free)
						//{
						//
						//}
						_n0.get()[neg_pos] = int(Tt);
						_n12.get()[neg_pos] = xy;
						//_n2.get()[neg_pos] = xy.y;
						_n3.get()[neg_pos] = 1.0f;

						T[coord] = neg_pos;
					}
				}

			}
		}
	}


	for (int itt = 0; (neg_pos != -1); itt++)
	{
		auto index = argmin(_n0.get(), neg_pos + 1);

		if (isfinite(_n0.get()[index]) == false)
			break;

		auto src = _n12.get()[index];

		//if (itt % 1000 == 0 && itt != 0)
		//    qDebug() << itt;
		auto _coord = pos(src);

		Frozen.get()[_coord] = true;
		T[_coord] = _n0.get()[index];
		Y[_coord] = _n3.get()[index];

		if (index < neg_pos)
		{
			_n0.get()[index] = _n0.get()[neg_pos];
			_n12.get()[index] = _n12.get()[neg_pos];
			_n3.get()[index] = _n0.get()[neg_pos];

			T[pos(_n12.get()[index])] = index;
		}

		neg_pos--;
		for (auto& ij : ne)
		{
			int2 xy = make_int2(src.x + ij.x, src.y + ij.y);
			auto coord = pos(xy);
			if (isInside(xy))
			{
				if (!Frozen.get()[coord])
				{
					int cidx = T[coord];
					auto tt = CalculateDistance<float>(T, F[coord], xy, useSec, useCross, Frozen.get(), wh);
					auto ty = CalculateDistance<int>(Y, 1, xy, useSec, useCross, Frozen.get(), wh);

					if (T[coord] > 0)
					{
						_n0.get()[cidx] = fminf(tt, _n0.get()[cidx]);
						_n3.get()[cidx] = fminf(ty, _n3.get()[cidx]);
					}
					else
					{
						neg_pos++;
						_n0.get()[neg_pos] = tt;
						_n12.get()[neg_pos] = xy;
						_n3.get()[neg_pos] = ty;

						T[coord] = neg_pos;


					}
				}
			}
		}
	}
}

float Angio_Algorithm::get_distance_points(const float2& a, const float2& b) { return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)); };

float2 Angio_Algorithm::get_rotation_point(const float& radian, const float2& input_point_a, const float2& input_point_b)
{
	float2 output_point;
	output_point.x = cosf(radian) * (input_point_a.x - input_point_b.x) - sinf(radian) * (input_point_a.y - input_point_b.y);
	output_point.y = sinf(radian) * (input_point_a.x - input_point_b.x) + cosf(radian) * (input_point_a.y - input_point_b.y);
	output_point = make_float2((output_point.x + input_point_b.x), (output_point.y + input_point_b.y));
	return output_point;
}


bool Angio_Algorithm::get_intersect_point(float2 AP1, float2 AP2, float2 BP1, float2 BP2, float2& IP)
{
	double t;
	double s;
	double under = (BP2.y - BP1.y) * (AP2.x - AP1.x) - (BP2.x - BP1.x) * (AP2.y - AP1.y);
	if (under == 0) return false; //평행

	double _t = (BP2.x - BP1.x) * (AP1.y - BP1.y) - (BP2.y - BP1.y) * (AP1.x - BP1.x);
	double _s = (AP2.x - AP1.x) * (AP1.y - BP1.y) - (AP2.y - AP1.y) * (AP1.x - BP1.x);

	t = _t / under;
	s = _s / under;

	if (t < 0.0 || t>1.0 || s < 0.0 || s>1.0) return false;
	if (_t == 0 && _s == 0) return false;

	IP.x = AP1.x + t * (double)(AP2.x - AP1.x);
	IP.y = AP1.y + t * (double)(AP2.y - AP1.y);

	return true;
}

template<typename T>
T adjust_data(T value, int min, int max)
{
	if (value < min) 
		return min;
	else if (value > max) 
		return max;
	return value;
};

void Angio_Algorithm::create_buffer(int w, int h)
{
	auto initbuffer = [&](model_info::image_float_type type)
	{
		vector<float> data;
		data.resize(w * h);
		image_buffers_.insert(make_pair(type, data));
	};
	delete_buffer();
	int size = w * h;

	Frozen_ = new bool[w * h];

	initbuffer(model_info::image_float_type::ORIGIN_IMAGE);
	initbuffer(model_info::image_float_type::CENTLINE_IMAGE);
	initbuffer(model_info::image_float_type::LEBELING_IMAGE);
	initbuffer(model_info::image_float_type::SPEED_IMAGE);
	initbuffer(model_info::image_float_type::BOUNDARY_IMAGE);
	
	set_width(w);
	set_height(h);
}

void Angio_Algorithm::delete_buffer()
{
	for_each(image_buffers_.begin(), image_buffers_.end(), [](pair<model_info::image_float_type, vector<float>> var) { var.second.clear(); });
	image_buffers_.clear();


	if (Frozen_)
	{
		delete Frozen_;
		Frozen_ = nullptr;
	}
}

float* Angio_Algorithm::get_float_buffer(model_info::image_float_type key)
{
	auto it = image_buffers_.find(key);
	if (it != image_buffers_.end())
		return (it->second.data());
	else
		return nullptr;
}

void Angio_Algorithm::run_AI_segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type)
{
	const char* applicationName = R"(../segmentation/main_test_for_build.exe)";
	std::string commandLine;

	if (run_type == model_info::segmentation_exe_type::run_centerline)
	{
		applicationName = R"(../segmentation/main_test_for_build.exe)";
		commandLine = "\"../segmentation/main_test_for_build.exe\" ../data/ ../output/ ";
	}
	else if (run_type == model_info::segmentation_exe_type::run_outlines)
	{
		applicationName = R"(../segmentation/main_test_for_build_outline.exe)";
		commandLine = "\"../segmentation/main_test_for_build_outline.exe\" ../data/ ../output/ ";
	}
	else if (run_type == model_info::segmentation_exe_type::run_endpoints)
	{
		applicationName = R"(../segmentation/main_test_for_build_spd.exe)";
		commandLine = "\"../segmentation/main_test_for_build_spd.exe\" ../data/ ../output/ ";
	}

	if (model_type == model_info::segmentation_model_type::rca)
		commandLine += "../model RCA";
	else if (model_type == model_info::segmentation_model_type::lcx)
		commandLine += "../model LCX";
	else
		commandLine += "../model LAD";

	std::vector<char> lpstr(commandLine.begin(), commandLine.end());
	lpstr.push_back('\0'); // null terminator 추가
	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));
	bool state = CreateProcessA(applicationName, lpstr.data(), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);
	if (state)
	{
		WaitForSingleObject(pi.hProcess, INFINITE);
		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
	}
	else
	{
		DWORD errorCode = GetLastError();
		std::cerr << "Error: Failed to create process. Error code: " << errorCode << std::endl;

	}
}


bool Angio_Algorithm::apply_filters(float* buffer, float* buffer1, float* buffer5, float* filter_image)
{
	int w = get_width(); int h = get_height();
	int rect_size = w * h;

	if (inspect_roi_.size() != 0)
	{
		auto point_lt = inspect_roi_[0][0];
		auto end_roi = inspect_roi_[inspect_roi_.size() - 1];
		auto point_rb = end_roi[end_roi.size() - 1];
		rect_size = fabs((point_lt.x - point_rb.x) * (point_lt.y - point_rb.y));
	}

	const float weights[] = { 1.0f / 3.0f,1.0f / 3.0f,1.0f / 3.0f };
	// 필터링 함수
	auto filtering = [&](float val)
	{
		const float sigmas[] = { val, 80.0f, 200.0f };
		
		//// 버퍼 초기화
		std::fill_n(buffer,  w * h, 0);
		std::fill_n(buffer1, w * h, 0);
		std::fill_n(buffer5, w * h, 0);

		// MSRCR 필터 적용 및 후속 필터링
		test_gpuMSRCR2(filter_image, buffer5, weights, sigmas, 120, -50, _countof(weights), w, h);
		gpu_diffusefilt_float(buffer5, buffer, w, h, 10, 1.0f, 4, 1, 0.001f, 1, 1, 3);
		gpu_frangifilt_float(buffer, buffer1, buffer5, w, h, 1.0f, 4.0f, 0.1f, 1.0f, 3.5f, 0.15f, true);
				
		// ROI에서 필터 결과가 의미 있는지 판단
		int valid_count = std::count_if(buffer1, buffer1 + w * h, [](float val) { return val != 0; });
		return valid_count > rect_size * 0.03;
	};

	bool is_filt_found = false;

	for (int i = 0; i < 2; i++)
	{
		if (filtering(i * 15))
		{
			cvAlgorithm_.create_filter_image(buffer1, filter_image, w, h);
			for (int rep = 0; rep < w * h; rep++)
			{
				if (filter_image[rep] == 0)
				{
					buffer1[rep] = 0;
					buffer5[rep] = 0;
				}
			}
			is_filt_found = true;
			break;
		}
	}

	return is_filt_found;
}

void Angio_Algorithm::processImage(float* buffer, float* buffer1, float* buffer5)
{
	int w = get_width();
	int h = get_height();

	auto _testFrozen = make_unique<bool[]>(w * h);

	gpu_speedImage_float(buffer1, buffer, _testFrozen.get(), w, h, 4, true, false);
	int valid_count = std::count_if(buffer, buffer + w * h, [](float val) { return val != 0; });
	if (valid_count == 0)
	{
		Mat opencvImage(Size(w, h), CV_32FC1);
		cvAlgorithm_.convert_image(buffer1, opencvImage, w, h);
		opencvImage = cvAlgorithm_.dilate_image(opencvImage, 3);
		cvAlgorithm_.convert_image(opencvImage, buffer, w, h);

		gpu_speedImage_float(buffer, buffer, _testFrozen.get(), w, h, 4, true, false);

		std::transform(buffer, buffer + (w * h), _testFrozen.get(),
			[](float val) -> bool {return (std::fabs(val) > 0) ? 0 : 1; });
	}

	std::copy(_testFrozen.get(), _testFrozen.get() + w * h, Frozen_);
	std::copy(buffer, buffer + w * h, get_float_buffer(model_info::image_float_type::SPEED_IMAGE));
	cvAlgorithm_.save_image(buffer, "\\test\\SPEED_IMAGE", w, h);

	std::copy(buffer5, buffer5 + w * h, get_float_buffer(model_info::image_float_type::LEBELING_IMAGE));
	cvAlgorithm_.save_image(buffer5, "\\test\\LEBELING_IMAGE", w, h);
}

bool Angio_Algorithm::manual_segmentation(bool find_line)
{
	int w = get_width();
	int h = get_height();

	auto buffer = std::make_unique<float[]>(w * h);
	auto buffer1 = std::make_unique<float[]>(w * h);
	auto buffer5 = std::make_unique<float[]>(w * h);

	int N = 17;
	int range_x = w / N;
	int range_y = h / N;
	float except_x = (w - (range_x * N)) / 2;
	float except_y = (h - (range_y * N)) / 2;
	int2 LT = make_int2(1, 3);
	int2 RB = make_int2(16, 15);
	// 좌상단과 우하단 좌표를 생성
	float2 point_lt = make_float2(except_x + (range_x * LT.x), except_y + (range_y * LT.y));
	float2 point_rb = make_float2(except_x + (range_x * RB.x), except_y + (range_y * RB.y));

	auto filter_image = get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE);
	for (int i = 0; i < (w * h); i++)
		filter_image[i] = filter_image[i] / 255.0;

	// ROI를 위한 좌표 그리드 생성
	inspect_roi_.clear();
	for (int i = 0; i <= RB.y - LT.y; ++i)
	{
		std::vector<float2> pos;
		for (int j = 0; j <= RB.x - LT.x; ++j)
		{
			float2 pt = make_float2(point_lt.x + range_x * j, point_lt.y + range_y * i);
			pos.push_back(pt);
		}
		inspect_roi_.push_back(std::move(pos)); // pos 벡터 이동을 통해 복사 비용 절감
	}


	auto start_point = make_float2(0, 0);
	int size = inspect_roi_[inspect_roi_.size() - 1].size() - 1;
	auto end_point = inspect_roi_[inspect_roi_.size() - 1][size / 2];

	// 최대 ROI 찾기 함수
	auto find_max_roi = [&](float* image, int& start_id, int& end_id, int roi_cnt)
	{
		std::vector<int> vec_sum_start, vec_sum_end; //(inspect_roi.size(), 0), vec_sum_end(inspect_roi.size(), 0);
		vector<cv::Rect> rcRoI;
		// ROI 순회하며 시작 및 종료 좌표 계산
		for (int i = 0; i < inspect_roi_.size() - 1; ++i)
		{
			for (int j = 0; j < inspect_roi_[i].size() - 1; ++j)
			{
				auto pos_lt = inspect_roi_[i][j];
				auto pos_rb = inspect_roi_[i + 1][j + 1];
				int sum_s = 0, sum_e = 0;
				for (int x = pos_lt.x; x < pos_rb.x; ++x) {
					for (int y = pos_lt.y; y < pos_rb.y; ++y) {
						if (image[y * h + x] != 0) {
							sum_s++;
							sum_e++;
						}
					}
				}
				vec_sum_start.push_back((i <= inspect_roi_.size() / 2 - 1 && j < roi_cnt) ? sum_s : -1);
				vec_sum_end.push_back((dcm_info_.primary_angle < 0 && dcm_info_.secondary_angle > 0) ?
					(i > inspect_roi_.size() / 2 - 1 && j >= roi_cnt) ? sum_e : -1 : sum_e);

				rcRoI.push_back(cv::Rect(pos_lt.x, pos_lt.y, pos_rb.x - pos_lt.x, pos_rb.y - pos_lt.y));
			}
		}

		start_id = std::max_element(vec_sum_start.begin(), vec_sum_start.end()) - vec_sum_start.begin();
		end_id = std::max_element(vec_sum_end.begin(), vec_sum_end.end()) - vec_sum_end.begin();
		return rcRoI;
	};

	//필터이미지
	auto find_image = apply_filters(buffer.get(), buffer1.get(), buffer5.get(), filter_image);
	if (!find_image)
		return false;

	if (find_line)
	{
		auto testFrozen = std::make_unique<bool[]>(w * h);
		auto point_instance = get_points_instance();
		point_instance.clear_points();

		//끝점 찾기
		auto endPoint_Pretreatment = [&](float* pTracker, cv::Rect maxRoi)
		{
			vector<float2> _srcPnt;
			for (int x = maxRoi.x; x < maxRoi.x + maxRoi.width; ++x) {
				for (int y = maxRoi.y; y < maxRoi.y + maxRoi.height; ++y) {
					if (filter_image[(y * h) + x] != 0)
						_srcPnt.push_back(make_float2(x, y));
				}
			}

			if (_srcPnt.empty()) {
				std::copy(filter_image, filter_image + w * h, pTracker);
				return;
			}

			// 트래커 초기화 및 처리
			auto T1 = std::make_unique<float[]>(w * h);
			auto Y1 = std::make_unique<int[]>(w * h);
			std::fill_n(T1.get(), w * h, -1.0f);
			std::fill_n(Y1.get(), w * h, 0);

			custom_tracker2(T1.get(), Y1.get(), filter_image, testFrozen.get(), _srcPnt, _srcPnt[0], 15000, make_uint2(w, h), false, false);

			// 처리된 트래커 값을 pTracker에 복사
			std::copy(T1.get(), T1.get() + w * h, pTracker);
		};

		auto find_end_point = [&](std::vector<float2>& roi_pos, distance_and_id& distance_metric, int roi_cnt)
		{
			if (roi_pos.empty()) return;

			vector<pair<int, distance_and_id>> end_points_roi;
			auto compare = [](const pair<int, distance_and_id>& a, const pair<int, distance_and_id>& b) {
				return a.first < b.first;
			};

			if (dcm_info_.primary_angle < 0 && dcm_info_.secondary_angle > 0) {  // RAO CRA 우하
				for (int i = roi_cnt; i < inspect_roi_.back().size(); ++i) {
					auto end_pt = inspect_roi_.back()[i];
					auto distance_roi = get_distance_end(roi_pos, end_pt);
					end_points_roi.push_back({ distance_roi.distance, distance_roi });
				}
			}
			else if ((dcm_info_.primary_angle > 0 && dcm_info_.secondary_angle > 0) ||
				(dcm_info_.primary_angle > 0 && dcm_info_.secondary_angle < 0 && fabs(dcm_info_.primary_angle) > fabs(dcm_info_.secondary_angle))) {  // LAO CRA
				for (int i = 0; i <= roi_cnt; ++i) {  // 좌하
					auto end_pt = inspect_roi_.back()[i];
					auto distance_roi = get_distance_end(roi_pos, end_pt);
					end_points_roi.push_back({ distance_roi.distance, distance_roi });
				}
				for (int i = 0; i <= inspect_roi_.size() / 2; ++i) {  // 우상
					auto end_pt = inspect_roi_[i].back();
					auto distance_roi = get_distance_end(roi_pos, end_pt);
					if (dcm_info_.primary_angle > 0 && dcm_info_.secondary_angle > 0 && distance_roi.distance > 100) {
						end_points_roi.push_back({ distance_roi.distance, distance_roi });
					}
					else {
						end_points_roi.push_back({ distance_roi.distance, distance_roi });
					}
				}
			}
			else {  // 좌하 우하
				for (int i = 0; i < inspect_roi_.back().size(); ++i) {
					auto end_pt = inspect_roi_.back()[i];
					auto distance_roi = get_distance_end(roi_pos, end_pt);
					end_points_roi.push_back({ distance_roi.distance, distance_roi });
				}
			}

			std::sort(end_points_roi.begin(), end_points_roi.end(), compare);
			end_point = end_points_roi.front().second.pos;
			distance_metric = end_points_roi.front().second;
		};
		

		//최대 혈관 ROI찾기
		vector <cv::Rect> rcRoI;
		int start_max_id = 0, end_max_id = 0;
		int roi_x = 9;
		auto skeleton = std::make_unique<float[]>(w * h);
		for (int rep = 0; rep < w * h; rep++)
			skeleton[rep] = filter_image[rep];

		cvAlgorithm_.create_skeleton_image(skeleton.get(), skeleton.get(), w, h, false);
		rcRoI = find_max_roi(skeleton.get(), start_max_id, end_max_id, roi_x);

		/////////////////////////////////////혈관, 끝점 시작점찾기//////////////////////////////////////////////////////
		{
			auto startToend_tracker = std::make_unique<float[]>(w * h);
			auto endTostart_tracker = std::make_unique<float[]>(w * h);

			auto skeleton_se = std::make_unique<float[]>(w * h);
			auto skeleton_es = std::make_unique<float[]>(w * h);
			

			for (int rep = 0; rep < w * h; rep++)
			{
				if (filter_image[rep] == 0)
					testFrozen.get()[rep] = 1;
				else
					testFrozen.get()[rep] = 0;
			}

			endPoint_Pretreatment(endTostart_tracker.get(), rcRoI[end_max_id]);
			endPoint_Pretreatment(startToend_tracker.get(), rcRoI[start_max_id]);


			cvAlgorithm_.create_skeleton_image(startToend_tracker.get(), skeleton_se.get(), w, h, true);
			cvAlgorithm_.create_skeleton_image(endTostart_tracker.get(), skeleton_es.get(), w, h, true);

			cvAlgorithm_.save_image(skeleton_se.get(), "\\test\\skeleton_se", w, h);
			cvAlgorithm_.save_image(skeleton_es.get(), "\\test\\skeleton_es", w, h);

			// 벡터 초기화
			std::vector<float2> all_pos_se, all_pos_es, roi_pos_se, roi_pos_es;

			// ROI 내에서 스켈레톤 점 찾기
			for (int y = 0; y < h; ++y) {
				for (int x = 0; x < w; ++x) {
					if (skeleton_se[y * h + x] != 0) {
						all_pos_se.push_back(make_float2(x, y));
						if (x > point_lt.x && x < point_rb.x && y > point_lt.y && y < point_rb.y)
							roi_pos_se.push_back(make_float2(x, y));
					}
					if (skeleton_es[y * h + x] != 0) {
						all_pos_es.push_back(make_float2(x, y));
						if (x > point_lt.x && x < point_rb.x && y > point_lt.y && y < point_rb.y)
							roi_pos_es.push_back(make_float2(x, y));
					}
				}
			}

			// 시작점 계산
			auto start_distance_roi = get_distance_end(roi_pos_se, start_point);
			auto start_distance_all = get_distance_end(all_pos_se, start_point);
			start_point = (start_distance_roi.distance < start_distance_all.distance) ? start_distance_roi.pos : start_distance_all.pos;

			distance_and_id start_distance_e, end_distance_e;
			find_end_point(roi_pos_es, end_distance_e, roi_x);
			find_end_point(roi_pos_se, start_distance_e, roi_x);
			end_point = (dcm_info_.primary_angle < 0 && dcm_info_.secondary_angle > 0 && start_distance_e.distance < end_distance_e.distance) ? start_distance_e.pos : end_distance_e.pos;
		}
		point_instance.start_point = start_point;
		point_instance.end_point = end_point;
		set_points_instance(point_instance);
	}

	processImage(buffer.get(), buffer1.get(), buffer5.get());

	return true;
}

float Angio_Algorithm::AppendParabolaLine(vector<float2>& Line, vector<float2>& ParabolaLine, float2 ptStart, float2 ptEnd, vector<float2> LineForward, vector<float2> LineBackward, float lfDisForward, float lfDisBackward)
{
	auto SortLine = [&](vector<float2>& centerline, const vector<float2>& LineForward, const vector<float2>& LineBackward, bool isForward) 
	{
		int n = 1;
		if (isForward) {
			for (int j = LineBackward.size() - 1; j >= 0; --j) {
				if (!LineForward.empty() && get_distance_points(LineForward.back(), LineBackward[j]) < 5.0f) {
					n = j;
					break;
				}
			}
			for (int j = n; j >= 0; --j) {
				centerline.push_back(LineBackward[j]);
			}
		}
		else {
			for (int j = 0; j < LineForward.size(); ++j) {
				if (!LineBackward.empty() && get_distance_points(LineBackward.front(), LineForward[j]) < 5.0f) {
					n = j;
					break;
				}
			}
			for (int j = n; j < LineForward.size(); ++j) {
				centerline.push_back(LineForward[j]);
			}
		}
	};

	// 거리 계산 및 변수 초기화
	bool isForward = true;
	float2 ptS = ptStart, ptE = ptEnd;
	float disForward = LineForward.empty() ? get_distance_points(ptStart, ptEnd) : get_distance_points(ptStart, LineForward[0]);
	float disBackward = LineBackward.empty() ? get_distance_points(ptStart, ptEnd) : get_distance_points(ptStart, LineBackward.back());
	float dis = get_distance_points(ptEnd, ptStart);

	// 선이 끊어졌는지 확인
	if (disForward > 5.0f || disBackward > 5.0f) {
		if (disForward > disBackward) {
			isForward = false;
			dis = disBackward;
			Line.insert(Line.end(), LineBackward.rbegin(), LineBackward.rend());
			if (!LineForward.empty()) {
				auto _dm = get_distance_end(LineForward, Line.back());
				ptE = _dm.pos;
			}
		}
		else {
			dis = disForward;
			Line.insert(Line.end(), LineForward.begin(), LineForward.end());
			if (!Line.empty()) {
				auto _dm = get_distance_end(LineBackward, Line.back());
				ptE = _dm.pos;
			}
			else {
				ptE = LineBackward.back();
			}
		}

		// 곡선 추가
		if (!Line.empty() && get_distance_points(Line.back(), ptEnd) > 10.0f) {
			ptS = Line.back();
			vector<float2> vecpt3 = { ptStart, ptS, ptE };
			make_parabola_data(vecpt3, ParabolaLine, 1, 2);
			if (ParabolaLine.size() != 0)
			{
				Line.insert(Line.end(), ParabolaLine.begin(), ParabolaLine.end());
				if (!isForward)
				{
					// LineBackward 비우기
					std::vector<float2> OriginalLine = LineBackward;
					LineBackward.clear();
					LineBackward.insert(LineBackward.end(), ParabolaLine.rbegin(), ParabolaLine.rend());
					LineBackward.insert(LineBackward.end(), OriginalLine.begin(), OriginalLine.end());
				}
				else
					LineForward.insert(LineForward.end(), ParabolaLine.begin(), ParabolaLine.end());

			}
		}
		else if (get_distance_points(Line[0], ptStart) > 10.0f) {
			ptS = Line[0];
			vector<float2> vecpt3 = { ptStart, ptS, ptE };
			make_parabola_data(vecpt3, ParabolaLine, 0, 1);
			if (ParabolaLine.size() != 0)
			{
				std::vector<float2> OriginalLine = Line;
				Line.clear();
				Line.insert(Line.end(), ParabolaLine.begin(), ParabolaLine.end());
				Line.insert(Line.end(), OriginalLine.begin(), OriginalLine.end());


				if (!isForward)
				{
					// LineBackward 비우기
					std::vector<float2> backup_line = LineBackward;
					LineBackward.clear();
					LineBackward.insert(LineBackward.end(), ParabolaLine.rbegin(), ParabolaLine.rend());
					LineBackward.insert(LineBackward.end(), backup_line.begin(), backup_line.end());
				}
				else
				{
					std::vector<float2> backup_line = LineForward;
					LineForward.clear();
					LineForward.insert(LineForward.end(), ParabolaLine.rbegin(), ParabolaLine.rend());
					LineForward.insert(LineForward.end(), backup_line.begin(), backup_line.end());

				}
			}
		}

		// 양방향 선 정렬
		if (!LineForward.empty() && !LineBackward.empty()) {
			SortLine(Line, LineForward, LineBackward, isForward);
		}
	}

	return dis;
}

void Angio_Algorithm::segmentation(model_info::segmentation_manual_type type)
{
	auto testFrozen = (Frozen_);
	auto buffer = get_float_buffer(model_info::image_float_type::SPEED_IMAGE);
	vector<float2>  center_line, left_line, right_line;
	int w = get_width();
	int h = get_height();

	auto search_line = [&](float* image, vector<float2>& centerline, float2 start_point, float2 end_point)
	{
		float lfptCenterDisForward = 0, lfptCenterDisBackward = 0;
		float lfptEndDisForward = 0, lfptEndDisBackward = 0;
		vector<float2> ptCenterLine, ptCenterParabolaLine, ptCenterForwardLine, ptCenterBackwardLine;
		vector<float2> ptEndLine, ptEndParabolaLine, ptEndForwardLine, ptEndBackwardLine;

		FindCenterLine(image, testFrozen, start_point, end_point, ptCenterForwardLine, ptCenterBackwardLine, lfptCenterDisForward, lfptCenterDisBackward, 0);
		FindCenterLine(image, testFrozen, start_point, end_point, ptEndForwardLine, ptEndBackwardLine, lfptEndDisForward, lfptEndDisBackward, 1);
		if (ptCenterForwardLine.size() == 0 && ptCenterBackwardLine.size() == 0 &&
			ptEndForwardLine.size() == 0 && ptEndBackwardLine.size() == 0)
		{
			return;
		}

		vector<float> dis;
		{
			auto L = get_distance_points(start_point, end_point);
			float disEnd = L, disCenter = L;
			if (ptCenterForwardLine.size() != 0 && ptCenterBackwardLine.size() != 0)
				disCenter = AppendParabolaLine(ptCenterLine, ptCenterParabolaLine, start_point, end_point, ptCenterForwardLine, ptCenterBackwardLine, lfptCenterDisForward, lfptEndDisBackward);
			else
			{
				if (ptCenterForwardLine.size() != 0)
					disCenter = get_distance_points(start_point, ptCenterForwardLine[0]);
				else if (ptCenterBackwardLine.size() != 0)
					disCenter = get_distance_points(start_point, ptCenterBackwardLine[ptCenterBackwardLine.size() - 1]);
			}

			if (ptEndForwardLine.size() != 0 && ptEndBackwardLine.size() != 0)
				disEnd = AppendParabolaLine(ptEndLine, ptEndParabolaLine, start_point, end_point, ptEndForwardLine, ptEndBackwardLine, lfptEndDisForward, lfptEndDisBackward);
			else
			{
				if (ptEndForwardLine.size() != 0)
					disEnd = get_distance_points(start_point, ptEndForwardLine[0]);
				else if (ptEndBackwardLine.size() != 0)
					disEnd = get_distance_points(start_point, ptEndBackwardLine[ptEndBackwardLine.size() - 1]);
			}
			dis.push_back(disCenter);
			dis.push_back(disEnd);
		}


		int min_index = std::min_element(dis.begin(), dis.end()) - dis.begin();
		if (min_index == 0)
		{
			cvAlgorithm_.draw_line(get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE), ptCenterForwardLine, make_int3(0, 0, 255), w, h, "Forward");
			cvAlgorithm_.draw_line(get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE), ptCenterBackwardLine, make_int3(0, 0, 255), w, h, "BackwardLine");
			for (int j = 0; j < ptCenterLine.size(); j++)
				centerline.push_back(ptCenterLine[j]);

			if (ptCenterLine.size() == 0)
			{
				if (ptCenterForwardLine.size() > ptCenterBackwardLine.size())
				{
					for (int j = 0; j < ptCenterForwardLine.size(); j++)
						centerline.push_back(ptCenterForwardLine[j]);
				}
				else
				{
					for (int j = ptCenterBackwardLine.size() - 1; j >= 0; j--)
						centerline.push_back(ptCenterBackwardLine[j]);
				}
			}
		}
		else if (min_index == 1)
		{
			cvAlgorithm_.draw_line(get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE), ptEndForwardLine, make_int3(0, 0, 255), w, h, "Forward");
			cvAlgorithm_.draw_line(get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE), ptEndBackwardLine, make_int3(0, 0, 255), w, h, "BackwardLine");

			for (int j = 0; j < ptEndLine.size(); j++)
				centerline.push_back(ptEndLine[j]);

			if (ptEndLine.size() == 0)
			{
				if (ptEndForwardLine.size() > ptEndBackwardLine.size())
				{
					for (int j = 0; j < ptEndForwardLine.size(); j++)
						centerline.push_back(ptEndForwardLine[j]);
				}
				else
				{
					for (int j = ptEndBackwardLine.size() - 1; j >= 0; j--)
						centerline.push_back(ptEndBackwardLine[j]);
				}
			}
		}

		cvAlgorithm_.draw_line(get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE), centerline, make_int3(0, 0, 255), w, h, "centerline");
	};

	auto point_instance = get_points_instance();


	if (type == model_info::segmentation_manual_type::find_centerline)
	{
		vector<float2> pnts;
		pnts.push_back(point_instance.start_point);
		pnts.push_back(point_instance.branch_point[0].second);
		pnts.push_back(point_instance.branch_point[1].second);
		pnts.push_back(point_instance.end_point);

		vector<vector<float2>> centerlines;
		for (auto i = 0; i < pnts.size() - 1; i++)
		{
			vector<float2> Line;
			search_line(buffer, center_line, pnts[i], pnts[i + 1]);
			centerlines.push_back(Line);
		}

		if (centerlines.size() == 3)
		{
			vector<float> D;
			for (int i = 0; i < centerlines.size() - 1; i++)
			{
				if (centerlines[i].size() < 2 || centerlines[i + 1].size() < 2)
					break;
				float2 pt1 = centerlines[i][centerlines[i].size() - 2];
				float2 pt2 = centerlines[i][centerlines[i].size() - 1];
				float2 pt3 = centerlines[i + 1][0];
				float2 pt4 = centerlines[i + 1][1];
				float2 v1 = make_float2((pt2.x - pt1.x), (pt2.y - pt1.y));
				float2 v2 = make_float2((pt4.x - pt3.x), (pt4.y - pt3.y));
				D.push_back((v1.x * v2.x) + (v1.y * v2.y));
			}
			if (D.size() == 2)
			{
				if (D[0] < 0)
				{
					auto dm_ = get_distance_end(centerlines[0], point_instance.branch_point[0].second);
					if (dm_.distance < 20)
					{
						centerlines.clear();
						pnts[1] = point_instance.branch_point[1].second;
						pnts[2] = point_instance.branch_point[0].second;
						for (auto i = 0; i < pnts.size() - 1; i++)
						{
							vector<float2> Line;
							search_line(buffer, center_line, pnts[i], pnts[i + 1]);
							centerlines.push_back(Line);
						}
					}
				}

			}
		}


		for (int i = 0; i < centerlines.size(); i++)
		{
			for (int j = 0; j < centerlines[i].size(); j++)
			{
				center_line.push_back(centerlines[i][j]);
			}
		}
		manual_segmentation_edgeline(center_line);
	}
	else
	{
		search_line(buffer, center_line, point_instance.start_point, point_instance.end_point);
		int vessnum = center_line.size();
		if (center_line.size() != 0)
		{
			if (type == model_info::segmentation_manual_type::start_end_findline)
			{
				float _min = FLT_MAX;
				auto ptFind = point_instance.start_point;
				for (int i = 0; i < vessnum - 1; i++)
				{
					float2 ptIntersect;
					bool isintersect = get_intersect_point(center_line[i], center_line[i + 1], make_float2(0, inspect_roi_[0][0].y), make_float2(w, inspect_roi_[0][0].y), ptIntersect);
					if (isintersect)
					{
						auto _val = get_distance_points(ptIntersect, point_instance.start_point);
						if (_val < _min)
						{
							_min = _val;
							ptFind = ptIntersect;
						}
					}
				}
				point_instance.start_point = ptFind;
			}
			else if (type == model_info::segmentation_manual_type::branch_findline)
			{
				manual_segmentation_edgeline(center_line);
				result_info::segmentation_line2D_instance segmentation_line2D_instance_ = angio_result_.get_segmentation_line2D_instance();
				left_line = segmentation_line2D_instance_.get_line(result_info::line_position::LEFT_LINE);
				right_line = segmentation_line2D_instance_.get_line(result_info::line_position::RIGHT_LINE);
				vector<float> distances_lines;
				for (int i = 0; i < vessnum; i++)
				{
					auto d = get_distance_points(left_line[i], right_line[i]);
					distances_lines.push_back(d);
				}

				// 가장 작은 거리를 가진 인덱스 계산
				int minIdx = std::min_element(distances_lines.begin(), distances_lines.end()) - distances_lines.begin();
				minIdx = std::clamp(minIdx, 1, vessnum - 2);

				// 협착점 시작 및 끝 인덱스
				int Idx = (minIdx < std::abs(vessnum - minIdx)) ? vessnum * 0.75 : vessnum * 0.25;

				auto minpos = center_line[minIdx];
				auto pos = center_line[Idx];

				auto d1 = get_distance_points(point_instance.end_point, pos);
				auto d2 = get_distance_points(point_instance.end_point, minpos);
				if (d1 > d2)//	if (nId < min) // 협착점이 아래인경우
				{
					point_instance.branch_point.push_back(make_pair(false, pos));
					point_instance.branch_point.push_back(make_pair(true, minpos));
				}
				else
				{
					point_instance.branch_point.push_back(make_pair(true, minpos));
					point_instance.branch_point.push_back(make_pair(false, pos));
				}
			}
		}
		else
		{
			if (type == model_info::segmentation_manual_type::branch_findline)
			{
				// 중심선이 없는 경우 중간점과 임의의 좌표 추가
				auto pos = make_float2((point_instance.start_point.x + point_instance.end_point.x) / 2, (point_instance.start_point.y + point_instance.end_point.y) / 2);
				auto minpos = make_float2((pos.x + 10), (pos.y + 10));
				point_instance.branch_point.push_back(make_pair(false, pos));
				point_instance.branch_point.push_back(make_pair(true, minpos));
			}
		}
		set_points_instance(point_instance);
	}
}

void Angio_Algorithm::manual_segmentation_edgeline(vector<float2>& center_line)
{
	if (center_line.size() == 0)
		return;
	int w = get_width();
	int h = get_height();

	result_info::segmentation_line2D_instance segmentation_line2D_instance_ = angio_result_.get_segmentation_line2D_instance();
	segmentation_line2D_instance_.clear();

	vector<float> buffer2(w * h);
	vector<float> buffer3(w * h);
	vector<float> buffer4(w * h);
	int vessnum = center_line.size();

	vector<float2> tempMedial(vessnum);
	vector<float2> tempLeft(vessnum);
	vector<float2> tempRight(vessnum);
	vector<float> tempRadius(vessnum);

	gpu_eigenimg_float(get_float_buffer(model_info::image_float_type::LEBELING_IMAGE), get_float_buffer(model_info::image_float_type::BOUNDARY_IMAGE), buffer2.data(), buffer3.data(), buffer4.data(), 0.3f, w, h);
	cvAlgorithm_.save_image(get_float_buffer(model_info::image_float_type::BOUNDARY_IMAGE), "\\test\\BOUNDARY_IMAGE", w, h);

	test_gpuBASOC(get_float_buffer(model_info::image_float_type::BOUNDARY_IMAGE), center_line.data(), tempMedial.data(), tempLeft.data(), tempRight.data(), tempRadius.data(), vessnum, 12, w, h);
	center_line.clear();
	for (int i = 0; i < vessnum; i++)
	{
		float2 C = make_float2((tempLeft.data()[i].x + tempRight.data()[i].x) / 2, (tempLeft.data()[i].y + tempRight.data()[i].y) / 2);
		center_line.push_back(C);
	}

	segmentation_line2D_instance_.set_line(result_info::line_position::LEFT_LINE, tempLeft);
	segmentation_line2D_instance_.set_line(result_info::line_position::RIGHT_LINE, tempRight);
	segmentation_line2D_instance_.set_line(result_info::line_position::CENTER_LINE, center_line);
	angio_result_.set_segmentation_line2D_instance(segmentation_line2D_instance_);
}

void Angio_Algorithm::segmentation_output(model_info::segmentation_exe_type run_type, int index)
{
	result_info::segmentation_line2D_instance segmentation_line2D_instance_ = angio_result_.get_segmentation_line2D_instance();

	if (run_type == model_info::segmentation_exe_type::run_centerline)
	{
		return;
		std::string output_filename = (index == 0) ? "../output/output1.nii.gz" : "../output/output2.nii.gz";
		result_info::segmentation_instance segmentation_instance_ = angio_result_.get_segmentation_instance();
		segmentation_instance_.clear();

		bool state = get_segmentation_file_niigz(output_filename, segmentation_instance_, index);
		if (!state)
			return;

		int optimal_imgage_id = segmentation_instance_.optimal_image_id;

		int w = get_width(); int h = get_height();
		vector<float2>  centline, leftline, rightline;

		for (int i = 0; i < segmentation_instance_.get_center_line_points().size(); i++)
		{
			auto pos = segmentation_instance_.get_center_line_points()[i];

			get_float_buffer(model_info::image_float_type::CENTLINE_IMAGE)[int(pos.y * h + pos.x)] = 1.0;
			centline.push_back(pos);
		}

		for (int i = 0; i < segmentation_instance_.get_labeling_point(optimal_imgage_id).second.size(); i++)
		{
			auto pos = segmentation_instance_.get_labeling_point(optimal_imgage_id).second[i];
			get_float_buffer(model_info::image_float_type::LEBELING_IMAGE)[int(pos.y * h + pos.x)] = 1.0;
		}
		cvAlgorithm_.save_image(get_float_buffer(model_info::image_float_type::LEBELING_IMAGE), "test\\LEBELING_IMAGE", w, h);
		auto find_endpoints = [&](float* image, vector<float2> input_points)
		{
			vector<float2> output_endpoints;
			int w = get_width(); int h = get_height();
			for (int i = 0; i < input_points.size(); i++)
			{
				int x = input_points[i].x;
				int y = input_points[i].y;
				int count = 0;
				for (int dx = -1; dx <= 1; ++dx)
				{
					for (int dy = -1; dy <= 1; ++dy)
					{
						if (dx != 0 || dy != 0)
						{
							if (image[(x + dx) + (y + dy) * h] == 1 || image[(x + dx) + (y + dy) * h] == 255)
							{
								count++;
							}
						}
					}
				}
				if (count == 1) {
					output_endpoints.push_back(make_float2(x, y));
				}
			}
			return output_endpoints;
		};

		auto getNeighbors = [&](const float2& p, const float* image)
		{
			vector<float2> neighbors;
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					if (dx == 0 && dy == 0) continue;
					int nx = p.x + dx;
					int ny = p.y + dy;
					if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
						if (image[ny * h + nx] == 1 || image[ny * h + nx] == 255) {
							neighbors.push_back(make_float2(nx, ny));
						}
					}
				}
			}
			return neighbors;
		};

		auto traceCenterline = [&](float* center_image, vector<float2> points)
		{
			int w = get_width(); int h = get_height();
			Mat visited = Mat::zeros(Size(w, h), CV_8UC1);
			vector<float2> centerline;
			auto start = points[0];
			auto end = points[1];
			stack<float2> s;
			s.push(start);
			while (!s.empty()) {
				float2 p = s.top();
				s.pop();
				if (visited.at<uchar>(p.x, p.y) == 1) continue;
				visited.at<uchar>(p.x, p.y) = 1;
				centerline.push_back(p);
				if (p.x == end.x && p.y == end.y) {
					break;
				}
				vector<float2> neighbors = getNeighbors(p, center_image);
				if (neighbors.empty())
				{
					//float min_distance = FLT_MAX;
					//for (const float2& pt : points) {
					//	float dist = hypot(pt.x - p.x, pt.y - p.y);
					//	if (dist < min_distance && dist > 0) { // 자기 자신 제외
					//		min_distance = dist;
					//		nearest_point = pt;
					//	}
					//}
					//if (min_distance != FLT_MAX) {
					//	s.push(nearest_point);
					//}
				}
				else
				{
					stack<float2> ss;
					for (const float2& neighbor : neighbors) {
						if (visited.at<uchar>(neighbor.x, neighbor.y) == 0) {
							s.push(neighbor);
							ss.push(neighbor);
						}
					}

					if (ss.size() == 0)
					{
						break;
					}
				}
			}
			return centerline;
		};

		auto end_points = find_endpoints(get_float_buffer(model_info::image_float_type::CENTLINE_IMAGE), centline);
		centline.clear();
		centline = traceCenterline(get_float_buffer(model_info::image_float_type::CENTLINE_IMAGE), end_points);
		int vessnum = centline.size();
		if (vessnum == 0)
			return;

		{
			auto image = get_float_buffer(model_info::image_float_type::LEBELING_IMAGE);

			vector<float2> forward_line, backward_line;
			vector<float> foz;
			foz.resize(w * h);
			auto testFrozen = make_unique<bool[]>(w * h);
			//	std::transform(image, image + (w * h), testFrozen.get(), [](const float& val) { return val == 0 ? true : false; });
			//	std::transform(image, image + (w * h), foz.data(), [](const float& val) { return val == 0 ? 1.0 : 0.0; });
			float dis_forward = 0, dis_backward = 0;

			gpu_speedImage_float(image, get_float_buffer(model_info::image_float_type::SPEED_IMAGE), testFrozen.get(), w, h, 4, true, false);
			cvAlgorithm_.save_image(get_float_buffer(model_info::image_float_type::SPEED_IMAGE), "Test\\SPEED_IMAGE");

			FindCenterLine(get_float_buffer(model_info::image_float_type::SPEED_IMAGE), testFrozen.get(), centline[0], centline[centline.size() - 1], forward_line, backward_line, dis_forward, dis_backward,0);

			//Mat img(Size(w, h), CV_8UC3, Scalar(0));
			//for (int y = 0; y < h; y++)
			//{
			//	for (int x = 0; x < w; x++)
			//	{
			//		if (image[y * h + x] == 1)
			//			img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			//	}
			//}
			//for (int i = 0; i < forward_line.size() - 1; i++)
			//	line(img, Point(forward_line[i].x, forward_line[i].y), Point(forward_line[i + 1].x, forward_line[i + 1].y), Scalar(0, 0, 255), 1, 8);
			//
			//cvAlgorithm_.save_image(img, "Test\\img");
		}

		{
			result_info::segmentation_line2D_instance segmentation_line2D_instance_ = angio_result_.get_segmentation_line2D_instance();
			segmentation_line2D_instance_.clear();

			vector<float> buffer2(w * h);
			vector<float> buffer3(w * h);
			vector<float> buffer4(w * h);

			vector<float2> tempMedial(vessnum);
			vector<float2> tempLeft(vessnum);
			vector<float2> tempRight(vessnum);
			vector<float> tempRadius(vessnum);

			gpu_eigenimg_float(get_float_buffer(model_info::image_float_type::LEBELING_IMAGE), get_float_buffer(model_info::image_float_type::BOUNDARY_IMAGE), buffer2.data(), buffer3.data(), buffer4.data(), 0.3f, w, h);
			centline.clear();
			test_gpuBASOC(get_float_buffer(model_info::image_float_type::BOUNDARY_IMAGE), centline.data(), tempMedial.data(), tempLeft.data(), tempRight.data(), tempRadius.data(), vessnum, 7, w, h);
			for (int i = 0; i < vessnum; i++)
			{
				float2 C = make_float2((tempLeft.data()[i].x + tempRight.data()[i].x) / 2, (tempLeft.data()[i].y + tempRight.data()[i].y) / 2);
				centline.push_back(C);
				leftline.push_back(tempLeft[i]);
				rightline.push_back(tempRight[i]);
			}

			segmentation_line2D_instance_.set_line(result_info::line_position::LEFT_LINE, leftline);
			segmentation_line2D_instance_.set_line(result_info::line_position::RIGHT_LINE, rightline);
			segmentation_line2D_instance_.set_line(result_info::line_position::CENTER_LINE, centline);
			angio_result_.set_segmentation_line2D_instance(segmentation_line2D_instance_);
		}
		angio_result_.set_segmentation_instance(segmentation_instance_);
		segmentation_instance_ = angio_result_.get_segmentation_instance();
	}
	else
	{
		std::string file_path = (run_type == model_info::segmentation_exe_type::run_outlines) ?  R"(../output/lines)" : R"(../output/points)";
		file_path += (index == 0) ? R"(1.npy)" : R"(2.npy)";

		// npy 파일 읽기
		cnpy::NpyArray arr = cnpy::npy_load(file_path);
		std::vector<size_t> shape = arr.shape;

		double* data = arr.data<double>();
		size_t total_elements = 1;
		size_t rows = shape[0];
		size_t cols = shape[1];

		// 2차원 배열 형식으로 데이터 출력
		std::cout << "2D Array Data: " << std::endl;

		std::vector<float2> points;

		// 2차원 배열 데이터를 float2로 변환하여 저장
		for (size_t i = 0; i < rows; ++i) {
			float2 point;
			point.x = static_cast<float>(data[i * cols + 0]);  // 첫 번째 값 (x)
			point.y = static_cast<float>(data[i * cols + 1]);  // 두 번째 값 (y)
			points.push_back(point);
		}

		if (run_type == model_info::segmentation_exe_type::run_outlines)
		{
			vector<float2> l,r,c,nc;
			int len = (points.size() / 3);

			for (size_t i = 0; i < points.size() / 3; i++)
			{
				l.push_back(points[(len * 0) + i]);
				r.push_back(points[(len * 1) + i]);
				c.push_back(points[(len * 2) + i]);
			}

			CubicSpline spline;
			spline.set_points(c);
			// t값에 따른 보간된 좌표 출력
			std::cout << "Interpolated points along the spline:" << std::endl;
			for (float t = 0.0f; t < c.size(); t += 0.5f) {
				float2 point = spline.get_point(t);
				nc.push_back(point);
				std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
			}

			segmentation_line2D_instance_.clear();
			segmentation_line2D_instance_.set_line(result_info::line_position::LEFT_LINE, l);
			segmentation_line2D_instance_.set_line(result_info::line_position::RIGHT_LINE, r);
			segmentation_line2D_instance_.set_line(result_info::line_position::CENTER_LINE, c);
			angio_result_.set_segmentation_line2D_instance(segmentation_line2D_instance_);
			history_lines_.save_state(segmentation_line2D_instance_.get_lines());
		}
		else if(run_type == model_info::segmentation_exe_type::run_endpoints)
		{
			vector<float> frame_id, dis_centerlines;
			vector<float2> end_points;
			int len = (points.size() / 3);

			for (size_t i = 0; i < points.size() / 3; i++)
			{
				frame_id.push_back(points[(len * 0) + i].y);
				end_points.push_back(points[(len * 1) + i]);
				dis_centerlines.push_back(points[(len * 2) + i].y);
			}
			for (int i = 0; i < frame_id.size(); i++)
			{
				segmentation_line2D_instance_.dis_line2D_points.push_back(std::pair{ frame_id[i], dis_centerlines[i] });
				segmentation_line2D_instance_.end_points.push_back(std::pair{ frame_id[i], end_points[i] }) ;
			}
			angio_result_.set_segmentation_line2D_instance(segmentation_line2D_instance_);
			//std::string output_filename = (index == 0) ? R"(../output/points1.npy)" : R"(../output/points2.npy)";
			//std::filesystem::copy(file_path, output_filename, std::filesystem::copy_options::overwrite_existing);
		}
	}

}

void Angio_Algorithm::FindCenterLine(float* mapp, bool* testFrozen, float2 ptStart, float2 ptEnd, vector<float2>& Start_Line, vector<float2>& End_Line, float& lfDisForward, float& lfDisBackward, int nType)
{
	int h = get_height();
	int w = get_width();
	int size = w * h;

	int rep = 15000;

	vector<float2> EndPnts;
	vector<float2> StartPnts;
	
	EndPnts.push_back(make_float2(ptEnd.x, ptEnd.y));
	StartPnts.push_back(make_float2(ptStart.x, ptStart.y));

	auto pos = [&](const int2& pnt)->unsigned int {return pnt.x + pnt.y * w; };

	auto T1 = std::make_unique<float[]>(size);
	auto Y1 = std::make_unique<int[]>(size);
	memset(T1.get(), -1, sizeof(float) * size);
	memset(Y1.get(), 0, sizeof(int) * size);
	custom_tracker2(T1.get(), Y1.get(), mapp, testFrozen, EndPnts, ptStart, rep, make_uint2(w, h), false, false);

	auto T2 = std::make_unique<float[]>(w * h);
	auto Y2 = std::make_unique<int[]>(w * h);
	memset(T2.get(), -1, sizeof(float) * size);
	memset(Y2.get(), 0, sizeof(int) * size);
	custom_tracker2(T2.get(), Y2.get(), mapp, testFrozen, StartPnts, ptEnd, rep, make_uint2(w, h), false, false);

	Start_Line = ShortestPath(T1.get(), ptStart, EndPnts, 1.0f, w, h);
	End_Line = ShortestPath(T2.get(), ptEnd, StartPnts, 1.0f, w, h);

	if ((Start_Line.size() == 0 || End_Line.size() == 0))
	{
		float2 ptFind = make_float2((ptStart.x + ptEnd.x) / 2, (ptStart.y + ptEnd.y) / 2);
		ptFind = make_float2((ptFind.x + ptEnd.x) / 2, (ptFind.y + ptEnd.y) / 2);
		std::vector<float2> vecpos1;
		std::vector<float2> vecpos2;
		for (int y = 0; y < w * h; y++)
		{
			auto pos = make_float2(y % (int)h, (y / (int)w));
			if (fabs(T2.get()[y]) > 0)
				vecpos1.push_back(pos);

			if (fabs(T1.get()[y]) > 0)
				vecpos2.push_back(pos);
		}

		float2 newEndpos, newStartpos, newpos;
		if (nType == 0)
		{
			if (vecpos2.size() != 0)
			{
				auto id = get_distance_end(vecpos2, ptStart).id;
				newpos = vecpos2[id];
			}
			if (vecpos1.size() != 0)
			{
				auto id = get_distance_end(vecpos1, newpos).id;
				ptFind = vecpos1[id];
			}
		}
		else if (nType == 1)
		{
			if (vecpos2.size() != 0)
			{
				auto id = get_distance_end(vecpos2, ptEnd).id;
				newpos = vecpos2[id];
			}
			if (vecpos1.size() != 0)
			{
				auto id = get_distance_end(vecpos1, newpos).id;
				ptFind = vecpos1[id];
			}
		}
		if (Start_Line.size() == 0)
		{
			if (nType == 2)
				ptFind = ptStart;
			Start_Line = ShortestPath(T1.get(), ptFind, EndPnts, 1.0f, w, h, Start_Line.size() == 0);
		}

		if (End_Line.size() == 0)
		{
			if (nType == 2)
				ptFind = ptEnd;
			End_Line = ShortestPath(T2.get(), ptFind, StartPnts, 1.0f, w, h, End_Line.size() == 0);
		}
	}

	auto linelength = [](const vector<float2> pnts)->float {
		float xb, yb;
		float sum = 0;
		if (pnts.size() == 0)
			return sum;
		for (auto i = 0; i < pnts.size() - 1; i++)
		{
			xb = (pnts[i].x - pnts[i + 1].x);
			yb = (pnts[i].y - pnts[i + 1].y);
			sum += sqrtf(xb * xb + yb * yb);
		}
		return sum;
	};

	lfDisForward = linelength(Start_Line);
	lfDisBackward = linelength(End_Line);
}

void Angio_Algorithm::segmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type,int index)
{
	std::string output_filename = (index == 0) ? "../output/output1.nii.gz" : "../output/output2.nii.gz";

	std::string npy_filename;
	{
		run_AI_segmentation(model_type, run_type);
		if (run_type == model_info::segmentation_exe_type::run_outlines)
		{
			if (!std::filesystem::exists(R"(../output/output.nii.gz)"))
				return;
			std::filesystem::copy(R"(../output/output.nii.gz)", output_filename, std::filesystem::copy_options::overwrite_existing);
			npy_filename = (index == 0) ? R"(../output/lines1.npy)" : R"(../output/lines2.npy)";
			std::filesystem::copy(R"(../output/lines.npy)", npy_filename, std::filesystem::copy_options::overwrite_existing);

			result_info::segmentation_instance segmentation_instance_ = angio_result_.get_segmentation_instance();
			auto id = angio_result_.get_segmentation_instance().optimal_image_id;
			segmentation_instance_.clear();
			bool state = get_segmentation_file_niigz(output_filename, segmentation_instance_, index);
			if (!state)
				return;
			angio_result_.set_segmentation_instance(segmentation_instance_);
		}
		else if (run_type == model_info::segmentation_exe_type::run_centerline)
		{
			if (!std::filesystem::exists(R"(../output/output.nii.gz)"))
				return;
			std::filesystem::copy(R"(../output/output.nii.gz)", output_filename, std::filesystem::copy_options::overwrite_existing);
			result_info::segmentation_instance segmentation_instance_ = angio_result_.get_segmentation_instance();
			auto id = angio_result_.get_segmentation_instance().optimal_image_id;
			segmentation_instance_.clear();
			bool state = get_segmentation_file_niigz(output_filename, segmentation_instance_, index);
			if (!state)
				return;
			segmentation_instance_.set_optimal_image_id(id);
			angio_result_.set_segmentation_instance(segmentation_instance_);
		}
		else if (run_type == model_info::segmentation_exe_type::run_endpoints)
		{
			if (!std::filesystem::exists(R"(../output/points1.npy)"))
				return;
			npy_filename = (index == 0) ? R"(../output/points1.npy)" : R"(../output/points2.npy)";
			std::filesystem::copy(R"(../output/points.npy)", npy_filename, std::filesystem::copy_options::overwrite_existing);
		}
	}
	segmentation_output(run_type, index);
}

void Angio_Algorithm::set_file_niigz(int view_id , dcmHelper* dcm_helper)
{
	save_dcm_file(view_id, dcm_helper);
	const char* nifti_file = R"(../data/input.nii.gz)";
	nifti_image* nim = nifti_simple_init_nim();
	//auto image = nifti_simple_init_nim();
	nim->ndim = 3;
	nim->nx = dcm_helper->getCols();
	nim->ny = dcm_helper->getRows();
	nim->nz = dcm_helper->getNumberOfFrames();
	nim->nu = 1.0;
	nim->nt = 1.0;
	nim->nv = 1.0;
	nim->nw = 1.0;

	nim->nvox = (nim->nx * nim->ny * nim->nz);
	nim->datatype = NIFTI_TYPE_UINT8;
	nim->nbyper = sizeof(unsigned char);

	std::unique_ptr<uchar[]> _data(new uchar[nim->nvox]);
	//std::copy_n(dcm_helper->Data(), nim->nvox, _data.get());
	for (int i = 0; i < nim->nvox; i++)
		_data.get()[i] = dcm_helper->Data()[i];
	nim->data = _data.get();

	nim->dim[1] = dcm_helper->getCols();
	nim->dim[2] = dcm_helper->getRows();
	nim->dim[3] = dcm_helper->getNumberOfFrames();
	for (int i = 4; i < 8; i++)
		nim->dim[i] = 1.0;

	nim->dx = dcm_helper->getPixelSpacing()[0];
	nim->dy = dcm_helper->getPixelSpacing()[1];
	nim->dz = dcm_helper->getFrameTime();
	nim->dt = 0.0;
	nim->du = 0.0;
	nim->dv = 0.0;
	nim->dw = 0.0;

	nim->pixdim[1] = dcm_helper->getPixelSpacing()[0];
	nim->pixdim[2] = dcm_helper->getPixelSpacing()[1];
	nim->pixdim[3] = dcm_helper->getFrameTime();

	nim->qfac = 1;
	nim->qform_code = 1;
	nim->sform_code = 1;
	nim->xyz_units = NIFTI_UNITS_MM;
	nim->time_units = NIFTI_UNITS_UNKNOWN;

	nim->quatern_d = 1;


	if (nifti_set_filenames(nim, nifti_file, 0, 1) != 0)
		std::cerr << "Error: cannot set NIfTI filenames" << std::endl;
	else
		nifti_image_write(nim);
	_data.release();
	nifti_image_free(nim);

	dcm_info_.dcm_file = true;
	dcm_info_.detector = dcm_helper->getDistanceSourceToDetector();
	dcm_info_.Patient = dcm_helper->getDistanceSourceToPatient();
	dcm_info_.secondary_angle = dcm_helper->getPositionerSecondaryAngle();
	dcm_info_.primary_angle= dcm_helper->getPrimaryAngle();
}

bool Angio_Algorithm::get_segmentation_file_niigz(std::string path,result_info::segmentation_instance& segmentation_instance_, int index)
{
	nifti2_image* image = nifti_image_read(path.c_str(), 1);

	if (!image || nifti_image_load(image) < 0)
	{
		if (image) nifti_image_free(image);
		return false;
	}

	char* data = reinterpret_cast<char*>(image->data);
	int optimal_imgage_id = -1;

	for (int z = 0; z < image->nz; ++z)
	{
		vector<float2> labeling_points;
		vector<float2> center_line_points;
		for (int y = 0; y < image->ny; ++y)
		{
			for (int x = 0; x < image->nx; ++x)
			{
				int idx = x + y * image->nx + z * image->nx * image->ny;
				if ((data[idx]) == 127) //색칠영역
				{
					labeling_points.push_back(make_float2(x, y));
					optimal_imgage_id = z;
				}
				else if ((data[idx]) == -1) //센터라인값
				{
					labeling_points.push_back(make_float2(x, y));
					center_line_points.push_back(make_float2(x, y));
				}
				else if (data[idx] != 0) //배경
					std::cout << data[idx] << " ";
			}
		}
		//instance.id = z;
		segmentation_instance_.set_center_line_points(center_line_points);
		segmentation_instance_.set_labeling_points(std::pair{ z, labeling_points });
	}

	nifti_image_free(image);

	if (optimal_imgage_id == -1)
		return false;

	segmentation_instance_.set_optimal_image_id(optimal_imgage_id);
	segmentation_instance_.set_bpm(image->descrip);
	return true;
}


//데이터 균등화
void Angio_Algorithm::set_verticality_Distance(vector<float2> input_line_points, int id_s, int id_e, int& nminIndex, bool equal)
{
	vector<float> output_distance_points;
	vector<float2> new_line_points_c;
	vector<float2> new_line_points_l;
	vector<float2> new_line_points_r;

	vector<float2>line_points_l = get_lines2D(result_info::line_position::LEFT_LINE);
	vector<float2>line_points_r = get_lines2D(result_info::line_position::RIGHT_LINE);
	int N = 100;

	int id_s_l = 0;
	int id_s_r = 0;
	for (int i = 0; i < input_line_points.size(); i++)
	{
		float radian = 0;
		float dbRadian3 = 0;
		float2 ptS = make_float2(0, 0);
		float2 ptE = make_float2(0, 0);
		auto ptCenterL = make_float2(0, 0);
		auto ptCenterR = make_float2(0, 0);
		auto ptCenterVertical = make_float2(0, 0);
		if (i == 0 && id_s == 0)
		{
			radian = atan2(input_line_points[i].y - input_line_points[i + 1].y, input_line_points[i].x - input_line_points[i + 1].x) + (M_PI * 180 / 180);
			ptS = get_rotation_point(-radian, input_line_points[i], input_line_points[i]);
			ptE = get_rotation_point(-radian, input_line_points[i + 1], input_line_points[i]);
			dbRadian3 = atan2((ptS.y - ptE.y), (ptS.x - ptE.x)) / 2;
			float2 pt = get_rotation_point(dbRadian3, input_line_points[i + 1], input_line_points[i]);
			dbRadian3 = atan2(pt.y - input_line_points[i].y, pt.x - input_line_points[i].x);
			ptCenterVertical = get_rotation_point(-dbRadian3, pt, input_line_points[i]);
		}
		else if (i == input_line_points.size() - 1 && (id_e == input_line_points.size() - 1 || id_e == 0))
		{
			radian = atan2(input_line_points[i - 1].y - input_line_points[i].y, input_line_points[i - 1].x - input_line_points[i].x) + (M_PI * 180 / 180);
			ptS = get_rotation_point(-radian, input_line_points[i - 1], input_line_points[i]);
			ptE = get_rotation_point(-radian, input_line_points[i], input_line_points[i]);
			dbRadian3 = atan2((ptS.y - ptE.y), (ptS.x - ptE.x)) / 2;
			float2 pt = get_rotation_point(dbRadian3, input_line_points[i - 1], input_line_points[i]);
			dbRadian3 = atan2(pt.y - input_line_points[i].y, pt.x - input_line_points[i].x);
			ptCenterVertical = get_rotation_point(-dbRadian3, pt, input_line_points[i]);
		}
		else if (i > 0 && i < input_line_points.size() - 1)
		{
			radian = atan2(input_line_points[i].y - input_line_points[i + 1].y, input_line_points[i].x - input_line_points[i + 1].x) + (M_PI * 180 / 180);
			ptS = get_rotation_point(-radian, input_line_points[i - 1], input_line_points[i]);
			ptE = get_rotation_point(-radian, input_line_points[i + 1], input_line_points[i]);
			float Q2 = atan2((ptS.y - input_line_points[i].y), (ptS.x - input_line_points[i].x));
			float Q1 = atan2((ptE.y - input_line_points[i].y), (ptE.x - input_line_points[i].x));
			dbRadian3 = (Q2 - Q1) / 2;
			float2 pt = get_rotation_point(dbRadian3, ptE, input_line_points[i]);
			pt = get_rotation_point(radian, pt, input_line_points[i]);
			dbRadian3 = atan2(pt.y - input_line_points[i].y, pt.x - input_line_points[i].x);
			ptCenterVertical = get_rotation_point(-dbRadian3, pt, input_line_points[i]);
		}
		else
		{
			continue;
		}
		if (dbRadian3 < 0)
		{
			ptCenterL = make_float2(ptCenterVertical.x + N, ptCenterVertical.y);
			ptCenterR = make_float2(ptCenterVertical.x - N, ptCenterVertical.y);
		}
		else
		{
			ptCenterL = make_float2(ptCenterVertical.x - N, ptCenterVertical.y);
			ptCenterR = make_float2(ptCenterVertical.x + N, ptCenterVertical.y);
		}

		//tCenterL = make_float2(ptCenterVertical.x - N, ptCenterVertical.y);
		//tCenterR = make_float2(ptCenterVertical.x + N, ptCenterVertical.y);
		//{
		//	vector<float2> line;
		//	line.push_back(ptCenterL);
		//	line.push_back(ptCenterR);
		//	cvAlgorithm_.draw_line(cvimage2, line, ORANGE_COLOR, 512, 512, "set_verticality_Distance" + std::to_string(int(1)));
		//}

		ptCenterL = get_rotation_point((dbRadian3), ptCenterL, input_line_points[i]);
		ptCenterR = get_rotation_point((dbRadian3), ptCenterR, input_line_points[i]);
		//{
		//	vector<float2> line;
		//	line.push_back(ptCenterL);
		//	line.push_back(ptCenterR);
		//	cvAlgorithm_.draw_line(cvimage2, line, ORANGE_COLOR, 512, 512, "set_verticality_Distance" + std::to_string(int(2)));
		//}
		float2 ptIntersectL = make_float2(0, 0);
		float2 ptIntersectR = make_float2(0, 0);
		float _minLD = FLT_MAX;
		float _minRD = FLT_MAX;
		bool bFindL = false;
		bool bFindR = false;
		for (int n = 0; n < line_points_l.size() - 1; n++)
		{
			float2 ptIntersect = make_float2(0, 0);
			if (get_intersect_point(ptCenterL, ptCenterR, line_points_l[n], line_points_l[n + 1], ptIntersect))
			{
				auto _val = get_distance_points(ptIntersect, input_line_points[i]);
				if (_val < _minLD)
				{
					_minLD = _val;
					ptIntersectL = ptIntersect;
					bFindL = true;
					id_s_l = n;
				}
			}

			if (get_intersect_point(ptCenterL, ptCenterR, line_points_l[n], line_points_l[n + 1], ptIntersect))
			{
			}
		}
		for (int n = 0; n < line_points_r.size() - 1; n++)
		{
			float2 ptIntersect = make_float2(0, 0);
			if (get_intersect_point(ptCenterR, ptCenterL, line_points_r[n], line_points_r[n + 1], ptIntersect))
			{
				auto _val = get_distance_points(ptIntersect, input_line_points[i]);
				if (_val < _minRD)
				{
					_minRD = _val;
					ptIntersectR = ptIntersect;
					bFindR = true;
					id_s_r = n;
				}
			}
		}

		if (!bFindR && !bFindL)
		{
			for (int n = 0; n < line_points_l.size() - 1; n++)
			{
				float2 ptIntersect = make_float2(0, 0);
				if (get_intersect_point(ptCenterR, input_line_points[i], line_points_l[n], line_points_l[n + 1], ptIntersect))
				{
					auto _val = get_distance_points(ptIntersect, input_line_points[i]);
					if (_val < _minLD)
					{
						_minLD = _val;
						ptIntersectL = ptIntersect;
						bFindL = true;
					}
				}
			}
			for (int n = 0; n < line_points_r.size() - 1; n++)
			{
				float2 ptIntersect = make_float2(0, 0);
				if (get_intersect_point(ptCenterL, input_line_points[i], line_points_r[n], line_points_r[n + 1], ptIntersect))
				{
					auto _val = get_distance_points(ptIntersect, input_line_points[i]);
					if (_val < _minRD)
					{
						_minRD = _val;
						ptIntersectR = ptIntersect;
						bFindR = true;
					}
				}
			}
		}
		else if (!bFindR)
			ptIntersectR = get_distance_end(line_points_r, input_line_points[i]).pos;
		else if (!bFindL)
			ptIntersectL = get_distance_end(line_points_l, input_line_points[i]).pos;

		float lfDL = get_distance_points(ptIntersectL, input_line_points[i]);
		float lfDR = get_distance_points(ptIntersectR, input_line_points[i]);

		if (lfDL * 1.3 < lfDR || !bFindR)
		{
			auto _radian = atan2(ptIntersectL.y - input_line_points[i].y, ptIntersectL.x - input_line_points[i].x) + (M_PI * 180 / 180);
			auto ptR = input_line_points[i];
			ptR.x += lfDL;
			ptIntersectR = get_rotation_point(_radian, ptR, input_line_points[i]);
			lfDR = get_distance_points(ptR, input_line_points[i]);
		}
		else if (lfDL > lfDR * 1.3 || !bFindL)
		{
			auto _radian = atan2(ptIntersectR.y - input_line_points[i].y, ptIntersectR.x - input_line_points[i].x) + (M_PI * 180 / 180);
			auto ptL = input_line_points[i];
			ptL.x += lfDR;
			ptIntersectL = get_rotation_point(_radian, ptL, input_line_points[i]);
			lfDL = get_distance_points(ptL, input_line_points[i]);
		}
		new_line_points_l.push_back(ptIntersectL);
		new_line_points_r.push_back(ptIntersectR);
		new_line_points_c.push_back(input_line_points[i]);
		output_distance_points.push_back((lfDL + lfDR));
	}
	float _min = FLT_MAX;
	for (int i = 0; i < output_distance_points.size(); i++)
	{
		auto _val = output_distance_points[i];
		if (_val < _min)
		{
			_min = _val;
			nminIndex = i;
		}
	}
	if (nminIndex == 0 || nminIndex == output_distance_points.size() - 1)
		nminIndex = output_distance_points.size() / 2;

	if (equal)
	{
		set_lines2D(result_info::line_position::LEFT_LINE, new_line_points_l, equal);
		set_lines2D(result_info::line_position::RIGHT_LINE, new_line_points_r, equal);
		set_lines2D(result_info::line_position::CENTER_LINE, new_line_points_c, equal);
		auto segmentation_line2D_instance_ = get_segmentation_line2D_instance();;
		segmentation_line2D_instance_.set_equal_interval_dis(output_distance_points);
		set_segmentation_line2D_instance(segmentation_line2D_instance_);
	}
}

vector<float> Angio_Algorithm::get_minimum_radius()
{
	vector<float> vec_percents;
	if (get_points_instance().branch_point.size() == 0)
		return vec_percents;
	vector<float2>line_points_c = get_lines2D(result_info::line_position::CENTER_LINE);
	auto vessnum = line_points_c.size();

	float D = 0;
	int _nStart = 0;
	float lfD = 0;
	for (int i = 0; i < vessnum - 1; i++)
		lfD += get_distance_points(line_points_c[i], line_points_c[i + 1]);


	auto points_instance = get_points_instance();
	for (int i = 0; i < points_instance.branch_point.size(); i++) {
		auto pair = points_instance.branch_point[i];
		int id = get_distance_end(line_points_c, pair.second).id;
		points_instance.branch_point[i] = std::make_pair(id, pair.second);
	}

	std::sort(points_instance.branch_point.begin(), points_instance.branch_point.end(),
		[](const std::pair<int, float2>& a, const std::pair<int, float2>& b) {
			return std::tie(a.first) < std::tie(b.first);
		});

	int N = 12;
	for (auto& pair : points_instance.branch_point)
	{
		int nStart = pair.first - N;
		int nEnd = pair.first + N;
		if (nStart - 1 < 0) nStart = 0;
		if (nEnd > vessnum) nEnd = vessnum - 1;

		std::vector<float2> line_c; // 매칭포인트 내부

		for (int j = nStart - 1; j <= nEnd + 1; j++)
		{
			if (j == -1)
				continue;
			else if (j == vessnum - 1)
				break;
			line_c.push_back(line_points_c[j]);
		}
		int id_branch_min_point = 0;
		set_verticality_Distance(line_c, nStart, nEnd, id_branch_min_point, false);

		int nMinid = id_branch_min_point + nStart;
		if (id_branch_min_point == 0 || id_branch_min_point == line_c.size() - 1)
			nMinid = pair.first;


		pair.first = nMinid;
		float D = 0;
		for (int j = _nStart; j < nMinid; j++)
			D += get_distance_points(line_points_c[j], line_points_c[j + 1]);
		D = D / (float)lfD * 100;
		if (!isnan(D))
			vec_percents.push_back(D);
		_nStart = nMinid;
	}

	D = 0;
	for (int i = _nStart; i < vessnum - 1; i++)
		D += get_distance_points(line_points_c[i], line_points_c[i + 1]);
	D = D / (float)lfD * 100;
	if (!isnan(D))
		vec_percents.push_back(D);

	set_points_instance(points_instance);
	return vec_percents;
}


vector<float2> Angio_Algorithm::get_equilateral_line(vector<float2> input_line_points, int percent, int nIndex)
{
	vector<float2> new_line_point;

	float lfD = 0;
	int nS, nE;
	int vessnum = input_line_points.size();

	if (nIndex == get_points_instance().branch_point.size())
	{
		nS = (get_points_instance().branch_point.empty()) ? 0 : get_points_instance().branch_point.back().first;
		nS = (get_points_instance().branch_point.empty()) ? 0 : get_points_instance().branch_point[nIndex - 1].first;

		for (int i = nS; i < vessnum - 1; i++)
			lfD += get_distance_points(input_line_points[i], input_line_points[i + 1]);
		equal_spacing(input_line_points, new_line_point, percent, nS, vessnum - 1);
		new_line_point[new_line_point.size() - 1] = input_line_points[vessnum - 1];
	}
	else
	{
		nS = (nIndex == 0) ? 0 : get_points_instance().branch_point[nIndex - 1].first;
		nE = get_points_instance().branch_point[nIndex].first;
		for (int i = nS; i < vessnum - 1; i++)
			lfD += get_distance_points(input_line_points[i], input_line_points[i + 1]);
		equal_spacing(input_line_points, new_line_point, percent, nS, nE);
	}
	return new_line_point;
}



bool Angio_Algorithm::line_undo()
{
	result_info::segmentation_line2D_instance _segmentation_line2D_instance = angio_result_.get_segmentation_line2D_instance();
	auto lines = _segmentation_line2D_instance.get_lines();
	if (history_lines_.undo(lines))
	{
		result_info::segmentation_line2D_instance _segmentation_line2D_instance = angio_result_.get_segmentation_line2D_instance();
		_segmentation_line2D_instance.set_lines(lines);
		angio_result_.set_segmentation_line2D_instance(_segmentation_line2D_instance);
		return true;
	}
	return false;
}

vector<float2> Angio_Algorithm::modify_line(const bool& release_button, const int& select_line_id, const float2& pick_point, const float2& mouse_point, int move_range)
{
	result_info::segmentation_line2D_instance _segmentation_line2D_instance = angio_result_.get_segmentation_line2D_instance();
	auto lines = _segmentation_line2D_instance.get_lines();
	if(lines.size() == 0)
		return vector<float2>();
	const int w = get_width();
	const int h = get_height();

	vector<float2> select_line, no_select_line;
	if (select_line_id == int(result_info::line_position::LEFT_LINE))
	{
		select_line = lines[int(result_info::line_position::LEFT_LINE)];
		no_select_line = lines[int(result_info::line_position::RIGHT_LINE)];
	}
	else
	{
		select_line = lines[int(result_info::line_position::RIGHT_LINE)];
		no_select_line = lines[int(result_info::line_position::LEFT_LINE)];
	}

	distance_and_id _dm = get_distance_end(select_line, pick_point);
	int id_pick = _dm.id;
	int vessnum = select_line.size();
	if (id_pick == -1 || select_line.size() == 0)
		return vector<float2>();

	int id_end = vessnum - 1;
	int id_start = 0;
	bool both_ends = (id_pick < 5 || 5 > id_end - id_pick) ? true : false; // 양 끝단인지 중간인지 (true 양끝 / false 중간)
	bool ends = id_pick > id_end / 2; // isEnd = 1 시작점 / 끝단 isEnd = 0 
	vector<float2> point_parabola, line_parabola, modify_line;
	int id_index = move_range;
	int s_id = 0, e_id = 0;

	if (both_ends)
	{
		int _id = (ends == true ? id_end : id_start);
		if (move_range == 0)
		{
			move_range = qMax(abs(mouse_point.x - select_line[_id].x), abs(mouse_point.y - select_line[_id].y));
			move_range = adjust_data(move_range, 3, 20);

			float dis = get_distance_points(select_line[move_range], select_line[move_range + 1]);
			float dis2 = get_distance_points(select_line[move_range + 1], mouse_point);
			move_range = int(dis + dis2);
			id_index = (ends == false ? move_range : id_end - move_range);  //마우스가 움직일 만큼 변경될 범위 설정
			id_index = adjust_data(id_index, 0, vessnum - 2);
		}

		if (ends)
		{
			point_parabola.push_back(select_line[id_index]);
			point_parabola.push_back(select_line[id_index + 1]);
			point_parabola.push_back(mouse_point);
		}
		else
		{
			point_parabola.push_back(mouse_point);
			point_parabola.push_back(select_line[id_index]);
			point_parabola.push_back(select_line[id_index + 1]);
		}
		add_parabola_data(point_parabola, line_parabola, 0, 2, 0, w);

		if(line_parabola.size() == 0)
			return vector<float2>();

		if (ends)
		{
			for (int i = 0; i < id_index; i++)
				modify_line.push_back(select_line[i]);

			for (int i = 0; i < line_parabola.size(); i++)
				modify_line.push_back(line_parabola[i]);

			modify_line[modify_line.size() - 1] = mouse_point;
		}
		else
		{
			for (int i = 0; i < line_parabola.size(); i++)
				modify_line.push_back(line_parabola[i]);

			for (int i = id_index; i < id_end; i++)
				modify_line.push_back(select_line[i]);
			modify_line[0] = mouse_point;
		}

		s_id = (ends == false ? id_start : get_distance_end(modify_line, select_line[id_index]).id);
		e_id = (ends == false ? get_distance_end(modify_line, select_line[id_index]).id : id_end);
		s_id = adjust_data(s_id, 0, e_id);
		e_id = adjust_data(e_id, 0, modify_line.size());
	}
	else
	{
		bool inside = true;
		float2 range_point = make_float2(select_line[id_pick].x - mouse_point.x, select_line[id_pick].y - mouse_point.y);
		if (move_range == 0)
			move_range = get_distance_points(select_line[id_pick], mouse_point);//qMax(abs(range_point.x) , abs(range_point.y));
		if (validate(move_range, 0, 2))
		{
			std::cerr << "Error: (modify_line) move_range is within invalid range [0, 2]" << std::endl;
			return vector<float2>();
		}
		s_id = id_pick - move_range;
		e_id = id_pick + move_range;
		if (s_id < 0)
			s_id = 0;
		if (e_id > id_end)
			e_id = id_end;
		point_parabola.push_back(select_line[s_id]);
		point_parabola.push_back(select_line[s_id + 1]);
		if (abs(select_line[s_id].x - select_line[e_id].x) > abs(select_line[s_id].y - select_line[e_id].y))
		{
			point_parabola.push_back(make_float2(select_line[id_pick].x, select_line[id_pick].y - range_point.y));
			inside = range_point.y < 0 ? true : false;
		}
		else
		{
			point_parabola.push_back(make_float2(select_line[id_pick].x - range_point.x, select_line[id_pick].y));
			inside = range_point.x < 0 ? true : false;
		}
		point_parabola.push_back(select_line[e_id - 1]);
		point_parabola.push_back(select_line[e_id]);

		for (int i = 0; i <= id_end; i++)
			line_parabola.push_back(select_line[i]);

		set_parabola_data(point_parabola, line_parabola, select_line[id_pick], modify_line, s_id, e_id, inside, 0, w);
		if (modify_line.size() == 0)
			return vector<float2>();
	}

	

	if (release_button)
	{
		if (both_ends)
		{
			vector<float2> output_modifyline;
			int move_id_range = e_id - s_id;
			for (int i = s_id; i < move_id_range; i++)
				output_modifyline.push_back(modify_line[i]);
			return output_modifyline;
		}
		else
		{
			return modify_line;
		}
	}
	else
	{
		if (both_ends)
		{
			if (ends)
			{
				if (e_id == 0) e_id = 1;
				if (s_id == e_id)
					s_id = s_id - 1;
			}
			else
			{
				if (e_id > select_line.size() - 1)
					e_id = select_line.size() - 1;

				if (s_id == e_id)
					s_id = s_id - 1;
			}
			float dis_ = 0;
			for (int n = s_id; n < e_id; n++)
			{
				if (modify_line[n].x > 0 && modify_line[n].y > 0)
					dis_ += get_distance_points(modify_line[n], modify_line[n + 1]);
			}

			int LineIndex = int(dis_);
			if (LineIndex == 1)
				LineIndex = 2;

			vector<float2> vecNewLinePoints;
			vector<float2> vecNewLine2Points;

			{
				//선택한 라인 좌표값 균등화 
				vector<float2> new_data;
				equal_spacing(modify_line, new_data, LineIndex, 0, e_id);
				new_data[new_data.size() - 1] = modify_line[e_id];
				//좌표 재배치
				for (int i = 0; i < s_id; i++)
					vecNewLinePoints.push_back(modify_line[i]);
				for (int i = 0; i < new_data.size(); i++)
					vecNewLinePoints.push_back(new_data[i]);
				for (int i = e_id; i < modify_line.size(); i++)
					vecNewLinePoints.push_back(modify_line[i]);
			}

			if (select_line_id == int(result_info::line_position::LEFT_LINE))
				_segmentation_line2D_instance.set_line(result_info::line_position::LEFT_LINE, vecNewLinePoints);
			else
				_segmentation_line2D_instance.set_line(result_info::line_position::RIGHT_LINE, vecNewLinePoints);

			angio_result_.set_segmentation_line2D_instance(_segmentation_line2D_instance);
			history_lines_.save_state(_segmentation_line2D_instance.get_lines());
			return vecNewLinePoints;
		}
		else
		{
			vector<float2> new_line = select_line;
			for (int i = 0; i < modify_line.size(); i++)
				new_line[i + s_id] = (modify_line[i]);

			if (select_line_id == int(result_info::line_position::LEFT_LINE))
				_segmentation_line2D_instance.set_line(result_info::line_position::LEFT_LINE, new_line);
			else
				_segmentation_line2D_instance.set_line(result_info::line_position::RIGHT_LINE, new_line);
			angio_result_.set_segmentation_line2D_instance(_segmentation_line2D_instance);
			history_lines_.save_state(_segmentation_line2D_instance.get_lines());
			return new_line;
		}
	}
}

void Angio_Algorithm::make_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end)
{
	int N = input_points.size();
	float2 pick_start = input_points[0];
	float2 pick_end = input_points[N - 1];
	auto dir = fabs(input_points[0].x - input_points[N - 1].x) > fabs((input_points[0].y - input_points[N - 1].y));
	auto direction = (dir != true) ? direction::x_direction : direction::y_direction;//calculaterrror_parabola(input_points);
	vector< float2> points;
	float radian = 0;

	if (direction == direction::x_direction)
	{
		radian = (input_points[0].y > input_points[N - 1].y) ? atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) : atan2(pick_end.y - pick_start.y, pick_end.x - pick_start.x);
		radian = (M_PI * 90 / 180) - radian;
	}
	else
	{
		radian = (input_points[0].x < input_points[N - 1].x) ? atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) : atan2(pick_end.y - pick_start.y, pick_end.x - pick_start.x);
		radian = (M_PI * 180 / 180) - radian;
	}

	for (int i = 0; i < input_points.size(); i++)
	{
		float2 rotpoint = input_points[i];
		rotpoint = get_rotation_point(radian, input_points[i], pick_start);
		points.push_back(rotpoint);
	}

	float total_distance = 0;
	for (int i = id_start; i < id_end; i++)
		total_distance += get_distance_points(input_points[i], input_points[i + 1]);

	equal_spacing(input_points, output_parabola_line, total_distance, id_start, id_end);
	//new_data[spaced_data.size() - 1] = points[id_End];
}

void Angio_Algorithm::set_parabola_data(const vector<float2>& input_points, vector<float2>& input_line, const float2 pick_point, vector<float2>& output_parabola_line, int id_Start, int id_End, bool inside_range, int min, int max)
{
	int N = input_points.size();
	float2 pick_start = input_points[0];
	float2 pick_end = input_points[N - 1];
	auto dir = fabs(input_points[0].x - input_points[N - 1].x) > fabs((input_points[0].y - input_points[N - 1].y));
	auto direction = (dir != true) ? direction::x_direction : direction::y_direction;//calculaterrror_parabola(input_points);
	vector< float2> points;
	float radian = 0;

	if (direction == direction::x_direction)
	{
		radian = (input_points[0].y > input_points[N - 1].y) ? atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) : atan2(pick_end.y - pick_start.y, pick_end.x - pick_start.x);
		radian = (M_PI * 90 / 180) - radian;
	}
	else
	{
		radian = (input_points[0].x < input_points[N - 1].x) ? atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) : atan2(pick_end.y - pick_start.y, pick_end.x - pick_start.x);
		radian = (M_PI * 180 / 180) - radian;
	}
	for (int i = 0; i < input_points.size(); i++)
	{
		float2 rotpoint = input_points[i];
		if (i != 2)
			rotpoint = get_rotation_point(radian, input_points[i], pick_point);
		points.push_back(rotpoint);
	}

	vector<float2> rot_line = input_line;
	for (int i = id_Start; i <= id_End; i++)
		rot_line[i] = get_rotation_point(radian, input_line[i], pick_point);

	get_parabola(points, rot_line, rot_line.size(), direction, id_Start, id_End, points.size(), inside_range, points.size() == 5);

	std::vector<float2> tt;
	for (int i = id_Start; i <= id_End; i++)
		tt.push_back(rot_line[i]);

	int n = 0;
	for (int i = id_Start; i <= id_End; i++)
	{
		float2 pt = rot_line[i];
		pt = get_rotation_point(-radian, rot_line[i], pick_point);
		output_parabola_line.push_back(pt);
		tt[n++] = pt;
	}

	if (!(min == 0 && max == 0))  //설정한경우
	{
		vector< float2> org = output_parabola_line;
		output_parabola_line.clear();

		for (int i = 0; i < org.size(); i++)
		{
			if (validate(org[i].x, float(min), float(max)) && validate(org[i].y, float(min), float(max)))
				output_parabola_line.push_back(org[i]);
		}
	}
}

Angio_Algorithm::distance_and_id Angio_Algorithm::get_distance_end(const vector<float2>&sources, const float2 & end)
{
	float minDist = FLT_MAX;
	int minID = 0;
	float2 pos;
	for (int i = 0; i < sources.size(); i++)
	{
		float dist = (sources[i].x - end.x) * (sources[i].x - end.x) + (sources[i].y - end.y) * (sources[i].y - end.y);
		if (dist < minDist) {
			minDist = dist;
			minID = i;
			pos = sources[i];
		}
	}
	return distance_and_id(sqrtf(minDist), minID, pos);
}



void Angio_Algorithm::add_parabola_data(const vector<float2>& input_points, vector<float2>& output_parabola_line, int id_start, int id_end, int min, int max)
{
	int N = input_points.size();
	float2 pick_start = input_points[0];
	float2 pick_end = input_points[N - 1];
	auto direction = fabs(input_points[0].x - input_points[N - 1].x) > fabs((input_points[0].y - input_points[N - 1].y));
	vector< float2> points;
	float radian = 0;
	float dis_range = 0;
	if (!direction)
		radian = atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) + (M_PI * 90 / 180);
	else
		radian = atan2(pick_start.y - pick_end.y, pick_start.x - pick_end.x) + (M_PI * 180 / 180);

	for (int i = 0; i < input_points.size(); i++)
	{
		float2 rotpoint = get_rotation_point(-radian, input_points[i], pick_start);
		points.push_back(rotpoint);
	}

	for (int i = id_start; i < id_end; i++)
		dis_range += get_distance_points(points[i], points[i + 1]);
	if (dis_range == 0)
		dis_range = 1;
	vector<float2> new_data;
	equal_spacing(points, new_data, round(dis_range), id_start, id_end);
	if (new_data.size() == 0)
		return;
	new_data[new_data.size() - 1] = points[id_end];

	get_parabola(points, new_data, dis_range, (direction == 0) ? direction::x_direction : direction::y_direction, 0, dis_range, points.size());

	for (int i = 0; i < dis_range; i++)
	{
		float2 pt = get_rotation_point(radian, new_data.data()[i], pick_start);
		output_parabola_line.push_back(pt);
	}

	if (!(min == 0 && max == 0))  //설정한경우
	{
		vector< float2> org = output_parabola_line;
		output_parabola_line.clear();

		for (int i = 0; i < dis_range; i++)
		{
			if (validate(org[i].x, float(min), float(max)) && validate(org[i].y, float(min), float(max)))
				output_parabola_line.push_back(org[i]);
		}
	}
}


void Angio_Algorithm::equal_spacing(const vector<float2> input_data, vector<float2>& output_data, int n, int id_start, int id_end)
{
	double total_len = 0;
	std::vector<float2> dv{};
	std::vector<float2> _Data{};
	std::vector<float> dvl{};

	for (int i = id_start; i < id_end; i++)
	{
		float2 data_v;
		data_v.x = input_data[i + 1].x - input_data[i].x;
		data_v.y = input_data[i + 1].y - input_data[i].y;
		double data_l = sqrt((pow(data_v.x, 2) + pow(data_v.y, 2)));
		total_len += data_l;
		dv.push_back(data_v);
		dvl.push_back(data_l);
	}

	double v = 0;
	double es = total_len / (n);
	for (int i = 0; i < n; i++)
	{
		double es_cum = i * es;
		int j = 0;
		double dvl_cum = 0;
		double delta = 0;
		while (1)
		{
			dvl_cum += dvl[j];
			j += 1;
			delta = dvl_cum - es_cum;
			if (delta >= 0)
				break;
		}
		output_data.push_back(make_float2(input_data[j + id_start].x - delta / dvl[j - 1] * dv[j - 1].x, input_data[j + id_start].y - delta / dvl[j - 1] * dv[j - 1].y));
	}
}


void Angio_Algorithm::get_parabola(vector<float2>& input_points, vector<float2>& out_parapoint, int point_cnt, direction dir, int id_start, int id_end, int N, bool inside_range, bool bCompare)
{
	double** matrixX = new double* [N];
	double** matrixY = new double* [N];
	double** invmatrix = new double* [N];
	double** matrixXY = new double* [N];
	double** matrixA = new double* [N];
	for (int i = 0; i < N; i++)
	{
		matrixX[i] = new double[N];
		matrixY[i] = new double[N];
		invmatrix[i] = new double[N];
		matrixXY[i] = new double[N];
		matrixA[i] = new double[N];
		memset(matrixX[i], 0, sizeof(double) * N);
		memset(matrixY[i], 0, sizeof(double) * N);
		memset(invmatrix[i], 0, sizeof(double) * N);
		memset(matrixXY[i], 0, sizeof(double) * N);
		memset(matrixA[i], 0, sizeof(double) * N);
	}

	vector<float> x, y;

	if (dir == direction::x_direction) // x = y^2 +y +b;
	{
		for (int i = 0; i < N; i++)
		{
			matrixX[i][0] = ((input_points[i].x));
			for (int j = 0; j < N; j++)
				matrixY[j][i] = pow((input_points[j].y), (N - 1) - i);
		}
		inv_matrix(N, matrixY, invmatrix);

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				matrixXY[i][0] += invmatrix[i][j] * matrixX[j][0];
		}
		if (matrixXY[0][0] == 0.0)	matrixXY[0][0] = 0.00000001;

		for (int i = id_start; i <= id_end; i++)
		{
			if (i < 0)
				continue;
			if (i >= (point_cnt - 1))
				break;
			double lfX = 0;
			for (int n = 0; n < N; n++)
			{
				if (((N - 1) - n) == 0)
					lfX += (matrixXY[n][0]);
				else
					lfX += matrixXY[n][0] * pow(out_parapoint[i].y, (N - 1) - n);
			}
			if (bCompare)
			{
				if (inside_range) //기존 좌표 값 보다 작으면 기존값
				{
					if (lfX < out_parapoint[i].x)
						lfX = out_parapoint[i].x;
					else
						int sss = 0;
				}
				else
				{
					if (lfX > out_parapoint[i].x)
						lfX = out_parapoint[i].x;
				}
			}
			out_parapoint[i].x = lfX;
		}
	}
	else
	{
		for (int i = 0; i < N; i++)
		{
			matrixY[i][0] = (input_points[i].y);
			for (int j = 0; j < N; j++)
				matrixX[j][i] = pow((input_points[j].x), (N - 1) - i);
		}
		inv_matrix(N, matrixX, invmatrix);

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				matrixXY[i][0] += invmatrix[i][j] * matrixY[j][0];
		}

		if (matrixXY[0][0] == 0.0) matrixXY[0][0] = 0.00000001;

		for (int i = id_start; i <= id_end; i++)
		{
			if (i < 0)
				continue;
			if (i >= (point_cnt - 1))
				break;

			double lfY = 0;
			for (int n = 0; n < N; n++)
			{
				if (((N - 1) - n) == 0)
					lfY += (matrixXY[n][0]);
				else
					lfY += matrixXY[n][0] * pow(out_parapoint[i].x, (N - 1) - n);
			}
			if (bCompare)
			{
				if (inside_range)
				{
					if (lfY < out_parapoint[i].y)
						lfY = out_parapoint[i].y;
				}
				else
				{
					if (lfY > out_parapoint[i].y)
						lfY = out_parapoint[i].y;
				}
			}
			out_parapoint[i].y = lfY;
		}
	}
	for (int i = 0; i < N; i++)
	{
		delete[] matrixX[i];
		delete[] matrixY[i];
		delete[] invmatrix[i];
		delete[] matrixXY[i];
		delete[] matrixA[i];
	}
	delete[] matrixX;
	delete[] matrixY;
	delete[] invmatrix;
	delete[] matrixXY;
	delete[] matrixA;
}


void Angio_Algorithm::inv_matrix(int n, double** A, double** B)
{
	double m;
	register int i, j, k;
	double* a = new double[n * n];
	double* b = new double[n * n];

	int num = 0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			a[num] = A[i][j];
			b[num] = 0;
			num++;
		}
	}

	for (j = 0; j < n; j++)

	{
		for (i = 0; i < n; i++)
		{
			b[i * n + j] = (i == j) ? 1. : 0.;
		}
	}
	for (i = 0; i < n; i++)
	{
		if (a[i * n + i] == 0.)
		{
			if (i == n - 1)
			{
				delete[] a;
				delete[] b;
				return;
			}
			for (k = 1; i + k < n; k++)
			{
				if (a[i * n + i + k] != 0.)
					break;
			}
			if (i + k >= n)
			{
				delete[] a;
				delete[] b;
				return;
			}
			for (j = 0; j < n; j++)
			{
				m = a[i * n + j];
				a[i * n + j] = a[(i + k) * n + j];
				a[(i + k) * n + j] = m;
				m = b[i * n + j];
				b[i * n + j] = b[(i + k) * n + j];
				b[(i + k) * n + j] = m;
			}
		}
		m = a[i * n + i];
		for (j = 0; j < n; j++)
		{
			a[i * n + j] /= m;
			b[i * n + j] /= m;
		}
		for (j = 0; j < n; j++)
		{
			if (i == j)
				continue;

			m = a[j * n + i];
			for (k = 0; k < n; k++)
			{
				a[j * n + k] -= a[i * n + k] * m;
				b[j * n + k] -= b[i * n + k] * m;
			}
		}
	}
	num = 0;
	for (int i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			B[i][j] = b[num];
			num++;
		}
	}
	delete[] a;
	delete[] b;
}




void Angio_Algorithm::set_end_points(model_info::find_endpoint_type type, std::string fileName)
{
	auto line = get_lines2D(result_info::line_position::CENTER_LINE, true); //흠
	//std::vector<endPoint_result_Instance> end_points;
	
	auto instance = get_segmentation_Instance();

	vector<result_info::endPoint_result_Instance> end_points;
	if (type == model_info::find_endpoint_type::AI_lebeling_data)
	{
		instance.end_points.clear();
		vector<vector<float2>> points;
		for (int i = 0; i < get_segmentation_Instance().get_labeling_points().size(); i++)
			points.push_back(get_segmentation_Instance().get_labeling_point(i).second);

		end_points = cvAlgorithm_.EndPoints(points, line, get_width(), get_height());
	}
	else if(type == model_info::find_endpoint_type::image_open)
	{
		instance.end_points.clear();
		end_points = cvAlgorithm_.image_EndPoints(line, fileName);
	}
	else
	{
		end_points = instance.end_points;
	}

	auto file_path = program_path_ + ("\\data\\EndPointIDs.dat");
	FILE* f = fopen(file_path.c_str(), "wt");
	fprintf(f, "%d\n", end_points.size());
	for (int i = 0; i < end_points.size(); i++)
		fprintf(f, "%d,%d\n", end_points[i].frame_id, end_points[i].end_center_line_id);// 이미지 frame, 끝점과 가까운 센터라인 인덱스
	fclose(f);
}

void Angio_Algorithm::save_dcm_file(int view_id, dcmHelper* dcm_helper)
{
	std::string path = program_path_ + "\\data\\dcm_attr_"+ std::to_string(view_id) + ".txt";
	FILE* f = fopen(path.c_str(), "wt");
	int w = get_width();
	if (f)
	{
		auto pixelsize = (dcm_helper->getPixelSpacing());// *m_lfWidth* sqrt(2);
		fprintf(f, "IntensifierSize:%.1f\n", (pixelsize[0] * w * sqrt(2)));
		fprintf(f, "DistanceSourceToDetector:%d\n", dcm_helper->getDistanceSourceToDetector());
		fprintf(f, "DistanceSourceToPatient:\n", dcm_helper->getDistanceSourceToPatient());
		fprintf(f, "PositinerPrimaryAngle:%.1f\n", dcm_helper->getPositionerPrimaryAngle());
		fprintf(f, "PositinerSecondaryAngle:%.1f\n", dcm_helper->getPositionerSecondaryAngle());
		fclose(f);
	}
}

void Angio_Algorithm::read_data_in(int view_id, std::string strExtension)
{
	std::string strName;
	if (view_id == 0)
		strName = "_0.dat";
	else
		strName = "_1.dat";

	auto insert_line = [&](vector<vector<float2>> line, bool equal)
	{
		result_info::segmentation_line2D_instance _segmentation_line2D_instance = angio_result_.get_segmentation_line2D_instance();
		_segmentation_line2D_instance.clear(equal);
		for (int i = 0; i < line.size(); i++)
			_segmentation_line2D_instance.set_line((result_info::line_position)i, line[i], equal);
		angio_result_.set_segmentation_line2D_instance(_segmentation_line2D_instance);
		if (!equal)
		{
			history_lines_.clear();
			history_lines_.save_state(line);
		}
	};

	{
		//센터라인
		std::string path = strExtension + "\\data\\line" + strName;

		FILE* fp_in1 = fopen(path.c_str(), "r");
		if (fp_in1)
		{
			int nc = 0;
			fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.
			if (nc != 0)
			{
				vector<float2> l_line, r_line, c_line;
				l_line.resize(nc);
				r_line.resize(nc);
				c_line.resize(nc);

				for (int i = 0; i < nc; i++)
				{
					float2 l, r, c;
					//fscanf(fp_in1, "%f,%f,%f,%f,%f,%f\n", &l_line[i].x, &l_line[i].y, &r_line[i].x, &r_line[i].y, c_line[i].x, &c_line[i].y);
					fscanf(fp_in1, "%f,%f,%f,%f,%f,%f\n", &l.x, &l.y, &r.x, &r.y, &c.x, &c.y);
					l_line[i] = l;
					r_line[i] = r;
					c_line[i] = c;
				}
				fclose(fp_in1);
				vector<vector<float2>> lines;
				lines.push_back(l_line);
				lines.push_back(r_line);
				lines.push_back(c_line);
				insert_line(lines, false);
			}
		}
	}

	{
		//균등 데이터
		std::string path = strExtension + "\\data\\equalline" + strName;

		FILE* fp_in1 = fopen(path.c_str(), "r");
		if (fp_in1)
		{
			int nc = 0;
			fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.
			if (nc != 0)
			{
				vector<float2> l_line, r_line, c_line;
				l_line.resize(nc);
				r_line.resize(nc);
				c_line.resize(nc);

				for (int i = 0; i < nc; i++)
				{
					float2 l, r, c;
					fscanf(fp_in1, "%f,%f,%f,%f,%f,%f\n", &l.x, &l.y, &r.x, &r.y, &c.x, &c.y);
					l_line[i] = l;
					r_line[i] = r;
					c_line[i] = c;
				}
				fclose(fp_in1);
				vector<vector<float2>> lines;
				lines.push_back(l_line);
				lines.push_back(r_line);
				lines.push_back(c_line);
				insert_line(lines, true);
			}
		}
	}

	{
		if (view_id == 0)
			strName = "0.dat";
		else
			strName = "1.dat";
		std::string path = strExtension + "\\data\\data" + strName;
		FILE* fp_in1 = fopen(path.c_str(), "r");
		if (fp_in1)
		{
			vector<float> point_dis;

			int nc = 100;
			//fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.
			if (nc != 0)
			{
				vector<float2> l_line, r_line, c_line;
				l_line.resize(nc);
				r_line.resize(nc);
				c_line.resize(nc);
				for (int i = 0; i < nc; i++)
				{
					float2 l, r, c;
					float d;
					fscanf(fp_in1, "%f,%f,%f,%f,%f,%f,%f\n", &l.x, &l.y, &r.x, &r.y, &c.x, &c.y, &d);
					point_dis.push_back(d);
					l_line[i] = l;
					r_line[i] = r;
					c_line[i] = c;
				}
				
				auto instance = angio_result_.get_segmentation_line2D_instance();
				instance.set_equal_interval_dis(point_dis);
				angio_result_.set_segmentation_line2D_instance(instance);
			}
			fclose(fp_in1);
		}
	}

	{
		//점도...
		std::string path = strExtension + "\\data\\match_point.txt";

		FILE* fp_in1 = fopen(path.c_str(), "r");
		if (fp_in1)
		{
			char label[10];
			vector<float2> points;
			int s_id = view_id * 4;
			int e_id = 4 + s_id;
			for (int i = 0; i < e_id; i++)
			{
				int2 coord;
				fscanf(fp_in1, "%s %d, %d\n", label, &coord.x, &coord.y);
				if (view_id == 0)
					points.push_back(make_float2(coord.x, coord.y));
				else
				{
					if (i >= s_id)
						points.push_back(make_float2(coord.x, coord.y));
				}
			}

			auto points_instance = angio_result_.get_points_instance();
			points_instance.clear_points();
			points_instance.start_point = points[0];
			points_instance.branch_point.push_back(make_pair(0, points[1]));
			points_instance.branch_point.push_back(make_pair(1, points[2]));
			points_instance.end_point = points[3];
			angio_result_.set_points_instance(points_instance);
			fclose(fp_in1);
		}
	}

	{
		auto filename = (view_id == 0) ? R"(../output/lines1.npy)" : R"(../output/lines2.npy)";
		if (std::filesystem::exists(filename))
		{
			result_info::segmentation_instance segmentation_instance_ = angio_result_.get_segmentation_instance();
			auto id = angio_result_.get_segmentation_instance().optimal_image_id;
			segmentation_instance_.clear();
			bool state = get_segmentation_file_niigz(filename, segmentation_instance_, view_id);
			//get_file_n
		}
	}
}

void Angio_Algorithm::save_result_out(int view_id, std::string file_name)
{
	{// 100개 균등화
		if (angio_result_.get_segmentation_line2D_instance().get_lines(true).size() == 0)
			return;

		auto _center = get_lines2D(result_info::line_position::CENTER_LINE, true);
		auto _left = get_lines2D(result_info::line_position::LEFT_LINE, true);
		auto _right = get_lines2D(result_info::line_position::RIGHT_LINE, true);
		auto path = program_path_+ "\\result\\ " + file_name + ".dat";
		
		FILE* f = fopen(path.c_str(), "wt");

		auto point_dis = angio_result_.get_segmentation_line2D_instance().get_equal_interval_dis();
		int cnt = point_dis.size();
		if (f)
		{
			for (auto i = 0; i < cnt; i++)
			{
				float D = point_dis[i];
				fprintf(f, "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", _left[i].x, _left[i].y, _right[i].x, _right[i].y, _center[i].x, _center[i].y, D);
			}
			fclose(f);
		}
		path = "\\Result\\ " + file_name;

		cvAlgorithm_.save_image(get_float_buffer(model_info::image_float_type::LEBELING_IMAGE), path, get_width(), get_height());
	}
}

void Angio_Algorithm::save_data_out(int view_id)
{
	std::string strName;
	if (view_id == 0)
		strName = "_0";
	else
		strName = "_1";

	{//라인
		std::string path = program_path_ + "\\data\\line" + strName +".dat";

		auto _center = get_lines2D(result_info::line_position::CENTER_LINE);
		auto _left = get_lines2D(result_info::line_position::LEFT_LINE);
		auto _right = get_lines2D(result_info::line_position::RIGHT_LINE);
		int cnt = _center.size();
		FILE* f = fopen(path.c_str(), "wt");
		auto point_dis = angio_result_.get_segmentation_line2D_instance().get_equal_interval_dis();
		fprintf(f, "%d\n", cnt);
		for (auto i = 0; i < cnt; i++)
			fprintf(f, "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", _left[i].x, _left[i].y, _right[i].x, _right[i].y, _center[i].x, _center[i].y);
		fclose(f);
	}

	{// 100개 균등화

		auto _center = get_lines2D(result_info::line_position::CENTER_LINE, true);
		auto _left = get_lines2D(result_info::line_position::LEFT_LINE, true);
		auto _right = get_lines2D(result_info::line_position::RIGHT_LINE, true);

		std::string path = program_path_ + "\\data\\equalline" + strName + ".dat";
		int cnt = _center.size();
		FILE* f = fopen(path.c_str(), "wt");

		fprintf(f, "%d\n", cnt);
		for (auto i = 0; i < cnt; i++)
			fprintf(f, "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", _left[i].x, _left[i].y, _right[i].x, _right[i].y, _center[i].x, _center[i].y);
		fclose(f);
	}


	{
		if (view_id == 0)
			strName = "0";
		else
			strName = "1";
		auto _center = get_lines2D(result_info::line_position::CENTER_LINE, true);
		auto _left = get_lines2D(result_info::line_position::LEFT_LINE, true);
		auto _right = get_lines2D(result_info::line_position::RIGHT_LINE, true);

		std::string path = program_path_ + "\\data\\data" + strName + ".dat";

		auto point_dis = angio_result_.get_segmentation_line2D_instance().get_equal_interval_dis();
		int cnt = point_dis.size();

		FILE* f = fopen(path.c_str(), "wt");
		if (f)
		{
			for (auto i = 0; i < cnt; i++)
			{
				float D = point_dis[i];
				fprintf(f, "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", _left[i].x, _left[i].y, _right[i].x, _right[i].y, _center[i].x, _center[i].y, D);
			}
			fclose(f);
		}
	}
}



void Angio_Algorithm::set_center_points_3D(float3 center_point)
{
	auto Get3DCoord = [&](float3 pos)
	{
		float3 pos_coord;
		pos_coord.x = pos.x / 0.05 + center_point.x;
		pos_coord.y = pos.y / 0.05 + center_point.y;
		pos_coord.z = -pos.z / 0.05 + center_point.z;
		return pos_coord;
	};

	auto instance_3D = get_endPoint3D_result_Instance();
	instance_3D.center_line_points.clear();

	auto centerlineFilePath = program_path_ + ("\\output\\centerline_c3r22_P_SW1.dat");
	int nc = 0;
	FILE* fp_in1 = fopen(centerlineFilePath.c_str(), "r");
	if (!fp_in1)
		return;
	fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.
	double* xc = (double*)malloc(sizeof(double) * (nc + 1));
	double* yc = (double*)malloc(sizeof(double) * (nc + 1));
	double* zc = (double*)malloc(sizeof(double) * (nc + 1));

	for (int i = 0; i < nc; i++)
	{
		float3 pos = make_float3(0, 0, 0);
		fscanf(fp_in1, "%lf %lf %lf\n", &xc[i], &yc[i], &zc[i]);
		pos = make_float3(xc[i], yc[i], zc[i]);
		xc[i] = (pos.x - center_point.x) * 0.05f;
		yc[i] = (pos.y - center_point.y) * 0.05f;
		zc[i] = -(pos.z - center_point.z) * 0.05f;
		pos = make_float3(xc[i], yc[i], zc[i]);
		instance_3D.center_line_points.push_back(pos);
	}


	FILE* f = fopen((program_path_ + ("\\data\\EndPointIDs.dat")).c_str(), "r");
	vector<pair<int, int>> vecEndPointIds;
	if (f)
	{
		int n = 0;
		fscanf(f, "%d\n", &n);
		for (int i = 0; i < n; i++)
		{
			int id = 0, imgIndex = 0;
			fscanf(f, "%d,%d\n", &imgIndex, &id);
			vecEndPointIds.push_back(make_pair(imgIndex, id));
		}
	}
	else
		return;

	if (vecEndPointIds.size() == 0)
		return;

	// 두 3D 점 사이의 거리 계산 함수
	auto GetDistance = [&](float3 p1, float3 p2)
	{
		float dx = p2.x - p1.x;
		float dy = p2.y - p1.y;
		float dz = p2.z - p1.z;
		return std::sqrt(dx * dx + dy * dy + dz * dz);
	};

	auto CalculateAndStore3DPointData = [&](int idx, int frame, int nStartId, int nEndId) //idx 이미지 frame 번호 
	{
		float _dis = 0;
		auto idMax = qMax(nStartId, nEndId);
		auto idMin = qMin(nStartId, nEndId);

		for (int j = idMin; j < idMax; j++)
			_dis += GetDistance(Get3DCoord(instance_3D.center_line_points[j]), Get3DCoord(instance_3D.center_line_points[j + 1]));

		if (nEndId - nStartId < 0)
			_dis *= -1;
		_dis = _dis * (frame * 15);
		result_info::end_point_info info(_dis, idx, nEndId, instance_3D.center_line_points[nEndId]);
		instance_3D.sort_id_point.push_back(info);
		instance_3D.sort_center_id_point.push_back(info);
		_dis = 0;
	};
	
	int frame = 1;

	if (vecEndPointIds[0].second != 0)
	{
		CalculateAndStore3DPointData(vecEndPointIds[0].first, frame, 0, vecEndPointIds[0].second);

		for (int i = 1; i < vecEndPointIds.size(); i++)
		{
			frame = vecEndPointIds[i].first - vecEndPointIds[i - 1].first;
			CalculateAndStore3DPointData(vecEndPointIds[i].first, frame, vecEndPointIds[i - 1].second, vecEndPointIds[i].second);
		}
	}
	else
	{
		for (int i = 1; i < vecEndPointIds.size(); i++)
		{
			frame = vecEndPointIds[i].first - vecEndPointIds[i - 1].first;
			CalculateAndStore3DPointData(vecEndPointIds[i].first, frame, vecEndPointIds[i - 1].second, vecEndPointIds[i].second);
		}
	}

	sort(instance_3D.sort_center_id_point.begin(), instance_3D.sort_center_id_point.end(),
		[](const result_info::end_point_info& a, const result_info::end_point_info& b) {
			return a.center_line_id < b.center_line_id;
		});
	sort(instance_3D.sort_id_point.begin(), instance_3D.sort_id_point.end(),
		[](const result_info::end_point_info& a, const result_info::end_point_info& b) {
			return a.frame_id < b.frame_id;
		});

	{
		auto file_path = program_path_ + "\\output\\distance.dat";
		FILE* f1 = fopen(file_path.c_str(), "w");
		fprintf(f1, "%f\n", 0.0);
		for (int i = 0; i < instance_3D.sort_center_id_point.size(); i++)
		{
			float dis_sum = 0;
			for (int j = 0; j < instance_3D.sort_id_point[i].center_line_id; j++)
				dis_sum += GetDistance(Get3DCoord(instance_3D.center_line_points[j]), Get3DCoord(instance_3D.center_line_points[j + 1]));
			fprintf(f1, "%f\n", dis_sum);
			instance_3D.frame_end_point.push_back(instance_3D.sort_id_point[i].pos3D);
		}
		fclose(f1);
	}
	set_endPoint3D_result_Instance(instance_3D);
}
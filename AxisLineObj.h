#pragma once
#include "Object.h"
#include <QTimer>



class AxisLineObj :
	public Object
{
public:

	struct VertexType : public Object::VertexType
	{
		float r, g, b, w;
	};

	struct PosInfo
	{
		float x, y, z;
	};


	using Object::Object;

	void timer();

	~AxisLineObj();

	void initializeGL() override;
	void Render(const QMatrix4x4& world, const QMatrix4x4& view, const QMatrix4x4& proj);
	//void Render(const QMatrix4x4& world, const QMatrix4x4& view);
	//void Bind();
	//void Release();


	void Release() override;
	void ReleaseGLFunctions() override;
	void ReleaseBuffers() override;//void ShutDown() override;
	//void PushXY(const float2*);

	void SetWidth(unsigned val);
	void SetHeight(unsigned val);
	void SetAxisColor(int val);

	void storeData(void* data, const int& countx, const int& county = 0, const int& countz = 0) override;
	void* loadData() override;
	void PushVertex(const float2* _ary, const int& length, const float& r, const float& g, const float& b, const float& w);

	void moveData(void* data, int nCount, bool bChange = false) override;
	dm GetDistance2End(float2 end) override;
	float2* GetData();

	void DataOut(const int& id = 0) override;

	std::unique_ptr<float2[]> testDataOut(const float* ps) const;

	void SetOutLine(float3 Max, float3 Min);
	void SetCenterLine(std::vector<float3> Line);

	void SetEndPoints(std::vector<float3> Line);

	void SetCenterPoint(float3 pt);

	void SetPickObj(float3 pos);
	void SetScale(float val) { m_maxPlaneValue = val; };

	void SetPickData(float3 pos);
	void SetPoints(std::vector<float2> vecData);
	void SetRot(float3 rot) { m_rot = rot; };

	void SetCrossDisplay(bool bShow);

protected:
	void allocateVertices(void*)  override;

protected:

private:
	void makeObject()  override;

	VertexType* m_pVertices = nullptr;
	VertexType* m_pRaws = nullptr;
	PosInfo* m_pLinepos = nullptr;

	QPointF centeroid = { 0,0 };

	int m_projMatrixLoc = 0;
	int m_viewMatrixLoc = 0;
	int m_modelMatrixLoc = 0;
	int m_normalMatrixLoc = 0;
	int m_AxisColorLoc = 0;
	int m_valueLoc = 0;

	int m_axisColor = 0;

	std::vector<float3> m_pickData;
	static int circleNumber;
	static int arrowNumber;

	float3 m_OutLine_Max;
	float3 m_OutLine_Min;

	float m_maxPlaneValue;
	QTimer* m_ptime;
	float3 m_rot;
	std::vector<float3> m_vecCenterLine;
	std::vector<float3> m_vecEndPoints;
	float3 m_CenterPoints;
	bool m_bCrossShow = false;

	std::vector<float2> m_vecPoints;
};


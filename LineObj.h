#pragma once
#include "Object.h"
class LineObj :
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

    ~LineObj();

	void initializeGL() override;
	//void Bind();
	void Render(const QMatrix4x4& world) override;
	//void Release();

	
	void Release() override;
	void ReleaseGLFunctions() override;
	void ReleaseBuffers() override;//void ShutDown() override;
	//void PushXY(const float2*);

	void SetWidth(unsigned val);
	void SetHeight(unsigned val);

	void storeData(void* data, const int& countx, const int& county = 0, const int& countz = 0) override;
	void* loadData() override;
	void PushVertex(const float2* _ary, const int& length, const float& r, const float& g, const float& b, const float& w);
	void PushVertex(const float3*);
	
	void moveData(void* data ,int nCount,bool bChange =false) override;
	dm GetDistance2End(float2 end) override;
	float2* GetData();

	void DataOut(const int& id =0 ) override;

	std::unique_ptr<float2[]> testDataOut(const float* ps) const;

	void setArrowStart(const float2&);
	void setArrowEnd(const float2&);
	float2* getArrowStart();
	float2* getArrowEnd();

	void setCircleSize(const int nSize);
	int getCircleSize() { return m_nCircleSize; };
	
	void setLineColor(const float3&);

	void setCenteroid(const float&, const float&);
	QPointF getCenteroid() const;
	void SetPoints(std::vector<float2> vecData);
protected:
	void allocateVertices(void*)  override;

protected:

private:
	void makeObject()  override;

	VertexType* m_pVertices = nullptr;
	VertexType* m_pRaws = nullptr;
	PosInfo* m_pLinepos = nullptr;
	std::vector<float2> m_vecPoints;

	QPointF centeroid = { 0,0 };

	static int circleNumber;
	static int arrowNumber;
	static int crossNumber;
	
	std::unique_ptr<float4> _arpos;
	bool m_bSetColor  = false;
	float3 m_LineColor;
	int m_nCircleSize = 5;
public:
	Shape m_type = Shape::Line;
};


#pragma once
//#include<qopenglwidget.h>
//#include<qopenglfunctions.h>
#include<qopenglshaderprogram.h>
#include<qopenglbuffer.h>
#include<qopenglvertexarrayobject.h>
#include<qopenglfunctions.h>
#include<qstringlistmodel.h>

#include<vector_types.h>
#include<fstream>
#include<memory>
#include<algorithm>

#include<chrono>
#include<atomic>


#define SafeReleaseArray(X) {if((X)) delete[] (X);(X)=nullptr;}
#define SafeReleasePointer(X) {if((X)) delete (X);(X)=nullptr;}


QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)


#define PROGRAM_VERTEX_ATTRIBUTE 0
#define PROGRAM_TEXCOORD_ATTRIBUTE 1
#define PROGRAM_COLOR_ATTRIBUTE 2
#define PROGRAM_NORMAL_ATTRIBUTE 3
#define PROGRAM_MISCELLANEOUS_ATTRIBUTE 4


#define  COLOR_YELLOW float3{ 1.0f,1.0f,0.0f }
#define  COLOR_RED float3{ 1.0f,0.0f,0.0f };
#define  COLOR_GREEN float3{ 0.0f,1.0f,0.0f }
#define  COLOR_BLUE float3{ 0.0f,0.0f,1.0f }
#define  COLOR_BLACK float3{ 0.0f,0.0f,0.0f }
#define  COLOR_WHITE float3{ 1.0f,1.0f,1.0f }
#define  COLOR_ORANGE float3{ 1.0f,0.75f,0.0f }
#define  COLOR_PINK float3{ 1.0f,0.65f,1.0f }
#define  COLOR_CYAN float3{ 0.0f,0.85f,1.0f }
#define  COLOR_GRAY float3{ 0.75f,0.75f,0.75f }
#define  COLOR_PURPLE float3{ 0.5f,0.25f,0.85f }
#define  COLOR_GREEN2 float3{ 34/255.0f,177/255.0f,76/255.0f }

class Object: public QObject//: protected QOpenGLFunctions//:public QOpenGLWidget, 
{

private:
	template <typename Clock = std::chrono::high_resolution_clock>
	class Stopwatch
	{
		const typename Clock::time_point start_point;
	public:
		Stopwatch() :
			start_point(Clock::now())
		{}

		template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
		Rep elapsed_time() const
		{
			std::atomic_thread_fence(std::memory_order_relaxed);
			auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
			std::atomic_thread_fence(std::memory_order_relaxed);
			return static_cast<Rep>(counted_time);
		}
	};
	using precise_stopwatch = Stopwatch<>;
	using system_stopwatch = Stopwatch<std::chrono::system_clock>;
	using monotonic_stopwatch = Stopwatch<std::chrono::steady_clock>;

public:
	using Idx = unsigned int;

	struct dm
	{
		float d;
		int id;
		float2 pos;
		dm() {};
		dm(const float& D, const int& ID, const float2& POS) :d(D), id(ID),pos(POS) {};
	};

	enum Shape : int { Line, Arrow, Circle, Points };
	enum GraphicLines : int { Axis_X = 0, Axis_Y, Axis_Z, CenterLine3D, EndPoint3D, StenosisorBranch, EndPoint2D };

	enum GraphicType
	{
		LeftLine = 0,
		RightLine,
		CenterLine,
		Start_Point,
		End_Point,
		MatchingPoint,
		Manual_Point,
		Manual_Line,
		Modify_Line,
		Calibration_StartLine,
		Calibration_EndLine,
		GuidePoint,
		FrameEndPoints,
		default,
		default1,
		default2,
		default3,
		default4
	};

	enum ClearType { Clear_Line, Clear_ManualLine, Clear_CalibrationLine, Clear_EquilateralLine, Clear_All};

	enum GuideLinePosSetting { StartToM1, M1ToM2, M2ToEnd };

protected:
	struct VertexType
	{
		float x,y,z;
		//float4 color;
		//float3 normal;
	};

	//struct ModelType
	//{
	//	float x, y, z;
	//	float nx, ny, nz;
	//};

public:
	Object();
	Object(QOpenGLContext*);
	/*Object(float* position, float* color, float* normal, const int& vertexCount, const int& colorChannel);
	Object(float3* position, float4* color, float3* normal, const int& vertexCount, const int& colorChannel);*/
	Object(const Object&);
	~Object();

	virtual void initializeGL()= 0;

	/*void setData(float*);
	void setData(float4*);*/

	virtual void Render() {};
	virtual void Render(const QMatrix4x4& world) {};
	virtual void Render(QMatrix4x4& world, const QMatrix4x4& view) {};
	virtual void Render(const QMatrix4x4& world, const QMatrix4x4& view, const QMatrix4x4& projection) {};
	virtual void Render(QMatrix4x4& world, const QMatrix4x4& view, const QMatrix4x4& projection) {};
	void ShutDown();

	virtual void Bind();
	virtual void Release() = 0;
	virtual void ReleaseGLFunctions() = 0;
	virtual void ReleaseBuffers() = 0;

	void FileDataOut(int id) { DataOut(id);};

	void setData(void* data, const int& w, const int& h, const int& countx, const int& county = 0, const int& countz = 0) { SetWidth(w); SetHeight(h);  storeData(data, countx, county, countz); initializeGL(); };
	void MoveData(void* date ,int w, int h,int countx, bool bChange =false) { SetWidth(w); SetHeight(h);  moveData(date, countx, bChange);};

	dm Distance2End(float2 end) { return GetDistance2End(end); };

	void setGraphicType(GraphicType type) { m_GraphicType = type; };
	GraphicType getGraphicType() { return this->m_GraphicType; };

	void* getData() { return loadData(); }
	void setContext(QOpenGLContext* context);

	virtual bool Parsing(const QString&,const bool _ = false) { return true; };
	virtual const QList<QString>* getInformation() const { return nullptr; };


	virtual void moveScene(const int&) {};

	virtual const int getSceneWidth() const { return 0; };
	virtual const int getSceneHeight() const { return 0; };
	virtual const int getSceneDepth() const { return 0; };
	const int getVertexCount() const;
	const int getIndexCount() const;
	const int getWidth() const;
	const int getHeight() const;

	int getElipseTime()
	{
		return stopwatch.elapsed_time<unsigned int, std::chrono::microseconds>();
	}
	//virtual const QMatrix4x4 Model( QMatrix4x4& m = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)) = 0;


	//const void setModelMatrix(const QMatrix4x4&);
	//const void setScreenRatio(const float&);
	//
	//float getScreenRatio();
	//QMatrix4x4 getModelMatrix();



	//const int getColorChannel() const;

	//const void* getConstData() const;
	/*VertexType* getData();
	unsigned long* getIndices();*/

	//float4* getData();
	void setVertexCount(unsigned);
	void setIndexCount(const Idx&);
	

protected:

	virtual void allocateVertices(void*) = 0;

	void initializeShader(QOpenGLShader::ShaderTypeBit type, const char* src);

	void allocateIndices(Idx*);
	void copy(const Object&);

	virtual void makeObject() = 0;
private:
	virtual void storeData(void*, const int&, const int&, const int&) = 0;
	virtual void moveData(void*, int, bool) = 0;
	virtual dm GetDistance2End(float2) =0;
	virtual void* loadData() = 0;
	virtual void DataOut(const int&) = 0;
	virtual void SetWidth(unsigned)=0;
	virtual void SetHeight(unsigned)=0;

	//void allocate();

	//void setColorChannel(unsigned);

protected:

	QOpenGLBuffer m_vbo;
	QOpenGLVertexArrayObject m_vao;
	QOpenGLShaderProgram* m_program=nullptr;
	QOpenGLContext* m_Context;
	/*QOpenGLFunctions *m_pFunctions;*/


	Idx* m_pIndices = nullptr;
	Idx* m_pRawIndices = nullptr;
	int m_vertexCount, m_indexCount;
	float m_lfWidth = 0;
	float m_lfHeight = 0;

	GraphicType m_GraphicType = GraphicType::CenterLine;

	//float m_screenRatio = 1.0f;
	//QMatrix4x4 m_model = QMatrix4x4();
	//float4* m_Zoom;

private:
	//bool bBorder;
	//int m_colorChannel;
	//VertexType* m_pVertices;
private:
	//QVector3D* m_centre;
	//QVector3D* m_scale;
	static precise_stopwatch stopwatch;
};


static void qNormalizeAngle(int& angle)
{
	while (angle < 0)
		angle += 360 * 16;
	while (angle > 360 * 16)
		angle -= 360 * 16;

	//if (angle < -360)
	//	angle += 720;
	//else if (angle > 360)
	//	angle -= 720;
}


#include "LineObj.h"

int LineObj::circleNumber = 38;
int LineObj::arrowNumber = 9;
int LineObj::crossNumber = 20;


LineObj::~LineObj()
{
	ReleaseBuffers();
	SafeReleaseArray(m_pLinepos);
}

void LineObj::initializeGL()
{
	makeObject();

	//#define PROGRAM_VERTEX_ATTRIBUTE 0
	//#define PROGRAM_TEXCOORD_ATTRIBUTE 1
	//#define PROGRAM_COLOR_ATTRIBUTE 2
	//


	const char* vsrc = R"(
attribute highp vec4 vertex;
attribute lowp vec4 color;
varying lowp vec4 COLOR;
uniform mediump mat4 world;
uniform highp vec2 pos;

void main()
{
	//vec4 outpos = vertex;
	gl_Position = world*vec4(vertex.x + pos.x,vertex.y + pos.y,vertex.z,vertex.w);
	//gl_Position = world*vertex;
	//gl_Position.x +=pos.x;
	//gl_Position.y +=pos.y;
	COLOR = color;
}
)";

	const char* fsrc = R"(
varying lowp vec4 COLOR;
void main()
{
	gl_FragColor = COLOR;
}
)";

	initializeShader(QOpenGLShader::Vertex, vsrc);
	initializeShader(QOpenGLShader::Fragment, fsrc);
	m_program->bind();
	m_program->bindAttributeLocation("vertex", PROGRAM_VERTEX_ATTRIBUTE);
	m_program->bindAttributeLocation("color", PROGRAM_COLOR_ATTRIBUTE);

	m_program->link();

	m_program->release();
}

void LineObj::Render(const QMatrix4x4& world)
{
	Bind();
	m_Context->functions()->glDisable(GL_BLEND);
	m_Context->functions()->glDisable(GL_LINE_SMOOTH);
	m_Context->functions()->glEnable(GL_BLEND);
	m_Context->functions()->glDepthMask(false);
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	int screenWidth = viewport[2];
	int screenHeight = viewport[3];

	// 화면 너비 또는 높이에 기반한 선 두께 비율 설정
	float aspectRatio = (float)screenWidth / (float)screenHeight;
	float lineWidth = 2.0f * aspectRatio; // 원하는 비율로 조정
	if (m_type == Shape::Line)
	{
		m_Context->functions()->glLineWidth(1);
		m_Context->functions()->glEnable(GL_LINE_SMOOTH);
		m_Context->functions()->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		//m_Context->functions()->glBlendFunc(GL_SRC_ALPHA, GL_ZERO);
		m_Context->functions()->glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	}
	else if (m_type == Shape::Points)
	{
		//glPointSize(3.0f);
		m_Context->functions()->glEnable(GL_POINT_SMOOTH);
		m_Context->functions()->glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	}
	else if (m_type != Shape::Line)
	{
		m_Context->functions()->glEnable(GL_MULTISAMPLE);
		m_Context->functions()->glEnable(GL_POLYGON_SMOOTH);
		m_Context->functions()->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		m_Context->functions()->glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
		//m_Context->functions()->glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	}
	else if (m_vertexCount == 1)
	{

		m_Context->functions()->glEnable(GL_POINT_SMOOTH);
		m_Context->functions()->glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	}


	if (m_type == Shape::Circle)
		m_program->setUniformValue("pos", centeroid);

	m_program->setUniformValue("world", world);


	m_program->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
	m_program->enableAttributeArray(PROGRAM_COLOR_ATTRIBUTE);

	m_program->setAttributeBuffer(PROGRAM_VERTEX_ATTRIBUTE, GL_FLOAT, 0, 3, sizeof(float) * 7);
	m_program->setAttributeBuffer(PROGRAM_COLOR_ATTRIBUTE, GL_FLOAT, sizeof(float) * 3, 4, sizeof(float) * 7);
	
	if (m_type == Shape::Line)
	{
		m_Context->functions()->glDrawArrays(GL_LINE_STRIP, 0, m_vertexCount);

		glColor3f(1.0, 0.0, 0.0);
			//glLineWidth(2.0)  # 선의 두께

			//# Polyline 그리기
	//glBegin(GL_LINE_STRIP);
	//	for point in points :
	//	glVertex2f(point[0], point[1]);
	//	glEnd()
	//
		//m_Context->functions()->glDisable(GL_LINE_SMOOTH);
	}
	else if (m_type == Shape::Points)
	{
		if (m_GraphicType == GraphicType::Manual_Line)
		{
			glPointSize(1);

			m_Context->functions()->glDrawArrays(GL_POINTS, 0, m_vertexCount);
			m_Context->functions()->glDisable(GL_LINE_SMOOTH);
		}
		else
		{
			glPointSize(5);
			m_Context->functions()->glDrawArrays(GL_POINTS, 0, m_vertexCount);
			m_Context->functions()->glDisable(GL_LINE_SMOOTH);
				
		}
	}
	else if (m_type != Shape::Line)
	{
		m_Context->functions()->glDrawArrays(GL_TRIANGLE_FAN, 0, m_vertexCount);
		m_Context->functions()->glDisable(GL_LINE_SMOOTH);
		m_Context->functions()->glDisable(GL_MULTISAMPLE);
	}
	else if (m_vertexCount == 1)
		m_Context->functions()->glDrawArrays(GL_POINTS, 0, m_vertexCount);

	m_Context->functions()->glDepthMask(true);
	//m_Context->functions()->glDisable(GL_BLEND);
	//m_Context->functions()->glDisable(GL_LINE_SMOOTH);
	//m_Context->functions()->glDisable(GL_DEPTH_TEST);
	/*m_vbo.release();
	m_program->release();*/
	Release();
}

//void LineObj::ReleaseGLFunctions()
//{
//	Object::ReleaseGLFunctions();
//	SafeReleaseArray(m_pVertices);
//}

void LineObj::Release()
{
	m_program->release();
	m_vbo.release();
	m_vao.release();
}

void LineObj::ReleaseGLFunctions()
{
	m_vbo.destroy();
	m_vao.destroy();
	SafeReleasePointer(m_program);
}

void LineObj::ReleaseBuffers()
{
	/*for (auto& v : m_vpVertices)
		SafeReleaseArray(v);*/
	SafeReleaseArray(m_pRaws);
	SafeReleaseArray(m_pVertices);
}

//void LineObj::ShutDown()
//{
//	Release();
//	ReleaseGLFunctions();
//	ReleaseBuffers();
//}

void LineObj::SetWidth(unsigned val)
{
	m_lfWidth = val;
}

void LineObj::SetHeight(unsigned val)
{
	m_lfHeight = val;
}
void LineObj::SetPoints(std::vector<float2> vecData)
{
	m_vecPoints = vecData;
}


void LineObj::storeData(void* data, const int& count, const int& objType, const int& ShapeType)
{
	ReleaseGLFunctions();
	ReleaseBuffers();
	if (count == 0)
		return;
	int nCount = count;
	m_vertexCount = nCount;
	this->m_type = Shape(ShapeType);
	
	QSurfaceFormat format = m_Context->format();
	format.setSamples(4);
	m_Context->setFormat(format);
	float2* _ary = reinterpret_cast<float2*>(data);
	std::unique_ptr<float2[]> tempAry;
	const float _rd = 3.14159265358979323846 / 180.0f;

	if (ShapeType == Shape::Circle)
	{
		float radius = 0.5f;
		if (objType == default || Manual_Point == objType)
			radius = m_nCircleSize;
		tempAry = std::make_unique<float2[]>(circleNumber);
		const float2 base = { 0,0 };
		tempAry[0] = { base.x,base.y };
		const float tick = 360.0f / (circleNumber - 2);
		int a;
#pragma omp parallel for
		for (a = 1; a <= (circleNumber - 2); a++)
		{
			const float _a = (a * tick) * _rd;
			tempAry[a] = { base.x + radius * cosf(_a),base.y + radius * sinf(_a) };
		}
		const float _a = (tick)*_rd;
		tempAry[circleNumber - 1] = { base.x + radius * cosf(_a),base.y + radius * sinf(_a) };
		_ary = tempAry.get();
		m_vertexCount = circleNumber;
		nCount = circleNumber;
	}
	else if (ShapeType == Shape::Arrow)
	{
		/* data = x7
				x6	.	d3
				x5	d2			x4
		x7<-d0->x0<------d1---->x3
				x1				x2
				x8
		*/
		const float d[] = { 6.0f ,12.0f,0.5f,3.0f };
		const float a[] = { 1 * _rd,(1 - 90) * _rd,(1 + 90) * _rd };

		tempAry = std::make_unique<float2[]>(arrowNumber);
		tempAry[0] = { _ary->x + cosf(a[0]) * d[0],_ary->y + sinf(a[0]) * d[0] };
		const auto xy = &tempAry.get()[0];
		tempAry[1] = { xy->x + cosf(a[1]) * d[2],xy->y + sinf(a[1]) * d[2] };
		tempAry[2] = { tempAry[1].x + cosf(a[0]) * d[1],tempAry[1].y + sinf(a[0]) * d[1] };
		tempAry[3] = { xy->x + cosf(a[0]) * d[1],xy->y + sinf(a[0]) * d[1] };
		tempAry[4] = { tempAry[3].x + cosf(a[2]) * d[2],tempAry[3].y + sinf(a[2]) * d[2] };
		tempAry[5] = { xy->x + cosf(a[2]) * d[2],xy->y + sinf(a[2]) * d[2] };
		tempAry[6] = { xy->x + cosf(a[2]) * d[3],xy->y + sinf(a[2]) * d[3] };
		tempAry[7] = *_ary;
		tempAry[8] = { xy->x + cosf(a[1]) * d[3],xy->y + sinf(a[1]) * d[3] };
		_ary = tempAry.get();

		_arpos = std::make_unique<float4>();
		_arpos->x = _ary[3].x;
		_arpos->y = _ary[3].y;
		_arpos->z = _ary[7].x;
		_arpos->w = _ary[7].y;
		nCount = arrowNumber;
		m_vertexCount = arrowNumber;
	}
	m_pLinepos = new PosInfo[nCount];
	m_pRaws = new VertexType[nCount];
	float _alpha = 1.0f;
	float3 c = COLOR_WHITE;
	if (objType == GraphicType::LeftLine || objType == GraphicType::RightLine || objType == Calibration_StartLine)
		c = COLOR_YELLOW;
	else if (objType == GraphicType::CenterLine || objType == Calibration_EndLine)
		c = COLOR_GREEN;
	else if (objType == GraphicType::MatchingPoint)
		c = COLOR_BLUE;
	else if (objType == GraphicType::GuidePoint || objType == GraphicType::FrameEndPoints)
		c = COLOR_PURPLE; 
	else if (objType == GraphicType::default1)
		c = COLOR_ORANGE;
	else if (objType == GraphicType::default2)
		c = COLOR_PINK;
	else if (objType == GraphicType::default3)
		c = COLOR_CYAN;
	else if (objType == default4|| objType == GraphicType::Manual_Line)
		c = COLOR_GREEN2;
	else if (objType == GraphicType::End_Point || objType == GraphicType::Start_Point || objType == Modify_Line)
		c = COLOR_RED;

	if (objType == GraphicType::Manual_Point)
	{
		c = COLOR_RED;
		_alpha = 1.0;
	}

//#pragma omp parallel for^
	for (auto i = 0; i < m_vertexCount; i++)
	{
		m_pRaws[i].x = _ary[i].x / m_lfWidth - 0.5f;
		m_pRaws[i].y = (_ary[i].y / m_lfHeight) - 0.5f;
		m_pRaws[i].z = 0.0f;
		if (m_bSetColor)
			c = m_LineColor;
		m_pRaws[i].r = c.x;
		m_pRaws[i].g = c.y;
		m_pRaws[i].b = c.z;
		
		m_pRaws[i].w = _alpha;
		m_pLinepos[i].x = _ary[i].x;
		m_pLinepos[i].y = _ary[i].y;
	}

	setGraphicType(GraphicType(objType));
}

void* LineObj::loadData()
{
	return nullptr;
}

void LineObj::PushVertex(const float2* _ary,const int& length,const float& r, const float& g, const float& b, const float& w)
{
	m_pRaws = new VertexType[length];
	setVertexCount(length);
	for (auto i = 0; i < length; i++)
	{
		m_pRaws[i].x = _ary[i].x / 256.0f - 1.0f;
		m_pRaws[i].y = (_ary[i].y / 256.0f) - 1.0f;
		m_pRaws[i].z = 0.0f;
		m_pRaws[i].r = r;
		m_pRaws[i].g = g;
		m_pRaws[i].b = b;
		m_pRaws[i].w = w;
	}
}

//const QMatrix4x4 LineObj::Model(QMatrix4x4& model)
//{
//	if (m_Zoom)
//	{
//		m_Zoom = new float4();
//		m_Zoom->x = 1.0f;
//		m_Zoom->y = 1.0f;
//		m_Zoom->z = 1.0f;
//		m_Zoom->w = 1.0f;
//	}
//	if (m_screenRatio >= 1)
//		model.ortho(-(0.5f * m_screenRatio) / m_Zoom->x, (0.5f * m_screenRatio) / m_Zoom->y, 0.5f / m_Zoom->z, -0.5f / m_Zoom->w, -1.0f, 1.0f);
//	else
//		model.ortho(-0.5f / m_Zoom->x, 0.5f / m_Zoom->y, (0.5f / m_screenRatio) / m_Zoom->z, -(0.5f / m_screenRatio) / m_Zoom->w, -1.0f, 1.0f);
//	return model;
//}


//void LineObj::Bind()
//{
//	Object::Bind();
//	
//	//m_vao.bind();
//}

//void LineObj::Release()
//{
//	m_vbo.release();
//	
//	Object::Release();
//}

//void LineObj::setNumberContainer(const int& val)
//{
//	this->numberLine = val;
//}

void LineObj::moveData(void* data, int nCount, bool bChange)
{
	Release();
	ReleaseGLFunctions();
	float2* _ary = reinterpret_cast<float2*>(data);
	m_vertexCount = nCount;
	m_pRaws = new VertexType[m_vertexCount];

	float3 c{ 0.3,0.3,0.3f };
	float _alpha = 1.0f;
	switch (getGraphicType())
	{
	case GraphicType::LeftLine:
	case GraphicType::RightLine:
		c.x = 1.0f;
		c.y = 227 / 255.0f;
		c.z = 0.0f;
		break;
	case GraphicType::CenterLine:
		c.x = 0.0f;
		c.y = 1.0f;
		c.z = 0.0f;
		break;
	case GraphicType::default:
		_alpha = 0.7f;
		break;
	case GraphicType::MatchingPoint:
		c.x = 1.0f;
		c.y = 1.0f;
		c.z = 1.0f;
		break;
	case GraphicType::End_Point:
	case GraphicType::Start_Point:
	case -1:
		c.x = 1.0f;
		c.y = 0.0f;
		c.z = 0.0f;
		break;
	}
	for (auto i = 0; i < m_vertexCount; i++)
	{
		m_pRaws[i].x = _ary[i].x / m_lfWidth - 0.5f;
		m_pRaws[i].y = (_ary[i].y / m_lfHeight) - 0.5f;
		m_pRaws[i].z = 0.0f;
		if (m_bSetColor)
			c = m_LineColor;
		m_pRaws[i].r = c.x;
		m_pRaws[i].g = c.y;
		m_pRaws[i].b = c.z;
		m_pRaws[i].w = 1.0f;
		if (bChange)
		{
			m_pLinepos[i].x = _ary[i].x;
			m_pLinepos[i].y = _ary[i].y;
		}
	}
	initializeGL();
}

float2* LineObj::GetData()
{
	float2* _buffer = new float2[m_vertexCount];

	QPointF posOffset = QPointF(0, 0);
	if (this->m_type == Shape::Circle)
	{
		posOffset.setX(getCenteroid().x() * m_lfWidth);
		posOffset.setY(getCenteroid().y() * m_lfHeight);

	}

	for (auto i = 0; i < m_vertexCount; i++)
	{
		_buffer[i].x = m_pLinepos[i].x + posOffset.x();
		_buffer[i].y = m_pLinepos[i].y + posOffset.y();
	}
	return (float2*)_buffer;
}

Object::dm LineObj::GetDistance2End(float2 end)
{
	Object::dm _dm;
	float _min = FLT_MAX;
	int _minId = 0, current = 0;
	float2 pos;
	for (auto i = 0; i < m_vertexCount; i++)
	{
		auto f = m_pLinepos[i];
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
	return dm(sqrtf(_min), _minId, pos);
}

void LineObj::DataOut(const int& id)
{
	auto _buffer = std::make_unique<float2[]>(m_vertexCount);

	for (auto i = 0; i < m_vertexCount; i++)
	{
		_buffer.get()[i].x = (m_pRaws[i].x + 0.5f) * m_lfWidth;
		_buffer.get()[i].y = (m_pRaws[i].y + 0.5f) * m_lfHeight;
	}

	FILE* f = fopen(QString("./center%1.dat").arg(id).toStdString().c_str(), "wt");
	for (auto i = 0; i < m_vertexCount; i++)
	{
		fprintf(f, "%.8f,%.8f\n", _buffer.get()[i].x, _buffer.get()[i].y);
	}
	fclose(f);
}

std::unique_ptr<float2[]> LineObj::testDataOut(const float* ps) const
{
	auto _buffer = std::make_unique<float2[]>(m_vertexCount);
	if (ps)
	{
		auto psx = 1.0f / 256.0f * ps[0] / sqrtf(2.0f) / 2.0f;
		auto psy = 1.0f / 256.0f * ps[1] / sqrtf(2.0f) / 2.0f;;
#pragma omp parallel for
		for (auto i = 0; i < m_vertexCount; i++)
		{
			_buffer.get()[i].x = ((m_pRaws[i].x + 0.5f) * m_lfWidth  - (m_lfWidth  / 2.0)) * psx;
			_buffer.get()[i].y = ((m_pRaws[i].y + 0.5f) * m_lfHeight - (m_lfHeight / 2.0)) * psy;
		}
	}
	else
	{
#pragma omp parallel for
		for (auto i = 0; i < m_vertexCount; i++)
		{
			_buffer.get()[i].x = ((m_pRaws[i].x + 0.5f) * m_lfWidth  - (m_lfWidth / 2.0));
			_buffer.get()[i].y = ((m_pRaws[i].y + 0.5f) * m_lfHeight - (m_lfHeight / 2.0));
		}
	}
	return std::move(_buffer);
}

void LineObj::setArrowStart(const float2& v)
{
	_arpos->x = v.x; _arpos->y = v.y;
}

void LineObj::setArrowEnd(const float2& v)
{
	_arpos->z = v.x; _arpos->w = v.y;
}

float2* LineObj::getArrowStart()
{
	if (m_type != Shape::Arrow)
		return nullptr;
	else
	{
		return (float2*)(std::data({ _arpos->x,_arpos->y }));
	}
}

float2* LineObj::getArrowEnd()
{
	if (m_type != Shape::Arrow)
		return nullptr;
	else
	{
		return (float2*)(std::data({ _arpos->z,_arpos->w }));
	}
}

void LineObj::setCircleSize(const int nSize)
{
	m_nCircleSize = nSize;
}

void LineObj::setLineColor(const float3& c)
{
	m_bSetColor = true;
	this->m_LineColor = c;
}

void LineObj::setCenteroid(const float& x, const float& y)
{
	this->centeroid = QPointF(x / m_lfWidth, y / m_lfHeight);
}

QPointF LineObj::getCenteroid() const
{
	return this->centeroid;
}


void LineObj::allocateVertices(void*)
{
}

void LineObj::makeObject()
{
	m_bSetColor = false;
	SafeReleaseArray(m_pVertices);
	if (m_pRaws)
	{
		m_pVertices = new VertexType[m_vertexCount];
		//m_pVertices = m_pRaws;
		memcpy(m_pVertices, m_pRaws, sizeof(VertexType) * m_vertexCount);
		/*m_pVertices = new VertexType[1];

		for (int j = 0; j < 1; j++)
		{
			m_pVertices[j].x = float(j);
			m_pVertices[j].y = float(j);
			m_pVertices[j].z = 0.0f;
			m_pVertices[j].r = 1.0f;
			m_pVertices[j].g = 0.0f;
			m_pVertices[j].b = 0.0f;
			m_pVertices[j].w = 1.0f;

		}*/

	}

	m_vbo.create();
	m_vbo.bind();
	m_vbo.allocate(m_pVertices, m_vertexCount * sizeof(float) * 7);
	m_vbo.release();

	m_vao.create();
	m_vao.bind();

	m_vbo.bind();

	m_vbo.release();
	m_vao.release();
}


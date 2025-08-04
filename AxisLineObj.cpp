#include "AxisLineObj.h"
#include "TriObj.h"
#include <gl/glut.h>

int AxisLineObj::circleNumber = 38;
int AxisLineObj::arrowNumber = 9;


void AxisLineObj::timer()
{
//	glutTimerFunc(30, timer, 0); //다음 타이머 이벤트는 30밀리세컨트 후  호출됨.
}


AxisLineObj::~AxisLineObj()
{
	ReleaseBuffers();
	ReleaseGLFunctions();
}

void AxisLineObj::initializeGL()
{
	makeObject();

	//#define PROGRAM_VERTEX_ATTRIBUTE 0
	//#define PROGRAM_TEXCOORD_ATTRIBUTE 1
	//#define PROGRAM_COLOR_ATTRIBUTE 2
	//

	const char* vsrc =
		R"(
		attribute highp vec3 vertex;
		attribute highp vec3 normal;
		attribute highp float value;
		uniform  int AxisColor;
		
		uniform mat4 world;
		uniform mat4 view;
		
		uniform mat3 normalMatrix;
		uniform int contourType;
		uniform float T;
		uniform float timer;
		
		varying vec3 vtx;
		varying vec3 vtxNormal;
		varying highp vec4 color;
		
		void main()
		{
			vtx = vertex.xyz;
			vtxNormal = normalMatrix * normalize(normal.xyz);
			gl_Position = world *view * vec4(vtx,1);
			color = vec4(1,1,1,0.6);
			if(AxisColor == 0)
				color = vec4(1,0,0,0.6);
			else if(AxisColor == 1)
				color = vec4(1,1,0,0.6);
			else if(AxisColor == 2)
				color = vec4(0,1,0,0.6);
			else if(AxisColor == 3)
				color = vec4(1,1,1,1);
			else if(AxisColor == 4)
				color = vec4(255/255,255 /255,0 /255,1);
			else if(AxisColor == 5)
				color = vec4(1,1,1,0.5);
		}
	)";

	//color = vec4(241 / 255, 95 / 255, 95 / 255, 1.0);
	//vshader->compileSourceCode(vsrc);

	//QOpenGLShader* fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
	const char* fsrc =
		R"(
			varying highp vec3 vtx;
			varying highp vec3 vtxNormal;
			varying highp vec4 color;
			uniform highp vec3 lightPos;
			uniform int AxisColor;
 
			void main()
			{
				gl_FragColor = color;
			}
		)";
	//fshader->compileSourceCode(fsrc);
	initializeShader(QOpenGLShader::Vertex, vsrc);
	initializeShader(QOpenGLShader::Fragment, fsrc);

	m_program->bindAttributeLocation("vertex", 0);
	m_program->bindAttributeLocation("normal", 1);
	m_program->bindAttributeLocation("value", 2);
	m_program->bindAttributeLocation("AxisColor", 3);

	m_program->link();

	m_program->bind();

	//m_vao.create();
	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);
	m_vbo.bind();

	m_Context->functions()->glEnableVertexAttribArray(0);
	m_Context->functions()->glEnableVertexAttribArray(1);
	m_Context->functions()->glEnableVertexAttribArray(2);

	m_modelMatrixLoc = m_program->uniformLocation("world");
	m_viewMatrixLoc = m_program->uniformLocation("view");
	m_normalMatrixLoc = m_program->uniformLocation("normalMatrix");
	m_AxisColorLoc = m_program->uniformLocation("AxisColor");
	m_valueLoc = m_program->uniformLocation("T");

	m_program->bind();

	m_program->release();
	
//	connect(m_ptime, SIGNAL(timeout()), this, SLOT(fun));
//	gluLookAt(0, 0, 0, 0, 0, 2.0f, -1, 0, 0);

}

void AxisLineObj::SetCrossDisplay(bool bShow)
{
	m_bCrossShow = bShow;
}

void AxisLineObj::Render(const QMatrix4x4& world, const QMatrix4x4& view, const QMatrix4x4& proj)
{
	if (!m_program)
		return;
	float sizeW = 0.025 * m_maxPlaneValue;
	float sizeH = 0.045 * m_maxPlaneValue;

	float lenAxis = 0.3 * m_maxPlaneValue;
	float startpos = -0.5 * m_maxPlaneValue;

	Bind();
	m_Context->functions()->glEnable(GL_DEPTH_TEST);

	m_Context->functions()->glEnable(GL_BLEND);
	
	m_Context->functions()->glEnable(GL_LINE_SMOOTH);
	m_Context->functions()->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	m_Context->functions()->glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	m_Context->functions()->glLineWidth(2);
	
	m_program->setUniformValue(m_modelMatrixLoc, proj);
	m_program->setUniformValue(m_viewMatrixLoc, view * world);
	m_program->setUniformValue(m_normalMatrixLoc, world.normalMatrix());
	m_program->setUniformValue(m_valueLoc, 0);
	m_program->setUniformValue(m_AxisColorLoc, m_axisColor);

	m_program->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
	m_program->enableAttributeArray(PROGRAM_COLOR_ATTRIBUTE);
	
	
	m_Context->functions()->glDrawArrays(GL_LINE_STRIP, 0, m_vertexCount);

	m_Context->functions()->glDepthMask(true);
	//m_Context->functions()->glDisable(GL_BLEND);
	//m_Context->functions()->glDisable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	if (true)
	{
		if (m_axisColor == GraphicLines::CenterLine3D)
		{
			//for (int i = 0; i < m_vecCenterLine.size() - 1; i++)
			//{
			//	glBegin(GL_LINE_LOOP);
			//	glVertex3f(m_vecCenterLine[i].x, m_vecCenterLine[i].y, m_vecCenterLine[i].z);
			//	glVertex3f(m_vecCenterLine[i + 1].x, m_vecCenterLine[i + 1].y, m_vecCenterLine[i + 1].z);
			//
			//	//glVertex3f(m_vecCenterLine[i].x, m_vecCenterLine[i].y, m_vecCenterLine[i].z);
			//	//glVertex3f(m_vecCenterLine[i + 1].x, m_vecCenterLine[i + 1].y, m_vecCenterLine[i + 1].z);
			//	glEnd();
			//}

		}
		if (m_axisColor == GraphicLines::EndPoint2D)
		{
			glBegin(GL_LINE_LOOP);
			glVertex3f(0.5, 0.5, 0);
			glVertex3f(0.5,-0.5,0);
			glEnd();

		}
		else if (m_axisColor == GraphicLines::StenosisorBranch)
		{
			if(m_vecCenterLine.size() == 2)
			{
				float radius = 0.15;

				float radian = 3.141592 / 180;

				float maxAnlge = 360;
				
				//glPolygonMode(GL_FRONT, GL_FILL); // 앞 면을 채운다.
				//glPolygonMode(GL_BACK, GL_FILL); // 앞 면을 채운다.
				glPolygonMode(GL_FRONT, GL_LINE); // 뒷면은 선으로 이어진 다각형을 그린다.
				glPolygonMode(GL_BACK, GL_LINE); // 뒷면은 선으로 이어진 다각형을 그린다.

				for (int n = 0; n < m_vecCenterLine.size(); n++)
				{
					glBegin(GL_POLYGON);
					float3 pos = m_vecCenterLine[n];
					for (int i = 0; i < 360; ++i)
						glVertex3f(pos.x + radius * sin(i * (3.14152 / 180)), pos.y + radius * cos(i * (3.14152 / 180)), pos.z);
					glEnd();
				}
			
				//for (int i = 0; i <= maxAnlge; i++)
				//{
				//	glRotatef(i, 0, 1, 0);
				//	glBegin(GL_LINE_STRIP);
				//	glPushMatrix();
				//	for (int j = 0; j <= maxAnlge; j++)
				//	{
				//		radian += j;
				//		glVertex3f(m_vecCenterLine[0].x , m_vecCenterLine[0].y + sin(radian) * radius, m_vecCenterLine[0].z+ cos(radian) * radius);
				//	}
				//	glPopMatrix();
				//	glEnd();
				//} 
				//float xPos = 0;
				//float yPos = 0;
				//float radius = 0.5f;
				//float prevX = m_vecCenterLine[0].x;
				//float prevY = m_vecCenterLine[0].y - radius;
				//
				//for (int i = 0; i <= 360; i++)
				//{
				//	float newX = radius * sin(angle * i);
				//	float newY = -radius * cos(angle * i);
				//
				//	glBegin(GL_TRIANGLES);
				//	glColor3f(0, 0.5f, 0);
				//
				//	glVertex3f(0.0f, 0.0f, 0.0f);
				//	glVertex3f(prevX, prevY, 0.0f);
				//	glVertex3f(newX, newY, 0.0f);
				//
				//
				//
				//	prevX = newX;
				//	prevY = newY;
				//
				//
				//}

				//glPointSize(20);
				//for (int i = 0; i < m_vecCenterLine.size(); i++)
				//{
				//	glBegin(GL_POINTS);
				//	glVertex3f(m_vecCenterLine[i].x, m_vecCenterLine[i].y, m_vecCenterLine[i].z);
				//	glEnd();
				//	
				//}
			}
			//float radius = 1;
			//float radian = 3.141592 / 180;
			//float maxAnlge = 360;
			//for (int i = 0; i <= maxAnlge; i++)
			//{
			//
			//	glRotatef(i, 0, 1, 0);
			//	glBegin(GL_LINE_STRIP);
			//	glPushMatrix(); {    // 원 그리기
			//		for (int i = 0; i <= maxAnlge; i++) 
			//		{
			//			radian += i;
			//			glVertex3f(cos(radian) * radius, sin(radian) * radius, 0);
			//		}
			//
			//	}
			//
			//	glPopMatrix();
			//
			//	glEnd();
			//
			//}
		}
		else if (m_axisColor == GraphicLines::EndPoint3D)
		{
			auto x = 0.07 / m_maxPlaneValue;

			//glPointSize(x);
			if (m_bCrossShow)
			{
				glMatrixMode(GL_MODELVIEW);

				//glEnable(GL_LIGHTING);
				//GLfloat arLight[] = { 1.0, 1.0, 0, 1.0 };
				//glLightModelfv(GL_LIGHT_MODEL_AMBIENT, arLight);
				//glColorMaterial(GL_FRONT, GL_AMBIENT);
				for (int i = 0; i < m_vecEndPoints.size(); i++)
				{
					//	glBegin(GL_POINTS);
					//	glVertex3f(m_vecEndPoints[i].x, m_vecEndPoints[i].y, m_vecEndPoints[i].z);
					//	glEnd();
					auto z = m_vecEndPoints[i].z;

					glBegin(GL_LINE_LOOP);
					glVertex3f(m_vecEndPoints[i].x - sizeW, m_vecEndPoints[i].y, z);
					glVertex3f(m_vecEndPoints[i].x + sizeW, m_vecEndPoints[i].y, z);
					glEnd();

					glBegin(GL_LINE_LOOP);
					glVertex3f(m_vecEndPoints[i].x, m_vecEndPoints[i].y - sizeW, z);
					glVertex3f(m_vecEndPoints[i].x, m_vecEndPoints[i].y + sizeW, z);
					glEnd();

					glBegin(GL_LINE_LOOP);
					glVertex3f(m_vecEndPoints[i].x, m_vecEndPoints[i].y,  z-sizeW);
					glVertex3f(m_vecEndPoints[i].x, m_vecEndPoints[i].y,  z + sizeW);
					glEnd();
				}
			}
		}
		else 
		{
			glPolygonMode(GL_FRONT, GL_FILL); // 앞 면을 채운다.
			glPolygonMode(GL_BACK, GL_FILL); // 앞 면을 채운다.

			glPushMatrix(); //X축 붉은색
			glBegin(GL_LINE_LOOP);

			if (m_axisColor == GraphicLines::Axis_X) // X
			{
				glColor3f(1.0, 1.0, 0.0);
				glVertex3f(lenAxis, 0.0f, 0.0f);
				glVertex3f(0.0f, 0.0f, 0.0f);
				glEnd();

				glBegin(GL_TRIANGLES);
				glVertex3f(lenAxis, 0.0f, 0.0f);
				glVertex3f(lenAxis - sizeH, 0.0 - sizeW, sizeW);
				glVertex3f(lenAxis - sizeH, 0.0 + sizeW, sizeW);

				glVertex3f(lenAxis, 0.0f, 0.0f);
				glVertex3f(lenAxis - sizeH, 0.0 + sizeW, sizeW);
				glVertex3f(lenAxis - sizeH, 0.0 + sizeW, -sizeW);

				glVertex3f(lenAxis, 0.0f, 0.0f);
				glVertex3f(lenAxis - sizeH, 0.0 + sizeW, -sizeW);
				glVertex3f(lenAxis - sizeH, 0.0 - sizeW, -sizeW);

				glVertex3f(lenAxis, 0.0f, 0.0f);
				glVertex3f(lenAxis - sizeH, 0.0 - sizeW, -sizeW);
				glVertex3f(lenAxis - sizeH, 0.0 - sizeW, sizeW);
				glEnd();
			}
			else if (m_axisColor == GraphicLines::Axis_Y) // Y
			{
				glVertex3f(0.0f, lenAxis, 0.f);
				glVertex3f(0.0f, 0.0f, 0.f);
				glEnd();

				glBegin(GL_POLYGON);
				glVertex3f(0.0f, lenAxis, 0.f);
				glVertex3f(0.0 - sizeW, lenAxis - sizeH, sizeW);
				glVertex3f(0.0 + sizeW, lenAxis - sizeH, sizeW);

				glVertex3f(0.0f, lenAxis, 0.f);
				glVertex3f(0.0 + sizeW, lenAxis - sizeH, sizeW);
				glVertex3f(0.0 + sizeW, lenAxis - sizeH, -sizeW);

				glVertex3f(0.0f, lenAxis, 0.f);
				glVertex3f(0.0 + sizeW, lenAxis - sizeH, -sizeW);
				glVertex3f(0.0 - sizeW, lenAxis - sizeH, -sizeW);

				glVertex3f(0.0f, lenAxis, 0.f);
				glVertex3f(0.0 - sizeW, lenAxis - sizeH, -sizeW);
				glVertex3f(0.0 - sizeW, lenAxis - sizeH, sizeW);
				glEnd();
			}
			else if (m_axisColor == GraphicLines::Axis_Z) // Z
			{
				glVertex3f(0.0f, 0.0f, 0.0f);
				glVertex3f(0.0f, 0.0f, -lenAxis);
				glEnd();

				glBegin(GL_POLYGON);
				glVertex3f(0.0f, 0.0f, -lenAxis);
				glVertex3f(sizeW, 0.0 + sizeW, -lenAxis + sizeH);
				glVertex3f(sizeW, 0.0 - sizeW, -lenAxis + sizeH);

				glVertex3f(0.0f, 0.0f, -lenAxis);
				glVertex3f(-sizeW, 0.0 + sizeW, -lenAxis + sizeH);
				glVertex3f(-sizeW, 0.0 - sizeW, -lenAxis + sizeH);

				glVertex3f(0.0f, 0.0f, -lenAxis);
				glVertex3f(sizeW, 0.0 - sizeW, -lenAxis + sizeH);
				glVertex3f(-sizeW, 0.0 - sizeW, -lenAxis + sizeH);

				glVertex3f(0.0f, 0.0f, -lenAxis);
				glVertex3f(-sizeW, 0.0 + sizeW, -lenAxis + sizeH);
				glVertex3f(sizeW, 0.0 + sizeW, -lenAxis + sizeH);
				glEnd();
			}
			else
			{
				glVertex3f(m_OutLine_Min.x, m_OutLine_Min.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Min.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Max.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Max.y, m_OutLine_Min.z);
				glEnd();

				glBegin(GL_LINE_LOOP);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Min.y, m_OutLine_Max.z);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Min.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Min.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Min.y, m_OutLine_Max.z);
				glEnd();

				glBegin(GL_LINE_LOOP);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Max.y, m_OutLine_Max.z);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Max.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Max.y, m_OutLine_Min.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Max.y, m_OutLine_Max.z);
				glEnd();

				glBegin(GL_LINE_LOOP);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Min.y, m_OutLine_Max.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Min.y, m_OutLine_Max.z);
				glVertex3f(m_OutLine_Max.x, m_OutLine_Max.y, m_OutLine_Max.z);
				glVertex3f(m_OutLine_Min.x, m_OutLine_Max.y, m_OutLine_Max.z);
				glEnd();
			}
		}
	}


	glFlush();
	Release();
}

void AxisLineObj::Release()
{
	m_program->release();
	m_vbo.release();
	m_vao.release();
}

void AxisLineObj::ReleaseGLFunctions()
{
	m_vbo.destroy();
	m_vao.destroy();
	SafeReleasePointer(m_program);
}

void AxisLineObj::ReleaseBuffers()
{
	/*for (auto& v : m_vpVertices)
		SafeReleaseArray(v);*/
	SafeReleaseArray(m_pRaws);
	SafeReleaseArray(m_pVertices);
}

void AxisLineObj::SetWidth(unsigned val)
{
	m_lfWidth = val;
}

void AxisLineObj::SetHeight(unsigned val)
{
	m_lfHeight = val;
}

void AxisLineObj::SetAxisColor(int val)
{
	m_axisColor = val;
}


void AxisLineObj::storeData(void* data, const int& countx, const int& county, const int& countz)
{
	ReleaseGLFunctions();
	ReleaseBuffers();
}

void* AxisLineObj::loadData()
{
	return nullptr;
}

void AxisLineObj::PushVertex(const float2* _ary, const int& length, const float& r, const float& g, const float& b, const float& w)
{
} 


void AxisLineObj::moveData(void* data, int nCount, bool bChange)
{
}

float2* AxisLineObj::GetData()
{
	float2* _buffer = new float2[m_vertexCount];
	return (float2*)_buffer;
}

Object::dm AxisLineObj::GetDistance2End(float2 end)
{
	return dm(0, 0, end);
}

void AxisLineObj::DataOut(const int& id)
{
}

std::unique_ptr<float2[]> AxisLineObj::testDataOut(const float* ps) const
{
	auto _buffer = std::make_unique<float2[]>(m_vertexCount);
	return std::move(_buffer);
}

void AxisLineObj::SetOutLine(float3 Max, float3 Min)
{
	m_OutLine_Min = Min;
	m_OutLine_Max = Max;
}

void AxisLineObj::SetCenterLine(std::vector<float3> Line)
{
	m_vecCenterLine.clear();
	for (int i = 0; i<Line.size(); i++)
		m_vecCenterLine.push_back(Line[i]);

}

void AxisLineObj::SetEndPoints(std::vector<float3> Line)
{
	m_vecEndPoints.clear();
	for (int i = 0; i < Line.size(); i++)
		m_vecEndPoints.push_back(Line[i]);
}

void AxisLineObj::SetCenterPoint(float3 pt)
{
	m_CenterPoints = pt;
}

void AxisLineObj::SetPickObj(float3 pos)
{
}

void AxisLineObj::allocateVertices(void*)
{
}

void AxisLineObj::makeObject()
{
	SafeReleaseArray(m_pVertices);
	if (m_pRaws)
	{
		m_pVertices = new VertexType[m_vertexCount];
		memcpy(m_pVertices, m_pRaws, sizeof(VertexType) * m_vertexCount);
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


void AxisLineObj::SetPickData(float3 pos)
{
	for (int i = 0; i < 3; i++)
		m_pickData.push_back(pos);
}

void AxisLineObj::SetPoints(std::vector<float2> vecData)
{
	m_vecPoints = vecData;
}

 

#include "TriObj.h"

std::chrono::steady_clock::time_point TriObj::timer_prev = std::chrono::high_resolution_clock::now();

TriObj::~TriObj()
{
	ReleaseBuffers();
	ReleaseGLFunctions();
	//ShutDown();
	/*SafeReleaseArray(m_pVertices);
	SafeReleaseArray(m_pRawVertices);
	SafeReleaseArray(m_pIndices);
	SafeReleaseArray(m_pRawIndices);*/
}

void TriObj::initializeGL()
{
	/*if (!m_pRawIndices || !m_pRawVertices)
		return;*/

	makeObject();
	const char* vsrc = 
	R"(
		attribute highp vec3 vertex;
		attribute highp vec3 normal;
		attribute highp float value;
		
		uniform mat4 world;
		uniform mat4 view;
		
		uniform mat3 normalMatrix;
		uniform int contourType;
		uniform float T;
		uniform float timer;
		
		varying vec3 vtx;
		varying vec3 vtxNormal;
		varying highp vec4 color;
		
		vec3 float2RGB(in float value)
		{
			float H = mod(value,1.0);
			float R = abs(H * 6.0 - 3.0) - 1.0;
			float G = 2.0 - abs(H * 6.0 - 2.0);
			float B = 2.0 - abs(H * 6.0 - 4.0);  
			
			if (H < 0.5)
			{
				R = 0.705882 + H * 0.0318242;
				G = 0.015686 + H * 1.16986334;
				B = 0.149020 + H * 0.835199;
			}
			else
			{
				R = 0.865003 - H * 1.126726;
				G = 0.865003 - H * 1.13;
				B = 0.865003 - H * 0.0224124;
			}
			return clamp(vec3(R,G,B),0.0,1.0);
		}
		
		void main()
		{
			vtx = vertex.xyz;
			vtxNormal = normalMatrix * normalize(normal.xyz);
			gl_Position = world *view * vec4(vtx,1);
			if(contourType ==0)
			{
				color = vec4(1,1,1,1);
			}
			else 
			{
			
				float temp = clamp(value,0.0,1.0);
			
				if(value<timer)
					color = vec4(1,1,1,1);
				else
					color = vec4(float2RGB(1.0-value),1);
			}
		}
	)";

	//vshader->compileSourceCode(vsrc);

	//QOpenGLShader* fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
	const char* fsrc = 
		R"(
			varying highp vec3 vtx;
			varying highp vec3 vtxNormal;
			varying highp vec4 color;
			uniform highp vec3 lightPos;
			void main()
			{
				highp vec3 L = normalize(lightPos - vtx);
				highp float NL = max(dot(normalize(vtxNormal), L), 0.0);
				highp vec3 _color = color.xyz;
				highp vec3 col = clamp(_color* 0.2 + _color* 0.8 * NL, 0.0, 1.0);
				gl_FragColor = vec4(col,0.7);
			}
		)";
	//fshader->compileSourceCode(fsrc);
	initializeShader(QOpenGLShader::Vertex, vsrc);
	initializeShader(QOpenGLShader::Fragment, fsrc);

	m_program->bindAttributeLocation("vertex", 0);
	m_program->bindAttributeLocation("normal", 1);
	m_program->bindAttributeLocation("value", 2);
	
	m_program->link();

	m_program->bind();

	//m_vao.create();
	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);
	m_vbo.bind();
	m_Context->functions()->glEnableVertexAttribArray(0);
	m_Context->functions()->glEnableVertexAttribArray(1);
	m_Context->functions()->glEnableVertexAttribArray(2);


	m_Context->functions()->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VertexType::getBytes(), nullptr);
	m_Context->functions()->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VertexType::getBytes(), reinterpret_cast<void*>(3 * sizeof(float)));

	//m_Context->functions()->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
	//m_Context->functions()->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

	m_Context->functions()->glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, VertexType::getBytes(), reinterpret_cast<void*>(6 * sizeof(float)));
	//m_Context->functions()->glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), reinterpret_cast<void*>(6 * sizeof(float)));
	m_vbo.release();




	m_modelMatrixLoc = m_program->uniformLocation("world");
	m_viewMatrixLoc = m_program->uniformLocation("view");
	//m_projMatrixLoc = m_program->uniformLocation("projection");
	m_normalMatrixLoc = m_program->uniformLocation("normalMatrix");
	m_contourTypeLoc = m_program->uniformLocation("contourType");
	m_lightPosLoc = m_program->uniformLocation("lightPos");
	m_timerLoc = m_program->uniformLocation("timer");
	m_valueLoc = m_program->uniformLocation("T");
	/*m_program->setUniformValue("world", 0);
	m_program->setUniformValue("view", 0);
	m_program->setUniformValue("projection", 0);
	m_program->setUniformValue("normalMatrix", 0);
	m_program->setUniformValue("contourType", 0);
	m_program->setUniformValue("lightPos", 0);*/

	m_program->setUniformValue(m_lightPosLoc, QVector3D(1, 1, -5));
	m_program->setUniformValue(m_timerLoc, 0);

	m_program->release();

	timer_prev = std::chrono::high_resolution_clock::now();
	timer = 1;
}
//
//void TriObj::Bind()
//{
//	Object::Bind();
//}

void TriObj::Render(QMatrix4x4& model, const QMatrix4x4& view, const QMatrix4x4& projection)
{
	//Bind();
	/*QMatrix4x4 camera = model;
	camera.translate(m_center.x, m_center.y, m_center.z);*/
	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

	m_program->bind();
	//Bind();
	m_Context->functions()->glEnable(GL_DEPTH_TEST);
	m_Context->functions()->glEnable(GL_CULL_FACE);
	m_Context->functions()->glEnable(GL_FRONT_AND_BACK);

//	glMatrixMode(GL_PROJECTION);
//glColor3f(1.f, 0.f, 0.f);
//glBegin(GL_LINE_LOOP);
//glVertex3f(1.0f, 0.0f, 0.f);
//glVertex3f(-1.0f, 0.f, -0.f);
//glEnd();


	if (bFFRShow && this->m_contourType != 0 && (timer > 0))
	{

		auto _timer = std::chrono::high_resolution_clock::now();

		auto dd = _timer - timer_prev;

		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(dd);


		//qDebug() << d.count();



		if (d.count() < 5000)
		{
			//timer = 0;
			//timer_prev = _timer;
			timer =  -(d.count() / 3000.0f);
		}
		//else
		//	timer = 0.0f;

	}

	//m_program->setUniformValue(m_modelMatrixLoc, model);
	m_program->setUniformValue(m_modelMatrixLoc, projection);
	m_program->setUniformValue(m_viewMatrixLoc, view* model);
	//m_program->setUniformValue(m_projMatrixLoc, projection);
	m_program->setUniformValue(m_normalMatrixLoc, model.normalMatrix());
	m_program->setUniformValue(m_contourTypeLoc, bFFRShow? this->m_contourType : 0);
	m_program->setUniformValue(m_valueLoc, this->m_value);
	m_program->setUniformValue(m_timerLoc, this->timer);

//	m_program->setUniformValue(m_modelMatrixLoc, model);

	m_Context->functions()->glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, m_pIndices);

	


	m_Context->functions()->glDisable(GL_CULL_FACE);
	m_Context->functions()->glDisable(GL_DEPTH_TEST);

	m_program->release();
}

//void TriObj::Release()
//{
//	m_program->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
//	m_program->enableAttributeArray(PROGRAM_COLOR_ATTRIBUTE);
//}

//void TriObj::ReleaseGLFunctions()
//{
//	Object::ReleaseGLFunctions();
//	//m_ibo.destroy();
//}

//void TriObj::ShutDown()
//{
//	Release();
//	ReleaseGLFunctions();
//	ReleaseBuffers();
//}

void TriObj::Release()
{
	m_program->release();
	m_vbo.release();
	m_vao.release();
}

void TriObj::ReleaseGLFunctions()
{
	m_vbo.destroy();
	m_vao.destroy();
	SafeReleasePointer(m_program);
}

void TriObj::ReleaseBuffers()
{
	SafeReleaseArray(m_pVertices);
	SafeReleaseArray(m_pRawVertices);
	SafeReleaseArray(m_pIndices);
	SafeReleaseArray(m_pRawIndices);
}

void TriObj::SetWidth(unsigned val)
{
	m_lfWidth = val;
}

void TriObj::SetHeight(unsigned val)
{
	m_lfHeight = val;
}

void TriObj::storeData(void* data, const int& countx, const int& county, const int& countz)
{
	float4* vertices = reinterpret_cast<float4*>(data);

}

void* TriObj::loadData()
{
	return nullptr;
}

void TriObj::moveData(void* data, int nCount, bool bChange)
{
}

Object::dm TriObj::GetDistance2End(float2 end)
{
	return dm();
}

void TriObj::DataOut(const int& id)
{
}

void TriObj::moveScene(const int&)
{

}

bool TriObj::Parsing(const QString& str, const bool isTest)
{

	SetWidth(450);
	SetHeight(450);
	m_commonwidth = 450;
	m_commonheight = 450;
	if (isTest)
	{
		static const int coords[4][3] =
		{ { +1, -1, 0 }, { -1, -1, 0 }, { -1, +1, 0 }, { +1, +1, 0 } };

		//std::unique_ptr<VertexType[]> vertData(new VertexType[4]);
		SafeReleaseArray(m_pRawVertices);
		m_pRawVertices = new VertexType[4];
		m_vertexCount = 4;

		//0:qimage
		for (int j = 0; j < 4; j++)
		{
			m_pRawVertices[j].x = 0.5f * coords[j][0];
			m_pRawVertices[j].y = 0.5f * coords[j][1];
			m_pRawVertices[j].z = 0;
			m_pRawVertices[j].nx = 0.5f * coords[j][0];
			m_pRawVertices[j].ny = 0.5f * coords[j][1];
			m_pRawVertices[j].nz = 0.5f * coords[j][2];
		}
		m_pRawIndices = new Idx[6];
		auto aa = m_pRawIndices;
		*aa++ = 0;
		*aa++ = 1;
		*aa++ = 2;
		*aa++ = 0;
		*aa++ = 2;
		*aa++ = 3;

		m_indexCount = 6;
		makeObject();
		return true;
	}
	else
	{
		//tecplot case
		QFile qf(str);
		if (!qf.open(QIODevice::ReadOnly | QIODevice::Text | QIODevice::ExistingOnly))
			return false;

		QTextStream qts(&qf);
		QString line = qts.readLine();

		QRegExp rx("\\\"(.*)\\\"");
		int pos = 0, _adder = 0, _m_indexCount;
		{
			auto vl = line.split(",");
			for (auto& vll : vl)
			{
				rx.indexIn(vll, 0);
				m_variableName << rx.cap(1);
			}
			//while ((pos = rx.indexIn(line, pos)) != -1)
			//{
			//	m_variableName << rx.cap(1);
			//	pos += rx.matchedLength();
			//}
		}
		line = qts.readLine();
		// VARIABLES = "...", "...", "...", "..." ..
		rx.setPattern("\\\.*ET\\\s*=\\\s*(\\\w*)\s*,\\\s*");
		pos = 0;
		//ZONE F=..., ET=...,      N=        ...  , E=       ... 
		while ((pos = rx.indexIn(line, pos)) != -1)
		{
			if (m_facetEdge == 0)
			{
				if (rx.cap(1).startsWith("TRI", Qt::CaseInsensitive))
					m_facetEdge = 3;
				else if (rx.cap(1).startsWith("QUAD", Qt::CaseInsensitive))
					m_facetEdge = 4;
				//else if(rx.cap(1).startsWith("BRICK", Qt::CaseInsensitive))
				//	m_facetEdge = 6;
				else
					return false;
				_adder = rx.matchedLength();
				rx.setPattern("\\\.*N\\\s*=\\\s*(\\\d*)\\\s*,\\\s*");
			}
			else
			{
				if (getVertexCount() == 0)
				{
					m_vertexCount = rx.cap(1).toInt();
					if (m_vertexCount <= 0)
						return false;

					m_pRawVertices = new VertexType[m_vertexCount];
					_adder = rx.matchedLength();
					rx.setPattern("\\\.*E\\\s*=\\\s*(\\\d*)\\.*");
				}
				else
				{
					_m_indexCount = rx.cap(1).toInt();
					if (_m_indexCount <= 0)
						return false;
					m_indexCount = _m_indexCount * ((m_facetEdge == 4) ? 2 : 1) * 3;
					m_pRawIndices = new Idx[m_indexCount];
					_adder = rx.matchedLength();
					break;
				}
			}

			pos += _adder;
		}

		VertexType _center;
		_center.x = 0; _center.y = 0; _center.z = 0;

		auto _ll = qts.readAll();
		auto all = _ll.splitRef('\n', Qt::SplitBehaviorFlags::KeepEmptyParts);
	
		bool ok;
		float _max = FLT_MIN;
		float _min = FLT_MAX;

//#pragma omp parallel for
		for (int vi = 0; vi < m_vertexCount; vi++)
		{
			auto bll = all[vi].trimmed().split(' ');

			//const Qhar* stream = all[vi].data();
			//char* pc = nullptr;
			//char* cntx = nullptr;
			//
			//strtof()
			int _Ax = 0;
			for (int axis = 0; axis < bll.size(); axis++)
			{
				if (bll[axis].constData()->isSpace())
					continue;
				const float _value = bll[axis].toFloat();
				switch (_Ax)
				{
				case 0:m_pRawVertices[vi].x = _value;
#pragma omp atomic
					_center.x += m_pRawVertices[vi].x; _Ax++;
					break;
				case 1:m_pRawVertices[vi].y = _value;
#pragma omp atomic
					_center.y += m_pRawVertices[vi].y; _Ax++;
					break;
				case 2:m_pRawVertices[vi].z = _value;
#pragma omp atomic
					_center.z += m_pRawVertices[vi].z; _Ax++;
					break;
				case 3:
					m_pRawVertices[vi].v = _value;
					_max = qMax(_value, _max);
					_min = qMin(_value, _min);
					if (m_contourType == 0)
					{
						if (m_pRawVertices[vi].v != 0)
							m_contourType = 1;
					}
					break;
				default:
					break;
				}
			}
		}

		_center.x /= m_vertexCount; _center.y /= m_vertexCount; _center.z /= m_vertexCount;
		m_posCenter.x = _center.x;
		m_posCenter.y = _center.y;
		m_posCenter.z = _center.z;

		//m_centre = new QVector3D(_center.x, _center.y, _center.z);
		//m_scale = new QVector3D(0.1f, 0.1f, 0.1f);


		//(_center.x, _center.y, _center.z);

		{

			//_devX /= _repRatio;
			//_devY /= _repRatio;
			//_devZ /= _repRatio;
			auto nxv = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.v < right.v;
				});

			this->m_minVal = int(10 * nxv.first->v) / 10.0f;
			const float _dno = nxv.second->v - this->m_minVal;
#pragma omp parallel for
			for (auto vi = 0; vi < m_vertexCount; vi++)
			{
				m_pRawVertices[vi].x = (m_pRawVertices[vi].x - _center.x) * 0.05f;
				m_pRawVertices[vi].y = (m_pRawVertices[vi].y - _center.y) * 0.05f;
				m_pRawVertices[vi].z = (m_pRawVertices[vi].z - _center.z) * 0.05f;
				m_pRawVertices[vi].z = -m_pRawVertices[vi].z;
				m_pRawVertices[vi].v = (m_pRawVertices[vi].v - this->m_minVal) / (_dno);

			}

			auto nxx = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.x < right.x;
				});
			auto nxy = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.y < right.y;
				});
			auto nxz = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.z < right.z;
				});



			auto _devX = fmaxf(fabsf(nxx.second->x), fabsf(nxx.first->x));
			auto _devY = fmaxf(fabsf(nxy.second->y), fabsf(nxy.first->y));
			auto _devZ = fmaxf(fabsf(nxz.second->z), fabsf(nxz.first->z));

			this->m_maxPlaneValue = fmaxf(_devX, fmaxf(_devY, _devZ))*1.1;
			//this->m_maxPlaneValue = sqrtf(3 * this->m_maxPlaneValue * this->m_maxPlaneValue);
		}



#pragma omp parallel for
		for (int ii = 0; ii < _m_indexCount; ii++)
		{
			auto bll = all[ii + m_vertexCount].trimmed().split(' ');

			auto _idx = [&](const int& sz)->std::unique_ptr<int[]>
			{
				auto result = std::make_unique<int[]>(sz);

				int j = 0;
				for (int iii = 0; iii < bll.size(); iii++)
				{
					if (bll[iii].constData()->isSpace())
						continue;
					result[j++] = iii;
				}
				return std::move(result);
			};
			//std::unique_ptr<int> _tFace(new int[m_facetEdge]);
			if (m_facetEdge == 3)
			{
				auto ix = _idx(3);
				//if (!bFFRShow)
				//{
					m_pRawIndices[ii * 3 + 0] = bll[ix[0]].toInt()-1;//_tFace.get()[0];
					m_pRawIndices[ii * 3 + 1] = bll[ix[1]].toInt()-1;//_tFace.get()[2];
					m_pRawIndices[ii * 3 + 2] = bll[ix[2]].toInt()-1;//_tFace.get()[1];
				//}
			//	else
			//	{
			//		m_pRawIndices[ii * 3 + 0] = bll[ix[0]].toInt()-1;
			//		m_pRawIndices[ii * 3 + 1] = bll[ix[2]].toInt()-1;
			//		m_pRawIndices[ii * 3 + 2] = bll[ix[1]].toInt()-1;
			//	}
				cummulateNormalVector(m_pRawVertices[m_pRawIndices[ii * 3 + 0]],
					m_pRawVertices[m_pRawIndices[ii * 3 + 1]],
					m_pRawVertices[m_pRawIndices[ii * 3 + 2]]);
			}
			else if (m_facetEdge == 6)
			{
				auto ix = _idx(4);
				m_pRawIndices[ii * 6 + 0] = bll[ix[0]].toInt()-1;
				m_pRawIndices[ii * 6 + 1] = bll[ix[2]].toInt()-1;
				m_pRawIndices[ii * 6 + 2] = bll[ix[1]].toInt()-1;
				m_pRawIndices[ii * 6 + 3] = bll[ix[0]].toInt()-1;
				m_pRawIndices[ii * 6 + 4] = bll[ix[3]].toInt()-1;
				m_pRawIndices[ii * 6 + 5] = bll[ix[2]].toInt()-1;

				cummulateNormalVector(m_pRawVertices[m_pRawIndices[ii * 6 + 0]],
					m_pRawVertices[m_pRawIndices[ii * 6 + 1]],
					m_pRawVertices[m_pRawIndices[ii * 6 + 2]]);

				cummulateNormalVector(m_pRawVertices[m_pRawIndices[ii * 6 + 3]],
					m_pRawVertices[m_pRawIndices[ii * 6 + 4]],
					m_pRawVertices[m_pRawIndices[ii * 6 + 5]]);
			}

		}

		return true;

	}
	return false;
}

void TriObj::SetFFR(const QString& str)
{
	SetWidth(450);
	SetHeight(450);
	m_commonwidth = 450;
	m_commonheight = 450;
	if (true)
	{
		QFile qf(str);
		if (!qf.open(QIODevice::ReadOnly | QIODevice::Text | QIODevice::ExistingOnly))
			return;

		QTextStream qts(&qf);
		QString line = qts.readLine();

		QRegExp rx("\\\"(.*)\\\"");
		int pos = 0, _adder = 0, _m_indexCount;
		{
			auto vl = line.split(",");
			for (auto& vll : vl)
			{
				rx.indexIn(vll, 0);
				m_variableName << rx.cap(1);
			}
			//while ((pos = rx.indexIn(line, pos)) != -1)
			//{
			//	m_variableName << rx.cap(1);
			//	pos += rx.matchedLength();
			//}
		}
		line = qts.readLine();
		// VARIABLES = "...", "...", "...", "..." ..
		rx.setPattern("\\\.*ET\\\s*=\\\s*(\\\w*)\s*,\\\s*");
		pos = 0;
		//ZONE F=..., ET=...,      N=        ...  , E=       ... 
		while ((pos = rx.indexIn(line, pos)) != -1)
		{
			if (m_facetEdge == 0)
			{
				if (rx.cap(1).startsWith("BRICK", Qt::CaseInsensitive))
					m_facetEdge = 6;
				else
					return;
				_adder = rx.matchedLength();
				rx.setPattern("\\\.*N\\\s*=\\\s*(\\\d*)\\\s*,\\\s*");
			}
			else
			{
				if (getVertexCount() == 0)
				{
					m_vertexCount = rx.cap(1).toInt();
					if (m_vertexCount <= 0)
						return;

					m_pRawVertices = new VertexType[m_vertexCount];
					_adder = rx.matchedLength();
					rx.setPattern("\\\.*E\\\s*=\\\s*(\\\d*)\\.*");
				}
				else
				{
					_m_indexCount = rx.cap(1).toInt();
					if (_m_indexCount <= 0)
						return;
					m_indexCount = _m_indexCount * 3;
					m_pRawIndices = new Idx[m_indexCount];
					_adder = rx.matchedLength();
					break;
				}
			}

			pos += _adder;
		}

		VertexType _center;
		_center.x = 0; _center.y = 0; _center.z = 0;

		auto _ll = qts.readAll();
		auto all = _ll.splitRef('\n', Qt::SplitBehaviorFlags::KeepEmptyParts);

		bool ok;
		float _max = FLT_MIN;
		float _min = FLT_MAX;

		#pragma omp parallel for
		for (int vi = 0; vi < m_vertexCount; vi++)
		{
			auto bll = all[vi].trimmed().split(' ');
			int _Ax = 0;
			for (int axis = 0; axis < bll.size(); axis++)
			{
				if (bll[axis].constData()->isSpace())
					continue;
				const float _value = bll[axis].toFloat();
				switch (_Ax)
				{
				case 0:m_pRawVertices[vi].x = _value * 1000;
#pragma omp atomic
					_center.x += m_pRawVertices[vi].x; _Ax++;
					break;
				case 1:m_pRawVertices[vi].y = _value * 1000;
#pragma omp atomic
					_center.y += m_pRawVertices[vi].y; _Ax++;
					break;
				case 2:m_pRawVertices[vi].z = _value * 1000;
#pragma omp atomic
					_center.z += m_pRawVertices[vi].z; _Ax++;			
					break;
				case 6:
					m_pRawVertices[vi].v = _value;
					_max = qMax(_value, _max);
					_min = qMin(_value, _min);
					if (m_contourType == 0)
					{
						if (m_pRawVertices[vi].v != 0)
							m_contourType = 1;
					}
					break;
				default:
					_Ax++;
					break;
				}
			}
		}
		_center.x /= m_vertexCount; _center.y /= m_vertexCount; _center.z /= m_vertexCount;
		_center.nx /= m_vertexCount; _center.ny /= m_vertexCount; _center.nz /= m_vertexCount;

		m_posCenter.x = _center.x;
		m_posCenter.y = _center.y;
		m_posCenter.z = _center.z;

		m_value = _max - _min;
		m_valueMin = _min;
		{
			auto nxv = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.v < right.v;
				});

			this->m_minVal = int(10 * nxv.first->v) / 10.0f;
			const float _dno = nxv.second->v - this->m_minVal;

			for (auto vi = 0; vi < m_vertexCount; vi++)
			{
				m_pRawVertices[vi].x = (m_pRawVertices[vi].x - _center.x) * 0.05f;
				m_pRawVertices[vi].y = (m_pRawVertices[vi].y - _center.y) * 0.05f;
				m_pRawVertices[vi].z = (m_pRawVertices[vi].z - _center.z) * 0.05f;
				m_pRawVertices[vi].z = -m_pRawVertices[vi].z;
				m_pRawVertices[vi].v = (m_pRawVertices[vi].v - this->m_minVal) / (_dno);
			}

			auto nxx = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.x < right.x;
				});
			auto nxy = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.y < right.y;
				});
			auto nxz = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
				[=](const VertexType& left, const VertexType& right) {
					return left.z < right.z;
				});



			auto _devX = fmaxf(fabsf(nxx.second->x), fabsf(nxx.first->x));
			auto _devY = fmaxf(fabsf(nxy.second->y), fabsf(nxy.first->y));
			auto _devZ = fmaxf(fabsf(nxz.second->z), fabsf(nxz.first->z));

			this->m_maxPlaneValue = fmaxf(_devX, fmaxf(_devY, _devZ)) * 1.1;
			//this->m_maxPlaneValue = sqrtf(3 * this->m_maxPlaneValue * this->m_maxPlaneValue);
		}




		for (int ii = 0; ii < _m_indexCount; ii++)
		{
			auto bll = all[ii + m_vertexCount].trimmed().split(' ');

			auto result = std::make_unique<int[]>(3);
			int j = 0;
			int _Ax = 0;
			for (int iii = 0; iii < bll.size(); iii++)
			{
				if (bll[iii].constData()->isSpace())
					continue;
				const float _value = bll[iii].toInt();
				switch (_Ax)
				{
				case 0:m_pRawIndices[ii * 3 + 0] = _value-1;
					_Ax++;
					break;
				case 1:m_pRawIndices[ii * 3 + 1] = _value-1;
					_Ax++;
					break;
				case 2:m_pRawIndices[ii * 3 + 2] = _value-1;
					_Ax++;
				default:
					//_Ax++;
					break;
				}
			}
			cummulateNormalVector(m_pRawVertices[m_pRawIndices[ii * 3 + 0]],
				m_pRawVertices[m_pRawIndices[ii * 3 + 1]],
				m_pRawVertices[m_pRawIndices[ii * 3 + 2]]);
		}
	}
	else
	{

		QFile qf(str);
		if (!qf.open(QIODevice::ReadOnly | QIODevice::Text | QIODevice::ExistingOnly))
			return;

		QTextStream qts(&qf);
		QString line = qts.readLine();

		QRegExp rx("\\\"(.*)\\\"");
		int pos = 0, _adder = 0, _m_indexCount;
		{
			auto vl = line.split(",");
			for (auto& vll : vl)
			{
				rx.indexIn(vll, 0);
				m_variableName << rx.cap(1);
			}
		}
		line = qts.readLine();
		// VARIABLES = "...", "...", "...", "..." ..
		rx.setPattern("\\\.*ET\\\s*=\\\s*(\\\w*)\s*,\\\s*");
		pos = 0;
		//ZONE F=..., ET=...,      N=        ...  , E=       ... 

		auto _ll = qts.readAll();
		auto all = _ll.splitRef('\n', Qt::SplitBehaviorFlags::KeepEmptyParts);


		for (int vi = 0; vi < m_vertexCount; vi++)
		{
			auto bll = all[vi * 12].trimmed().split(' ');

			//const Qhar* stream = all[vi].data();
			//char* pc = nullptr;
			//char* cntx = nullptr;
			//
			//strtof()
			int _Ax = 0;
			for (int axis = 0; axis < bll.size(); axis++)
			{
				if (bll[axis].constData()->isSpace())
					continue;
				const float _value = bll[axis].toFloat();
				switch (_Ax)
				{
				case 0:
				case 1:
				case 2:
				case 3:
				case 4:
				case 5:
					_Ax++;
					break;
				case 6:
					m_pRawVertices[vi].v = _value;
					if (m_contourType == 0)
					{
						if (m_pRawVertices[vi].v != 0)
							m_contourType = 1;
					}
					break;
				default:
					break;
				}
			}
		}
		auto nxv = std::minmax_element(m_pRawVertices, m_pRawVertices + m_vertexCount,
			[=](const VertexType& left, const VertexType& right) {
				return left.v < right.v;
			});

		this->m_minVal = int(10 * nxv.first->v) / 10.0f;
		const float _dno = nxv.second->v - this->m_minVal;

		//#pragma omp parallel for
		for (auto vi = 0; vi < m_vertexCount; vi++)
		{
			m_pRawVertices[vi].v = (m_pRawVertices[vi].v - this->m_minVal) / (_dno);
		}
	}
}

const QStringList TriObj::getVariableName() const
{
	return this->m_variableName;
}

//const QMatrix4x4 TriObj::Model( QMatrix4x4& model)
//{
//	// TODO: insert return statement here
//	return model;
//}

void TriObj::allocateVertices(void*)
{
}


void TriObj::cummulateNormalVector(VertexType& a, VertexType& b, VertexType& c) const
{
	const float Ax = b.x - a.x;
	const float Ay = b.y - a.y;
	const float Az = b.z - a.z;
	const float Bx = c.x - a.x;
	const float By = c.y - a.y;
	const float Bz = c.z - a.z;

	const float Nx = (Ay * Bz - Az * By);
	const float Ny = (Az * Bx - Ax * Bz);
	const float Nz = (Ax * By - Ay * Bx);

	/*if ((a.nx == 0) && (a.ny == 0) && (a.nz == 0))
	{
		a.nx = Nx; a.ny = Ny; a.nz = Nz;
	}
	else
	{*/
	a.nx += Nx; a.ny += Ny; a.nz += Nz;
	b.nx += Nx; b.ny += Ny; b.nz += Nz;
	c.nx += Nx; c.ny += Ny; c.nz += Nz;
	//int pos = 0;
	//QRegExp rx("\\\s*(\\w*)\\\s*");
	//while ((pos = rx.indexIn(line, pos)) != -1)
	//{
	//	if(isVtx)
	//		m_pRawVertices.x rx.cap(1);
	//	else

	//	pos += rx.matchedLength();
	//}
	//return true;
}

void TriObj::makeObject()
{
	SafeReleaseArray(m_pVertices);
	SafeReleaseArray(m_pIndices);

	if (m_pRawIndices && m_pRawVertices)
	{
		//memccpy(m_pVertices,m_pRawVertices,sizeof)
		m_pVertices = new VertexType[m_vertexCount];
		m_pIndices = new Idx[m_indexCount];
		//m_pVertices = m_pRaws;
		/*m_pVertices = m_pRawVertices;
		m_pIndices = m_pRawIndices;*/
		memcpy(m_pVertices, m_pRawVertices, sizeof(VertexType) * m_vertexCount);
		memcpy(m_pIndices, m_pRawIndices, sizeof(Idx) * m_indexCount);
	}

	//if (!m_vao.isCreated())
	m_vao.create();
	//m_vao.destroy();
	QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

	//if (!m_vbo.isCreated())
	m_vbo.create();
	//m_vbo.destroy();
	m_vbo.bind();
	m_vbo.allocate(m_pVertices, m_vertexCount * VertexType::getBytes());
	//m_vbo.allocate(m_pVertices, m_vertexCount * sizeof(float) * 6);

	m_vbo.release();

	/*m_ibo.create();
	m_ibo.allocate(m_pIndices, m_indexCount * sizeof(Idx));
	m_ibo.release();*/
	m_vao.bind();

	m_vbo.bind();

	m_vbo.release();
	m_vao.release();
}

const int TriObj::getSceneWidth() const
{
	return m_commonwidth;
}

const int TriObj::getSceneHeight() const
{
	return m_commonheight;
}

#include "GraphicsWindow.h"
#include<qopenglshaderprogram.h>
#include<qcoreapplication.h>
#include<qmouseevent.h>

#include<math.h>

#include<memory>
#include<algorithm>
#include<map>
#include<unordered_map>
#include<set>
#include<algorithm>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>

//#include"tfHelper.h"
std::chrono::steady_clock::time_point GraphicsWindow::timer_prev = std::chrono::high_resolution_clock::now();
//float GraphicsWindow::timer = 0;

std::mutex GraphicsWindow::m_mtx;

////////////////////////////////////////////////////////
GraphicsWindow::GraphicsWindow(QWidget* parent) :QOpenGLWidget(parent)
{
	QSurfaceFormat format;
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setSamples(8);

	QSurfaceFormat::setDefaultFormat(format);

	this->setMouseTracking(true);

	setFocusPolicy(Qt::StrongFocus);
	m_camera.setToIdentity();
	m_camera.translate(0, 0, -1);

	m_model.setToIdentity();
	m_model.translate(0, 0, 0);


	m_zoomfactor.x = 1.0f;
	m_zoomfactor.y = 1.0f;
	m_zoomfactor.z = 1.0f;
	m_zoomfactor.w = 1.0f;
	m_zoomstack.push(m_zoomfactor);

	m_MousePos = make_float2(-1, -1);

	installEventFilter(this);

	m_guide = make_unique<QString[]>(4);

	m_codec = QTextCodec::codecForName("eucKR");
}

GraphicsWindow::~GraphicsWindow()
{
	DeleteObject(m_triDim);
	DeleteObject(m_texContainer);

	DeleteObject(m_ObjModifyLine);
	DeleteObject(m_ObjLines);

	DeleteObject(m_ObjCalibrationLines);

	DeleteObject(m_ObjStartPointCross);
	DeleteObject(m_ObjEndPointCross);
	DeleteObject(m_ObjMatchingPointsCross);

	DeleteObject(m_ObjEdge);
	
	DeleteObject(m_Objlebeling);
	
	DeleteObject(m_vecAxis);
	DeleteObject(m_vec3DLine);

	doneCurrent();
}

QSize GraphicsWindow::minimumSizeHint() const
{
	return QSize(50, 50);
}

QSize GraphicsWindow::sizeHint() const
{
	return QSize(m_lfWidth, m_lfHeight);
}

void GraphicsWindow::Initialize()
{
	initializeGL();
}

void GraphicsWindow::Render()
{
	this->paintGL();
}

void GraphicsWindow::ShutDown()
{

	if (m_texContainer)
		m_texContainer->ShutDown();

	if (m_triDim)
		m_triDim->ShutDown();

	if (m_ObjModifyLine)
		m_ObjModifyLine->ShutDown();

	for (auto& v : m_ObjLines)
		v->ShutDown();

	for (auto& v : m_ObjCalibrationLines)
		v->ShutDown();

	for (auto& v : m_ObjStartPointCross)
		v->ShutDown();

	for (auto& v : m_ObjEndPointCross)
		v->ShutDown();

	for (auto& v : m_ObjEdge)
		v->ShutDown();

	for (auto& v : m_ObjMatchingPointsCross)
		v->ShutDown();

	for (auto& v : m_Objlebeling)
		v->ShutDown();
	
	for (auto& v : m_vecAxis)
		v->ShutDown();

	for (auto& v : m_vec3DLine)
		v->ShutDown();


}

void GraphicsWindow::CleanUp()
{
	/*if (m_program == nullptr)
		return;*/
	makeCurrent();
	/*m_logoVbo.destroy();
	delete m_program;
	m_program = nullptr;*/

	DeleteObject(m_triDim);
	DeleteObject(m_texContainer);

	DeleteObject(m_ObjModifyLine);

	DeleteObject(m_ObjLines);
	DeleteObject(m_ObjStartPointCross);
	DeleteObject(m_ObjEndPointCross);
	DeleteObject(m_ObjMatchingPointsCross);
	DeleteObject(m_Objlebeling);

	doneCurrent();
}

bool GraphicsWindow::eventFilter(QObject* obj, QEvent* e)
{
	int key;
	QString objName = obj->objectName();
	QKeyEvent* keyEvent = static_cast<QKeyEvent*>(e);
	QEvent::Type eventType = keyEvent->type();

	if (eventType == QEvent::KeyPress)
	{
		if (objName == "pushButton") {
		}
	}
	return QObject::eventFilter(obj, e);
}

void GraphicsWindow::initializeGL()
{
	connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GraphicsWindow::CleanUp);
	initializeOpenGLFunctions();

	for (auto& v : m_ObjLines)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_ObjCalibrationLines)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_ObjStartPointCross)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_ObjEndPointCross)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_ObjEdge)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_ObjMatchingPointsCross)
	{
		v->setContext(context());
		v->initializeGL();
	}

	for (auto& v : m_Objlebeling)
	{
		v->setContext(context());
		v->initializeGL();
	}

	if (m_texContainer)
	{
		m_texContainer->setContext(context());
		m_texContainer->initializeGL();
	}

	if (m_triDim)
	{
		m_triDim->setContext(context());
		m_triDim->initializeGL();
	}

	for (int i = 0; i < m_vecAxis.size(); i++)
	{
		m_vecAxis[i]->setContext(context());
		m_vecAxis[i]->initializeGL();
	}

	for (int i = 0; i < m_vec3DLine.size(); i++)
	{
		m_vec3DLine[i]->setContext(context());
		m_vec3DLine[i]->initializeGL();
	}

}

void GraphicsWindow::resizeGL(int w, int h)
{
	//current_width = w;
	//current_height = h;
	m_screenRatio = float(w) / h;
	//m_screenRatio = 1;
	m_proj.setToIdentity();
	m_proj.perspective(0.0f, GLfloat(w) / h, 0.001f, 10000.0f);

	float _margin = 20;
	const float _w = this->width() / 2.0f - _margin;
	const float _h = this->height() / 2.0f - _margin;
	for (auto i = 0; i < 4; i++)
	{
		m_textBox[i].setX(i % 3 == 0 ? _margin : this->width() * 0.5f);
		m_textBox[i].setY(i < 2 ? _margin : this->height() * 0.5f);
		m_textBox[i].setWidth(_w);
		m_textBox[i].setHeight(_h);
	}

	if (m_texContainer)
	{
		getModelMat(m_lfScale);
	}
	else if (m_triDim)
	{
		InitScreen(m_bFFShow);
	}
	update();
	//m_triDim->setScreenRatio(m_screenRatio);
}

void GraphicsWindow::paintGL()
{
	context()->functions()->glClearColor(0, 0, 0, 0);
	context()->functions()->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	context()->functions()->glEnable(GL_DEPTH_TEST);
	auto renderText = [&](const QFont& font = QFont())->void
	{
		GLdouble glColor[4];
		glGetDoublev(GL_CURRENT_COLOR, glColor);
		QColor fontColor = QColor(255, 255, 255, 255);
		QColor dynaColor = QColor(0, 190, 245, 255);
		QColor PickPointColor = QColor(0, 255, 0, 255);
		// Render text
		QPainter painter(this);
		painter.beginNativePainting();
		painter.endNativePainting();
		painter.setPen(fontColor);
		painter.setFont(font);

		auto DrawPointName = [&](float2 pt, QString str)->void
		{
			float2 pos = GetScreenToPoint(pt);
			QRectF Linerect;
			Linerect.setX(pos.x);
			Linerect.setY(pos.y - 5);
			Linerect.setWidth(500);
			Linerect.setHeight(30);
			painter.drawText(Linerect, Qt::AlignLeft | Qt::AlignTop | Qt::TextExpandTabs, str);
		};

	
		QRectF rect = m_textBox[2];

		int2 pos = make_int2(m_MousePos.x, m_MousePos.y);
		rect.setY(m_textBox[2].y() - 20);
		painter.drawText(rect, Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs, tr("pos "));
		painter.setPen(dynaColor);

		if (pos.x < m_lfWidth && pos.y < m_lfHeight && pos.x > 0 && pos.y > 0)
		{
			int n = pos.x + pos.y * m_lfHeight;
			//painter.drawText(rect, Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs, tr("                       (%1 %2 %3)").
			//	arg(int(m_MousePos.x)).arg(int(m_MousePos.y)).arg(m_valuePixel));

			if (m_valuePixel == 0)
			{
				unsigned char* initScene = ((TexObj*)m_texContainer)->Scene();
				if (initScene)
				{
					auto val = int(initScene[n] * 1.0);
					painter.drawText(rect, Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs, tr("                       (%1 %2 %3)").
						arg(int(m_MousePos.x)).arg(int(m_MousePos.y)).arg(val));
				}
			}
			else
			{
				painter.drawText(rect, Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs, tr("                       (%1 %2 %3)").
					arg(int(m_MousePos.x)).arg(int(m_MousePos.y)).arg(m_valuePixel));
			}
		}
		else
		{
			painter.drawText(rect, Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs, tr("                       (%1 %2)").
				arg(int(m_MousePos.x)).arg(int(m_MousePos.y)));
		}

		painter.setPen(PickPointColor);

		painter.setPen(QColor(0, 255, 0, 255));

		//if(m_ModelState == model_info::model_type::line_2d || m_ModelState == model_info::model_type::equilateral_line_2d)
		{
			if (GetStartPoint().x != -1 && GetStartPoint().y != -1)
				DrawPointName(GetStartPoint(), " Start");
			if (GetEndPoint().x != -1 && GetEndPoint().y != -1)
				DrawPointName(GetEndPoint(), " End");
		}

		for (auto i = 0; i < 4; i++)
		{
			int _flag = 0;
			switch (i)
			{
			case 0:
				_flag = Qt::AlignLeft | Qt::AlignTop;
				break;
			case 1:_flag = Qt::AlignRight | Qt::AlignTop;
				break;
			case 2: _flag = Qt::AlignHCenter | Qt::AlignBottom | Qt::TextExpandTabs;
				break;
			case 3:
				_flag = Qt::AlignLeft | Qt::AlignBottom;
				break;
			}
			painter.setPen(fontColor);

			QRectF rect = m_textBox[i];
			painter.drawText(rect, _flag, m_guide.get()[i]);
			if (i == 3)
			{
				painter.setPen(dynaColor);
				painter.drawText(m_textBox[i], _flag, tr("           %1/%2\n\n\n\n\n").
					arg(currentImage + 1).
					arg(numberImage));
			}
			else if (i == 2)
			{
				int CC = ((TexObj*)m_texContainer)->getWindowCenter();
				int CW = ((TexObj*)m_texContainer)->getWindowWidth();
				painter.setPen(dynaColor);
				painter.drawText(m_textBox[i], _flag, tr("                     %1/%2\n").
					arg(CC).
					arg(CW));

			}
		}

		if (m_bSelectGraphic)
		{
			QColor Color = QColor(255, 0, 0, 255);
			painter.setPen(QPen(Color, 5));

			QRect rc;
			rc.setLeft(this->rect().x() + 2);
			rc.setRight(this->rect().x() + this->rect().width() - 2);
			rc.setTop(this->rect().y());
			rc.setBottom(this->rect().y() + this->rect().height() - 2);
			painter.drawRect(rc);
		}
	};

	if (m_texContainer)
	{
		if (m_isRunning)
			return;
		m_texContainer->Render(m_model);
		if (isPlay)
		{
			if (this->m_ObjLines.size() == 0)
			{
				clear_line();
				auto _timer = std::chrono::high_resolution_clock::now();
				auto dd = _timer - timer_prev;
				auto d = std::chrono::duration_cast<std::chrono::milliseconds>(dd);
				auto ggg = int(d.count() / (1000.0f / 15));
				if (ggg > timer)
				{
					currentImage++;
					if (rotCount <= 0 && myFrame == currentImage)
						isPlay = false;
					circleImage();
					m_texContainer->moveScene(currentImage);
					timer += 1;
				}
			}
			else
			{
				isPlay = false;
			}
		}
		if (rectDraw)
		{
			QPainter p(this);
			p.beginNativePainting();
			p.endNativePainting();
			p.fillRect(mRectBuffer, QBrush(QColor(255, 255, 255, 64)));
		}


		for (auto& v : m_ObjStartPointCross)
			v->Render(m_model);

		for (auto& v : m_ObjEndPointCross)
			v->Render(m_model);

		for (auto& v : m_ObjMatchingPointsCross)
			v->Render(m_model);

		for (auto& v : m_ObjCalibrationLines)
			v->Render(m_model);

		for (auto& v : m_Objlebeling)
			v->Render(m_model);

		if (m_ObjModifyLine)
			m_ObjModifyLine->Render(m_model);

		if (!pressedAlt)
		{
			for (auto& v : m_ObjEdge)
				v->Render(m_model);

			if (Angio_Algorithm_.get_segmentation_Instance().optimal_image_id != -1)
			{
				if (Angio_Algorithm_.get_segmentation_Instance().optimal_image_id == currentImage)
				{
					for (auto& v : m_ObjLines)
					{
						//if (v->getGraphicType() != Object::GraphicType::CenterLine)
							v->Render(m_model);
					}
				}
			}
			else
			{
				for (auto& v : m_ObjLines)
				{
					//if (v->getGraphicType() != Object::GraphicType::CenterLine)
					v->Render(m_model);
				}
			}
			renderText();
		}
	}

	else if (m_triDim)
	{
		auto end_point_info = Angio_Algorithm_.get_endPoint3D_result_Instance();
		if (end_point_info.sort_center_id_point.size() != 0)
		{
			glMatrixMode(GL_PROJECTION);
			
			vector<float> _dis;

			glColor3f(1.0, 1.0, 1.0);

			for (int i = 0; i < end_point_info.sort_id_point.size(); i++)
				Insert3DPointName(end_point_info.sort_id_point[i], i + 1);
		}
		
		float lenAxis = 0.25 * m_lfScale;


		glColor3f(1.0, 0.0, 0.0);
		DrawBitmapText("X", make_float3(lenAxis, 0, 0), TextType::AxisName);

		glColor3f(1.0, 1.0, 0.0);
		DrawBitmapText("Y", make_float3(0, lenAxis, 0), TextType::AxisName);

		glColor3f(0.0, 1.0, 0.0);
		DrawBitmapText("Z", make_float3(0, 0, -lenAxis), TextType::AxisName);


		glMatrixMode(GL_MODELVIEW);


		if (static_cast<TriObj*>(m_triDim)->getFFRShow())
		{
			QPainter p(this);
			p.beginNativePainting();
			p.endNativePainting();
			QRect rec = this->geometry();
			QRectF temp;

			auto per = rec.height() / 8.0f;
			temp.setX(rec.width() - 50);
			temp.setY(per * 2);
			temp.setWidth(10);
			temp.setHeight(rec.height() - per * 4);


			auto float2RGB = [](float value)->QColor
			{
				float H = 1.0 - value;
				if (H < 0.5)
				{
					float R = 0.705882 + H * 0.0318242;
					float G = 0.015686 + H * 1.16986334;
					float B = 0.149020 + H * 0.835199;

					return QColor(R * 255, G * 255, B * 255);
				}
				else
				{
					float R = 0.865003 - H * 1.126726;
					float G = 0.865003 - H * 1.13;
					float B = 0.865003 - H * 0.0224124;
					return QColor(R * 255, G * 255, B * 255);
				}

				return QColor(1.0 * 255, 1.0 * 255, 1.0 * 255);
			};

			QFont font = QFont("Arial");

			GLdouble glColor[4];
			glGetDoublev(GL_CURRENT_COLOR, glColor);
			QColor fontColor = QColor(255, 255, 255, 255);

			auto per2 = temp.height() / 11;

			p.setFont(font);
			int _flag = Qt::AlignHCenter | Qt::AlignTop | Qt::TextExpandTabs;
			p.setPen(fontColor);

			auto mn = ((TriObj*)m_triDim)->getMinValue();
			float slabel[] = { mn,(1.0f - mn) / 2.0f + mn,1.0f };
			for (auto i = 0; i < 3; i++)
			{
				p.drawText(rec.width() - 40, per * 2 + per2 * (i * 5), 40, per2 * 2, _flag, QString::number(slabel[2 - i], 'g', 2));
			}
			auto _def = font.pointSize();
			font.setPointSize(_def * 1.3);
			p.setFont(font);
			_flag = Qt::AlignRight | Qt::AlignTop | Qt::TextExpandTabs;

			auto study = static_cast<TriObj*>(m_triDim)->getVariableName().back();

			p.drawText(rec.width() - 70, per + 10, 45, per2 * 2, _flag, study);

			font.setPointSize(_def * 1.8);
			p.setFont(font);

			QLinearGradient gradient(temp.topLeft(), temp.bottomRight());
			QGradientStops stops;
			stops << QGradientStop(0.0, QColor(0.705882 * 255, 0.015686 * 255, 0.149020 * 255));
			stops << QGradientStop(0.5, QColor(0.865003 * 255, 0.865003 * 255, 0.865003 * 255));
			stops << QGradientStop(1.0, QColor(0.231373 * 255, 0.298039 * 255, 0.752941 * 255));
			gradient.setStops(stops);
			p.fillRect(temp, gradient);
			_flag = Qt::AlignLeft | Qt::AlignTop | Qt::TextExpandTabs;
			m_triDim->Render(m_model, m_camera, m_proj);
			for (int i = 0; i < m_vec3DLine.size(); i++)
				((AxisLineObj*)m_vec3DLine[i])->Render(m_model, m_camera, m_proj);

			for (int i = 0; i < m_vecAxis.size(); i++)
				((AxisLineObj*)m_vecAxis[i])->Render(m_AxisModel, m_AxisCamera, m_proj);
		}
		else
		{
			auto model = m_model;
			//model.rotate(0.1, 1, 0, 0);
			//model.rotate(5, 0, 1, 0);
			for (int i = 0; i < m_vec3DLine.size(); i++)
				((AxisLineObj*)m_vec3DLine[i])->Render(model, m_camera, m_proj);
		//	DrawBitmapText("+", make_float3(-0.44,0.08,-1.2), true);

			for (int i = 0; i < m_vecAxis.size(); i++)
				((AxisLineObj*)m_vecAxis[i])->Render(m_AxisModel, m_AxisCamera, m_proj);

			m_triDim->Render(m_model, m_camera, m_proj);
		}

	}

	update();
}

void GraphicsWindow::Insert3DPointName(result_info::end_point_info vecPoint3D, int nIndex)
{
	std::string strName;
	glColor3f(0.0, 1.0, 1.0);
	strName = "    " + QString::number(nIndex).toStdString() +" (" + QString::number(vecPoint3D.frame_id).toStdString() + ") :  ";
	float dis = floor(vecPoint3D.d) / 10; //5번째
	strName += QString::number(dis).toStdString() + " cm/s";
	float ypos = (nIndex - 1) * 0.3 - 3.0;
	DrawBitmapText(strName.c_str(), make_float3(ypos, 2.2, 0), TextType::fixTextName);

	glColor3f(1.0, 1.0, 1.0);
	strName = " " + QString::number(nIndex).toStdString();
	DrawBitmapText(strName.c_str(), vecPoint3D.pos3D, TextType::TextName);

}

//이미지 열기
bool GraphicsWindow::Parsing(const QString& s)
{
	program_path_ = std::filesystem::current_path().string();
	size_t pos = program_path_.find_last_of('\\');
	if (pos != std::string::npos)
		program_path_ = program_path_.substr(0, pos);
	FILE* f = NULL;

	using _prIQS = pair<int, QString>;
	static map<int, QString> _ext_list = { _prIQS(0,".png"),_prIQS(1,".bmp"),_prIQS(2,".jpg"),_prIQS(3,".jpeg"),_prIQS(4,".tlf"),_prIQS(5,".plt") };
	if (s.isEmpty())
		return false;
	this->myType = 1;
	int _chkVal = -1;
	for (auto& _chker : _ext_list)
	{
		if (s.endsWith(_chker.second))
		{
			_chkVal = _chker.first;
			break;
		}
	}

	if (_chkVal == 5)
	{
		DeleteObject(m_triDim);
		DeleteObject(m_vecAxis);
		DeleteObject(m_vec3DLine);

		m_triDim = new TriObj(context());
		reinterpret_cast<TriObj*>(m_triDim)->setFFRShow(m_bFFShow);
	
		if (m_bFFShow)
		{
			reinterpret_cast<TriObj*>(m_triDim)->SetFFR(s);
			m_lfScale = ((TriObj*)m_triDim)->getMaxPlaneValue();
		}
		else
		{
			bool b3D = m_triDim->Parsing(s);
			if (!b3D)
			{
				DeleteObject(m_triDim);
				return false;
			}
		}

		if (!m_triDim)
		{
			DeleteObject(m_triDim);
			return false;
		}

		m_lfScale = ((TriObj*)m_triDim)->getMaxPlaneValue();
		auto posCenter = ((TriObj*)m_triDim)->GetCenterPos();
	
		Get3DCenterline();

		auto end3D_points = Angio_Algorithm_.get_endPoint3D_result_Instance().frame_end_point;
		if (end3D_points.size() != 0)
		{
			auto point = new AxisLineObj(context());
			point->SetAxisColor(Object::GraphicLines::EndPoint3D);
			point->SetCrossDisplay(!m_bFFShow);
			point->SetScale(m_lfScale);
			vector<float3> vecEndPos;
			vector<int> vecEndPosIds;
			int id = -1;
			for (int i = 0; i < end3D_points.size(); i++)
				vecEndPos.push_back(end3D_points[i]);
			point->SetEndPoints(vecEndPos);
			point->SetCenterPoint(posCenter);
			m_vec3DLine.push_back(point);
		}

		
		for (int i = 0; i < 3; i++)
		{
			auto Axis = new AxisLineObj(context());
			Axis->SetAxisColor(i);
			Axis->SetScale(((TriObj*)m_triDim)->getMaxPlaneValue());
			m_vecAxis.push_back(Axis);
		}

		m_camera.setToIdentity();
		m_camera.lookAt(QVector3D(0, 0, -0.0), QVector3D(0, 0, 2.0f), QVector3D(-1, 0, 0));
	
		m_bInit = true;
	}
	else
	{
		m_texContainer = new TexObj(context());
		reinterpret_cast<TexObj*>(m_texContainer)->Parsing(s, false);
		this->numberImage = ((TexObj*)m_texContainer)->getNumberImages();
		this->myFrame = numberImage - 1;
		QString strExtension = s;
		this->m_fileAllPath = s;
		this->m_filePath = strExtension.section("\\\\", 0, -2);
		auto fileName = strExtension.section("\\\\", -1);
		this->m_fileName = fileName.section(".", 0, -2);
		this->m_datePath = QDir::currentPath().section("/", 0, -2) + QString("\\data");
		this->bDcmfile = false;
		this->m_nOrigWL = GetWindowCenter();
		this->m_nOrigWW = GetWindowWidth();


		auto folder = m_filePath.section("\\\\", 0, -2);


		m_lfWidth = m_texContainer->getSceneWidth();
		m_lfHeight = m_texContainer->getSceneHeight();

		timer_prev = std::chrono::high_resolution_clock::now();
		timer = 0;
		m_bInit = true;
	}
	repaint();
	return true;
}

void  GraphicsWindow::Parsing(unique_ptr<dcmHelper>& dh)
{
	m_bInit = true;

	program_path_ = std::filesystem::current_path().string();
	size_t pos = program_path_.find_last_of('\\');
	if (pos != std::string::npos)
		program_path_ = program_path_.substr(0, pos);

	mDCMHelper = std::move(dh);

	m_texContainer = new TexObj(context());
	reinterpret_cast<TexObj*>(m_texContainer)->Prepare(mDCMHelper.get());
	this->currentImage = ((TexObj*)m_texContainer)->getCurrentImages();
	this->numberImage = ((TexObj*)m_texContainer)->getNumberImages();
	this->myFrame = numberImage - 1;
	this->bDcmfile = true;
	this->m_nOrigWL = GetWindowCenter();
	this->m_nOrigWW = GetWindowWidth();
	QTextCodec* codec = QTextCodec::codecForName("eucKR");
	QString encodedString = codec->toUnicode(mDCMHelper.get()->getFile()->c_str());
	this->m_filePath = encodedString.section("\\\\", 0, -2);
	this->m_datePath = QDir::currentPath().section("/", 0, -2) + QString("\\data");
	auto fileName = encodedString.section("\\\\", -1);
	this->m_fileName = fileName.section(".", 0, -2);
	if (m_fileName.isEmpty())
		m_fileName = fileName;
	this->m_fileAllPath = encodedString;

	m_lfWidth = mDCMHelper.get()->getRows();
	m_lfHeight = mDCMHelper.get()->getCols();
	this->m_SeriesNumber = mDCMHelper.get()->getSeries();

	//edges of screen ordered by CW 
	for (auto i = 0; i < 4; i++)
		m_guide.get()[i] = mDCMHelper->getInformation(i);
	
	
	QString strExtension = QDir::currentPath().section("/", 0, -2);
	QString FilePath = strExtension + QString("\\data");
	if (!QFile::exists(FilePath))
		QDir().mkdir(FilePath);
	
	timer_prev = std::chrono::high_resolution_clock::now();
	timer = 0;
	SetOpenImageId(numberImage / 2);

	repaint();
}


//초기값
void GraphicsWindow::SetLineRange(int Vaule, bool bAuto)
{
	m_nFindLineRange = Vaule;
	m_bAutoRange = bAuto;
}

void GraphicsWindow::SetMulPSAngle(const float& _p0, const float& _s0, const float& _p1, const float& _s1, const float& _v)
{
	p0 = _p0;
	p1 = _p1;
	s0 = _s0;
	s1 = _s1;
	update();
}

void GraphicsWindow::getModelMat(const float& f)
{
	m_model = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
	auto _t = m_zoomstack.top();

	if (m_triDim)
	{
		const float _f = f * m_screenRatio;
		if (m_screenRatio < 1)
		{
			m_model.ortho(-(_f)*_t.x, (_f)*_t.y, _f * _t.z, -_f * _t.w, -2.0f * f, 2.0f * f);
		}
		else
		{
			//const float _f = f / m_screenRatio; 
			m_model.ortho(-f * _t.x, f * _t.y, (_f)*_t.z, -(_f)*_t.w, -2.0f * f, 2.0f * f);
		}
		m_AxisModel = m_model;
	}

	else
	{
		if (m_screenRatio >= 1)
		{
			const float _f = f * m_screenRatio;
			m_model.ortho(-(_f)*_t.x, (_f)*_t.y, f * _t.z, -f * _t.w, -2.0f * f, 2.0f * f);
		}
		else
		{
			const float _f = f / m_screenRatio;
			m_model.ortho(-f * _t.x, f * _t.y, (_f)*_t.z, -(_f)*_t.w, -2.0f * f, 2.0f * f);
		}
	}
	m_zoom = f;
}

void GraphicsWindow::circleImage()
{
	if (currentImage < 0)
		currentImage = numberImage - 1;
	else if (currentImage == numberImage)
	{
		currentImage = 0;
		rotCount--;
	}
}

void GraphicsWindow::SetOpenImageId(int nIndex)
{
	int nHalfIndex = nIndex;

	if (nIndex == -1)
		nHalfIndex = numberImage / 2.0f;
	//nHalfIndex = 20;
	currentImage = nHalfIndex;
	circleImage();
	if (m_texContainer)
		m_texContainer->moveScene(currentImage);
	update();
}


//마우스
void GraphicsWindow::mouseDoubleClickEvent(QMouseEvent* event)
{
	return;
}

void GraphicsWindow::mousePressEvent(QMouseEvent* event)
{
	if (!m_texContainer && !m_triDim)
		return;
	if (m_isRunning)
		return;
	m_LastPos = event->pos();

	float2 ptmouse = make_float2(event->pos().x(), event->pos().y());
	updata_pick_point_ = GetPointToScreen(ptmouse);

	if (m_texContainer)
	{
		m_fUpdateCursor(GetGridIndex());
		m_bSelectGraphic = true;

		if (event->button() == Qt::LeftButton)
		{
			auto axis = event->pos();
			if (!pressedlShft)
			{
				if ((mRectBuffer.left() == -1) && (mRectBuffer.top() == -1))
				{
					mRectBuffer.setTopLeft(axis);
					mRectBuffer.setBottomRight(axis);
				}
			}
		}
	}
	else if (m_triDim)
	{
		//m_fUpdateCursor(GetGridIndex());

		if (event->button() == Qt::LeftButton)
		{
			auto axis = event->pos();
			if ((mRectBuffer.left() == -1) && (mRectBuffer.top() == -1))
			{
				//mRectBuffer.setTopLeft(axis);
				//mRectBuffer.setBottomRight(axis);
			}
		}
	}
	m_Lbutton = event->button() == Qt::LeftButton;
	m_Rbutton = event->button() == Qt::RightButton;
	m_Mbutton = event->button() == Qt::MiddleButton;

	//pressedCtrl = event->button() == Qt::ControlModifier;
}

void GraphicsWindow::mouseReleaseEvent(QMouseEvent* event)
{
	if (m_isRunning)
		return;

	if (m_texContainer)
	{
		if (m_Lbutton)
		{
			if (is_move_line_)
			{
				if (m_ModelState == model_info::model_type::equilateral_line_2d || m_ModelState == model_info::model_type::line_2d)
					ModifyLine(false);
			}
			else if (is_move_pick_point_)
			{
				m_fUpdateLine(myGrid.x, is_move_pick_point_);
			}
			else if (rectDraw)
			{
				if ((mRectBuffer.left() != -1) && (mRectBuffer.top() != -1) && ((mRectBuffer.width() != 0) && (mRectBuffer.height() != 0)))
				{
					float4 _zf;
					DrawRectangle(mRectBuffer, _zf);
					if (mRectBuffer.width() < 0 && mRectBuffer.height() < 0)
					{
						if (m_zoomstack.size() != 1)
						{
							m_zoomstack.pop();
							getModelMat(m_lfScale);
							update();
						}
					}
					else
					{
						if (mRectBuffer.width() > 2 || mRectBuffer.height() > 2)
						{
							m_zoomstack.push(_zf);
							getModelMat(m_lfScale);
						}
					}
					update();
				}
			}
		}
	}
	mRectBuffer.setCoords(-1, -1, -1, -1);
	rectDraw = false;
	m_Lbutton = false;
	m_Rbutton = false;
	m_Mbutton = false;
	is_move_pick_point_ = false;
	is_move_line_ = false;
}

void GraphicsWindow::mouseMoveEvent(QMouseEvent* event)
{
	if (!m_texContainer && !m_triDim)
		return;
	if (m_isRunning)
		return;
	m_MousePos = make_float2(event->pos().x(), event->pos().y());

	UpdateModel();

	m_LastPos = event->pos();

	update();
	setFocus();
	setFocusPolicy(Qt::StrongFocus);

}

void GraphicsWindow::wheelEvent(QWheelEvent* event)
{
	if (m_isRunning)
		return;
	if (m_triDim)
	{
		auto lfRange = ((TriObj*)m_triDim)->getMaxPlaneValue() / 10;
		
		auto lfScale = 1.0;
		//lfScale += (event->delta() > 0) ? lfRange : -lfRange;

		lfScale += (event->delta() > 0) ? 0.1 : -0.1;
		if (lfScale < 0)
			return;
		m_scale = m_scale + lfScale;

		m_model.scale(lfScale);
	}
	else if (m_texContainer)
	{
		currentImage += (event->delta() > 0) ? 1 : -1;
		SetCurrentImage(currentImage);
		setCursor(Qt::ArrowCursor);
	}
	else
		return;

	setFocusPolicy(Qt::StrongFocus);
	update();
}



//확대,축소
void GraphicsWindow::DrawRectangle(QRectF& RectBuffer, float4& zf)
{
	if (!m_texContainer)
		return;
	float2 _center;
	_center.x = this->geometry().width() / 2.0f;
	_center.y = this->geometry().height() / 2.0f;

	RectBuffer.setX((RectBuffer.left() / 2.0f * (m_zoomstack.top().x + m_zoomstack.top().y) + this->geometry().width() / 2.0f * (1 - m_zoomstack.top().x)));
	RectBuffer.setY((RectBuffer.top() / 2.0f * (m_zoomstack.top().z + m_zoomstack.top().w) + this->geometry().height() / 2.0f * (1 - m_zoomstack.top().w)));
	RectBuffer.setRight((RectBuffer.right() / 2.0f * (m_zoomstack.top().x + m_zoomstack.top().y) + this->geometry().width() / 2.0f * (1 - m_zoomstack.top().x)));
	RectBuffer.setBottom((RectBuffer.bottom() / 2.0f * (m_zoomstack.top().z + m_zoomstack.top().w) + this->geometry().height() / 2.0f * (1 - m_zoomstack.top().w)));
	//LRBT
	float4 _zf;
	if (RectBuffer.width() < 0)
	{
		_zf.x = (_center.x - RectBuffer.right()) / _center.x;
		_zf.y = (RectBuffer.left() - _center.x) / _center.x;
	}
	else
	{
		_zf.x = (_center.x - RectBuffer.left()) / _center.x;
		_zf.y = (RectBuffer.right() - _center.x) / _center.x;
	}

	if (RectBuffer.height() < 0)
	{
		_zf.z = (RectBuffer.top() - _center.y) / _center.y;
		_zf.w = (_center.y - RectBuffer.bottom()) / _center.y;
	}
	else
	{
		_zf.z = (RectBuffer.bottom() - _center.y) / _center.y;
		_zf.w = (_center.y - RectBuffer.top()) / _center.y;
	}

	float _tx = (_zf.x + _zf.y);
	float _ty = (_zf.z + _zf.w);
	float _cen, _hf;
	if (_ty > _tx)
	{
		_hf = _ty / 2.0f;
		_cen = _tx / 2.0f - _zf.y;
		_zf.x = _cen + _hf;
		_zf.y = -_cen + _hf;
	}
	else
	{
		_hf = _tx / 2.0f;
		_cen = _ty / 2.0f - _zf.w;
		_zf.z = _cen + _hf;
		_zf.w = -_cen + _hf;
	}
	zf = _zf;
}

void GraphicsWindow::zoomFunc(bool bZoom, QRectF& RectBuffer)
{
	if (!bZoom)
		return;

	QPointF pos = m_LastPos;
	const float bRectCoeff = 10.0f;
	auto point = pos - RectBuffer.topLeft();
	if (point.manhattanLength() > bRectCoeff)
	{
		RectBuffer.setBottomRight(pos);
		if ((RectBuffer.width() < 0 && RectBuffer.height() > 0) || (RectBuffer.width() > 0 && RectBuffer.height() < 0))
		{
			rectDraw = false;
			RectBuffer = QRectF(-1, -1, -1, -1);
		}
		else
		{
			if ((RectBuffer.left() != -1) && (RectBuffer.top() != -1) && ((RectBuffer.width() != 0) && (RectBuffer.height() != 0)))
				rectDraw = true;
		}
	}
}

//위치 회전
void GraphicsWindow::NormalizeAngle(int& angle)
{
	while (angle < 0)
		angle += 360 * 16;
	while (angle > 360 * 16)
		angle -= 360 * 16;
}

void GraphicsWindow::SetCurrentImage(int nindex)
{
	currentImage = nindex;
	circleImage();

	if (m_texContainer)
	{
		m_texContainer->moveScene(currentImage);
		DeleteObject(m_Objlebeling);
		if (Angio_Algorithm_.get_segmentation_Instance().get_labeling_points().size() != 0)
		{
			//	CreateObject(m_Objlebeling, angioAlgorithm_.get_segmentation_Instance().points[i], angioAlgorithm_.get_segmentation_Instance().points[i].labeling_points.size(), Object::GraphicType::Manual_Point);
			if (currentImage < Angio_Algorithm_.get_segmentation_Instance().get_labeling_points().size())
			{
				auto lebeling_data = Angio_Algorithm_.get_segmentation_Instance().get_labeling_point(currentImage).second;
				CreateObject(m_Objlebeling, lebeling_data.data(), lebeling_data.size(), Object::GraphicType::Manual_Line, Object::Shape::Points);
			}
			auto end_points = Angio_Algorithm_.get_segmentation_line2D_instance().end_points;
			if (end_points.size() != 0)
			{
				int target = currentImage;
				auto it = std::find_if(end_points.begin(), end_points.end(),
					[target](const std::pair<int, float2>& element) {
						return element.first == target;
					});
				if (it != end_points.end())
				{
					vector<float2> point;
					point.push_back(make_float2(it->second.x, it->second.y));
					//point.push_back(make_float2(it->second.y, it->second.x));
					CreateObject(m_Objlebeling, point.data(), point.size(), Object::GraphicType::Manual_Point, Object::Shape::Points);
				}
			}
		}
		
	}
	
	update();
}

void GraphicsWindow::setXTranslation(const float& step)
{
	m_model.translate(step, 0, 0);
}

void GraphicsWindow::setYTranslation(const float& step)
{
	m_model.translate(0, step, 0);
}

void GraphicsWindow::setZTranslation(const float& step)
{
	m_model.translate(0, 0, step);
}

void GraphicsWindow::setXScaling(const float& step)
{
	m_model.scale(step, 1, 1);
}

void GraphicsWindow::setYScaling(const float& step)
{
	m_model.scale(1, step, 1);
	//m_model.rotate(15, 0, 1, 0);
}

void GraphicsWindow::setZScaling(const float& step)
{
	m_model.scale(1, 1, step);
	//m_model.rotate(-15, 0, 1, 0);
}

//키보드
void GraphicsWindow::keyPressEvent(QKeyEvent* event)
{
	//QWidget::keyPressEvent(event);
	float4 _zf = m_zoomstack.top();

	switch (event->key())
	{
	case Qt::Key_Plus:
	{
			if (!m_ObjLines.size() == 0)
			{
				m_nFindLineRange++;
				if (is_move_line_ && m_Lbutton)
					ModifyLine(true);
			}
	}
	break;
	case Qt::Key_Minus:
	{
		if (!m_ObjLines.size() == 0)
		{
			m_nFindLineRange--;
			if (is_move_line_ && m_Lbutton)
				ModifyLine(true);
		}
	}
	break;
	case Qt::Key_Control:
		pressedCtrl = true; break;
	case Qt::Key_Shift:
		pressedlShft = true; break;
	case Qt::Key_Alt:
		pressedAlt = true; break;
	case Qt::Key_M:
		break;
	case Qt::Key_N:
		break;
	case Qt::Key_W:
		break;
	case Qt::Key_Q:
	{
		//	DarwEdgeLine();
	}
	break;
	case Qt::Key_E:
	case Qt::Key_Enter:
		break;
	case Qt::Key_G:
		break;
	case Qt::Key_D:
	{
		//ClearPoints();
		//ClearLine();
	}
	break;
	case Qt::Key_T:
		break;
	case Qt::Key_C:
		break;
	case Qt::Key_1:
		break;
	case Qt::Key_2:
		break;
	case Qt::Key_3:
		break;
	case Qt::Key_L:
	{
		break;
	}
	case Qt::Key_Z:
	{
		m_fSetLineUndo();
		break;
	}
	case Qt::Key_Space:
	{
		InitScreen(!m_bFFShow);
		break;
	}
	case Qt::Key_Left:
	{
		if (m_texContainer)
		{
			_zf.x = m_zoomstack.top().x + 0.1f;
			_zf.y = m_zoomstack.top().y - 0.1f;
			m_zoomstack.top() = _zf;
			getModelMat(m_lfScale);
		}
		else if (m_triDim)
			m_camera.translate(0, 0.1, 0);

		break;
	}
	case Qt::Key_Right:
	{
		if (m_texContainer)
		{
			_zf.x = m_zoomstack.top().x - 0.1f;
			_zf.y = m_zoomstack.top().y + 0.1f;
			m_zoomstack.top() = _zf;
			getModelMat(m_lfScale);
		}
		else if (m_triDim)
		{
			m_camera.translate(0, -0.1, 0);
		}

		break;
	}
	case Qt::Key_Up:
	{
		m_camera.translate(-0.1, 0, 0);
		if (m_texContainer)
		{
			//_zf.z = m_zoomstack.top().z - 0.1f;
			//_zf.w = m_zoomstack.top().w + 0.1f;		
			//m_zoomstack.top() = _zf;
			//getModelMat(m_lfScale);
		}
		else if (m_triDim)
		{
			m_camera.translate(-0.1, 0, 0);

		}

		break;
	}
	case Qt::Key_Down:
	{
		if (m_texContainer)
		{
			_zf.z = m_zoomstack.top().z + 0.1f;
			_zf.w = m_zoomstack.top().w - 0.1f;
			m_zoomstack.top() = _zf;
			getModelMat(m_lfScale);
		}
		else if (m_triDim)
		{
			m_camera.translate(0.1, 0, 0);
		}

		break;
	}
	default:
		break;
	}
	setFocusPolicy(Qt::StrongFocus);
	update();
}

void GraphicsWindow::keyReleaseEvent(QKeyEvent* event)
{
	QWidget::keyPressEvent(event);

	switch (event->key())
	{
	case Qt::Key_Control:
		pressedCtrl = false; break;
	case Qt::Key_Shift:
		pressedlShft = false; break;
	case Qt::Key_Alt:
		pressedAlt = false; break;
	case Qt::Key_M:
		break;
	case Qt::Key_Space:
	{
		break;
	}
	default:
		break;
	}
}


//화면좌표 관련
bool GraphicsWindow::get_pick_point_move(model_info::points_type type, int Obj_id)
{
	bool move_point = false;
	float2 ptLT = make_float2(0, 0);
	float2 ptRB = make_float2(0, 0);

	float2* XAxisData = nullptr;
	float2* YAxisData = nullptr;
	int ncount = 0;
	if (type == model_info::points_type::start_point)
	{
		XAxisData = ((LineObj*)m_ObjStartPointCross[Obj_id])->GetData();
		YAxisData = ((LineObj*)m_ObjStartPointCross[Obj_id + 1])->GetData();
		ncount = ((LineObj*)m_ObjStartPointCross[Obj_id])->getVertexCount();
	}
	else if (type == model_info::points_type::end_point)
	{
		XAxisData = ((LineObj*)m_ObjEndPointCross[Obj_id])->GetData();
		YAxisData = ((LineObj*)m_ObjEndPointCross[Obj_id + 1])->GetData();
		ncount = ((LineObj*)m_ObjEndPointCross[Obj_id])->getVertexCount();
	}
	else if (type == model_info::points_type::branch_point)
	{
		XAxisData = ((LineObj*)m_ObjMatchingPointsCross[Obj_id])->GetData();
		YAxisData = ((LineObj*)m_ObjMatchingPointsCross[Obj_id + 1])->GetData();
		ncount = ((LineObj*)m_ObjMatchingPointsCross[Obj_id])->getVertexCount();
	}

	ptLT = XAxisData[0];
	ptRB = XAxisData[0];
	for (int i = 0; i < ncount; i++)
	{
		ptLT.x = min(ptLT.x, XAxisData[i].x);
		ptRB.x = max(ptRB.x, XAxisData[i].x);
		ptLT.y = min(ptLT.y, YAxisData[i].y);
		ptRB.y = max(ptRB.y, YAxisData[i].y);
	}

	if (m_ModelState == model_info::model_type::equilateral_line_2d || m_ModelState == model_info::model_type::line_2d)
	{
		ptLT = make_float2(ptLT.x + 3.5, ptLT.y + 3.5);
		ptRB = make_float2(ptRB.x - 3.5, ptRB.y - 3.5);
	}
	float2 pos = updata_pick_point_;
	if (pos.x >= ptLT.x && pos.x <= ptRB.x &&
		pos.y >= ptLT.y && pos.y <= ptRB.y)
	{
		move_point = true;
		updata_pick_point_ = m_MousePos;
		is_move_pick_point_ = true;
	}

	
	SafeReleaseArray(XAxisData);
	SafeReleaseArray(YAxisData);
	return move_point;
}

void GraphicsWindow::UpdateModel()
{
	if (m_texContainer)
	{
		m_MousePos = GetPointToScreen(m_MousePos);
		m_MousePos = make_float2(int(m_MousePos.x), int(m_MousePos.y));
		if (m_Lbutton)
		{
			if (m_ModelState == model_info::model_type::lint)
			{
				if (make_cross_object(make_float2(m_MousePos.x, m_MousePos.y), model_info::points_type::start_point))
					return;

				if (make_cross_object(make_float2(m_MousePos.x, m_MousePos.y), model_info::points_type::end_point))
					return;
			}
			else if (m_ModelState == model_info::model_type::branch_points)
			{
				if (make_cross_object(make_float2(m_MousePos.x, m_MousePos.y), model_info::points_type::branch_point))
					return;
			}
			else if (m_ModelState == model_info::model_type::line_2d || m_ModelState == model_info::model_type::equilateral_line_2d)
			{
				if (is_move_line_)
				{
					ModifyLine(true);
					return;
				}
				else
				{
					if (make_cross_object(make_float2(m_MousePos.x, m_MousePos.y), model_info::points_type::start_point))
						return;

					if (make_cross_object(make_float2(m_MousePos.x, m_MousePos.y), model_info::points_type::end_point))
						return;
				}
			}
			zoomFunc(true, mRectBuffer);
		}
		else if (m_Mbutton)
		{
			auto pos = updata_pick_point_;
			pos.x = (pos.x - m_MousePos.x) / m_lfWidth;
			pos.y = (pos.y - m_MousePos.y) / m_lfHeight;
			float4 _zf = m_zoomstack.top();
			if (m_texContainer)
			{
				_zf.x = m_zoomstack.top().x - pos.x;
				_zf.y = m_zoomstack.top().y + pos.x;
				_zf.z = m_zoomstack.top().z + pos.y;
				_zf.w = m_zoomstack.top().w - pos.y;
				m_zoomstack.top() = _zf;
				getModelMat(m_lfScale);
			}
			return;
		}
		else
		{
			if (m_ModelState == model_info::model_type::line_2d || m_ModelState == model_info::model_type::equilateral_line_2d)
			{
				if (Angio_Algorithm_.get_segmentation_line2D_instance().get_lines().size() == 0)
					return;
				auto pick_points_l = Angio_Algorithm_.get_distance_end(Angio_Algorithm_.get_lines2D(result_info::line_position::LEFT_LINE), m_MousePos);
				auto pick_points_r = Angio_Algorithm_.get_distance_end(Angio_Algorithm_.get_lines2D(result_info::line_position::RIGHT_LINE), m_MousePos);

				auto dis = -1.0;
				if (pick_points_l.distance < pick_points_r.distance)
				{
					dis = pick_points_l.distance;
					move_line_index_ = int(result_info::line_position::LEFT_LINE);
				}
				else
				{
					dis = pick_points_r.distance;
					move_line_index_ = int(result_info::line_position::RIGHT_LINE);
				}
				is_move_line_ = (dis >= 0 && dis < 3) ? true : false;
			}
		}
	}
	else if (m_triDim)
	{
		float2 pt = make_float2(m_LastPos.x(), m_LastPos.y());
		int dx1 = (m_MousePos.x - pt.x);
		int dy1 = (m_MousePos.y - pt.y);
		if (m_Lbutton)
		{
			if (pressedlShft)
			{
				m_model.rotate(dy1, 1.0f, 0.0f, 0.0f);
				m_model.rotate(dx1, 0.0f, 0.0f, 1.0f);

				m_AxisModel.rotate(dy1, 1.0f, 0.0f, 0.0f);
				m_AxisModel.rotate(dx1, 0.0f, 0.0f, 1.0f);
			}
			else if (pressedCtrl)
			{
				m_model.rotate(dy1, 0.0f, 0.0f, 1.0f);
				m_model.rotate(dx1, 0.0f, 1.0f, 0.0f);

				m_AxisModel.rotate(dy1, 0.0f, 0.0f, 1.0f);
				m_AxisModel.rotate(dx1, 0.0f, 1.0f, 0.0f);
			}
			else
			{
				m_model.rotate(dx1, 1, 0, 0);
				m_model.rotate(-dy1, 0, 1, 0);

				m_AxisModel.rotate(dx1, 1, 0, 0);
				m_AxisModel.rotate(-dy1, 0, 1, 0);
				//glRotatef(dy1, 1.0, 0.0, 0.0);
				//glRotatef(dx1, 0.0, 1.0, 0.0);

				//m_xRot += 180 * dy1;
				//m_yRot += 180 * dx1;
				//setXRotation(dx1);
				//setYRotation(dy1);
				update();
			}
			//	getModelMat(m_lfScale);
		}
		else if (m_Mbutton)
		{
			float ncdX = (2.0f * pt.x / this->geometry().width() - 1.0f);
			float ncdY = -1.0f * (2.0f * pt.y / this->geometry().height() - 1.0f);

			float ncdX1 = (2.0f * m_MousePos.x / this->geometry().width() - 1.0f);
			float ncdY1 = -1.0f * (2.0f * m_MousePos.y / this->geometry().height() - 1.0f);

			m_camera.translate(ncdY - ncdY1, ncdX - ncdX1, 0);

			//updata_pick_point_ = m_MousePos;
		}
		m_MousePos = GetPointToScreen(m_MousePos);
		m_MousePos = make_float2(int(m_MousePos.x), int(m_MousePos.y));
	}
}

float2 GraphicsWindow::GetScreenToPoint(float2 ptScreen)
{
	float2 pos;
	if (m_screenRatio >= 1)
	{
		auto _blank = (this->geometry().width() - this->geometry().height()) / 2.0f;
		pos.x = ptScreen.x / m_texContainer->getSceneWidth() * this->geometry().height() + _blank;
		pos.y = ptScreen.y / m_texContainer->getSceneHeight() * this->geometry().height();
	}
	else
	{
		auto _blank = (this->geometry().height() - this->geometry().width()) / 2.0f;
		pos.x = ptScreen.x / m_texContainer->getSceneWidth() * this->geometry().width();
		pos.y = ptScreen.y / m_texContainer->getSceneHeight() * this->geometry().width() + _blank;
	}
	pos.x = (pos.x - (this->geometry().width() / 2.0f * (1 - m_zoomstack.top().x))) * 2.0f / (m_zoomstack.top().x + m_zoomstack.top().y); //(2.0f * (1 - m_zoomstack.top().x) * 2.0f) / (m_zoomstack.top().x + m_zoomstack.top().y) - this->geometry().width();
	pos.y = (pos.y - (this->geometry().height() / 2.0f * (1 - m_zoomstack.top().w))) * 2.0f / (m_zoomstack.top().z + m_zoomstack.top().w); //(2.0f * (1 - m_zoomstack.top().x) * 2.0f) / (m_zoomstack.top().x + m_zoomstack.top().y) - this->geometry().width();
	return pos;
}

float2 GraphicsWindow::GetPointToScreen(float2 ptMouse)
{
	float2 pos = make_float2(0, 0);
	float w = 0, h = 0;
	if (m_texContainer)
	{
		w = m_texContainer->getSceneWidth();
		h = m_texContainer->getSceneHeight();
		pos.x = ptMouse.x / 2.0f * (m_zoomstack.top().x + m_zoomstack.top().y) + this->geometry().width() / 2.0f * (1 - m_zoomstack.top().x);
		pos.y = ptMouse.y / 2.0f * (m_zoomstack.top().z + m_zoomstack.top().w) + this->geometry().height() / 2.0f * (1 - m_zoomstack.top().w);

		if (m_screenRatio >= 1)
		{
			auto _blank = (this->geometry().width() - this->geometry().height()) / 2.0f;
			pos.x = float(pos.x - _blank) / this->geometry().height() * w;
			pos.y = float(pos.y) / this->geometry().height() * h;
		}
		else
		{
			auto _blank = (this->geometry().height() - this->geometry().width()) / 2.0f;
			pos.x = float(pos.x) / this->geometry().width() * w;
			pos.y = float(pos.y - _blank) / this->geometry().width() * h;
		}

	}
	else if (m_triDim)
	{
		w = m_triDim->getSceneWidth();
		h = m_triDim->getSceneHeight();
		pos.x = ptMouse.x / 2.0f * (m_zoomstack.top().z + m_zoomstack.top().w) + this->geometry().width() / 2.0f * (1 - m_zoomstack.top().w);
		pos.y = ptMouse.y / 2.0f * (m_zoomstack.top().x + m_zoomstack.top().y) + this->geometry().height() / 2.0f * (1 - m_zoomstack.top().x);

		if (m_screenRatio >= 1)
		{
			auto _blank = (this->geometry().width() - this->geometry().height()) / 2.0f;
			auto s = this->geometry().height();
			pos.x = float(pos.x - _blank) / this->geometry().height() * w;
			pos.y = float(pos.y) / this->geometry().height() * h;
		}
		else
		{
			auto _blank = (this->geometry().height() - this->geometry().width()) / 2.0f;
			pos.x = float(pos.x) / this->geometry().width() * w;
			pos.y = float(pos.y - _blank) / this->geometry().width() * h;
		}
	}
	else
	{
		return pos;
	}
	return pos;
}



void GraphicsWindow::SetEquilateraLines(std::vector<float2> CenterLine,int nViewNo)
{
	m_isRunning = true;
	int min = 0;
	Angio_Algorithm_.set_verticality_Distance(CenterLine, 0, 0, min, true);

	save_folder(model_info::folder_type::data_path);
	save_folder(model_info::folder_type::result_path);

	m_ModelState = model_info::model_type::equilateral_line_2d;
	m_isRunning = false;
}

void GraphicsWindow::ModifyLine(bool bChange)
{
	if (m_ModelState == model_info::model_type::lint || move_line_index_ == -1)
		return;
	if (Angio_Algorithm_.get_segmentation_Instance().optimal_image_id != currentImage)
		return;

	auto new_line = Angio_Algorithm_.modify_line(bChange, move_line_index_, updata_pick_point_, m_MousePos, m_nFindLineRange);

	if (m_ObjCalibrationLines.size() != 0)
		m_fUpdateLine(myGrid.x, false);
	
	DeleteObject(m_ObjModifyLine);
	if (bChange) // 마우스 왼쪽 버튼 클릭 상태
	{
		CreateObject(m_ObjModifyLine, new_line.data(), new_line.size(), Object::GraphicType::Modify_Line);
	}
	else
	{
		clear_line(model_info::clear_type::clear_equilateral_line);
		draw_lines();
	}
}

void GraphicsWindow::save_folder(model_info::folder_type type)
{
	if (type == model_info::folder_type::all_path)
	{
		save_folder(model_info::folder_type::data_path);
		save_folder(model_info::folder_type::output_path);
		save_folder(model_info::folder_type::result_path);

	}
	else if (type == model_info::folder_type::data_path)
	{
		Angio_Algorithm_.save_data_out(myGrid.x);
	}
	else if (type == model_info::folder_type::output_path)
	{
	}
	else if (type == model_info::folder_type::result_path)
	{
		std::string name = m_fileName.toStdString();
		Angio_Algorithm_.save_result_out(myGrid.x, name + "_" + std::to_string(currentImage + 1));
	}
}
void GraphicsWindow::save_dcm_file()
{
	Angio_Algorithm_.save_dcm_file(myGrid.x, mDCMHelper.get());
}

void GraphicsWindow::read_data_in(std::string strExtension)
{
	Angio_Algorithm_.read_data_in(myGrid.x, strExtension);

	{
		auto points = Angio_Algorithm_.get_points_instance();
		make_cross_object(points.start_point, model_info::points_type::start_point);
		DeleteObject(m_ObjMatchingPointsCross);
		for (int i = 0; i < points.branch_point.size(); i++)
			make_cross_object(points.branch_point[i].second, model_info::points_type::branch_point);
		make_cross_object(points.end_point, model_info::points_type::end_point);
		m_ModelState = model_info::model_type::segmentation_points;
	}

	{
		auto l_line = Angio_Algorithm_.get_lines2D(result_info::line_position::RIGHT_LINE);
		auto r_line = Angio_Algorithm_.get_lines2D(result_info::line_position::LEFT_LINE);
		auto c_line = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE);
		if (c_line.size() != 0)
		{
			DeleteObject(m_ObjLines);
			CreateObject(m_ObjLines, l_line.data(), l_line.size(), Object::GraphicType::LeftLine);
			CreateObject(m_ObjLines, r_line.data(), r_line.size(), Object::GraphicType::RightLine);
			CreateObject(m_ObjLines, c_line.data(), c_line.size(), Object::GraphicType::CenterLine);
			save_folder(model_info::folder_type::result_path);
			m_ModelState = model_info::model_type::line_2d;


			result_info::segmentation_instance segmentation_instance = Angio_Algorithm_.get_segmentation_Instance();
			segmentation_instance.set_optimal_image_id(currentImage);
			Angio_Algorithm_.set_segmentation_instance(segmentation_instance);

			auto line = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE, true);
			if (line.size() != 0)
				m_ModelState = model_info::model_type::equilateral_line_2d;
		}

	}
	
	
	//if (loadfile)
	//	m_ModelState = model_info::model_type::line_2d;
	//else
	//{
	//	if (m_ModelState != model_info::model_type::Logo_file)
	//		m_ModelState = model_info::model_type::equilateral_line_2d;
	//}
}

vector<float2> GraphicsWindow::get_equilateral(int nP, int nIndex)
{
	vector<float2>line_points_c = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE);
	auto new_line = Angio_Algorithm_.get_equilateral_line(line_points_c, nP, nIndex);
	return new_line;
}


//그래픽  
void GraphicsWindow::CreateObject(Object*& pObject, float2* _temp, int cnt, int GraphicType, int objType)
{
	if (!pObject)
		pObject = new LineObj(context());
	if (GraphicType == Object::GraphicType::Manual_Point)

	{
		((LineObj*)pObject)->setLineColor(COLOR_GREEN2);
		((LineObj*)pObject)->setCircleSize(m_nCircleSize);
	}
	((LineObj*)pObject)->setData(_temp, m_lfWidth, m_lfHeight, cnt, GraphicType, objType);
	if (objType == Object::Shape::Circle)
		((LineObj*)pObject)->setCenteroid(_temp[0].x, _temp[0].y);
}

void GraphicsWindow::CreateObject(vector<Object*>& vecObject, float2* _temp, int cnt, int GraphicType, int objType)
{
	vecObject.push_back(new LineObj(context()));
	auto _temporary1 = static_cast<LineObj*>(vecObject.back());
	if (objType == Object::Shape::Circle)
	{
		if (GraphicType == 0)
			_temporary1->setLineColor(COLOR_PINK);
		else if (GraphicType == 1)
			_temporary1->setLineColor(COLOR_ORANGE);
		else if (GraphicType == 2)
			_temporary1->setLineColor(COLOR_CYAN);
		else if (GraphicType == 6)
		{
			_temporary1->setLineColor(COLOR_GREEN2);
			_temporary1->setCircleSize(m_nCircleSize);
		}
		else
			_temporary1->setLineColor(COLOR_WHITE);

	}
	if(GraphicType == Object::GraphicType::Manual_Line)
		_temporary1->setCircleSize(m_nCircleSize);

	_temporary1->setData(_temp, m_lfWidth, m_lfHeight, cnt, GraphicType, objType);
	if (objType == Object::Shape::Circle)
		_temporary1->setCenteroid(_temp[0].x, _temp[0].y);
}

void GraphicsWindow::ModifyObject(Object*& pObject, float2* _temp, int cnt, int GraphicType, int objType)
{
	if (!pObject)
		return;

	((LineObj*)pObject)->setData(_temp, m_lfWidth, m_lfHeight, cnt, GraphicType);
}

void GraphicsWindow::ModifyObject(vector<Object*>& vecObject, float2* _temp, int nIndex, int cnt, int GraphicType, int objType)
{
	if (vecObject[nIndex])
	{
		auto _temporary1 = static_cast<LineObj*>(vecObject[nIndex]);
		_temporary1->setData(_temp, m_lfWidth, m_lfHeight, cnt, GraphicType, objType);
		if (objType == Object::Shape::Circle)
			_temporary1->setCenteroid(_temp[0].x, _temp[0].y);
	}
}

void GraphicsWindow::DeleteObject(vector<Object*>& vecObject)
{
	for (auto& v : vecObject)
	{
		v->ReleaseGLFunctions();
		v->ReleaseBuffers();
		SafeReleasePointer(v);
	}
	int cnt = vecObject.size();
	for (int i = 0; i < cnt; i++)
		vecObject.erase(vecObject.begin());
	vector<Object*>().swap(vecObject);
}

void GraphicsWindow::DeleteObject(Object*& pObject)
{
	if (pObject)
	{
		pObject->ReleaseGLFunctions();
		pObject->ReleaseBuffers();
		SafeReleasePointer(pObject);
	}
	pObject = nullptr;
}

bool GraphicsWindow::make_cross_object(float2 pos, model_info::points_type type)
{
	bool move_point = false;
	if (pos.x < 0)
		pos.x = 10;
	else if (pos.x > m_lfWidth)
		pos.x = m_lfWidth - 10;
	if (pos.y < 0)
		pos.y = 10;
	else if (pos.y > m_lfHeight)
		pos.y = m_lfHeight - 10;


	//	pos = make_float2(111, 249);
	int cnt = 11;
	int N = 5;

	float2* _temp1 = new float2[cnt];
	float2* _temp2 = new float2[cnt];
	int2 pt1 = make_int2(pos.x - N, pos.y);
	int2 pt2 = make_int2(pos.x, pos.y - N);
	for (int i = 0; i < cnt; i++)
	{
		_temp1[i] = make_float2(pt1.x + i, pt1.y);
		_temp2[i] = make_float2(pt2.x, pt2.y + i);
	}

	if (type == model_info::points_type::start_point)
	{
		if (m_ObjStartPointCross.size() != 0)
		{
			move_point = get_pick_point_move(type, 0);
			if(!move_point) return move_point;
			DeleteObject(m_ObjStartPointCross);
		}
		CreateObject(m_ObjStartPointCross, _temp1, cnt, Object::GraphicType::Start_Point);
		CreateObject(m_ObjStartPointCross, _temp2, cnt, Object::GraphicType::Start_Point);
	}
	else if (type == model_info::points_type::end_point)
	{
		if (m_ObjEndPointCross.size() != 0)
		{
			move_point = get_pick_point_move(type, 0);
			if (!move_point) return move_point;
			DeleteObject(m_ObjEndPointCross);
		}
		CreateObject(m_ObjEndPointCross, _temp1, cnt, Object::GraphicType::End_Point);
		CreateObject(m_ObjEndPointCross, _temp2, cnt, Object::GraphicType::End_Point);
	}
	else if (type == model_info::points_type::branch_point)
	{
		if (m_ObjMatchingPointsCross.size() < 4)
		{
			CreateObject(m_ObjMatchingPointsCross, _temp1, cnt, Object::GraphicType::MatchingPoint);
			CreateObject(m_ObjMatchingPointsCross, _temp2, cnt, Object::GraphicType::MatchingPoint);
		}
		else
		{
			int move_index = 0;
			for (int i = 0; i < m_ObjMatchingPointsCross.size() / 2; i++)
			{
				move_point = get_pick_point_move(type, i);
				if (move_point)
				{
					move_index = i;
					break;
				}
			}
			if (move_point)
			{
				ModifyObject(m_ObjMatchingPointsCross, _temp1, move_index * 2, cnt, Object::GraphicType::MatchingPoint);
				ModifyObject(m_ObjMatchingPointsCross, _temp2, (move_index * 2) + 1, cnt, Object::GraphicType::MatchingPoint);
			}
		}
	}
	SafeReleaseArray(_temp1);
	SafeReleaseArray(_temp2);

	return move_point;
}

void GraphicsWindow::clear_line(model_info::clear_type type)
{
	if (type == model_info::clear_type::clear_calibration_line)
	{
		DeleteObject(m_ObjCalibrationLines);
	}
	else if (type == model_info::clear_type::clear_line)
	{
		DeleteObject(m_ObjLines);
		m_ModelState = model_info::model_type::segmentation_points;
	}
	else if (type == model_info::clear_type::clear_equilateral_line)
	{
		DeleteObject(m_ObjLines);
		m_ModelState = model_info::model_type::line_2d;
	}
	else 
	{
		DeleteObject(m_ObjLines);
		DeleteObject(m_ObjEdge);
		DeleteObject(m_Objlebeling);
		DeleteObject(m_ObjCalibrationLines);
		m_ModelState = model_info::model_type::lint;
		Angio_Algorithm_.result_clear();
	}
	update();
}

void GraphicsWindow::clear_points(model_info::clear_type ntype)
{
	DeleteObject(m_ObjStartPointCross);
	DeleteObject(m_ObjEndPointCross);
	DeleteObject(m_ObjMatchingPointsCross);
}

void GraphicsWindow::draw_points()
{
	if (m_ModelState == model_info::model_type::lint)
		return;
	clear_points();

	for (int i = 0; i < Angio_Algorithm_.get_points_instance().branch_point.size(); i++)
		make_cross_object(Angio_Algorithm_.get_points_instance().branch_point[i].second, model_info::points_type::branch_point);

#ifdef _DEBUG
	float2 posStart = Angio_Algorithm_.get_points_instance().start_point;
	float2 posEnd = Angio_Algorithm_.get_points_instance().end_point;
	make_cross_object(posStart, model_info::points_type::start_point);
	make_cross_object(posEnd, model_info::points_type::end_point);
	return;
#endif
	if (m_ModelState == model_info::model_type::line_2d || m_ModelState == model_info::model_type::equilateral_line_2d)
	{
		float2 posStart = Angio_Algorithm_.get_points_instance().start_point;
		float2 posEnd = Angio_Algorithm_.get_points_instance().end_point;
		make_cross_object(posStart, model_info::points_type::start_point);
		make_cross_object(posEnd, model_info::points_type::end_point);
	}
}

void GraphicsWindow::draw_lines()
{
	if (m_ModelState == model_info::model_type::equilateral_line_2d  || m_ModelState == model_info::model_type::line_2d)
	{
		auto new_line_r = Angio_Algorithm_.get_lines2D(result_info::line_position::RIGHT_LINE);
		auto new_line_l = Angio_Algorithm_.get_lines2D(result_info::line_position::LEFT_LINE);
		auto new_line_c = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE);
		if (new_line_c.size() == 0)
			return;
	
		CreateObject(m_ObjLines, new_line_l.data(), new_line_l.size(), Object::GraphicType::LeftLine);
		CreateObject(m_ObjLines, new_line_r.data(), new_line_r.size(), Object::GraphicType::RightLine);
		CreateObject(m_ObjLines, new_line_c.data(), new_line_c.size(), Object::GraphicType::CenterLine);

		make_cross_object(new_line_c[new_line_c.size() / 4], model_info::points_type::branch_point);
		make_cross_object(new_line_c[new_line_c.size() / 2], model_info::points_type::branch_point);
	}
}

void GraphicsWindow::draw_calibration_line(vector<float2> INTCS, vector<float2> INTCE)
{
	DeleteObject(m_ObjCalibrationLines);
	CreateObject(m_ObjCalibrationLines, INTCS.data(), INTCS.size(), Object::GraphicType::Calibration_StartLine);
	CreateObject(m_ObjCalibrationLines, INTCE.data(), INTCE.size(), Object::GraphicType::Calibration_EndLine);

	auto point_instance = get_points_instance();
	point_instance.calibration_point.clear();
	point_instance.calibration_point.push_back(INTCS);
	point_instance.calibration_point.push_back(INTCE);
	Angio_Algorithm_.set_points_instance(point_instance);
	
}

void GraphicsWindow::moveAlongEpiline(vector<float2> points)
{
	if (points.size() < 2)
		return;
	
	if (points[0].x != -1 && points[0].y !=-1)
		make_cross_object(points[0], model_info::points_type::start_point);
	if (points[1].x != -1 && points[1].y != -1)
		make_cross_object(points[1], model_info::points_type::end_point);
}

bool GraphicsWindow::calculatePointOnEpiline(vector<float2>& output_points)
{
	auto bcheck = model_info::points_type::none_point;

	if (get_points_instance().calibration_point.size() == 0)
		return false;

	auto centline = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE);
	if (centline.size() == 0)
		return false;
	
	auto start_line = get_points_instance().calibration_point[0];
	auto end_line = get_points_instance().calibration_point[1];
	float2 ptIntersect_StartLine = make_float2(0, 0);
	float2 ptIntersect_EndLine = make_float2(0, 0);

	for (int i = 0; i < centline.size() - 1; i++)
	{
		bool isintersect = Angio_Algorithm_.get_intersect_point(centline[i], centline[i + 1], start_line[0], start_line[1], ptIntersect_StartLine);
		if (isintersect)
		{
			bcheck = model_info::points_type::start_point;
			output_points.push_back(ptIntersect_StartLine);
			break;
		}
	}

	vector<float2> end_points;
	for (int i = centline.size() - 1; i > 0; i--)
	{
		bool isintersect = Angio_Algorithm_.get_intersect_point(centline[i - 1], centline[i], end_line[0], end_line[1], ptIntersect_EndLine);
		if (isintersect)
			end_points.push_back(ptIntersect_EndLine);
	}

	if (end_points.size() != 0)
	{
		auto _dm = Angio_Algorithm_.get_distance_end(end_points, Angio_Algorithm_.get_points_instance().end_point);
		output_points.push_back(_dm.pos);
	}

	m_ModelState = model_info::model_type::epiline_2d;

	return bcheck == model_info::points_type::start_point;
}

void GraphicsWindow::SetStenosisorBranchPointIds(vector<int> vecIds)
{
	auto line_points = Angio_Algorithm_.get_endPoint3D_result_Instance().center_line_points;
	if (vecIds.size() < 3 || line_points.size() == 0)
		return;

	auto point = new AxisLineObj(context());
	point->SetAxisColor(Object::GraphicLines::StenosisorBranch);
	point->SetScale(m_lfScale);
	vector<float3> pts;
	pts.push_back(line_points[vecIds[1]]);
	pts.push_back(line_points[vecIds[2]]);
	point->SetCenterLine(pts);
	m_vec3DLine.push_back(point);
}

void GraphicsWindow::set_end_points(std::string fileName, model_info::find_endpoint_type type)
{
	if (m_ModelState == model_info::model_type::equilateral_line_2d)
		Angio_Algorithm_.set_end_points(type, fileName);
}

void GraphicsWindow::ManualSegmentation(model_info::segmentation_model_type model_type, model_info::segmentation_manual_type type)
{
	if (type == model_info::segmentation_manual_type::start_end_findline)
	{
		Angio_Algorithm_.create_buffer(m_lfWidth, m_lfHeight);
		Angio_Algorithm_.set_file_niigz(myGrid.x, mDCMHelper.get());

		auto initScene = ((TexObj*)m_texContainer)->Scene();

		for (int i = 0; i < m_lfWidth * m_lfHeight; i++)
			Angio_Algorithm_.get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE)[i] = (initScene[i] * 1.0f);

		if (Angio_Algorithm_.manual_segmentation(true))
		{
			Angio_Algorithm_.segmentation(model_info::segmentation_manual_type::start_end_findline);
			Angio_Algorithm_.segmentation(model_info::segmentation_manual_type::branch_findline);
			m_ModelState = model_info::model_type::branch_points;
		}
		return;
	}
	else if(type == model_info::segmentation_manual_type::find_centerline)
	{
		Angio_Algorithm_.segmentation(type);
		m_ModelState = model_info::model_type::line_2d;
		auto Instance = Angio_Algorithm_.get_segmentation_Instance();
		Instance.set_optimal_image_id(currentImage);
		Angio_Algorithm_.set_segmentation_instance(Instance);
	}
	else if (type == model_info::segmentation_manual_type::run_manual_ai)
	{
		Angio_Algorithm_.set_file_niigz(myGrid.x, mDCMHelper.get());
		if(model_type != model_info::segmentation_model_type::none)
			Angio_Algorithm_.segmentation(model_type, model_info::segmentation_exe_type::run_centerline, myGrid.x);
		else
			Angio_Algorithm_.segmentation_output(model_info::segmentation_exe_type::run_centerline, myGrid.x);
	}
	else if(type == model_info::segmentation_manual_type::load_file_line)
	{
		Angio_Algorithm_.create_buffer(m_lfWidth, m_lfHeight);
		Angio_Algorithm_.set_file_niigz(myGrid.x, mDCMHelper.get());
		auto initScene = ((TexObj*)m_texContainer)->Scene();

		for (int i = 0; i < m_lfWidth * m_lfHeight; i++)
			Angio_Algorithm_.get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE)[i] = (initScene[i] * 1.0f);

		Angio_Algorithm_.manual_segmentation(false);
	}
}

void GraphicsWindow::AutoSegmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type)
{
	Angio_Algorithm_.set_file_niigz(myGrid.x, mDCMHelper.get());
	Angio_Algorithm_.create_buffer(m_lfWidth, m_lfHeight);

	auto initScene = ((TexObj*)m_texContainer)->Scene();
	
	for (int i = 0; i < m_lfWidth * m_lfHeight; i++)
		Angio_Algorithm_.get_float_buffer(model_info::image_float_type::ORIGIN_IMAGE)[i] = (initScene[i] * 1.0f) / 255.0f;

	Angio_Algorithm_.segmentation(model_type, model_info::segmentation_exe_type::run_outlines, myGrid.x);
	Angio_Algorithm_.segmentation(model_type, model_info::segmentation_exe_type::run_endpoints, myGrid.x);

	int id = Angio_Algorithm_.get_segmentation_Instance().optimal_image_id;
	if (id == -1)
		return;

	SetOpenImageId(id);
	
	auto c = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE);
	auto r = Angio_Algorithm_.get_lines2D(result_info::line_position::RIGHT_LINE);
	auto l = Angio_Algorithm_.get_lines2D(result_info::line_position::LEFT_LINE);

	DeleteObject(m_Objlebeling);
	if (currentImage < Angio_Algorithm_.get_segmentation_Instance().get_labeling_points().size())
	{
		auto lebeling_data = Angio_Algorithm_.get_segmentation_Instance().get_labeling_point(currentImage).second;
		CreateObject(m_Objlebeling, lebeling_data.data(), lebeling_data.size(), Object::GraphicType::Manual_Line, Object::Shape::Points);
	}
	auto end_points = Angio_Algorithm_.get_segmentation_line2D_instance().end_points;
	int target = currentImage;
	auto it = std::find_if(end_points.begin(), end_points.end(),
		[target](const std::pair<int, float2>& element) {
			return element.first == target;
		});

	if (it != end_points.end()) 
	{
		vector<float2> point;
		point.push_back(it->second);
		CreateObject(m_Objlebeling, point.data(), point.size(), Object::GraphicType::Manual_Point, Object::Shape::Points);
	}

	m_ModelState = model_info::model_type::line_2d;
}

vector<float> GraphicsWindow::GetMinimumRadius()
{
	return Angio_Algorithm_.get_minimum_radius();
}

bool GraphicsWindow::LineUndo()
{
	if (Angio_Algorithm_.line_undo())
	{
		auto l = Angio_Algorithm_.get_lines2D(result_info::line_position::LEFT_LINE, false);//((LineObj*)this->m_ObjLines[Object::GraphicType::LeftLine])->GetData();
		auto r = Angio_Algorithm_.get_lines2D(result_info::line_position::RIGHT_LINE, false);//((LineObj*)this->m_ObjLines[Object::GraphicType::LeftLine])->GetData();
		auto c = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE, false);//((LineObj*)this->m_ObjLines[Object::GraphicType::LeftLine])->GetData();
		((LineObj*)this->m_ObjLines[Object::GraphicType::CenterLine])->setData(c.data(), m_lfWidth, m_lfHeight, c.size(), Object::GraphicType::CenterLine);
		((LineObj*)this->m_ObjLines[Object::GraphicType::LeftLine])->setData(l.data(), m_lfWidth, m_lfHeight, l.size(), Object::GraphicType::LeftLine);
		((LineObj*)this->m_ObjLines[Object::GraphicType::RightLine])->setData(r.data(), m_lfWidth, m_lfHeight, r.size(), Object::GraphicType::RightLine);
	}
	else
	{
		return false;
	}

	//if (m_ModelState == model_info::model_type::Epiline2D)
	//	return true;
	//int	N = 3;
	//for (int i = 0; i < N; i++)
	//{
	//	if (m_stacklineUndo.size() == 0)
	//		return false;
	//	auto line = m_stacklineUndo[m_stacklineUndo.size() - 1];
	//	int nLineIndex = line.GetGraphicType();
	//	set_line_data(AngioFFR_Define::line_position(nLineIndex), line.GetData());
	//	m_stacklineUndo.erase(m_stacklineUndo.end() - 1);
	//}
	//SetCenterLine();
	return true;
}


void GraphicsWindow::SetSelectWindow(bool bSelect)
{
	m_bSelectGraphic = bSelect;
}


std::vector<int> GraphicsWindow::calculate_chart_axis()
{
	auto _center = Angio_Algorithm_.get_lines2D(result_info::line_position::CENTER_LINE, true);
	if (_center.size() == 0)
		return std::vector<int>{};

	std::vector<int> X;
	auto id_start = Angio_Algorithm_.get_distance_end(_center, get_points_instance().start_point).id;
	auto id_end = Angio_Algorithm_.get_distance_end(_center, get_points_instance().end_point).id;
	X.push_back(id_start);
	for (int i = 0; i < get_points_instance().branch_point.size(); i++)
	{
		float2 pos = get_points_instance().branch_point[i].second;
		auto id = Angio_Algorithm_.get_distance_end(_center, pos).id;
		X.push_back(id);
	}
	X.push_back(id_end);
	return X;
}


//명암조절
void GraphicsWindow::SetWindowLevel(int nWC, int nWW)
{
	if (!m_texContainer)
		return;

	((TexObj*)m_texContainer)->setWindowCenter(nWC);
	((TexObj*)m_texContainer)->setWindowWidth(nWW);
}

int GraphicsWindow::GetWindowCenter()
{
	int nWL = WL;
	if (m_texContainer)
		nWL = ((TexObj*)m_texContainer)->getWindowCenter();
	return nWL;
}

int GraphicsWindow::GetWindowWidth()
{
	int nWW = WW;
	if (m_texContainer)
		nWW = ((TexObj*)m_texContainer)->getWindowWidth();
	return nWW;
}



void GraphicsWindow::SaveAs(QString dirName,int nStartIndex,int nEndIndex)
{
	if (nEndIndex == -1)
		nEndIndex = numberImage;
	if (nStartIndex == -1)
		nStartIndex = 0;

	for (int i = nStartIndex; i < nEndIndex; i++)
	{
		unsigned char* _ptr = new unsigned char[m_lfWidth * m_lfHeight];
		auto pData = ((TexObj*)m_texContainer)->getCurrentImage(i);

		for (int n = 0; n < m_lfWidth * m_lfHeight; n++)
		{
			auto val = (std::clamp(255.0 * ((pData[n] - (GetWindowCenter() - 0.5)) / (GetWindowWidth() - 1) + 0.5), 0.0, 255.0));
			_ptr[n] = val;
		}

		auto mImage = (new QImage(_ptr, m_lfWidth, m_lfHeight, QImage::Format::Format_Grayscale8));
		auto mPixmap = new QPixmap();
		mPixmap->convertFromImage(*mImage);

		auto FilePath = dirName  +  QString::number(i+1) + ".bmp";
		mImage->save(FilePath, "bmp");
		SafeReleaseArray(_ptr);
	}
}


//3D
void GraphicsWindow::Get3DCenterline()
{
	if (!m_triDim)
		return;
	Angio_Algorithm_.set_center_points_3D(((TriObj*)m_triDim)->GetCenterPos());
	//auto instance_3D = Angio_Algorithm_.get_endPoint3D_result_Instance();
	//instance_3D.center_line_points.clear();
	//
	//QString strExtension = QDir::currentPath().section("/", 0, -2);
	//QString centerlineFilePath = strExtension + QString("\\output\\centerline_c3r22_P_SW1.dat");
	//QString strName = strExtension + QString("\\data");
	//int nc = 0;
	//auto posCenter = ((TriObj*)m_triDim)->GetCenterPos();
	//
	//FILE* fp_in1 = fopen(centerlineFilePath.toStdString().c_str(), "r");
	//if (!fp_in1)
	//	return;
	//fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.
	//double* xc = (double*)malloc(sizeof(double) * (nc + 1));
	//double* yc = (double*)malloc(sizeof(double) * (nc + 1));
	//double* zc = (double*)malloc(sizeof(double) * (nc + 1));
	//
	//for (int i = 0; i < nc; i++)
	//{
	//	float3 pos = make_float3(0, 0, 0);
	//	//fscanf(fp_in1, "%lf %lf %lf\n", &pos.x, &pos.y, &pos.z);
	//	fscanf(fp_in1, "%lf %lf %lf\n", &xc[i], &yc[i], &zc[i]);
	//	pos = make_float3(xc[i], yc[i], zc[i]);
	//	xc[i] = (pos.x - posCenter.x) * 0.05f;
	//	yc[i] = (pos.y - posCenter.y) * 0.05f;
	//	zc[i] = -(pos.z - posCenter.z) * 0.05f;
	//	pos = make_float3(xc[i], yc[i], zc[i]);
	//	instance_3D.center_line_points.push_back(pos);
	//}
	//
	//
	//FILE* f = fopen(QString(strName + QString("\\EndPointIDs.dat")).toLocal8Bit().data(), "r");
	//vector<pair<int, int>> vecEndPointIds;
	//if (f)
	//{
	//	int n = 0;
	//	fscanf(f, "%d\n", &n);
	//	for (int i = 0; i < n; i++)
	//	{
	//		int id = 0, imgIndex = 0;
	//		fscanf(f, "%d,%d\n", &imgIndex, &id);
	//		vecEndPointIds.push_back(make_pair(imgIndex, id));
	//	}
	//}
	//else
	//	return;
	//
	//if (vecEndPointIds.size() == 0)
	//	return;
	//
	//// 두 3D 점 사이의 거리 계산 함수
	//auto GetDistance = [&](float3 p1, float3 p2)
	//{
	//	float dx = p2.x - p1.x;
	//	float dy = p2.y - p1.y;
	//	float dz = p2.z - p1.z;
	//	return std::sqrt(dx * dx + dy * dy + dz * dz);
	//};
	//
	//auto CalculateAndStore3DPointData = [&](int idx, int frame, int nStartId, int nEndId) //idx 이미지 frame 번호 
	//{
	//	float _dis = 0;
	//	auto idMax = qMax(nStartId, nEndId);
	//	auto idMin = qMin(nStartId, nEndId);
	//
	//	for (int j = idMin; j < idMax; j++)
	//		_dis += GetDistance(Get3DCoord(instance_3D.center_line_points[j]), Get3DCoord(instance_3D.center_line_points[j + 1]));
	//
	//	if (nEndId - nStartId < 0)
	//		_dis *= -1;
	//	_dis = _dis * (frame * 15);
	//	result_info::end_point_info info(_dis, idx, nEndId, instance_3D.center_line_points[nEndId]);
	//	instance_3D.sort_id_point.push_back(info);
	//	instance_3D.sort_center_id_point.push_back(info);
	//	_dis = 0;
	//};
	//
	//int frame = 1;
	//
	//if (vecEndPointIds[0].second != 0)
	//{
	//	CalculateAndStore3DPointData(vecEndPointIds[0].first, frame, 0, vecEndPointIds[0].second);
	//
	//	for (int i = 1; i < vecEndPointIds.size(); i++)
	//	{
	//		frame = vecEndPointIds[i].first - vecEndPointIds[i - 1].first;
	//		CalculateAndStore3DPointData(vecEndPointIds[i].first, frame, vecEndPointIds[i - 1].second, vecEndPointIds[i].second);
	//	}
	//}
	//else
	//{
	//	for (int i = 1; i < vecEndPointIds.size(); i++)
	//	{
	//		frame = vecEndPointIds[i].first - vecEndPointIds[i - 1].first;
	//		CalculateAndStore3DPointData(vecEndPointIds[i].first, frame, vecEndPointIds[i - 1].second, vecEndPointIds[i].second);
	//	}
	//}
	//
	//sort(instance_3D.sort_center_id_point.begin(), instance_3D.sort_center_id_point.end(),
	//	[](const result_info::end_point_info& a, const result_info::end_point_info& b) {
	//		return a.center_line_id < b.center_line_id;
	//	});
	//sort(instance_3D.sort_id_point.begin(), instance_3D.sort_id_point.end(),
	//	[](const result_info::end_point_info& a, const result_info::end_point_info& b) {
	//		return a.frame_id < b.frame_id;
	//	});
	//
	//{
	//	auto file_path = program_path_ + "\\output\\distance.dat";
	//	FILE* f1 = fopen(file_path.c_str(), "w");
	//	fprintf(f1, "%f\n", 0.0);
	//	for (int i = 0; i < instance_3D.sort_center_id_point.size(); i++)
	//	{
	//		float dis_sum = 0;
	//		for (int j = 0; j < instance_3D.sort_id_point[i].center_line_id; j++)
	//			dis_sum += GetDistance(Get3DCoord(instance_3D.center_line_points[j]), Get3DCoord(instance_3D.center_line_points[j + 1]));
	//		fprintf(f1, "%f\n", dis_sum);
	//		instance_3D.frame_end_point.push_back(instance_3D.sort_id_point[i].pos3D);
	//	}
	//	fclose(f1);
	//}
	//Angio_Algorithm_.set_endPoint3D_result_Instance(instance_3D);
}

float3 GraphicsWindow::GetWinCoord(float3 pos,int nAxis)
{
	auto cam= m_camera;
	auto mvp = m_proj * cam * m_model;
	if (nAxis == TextType::AxisName)
		mvp = m_proj * m_AxisCamera * m_AxisModel;
	else if(nAxis == TextType::fixTextName)
	{
		cam.setToIdentity();
		cam.lookAt(QVector3D(0, 0, -0.0),
			QVector3D(0, 0, 2.0f),
			QVector3D(-1, 0, 0));
		auto madel = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
		auto _t = m_zoomstack.top();
		auto f = 3.2;
		const float _f = f * m_screenRatio;
		if (m_screenRatio < 1)
			madel.ortho(-(_f)*_t.x, (_f)*_t.y, _f * _t.z, -_f * _t.w, -2.0f * f, 2.0f * f);
		else
			madel.ortho(-f * _t.x, f * _t.y, (_f)*_t.z, -(_f)*_t.w, -2.0f * f, 2.0f * f);
		mvp = m_proj * cam * madel;
	}
	auto cliping_pos = mvp * QVector4D(pos.x, pos.y, pos.z, 1);
	cliping_pos /= cliping_pos.w();
	cliping_pos.setZ(cliping_pos.z()); //보정된 좌표계는 z방향 다르다
	//if (!isAxis)
	//	cliping_pos = QVector4D(0,0,0,0);
	auto w= this->geometry().width();
	auto H = this->geometry().height();
	pos.x = (cliping_pos.x() + 1) * this->geometry().width() / 2.0f;
	pos.y = (cliping_pos.y() + 1) * this->geometry().height()/ 2.0f;
	pos.z = cliping_pos.z();
	return pos;
}

void GraphicsWindow::DrawBitmapText(const char* str, float3 pos, int nAxis)
{
	auto win_coord = GetWinCoord(pos, nAxis);
	glLoadIdentity();
	glOrtho(0, this->geometry().width(), 0, this->geometry().height(), -1, 1);
	auto x = 0.1 / m_lfScale;
	if (nAxis == TextType::AxisName)
		x = 0;
	
	glTranslatef(win_coord.x, win_coord.y, win_coord.z);

	glRasterPos3f(0, 0, 0);
	while (*str)
	{
		//GLUT_BITMAP_TIMES_ROMAN_24 폰트를 사용하여 문자열을 그린다.
		if (nAxis == TextType::AxisName)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *str++);
		else if(nAxis == TextType::fixTextName)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *str++);
		else
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *str++);
	}
	//glDisable(GL_LIGHTING);
	//glDisable(GL_LIGHT0);
	//glutSolidSphere(100, 10, 10);
}

float3 GraphicsWindow::Get3DPosition(float2 pt)
{
	auto ss = context();// ->functions();
	if (!context())
		return make_float3(0, 0, 0);
	int viewport[4];
	auto s = this->geometry();
	glGetIntegerv(GL_VIEWPORT, viewport);

	if (s.width() != viewport[2])
		viewport[2] = s.width();
	if (s.width() != viewport[3])
		viewport[3] = s.height();

	double winX = (double)pt.x;
	double winY = viewport[3] - (double)pt.y;


	QMatrix4x4 viewMatrix = m_camera * m_model;
	QVector3D Z(0, 0, m_lfScale * m_screenRatio); // instead of 0 for x and y i need worldPosition.x() and worldPosition.y() ....
	Z = Z.project(viewMatrix, m_proj, QRect(0, 0, viewport[2], viewport[3]));
	QVector3D worldPosition = QVector3D(winX, winY, Z.z()).unproject(viewMatrix, m_proj, QRect(viewport[0], viewport[1], viewport[2], viewport[3]));

	return make_float3(worldPosition.x(), worldPosition.y(), worldPosition.z());
}

void GraphicsWindow::InitScreen(bool is3d)
{
	//m_screenRatio = float(this) / h;
	m_zoomstack.clear();
	m_zoomstack.push(m_zoomfactor);
	m_camera.setToIdentity();

	QRect rec = this->geometry();

	if (m_triDim)
	{
		m_lfScale = ((TriObj*)m_triDim)->getMaxPlaneValue();
		getModelMat(m_lfScale);
		//if (p0 != 0 && p1 != 0 && s0 != 0 && s1 != 0)
		{
			int v = 1;

			if (p0 < 0 && p1 < 0)
				v = -1;
			else if (p0 * p1 < 0)
				v = -1;

			if (!m_bFFShow)
			{
				m_model.rotate(-p0 + p1, -1.0f, 0, v > 0 ? 1 : -1);
				m_model.rotate(-s0 + s1, 0.0f, -1.0f, 0.0f);
				m_AxisModel.rotate(-p0 + p1, -1.0f, 0, v > 0 ? 1 : -1);
				m_AxisModel.rotate(-s0 + s1, 0.0f, -1.0f, 0.0f);
			}
			else
			{
				m_model.rotate(-p0 + p1, -1.0f, 0, v < 0 ? 1 : -1);
				m_model.rotate(-s0 + s1, 0.0f, -1.0f, 0.0f);

				m_AxisModel.rotate(-p0 + p1, -1.0f, 0, v < 0 ? 1 : -1);
				m_AxisModel.rotate(-s0 + s1, 0.0f, -1.0f, 0.0f);
			}
		}
		m_camera.setToIdentity();
		m_camera.lookAt(QVector3D(0, 0, -0.0),
			QVector3D(0, 0, 2.0f),
			QVector3D(-1, 0, 0));

		m_AxisCamera = m_camera;
		m_AxisCamera.translate(0.5, 0.7, 0);

		float3 pos = Get3DPosition(make_float2(rec.width() / 12, rec.height() * 0.75));

		repaint();
		update();
	}

	if (m_texContainer)
		getModelMat(m_lfScale);
}

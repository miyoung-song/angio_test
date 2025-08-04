#include "windowFFR3D.h"

windowFFR3D::windowFFR3D(QWidget * mainWindow)
	: QDialog::QDialog(mainWindow, Qt::Dialog)
{
	setupUi(this);
	setFocusPolicy(Qt::FocusPolicy::ClickFocus);
	setAcceptDrops(true);
	uiPushButtonSave->setIcon(QIcon("icons/png/OutputFile_white_55x55.png"));
	QSize size(60, 60);
	uiPushButtonSave->setIconSize(size);
	m_nWidth = 700;
	m_nHeight = 600;
	this->setFixedSize(1400, 600);

//	layout()->setSizeConstraint(QLayout::SetFixedSize);
}

windowFFR3D::~windowFFR3D()
{
	clear_result_view();
}

void windowFFR3D::reject()
{
	if (isReject)
	{
		QDialog::reject();
	}
}

void windowFFR3D::closeEvent(QCloseEvent* event)
{
	m_bShow = false;
}

void windowFFR3D::clear_result_view()
{
	if (mGraphics)
	{
		delete mGraphics;
		mGraphics = nullptr;
	}

	if (mChartForm)
	{
		delete mChartForm;
		mChartForm = nullptr;
	}

	m_bShow = false;
	m_isCreate = false;
	this->accept();
}

void windowFFR3D::SetLog(FILE* (&f), QString FileName, QString str, int ntype, bool isclose)
{
	return;
	if (!f)
	{
		QString strExtension = QDir::currentPath().section("/", 0, -2);
		QString FilePath = strExtension + QString("\\data");
		if (!QFile::exists(FilePath))
			QDir().mkdir(FilePath);

		if (ntype == 1) //3d
			FileName += "_3D";
		else
			FileName += "_FFR";

		auto path = (FilePath + QString("\\Log_") + FileName + QString(".dat"));
		if (QFile::exists(path))
			QFile::remove(path);
		f = fopen((path.toLocal8Bit().data()), "wt");
	}
	fprintf(f, "%s\n", str.toLocal8Bit().data());
	if (isclose)
		fclose(f);
}

void windowFFR3D::SetResultFFRView(int ntype, QString filename)
{
	m_isCreate = false;
	FILE* f = NULL;

	if (!mGraphics)
	{
		SetLog(f, "Result", "new GraphicsWindow", ntype);

		mGraphics = new GraphicsWindow(gridLayoutWidget_3);
		mGraphics->SetGridIndex(0, m_nVeiwIndex);
		mGraphics->SetShowFFR(m_bFFRRun);

		if (!mGraphics->Parsing(filename))
		{
			//clear_result_view();
			//return;
		}
		Sleep(100);

		SetLog(f, "Result", "parsing", ntype);

		mGraphics->SetViewSize(m_nWidth, m_nHeight);
		mGraphics->setMaximumHeight(m_nHeight - 100);

		mGraphics->InitScreen(true);
		SetLog(f, "Result", "InitScreen", ntype);

		if (uiGridView1)
			uiGridView1->addWidget(mGraphics, 0, 0);
		SetLog(f, "Result", "add widget", ntype);
		m_isCreate = true;
	}
	else
	{
		mGraphics->InitScreen(true);
		SetLog(f, "Result", "InitScreen", ntype);
	}
	SetLog(f, "Result", "end", ntype,true);

	update();
}


void windowFFR3D::SetResultChart(bool b3D, std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY)
{
	//if (!m_isCreate)
	//	return;

	if (b3D)
	{
		if (mChartForm)
		{
			delete mChartForm;
			mChartForm = nullptr;
		}
		if (mGraphics && m_bFFRRun)
			mGraphics->SetStenosisorBranchPointIds(X);
		//mGraphics->SetStenosisorBranchPointIds(X);

		QChart* chart = new QChart();
		mChartForm = new ChartView(chart);
		mChartForm->setRenderHint(QPainter::Antialiasing);
		if(vecAxisX.size() != 0 && vecAxisY.size() != 0)
			mChartForm->Set3DChart(X, vecAxisX, vecAxisY);
		mChartForm->setMaximumWidth(m_nWidth + 150);
		mChartForm->setMaximumHeight(m_nHeight);
		uiGridView1->addWidget(mChartForm, 0, 1);
	}
	else
	{

		mChartForm->SetFFRChart(X, vecAxisX, vecAxisY);
	}
}

void windowFFR3D::SetMulPSAngle(const float& _p0, const float& _s0, const float& _p1, const float& _s1, const float& _v)
{
	if(mGraphics)
		mGraphics->SetMulPSAngle(_p0, _s0, _p1, _s1, _v);
}


void windowFFR3D::SetGridIndex(int viewIndex)
{
	m_nVeiwIndex = viewIndex;
}


void windowFFR3D::SetFFR(bool bRun)
{
	m_bFFRRun = bRun;
}

void windowFFR3D::Show(bool bshow)
{
	m_bShow = true;
}



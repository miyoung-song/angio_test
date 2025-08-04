#include "ViewClass.h"

ViewClass::ViewClass(QWidget* parent) : QWidget(parent)
{
	setupUi(this);
	setFocusPolicy(Qt::FocusPolicy::ClickFocus);
	setAcceptDrops(true);
	gridLayoutWidget_3->setAcceptDrops(true);
	m_nCursorView = make_int2(0, 0);
	mChecker = make_unique<bool[]>(256);

	connect(uiPushButtonFFR3D, &QAbstractButton::clicked, this, &ViewClass::SetFFR3D);

	uiPushButtonEndDiastolic->setIcon(QIcon("icons/png/End_white_48x48.png"));
	uiPushButtonFFR3D->setIcon(QIcon("icons/png/save_white_48x48.png"));

	QSize size(50, 50);
	uiPushButtonEndDiastolic->setIconSize(QSize(size.width() + 25, size.height() + 25));
	uiPushButtonFFR3D->setIconSize(QSize(size.width() + 50, size.height() + 100));
	uiPushButtonEndDiastolic->setVisible(false);

	for (int i = 0; i < VIEW_COL; i++)
		setEmptyViewer(i);

	auto program_path_ = std::filesystem::current_path().string();
	size_t pos = program_path_.find_last_of('\\');
	if (pos != std::string::npos)
		program_path_ = program_path_.substr(0, pos);

	m_strProgramPath = program_path_;

	QString strExtension = QDir::currentPath().section("/", 0, -2);
	strExtension = FromUnicode(strExtension);

	if (!QFile::exists(strExtension + "\\case"))
		QDir().mkdir(strExtension + "\\case");

	if (!QFile::exists(strExtension + "\\new_result"))
		QDir().mkdir(strExtension + "\\new_result");

}

std::string ViewClass::convertEucKrToUtf8(const std::string& eucKrStr)
{
	// eucKR -> wide string (wchar_t) 변환기

	std::string utf8Str;
	for (size_t i = 0; i < eucKrStr.length(); ++i) {
		unsigned char byte1 = static_cast<unsigned char>(eucKrStr[i]);

		if (byte1 >= 0xA1 && byte1 <= 0xFE) { // 두 바이트 문자 시작
			unsigned char byte2 = static_cast<unsigned char>(eucKrStr[++i]);
			int code = ((byte1 - 0xA1) * 0xBF) + (byte2 - 0xA1);

			// UTF-8 변환
			if (code < 0x800) {
				utf8Str.push_back(0xC0 | (code >> 6));
				utf8Str.push_back(0x80 | (code & 0x3F));
			}
			else {
				utf8Str.push_back(0xE0 | (code >> 12));
				utf8Str.push_back(0x80 | ((code >> 6) & 0x3F));
				utf8Str.push_back(0x80 | (code & 0x3F));
			}
		}
		else { // 단일 바이트 문자 처리
			utf8Str.push_back(byte1);
		}
	}
	return utf8Str;
}

QString ViewClass::FromUnicode(QString encodedString)
{
	QByteArray ary = encodedString.toUtf8();
	QTextCodec* codec = QTextCodec::codecForName("eucKR");
	encodedString = codec->toUnicode(encodedString.toLocal8Bit().constData());
	return encodedString;
}

//windowImage 클래스쪽 
void ViewClass::updateImageView(QString str)
{
	auto _dh = make_unique<dcmHelper>(str.toLocal8Bit().constData());
	windowImage dlg(this);
	dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
	dlg.setWindowTitle(str);
	QPoint position = mapToGlobal(this->frameGeometry().topLeft());
	dlg.move(position);
	dlg.Prepare(_dh.get());
	dlg.exec();
}

//windowBoundaryCondition 클래스쪽 
void ViewClass::SetOrigWindowLevel(int nIndex)
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;

	int nOrigWC = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetOrigWindowCenter();
	int nOrigWW = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetOrigWindowWidth();

	int nWC = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowCenter();
	int nWW = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowWidth();

	if (nIndex == dcmHelper::WindowLevel::WindowWidth)
		SetWindowLevel(nWC, nOrigWW);
	else if (nIndex == dcmHelper::WindowLevel::WindowCenter)
		SetWindowLevel(nOrigWC, nWW);

	emit requestWindowLevelControl(mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowCenter(), mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowWidth());
	emit requestControlEnable(true);
}

void ViewClass::SetWindowLevel(int nWC, int nWW)
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;
	mGraphics[m_nCursorView.x][m_nCursorView.y]->SetWindowLevel(nWC, nWW);
}

void ViewClass::clear_result_line()
{
	remove_folder(model_info::folder_type::output_path);
	remove_folder(model_info::folder_type::result_path);
	m_nCursorView = make_int2(0, 0);
	m_bFFRShow = false;
	m_b3DShow = false;
	if (mFFR3D) mFFR3D->clear_result_view();
}

void ViewClass::clear_result_view()
{
	for (auto r = 0; r < VIEW_ROW; r++)
	{
		for (auto c = 0; c < VIEW_COL; c++)
		{
			if (mGraphics[r][c])
			{
				delete mGraphics[r][c];
				mGraphics[r][c] = nullptr;
			}
		}
	}
}

void ViewClass::updateScene(dcmHelper* dh)
{
	if (!dh)
		return;
	//if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
	//	return;
	SetViewer(dh);
	QApplication::sendPostedEvents();
	//	mGraphics[View2D][LeftImage]->SetSelectWindow(true);
	//	mGraphics[View2D][RightImage]->SetSelectWindow(false);
}


void ViewClass::dragEnterEvent(QDragEnterEvent* e)
{
	const QMimeData* mimeData = e->mimeData();
	if (mimeData->hasFormat("application/x-locPath"))
		e->acceptProposedAction();
}

void ViewClass::dropEvent(QDropEvent* e)
{
	const QMimeData* mimeData = e->mimeData();

	if (mimeData->hasFormat("application/x-locPath"))
	{
		QByteArray encoded = mimeData->data("application/x-locPath");
		QDataStream stream(&encoded, QIODevice::ReadOnly);

		QString* _buffer = new QString;

		while (!stream.atEnd()) {
			stream >> *_buffer;
		}

		auto _dh = make_unique<dcmHelper>(_buffer->toLocal8Bit().toStdString());
		{
			auto curPos = e->pos();
			remove_folder(model_info::folder_type::all_path);
			for (auto r = 0; r < VIEW_ROW; r++)
			{
				for (auto c = 0; c < VIEW_COL; c++)
				{
					auto _rect = uiGridView->cellRect(r, c);
					if (_rect.contains(curPos))
					{
						setUpdatesEnabled(false);
						if (mFFR3D) mFFR3D->clear_result_view();

						if (mGraphics)
						{
							delete mGraphics[r][c];
							mGraphics[r][c] = nullptr;
						}
						SetViewer(_dh.get());
						setUpdatesEnabled(true);
					}
				}
			}
			update();
		}
	}
	else
		e->ignore();

}

void ViewClass::SetViewer(dcmHelper* dh)
{
	bool binit = false;
	if (!mGraphics[View2D][RightImage] || !mGraphics[View2D][RightImage])
		binit = true;

	for (int i = 0; i < VIEW_COL; i++)
	{
		if (mGraphics[View2D][i])
		{
			mGraphics[View2D][i]->clear_points();
			mGraphics[View2D][i]->clear_line();
			mGraphics[View2D][i]->SetSelectWindow(false);
		}
	}

	int r = 0, c = 0;
	GraphicsWindow* targ = getEmptyViewer(r, c);
	if (targ)
	{
		targ->setAcceptDrops(true);
		targ->SetGridIndex(r, c);
		auto _dh = make_unique<dcmHelper>(*dh->getFile());
		if (_dh)
		{
			m_strFilePath = _dh->getFile()->c_str();
			auto str = m_strFilePath.split(".");
			if (str.size() != 0)
				m_strFilePath = str[0];

			if (_dh.get()->loadFile(true))
			{
				if (_dh.get()->Data())
					targ->Parsing(_dh);
			}
			else
				targ->Parsing(_dh->getFile()->c_str());
		}
		uiGridView->addWidget(targ, r, c);
		targ->SetSelectWindow(true);
		QApplication::sendPostedEvents();
		if (loadfile_.open_loadfile)
		{
			if (c == LeftImage)
				targ->SetOpenImageId(loadfile_.frame_id_l - 1);
			else
				targ->SetOpenImageId(loadfile_.frame_id_r - 1);

			string_section section_handler(loadfile_.load_path, "/");
			std::string load_path = section_handler.get_section(0, -2);
			targ->ManualSegmentation(model_info::segmentation_model_type::none, model_info::segmentation_manual_type::load_file_line);
			targ->read_data_in(load_path);
		}
		else
		{
			targ->SetOpenImageId();
		}
		if (binit && c == 1)
			return;
		emit requestWindowLevelControl(targ->GetWindowCenter(), targ->GetWindowWidth());
	}
}

QDataStream& operator >> (QDataStream& s, dcmHelper*& dcmHelperptr)
{
	qulonglong ptrval;
	s >> ptrval;
	dcmHelperptr = *reinterpret_cast<dcmHelper**>(&ptrval);
	return s;
}

ViewClass::~ViewClass()
{
	clear_result_line();
	clear_result_view();
	if (mFFR3D)
		delete mFFR3D;

	if (mtestChart)
	{
		delete mtestChart;
		mtestChart = nullptr;
	}

	destroy();
	emit closed(this);
}

void ViewClass::testPause()
{
	if (mGraphics[View2D][LeftImage])mGraphics[View2D][LeftImage]->SetPlayState(false);
	if (mGraphics[View2D][RightImage])mGraphics[View2D][RightImage]->SetPlayState(false);
}

void ViewClass::testPlay()
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;

	if (mGraphics[View2D][RightImage]->GetSelectWindow())
	{
		mGraphics[View2D][RightImage]->SetPlayState(true);
	}
	else if (mGraphics[View2D][LeftImage]->GetSelectWindow())
	{
		mGraphics[View2D][LeftImage]->SetPlayState(true);
	}
}

GraphicsWindow* ViewClass::getEmptyViewer(int& _r, int& _c)
{
	for (auto r = 0; r < VIEW_ROW; r++)
	{
		for (auto c = 0; c < VIEW_COL; c++)
		{
			if (!mGraphics[r][c])
			{
				mGraphics[r][c] = new GraphicsWindow(gridLayoutWidget_3);
				std::function<void(int2)> fUpdateCursorView = std::bind(&ViewClass::UpdateCursorView, this, std::placeholders::_1);
				std::function<void(int id,bool is_move_point)> fUpdateLine = std::bind(&ViewClass::UpdateLine, this, std::placeholders::_1, std::placeholders::_2);
				std::function<void()> fUpdateLineUndo = std::bind(&ViewClass::SetLineUndo, this);
				mGraphics[r][c]->CallbackUpdateCursorView(fUpdateCursorView);
				mGraphics[r][c]->CallbackUpdateLine(fUpdateLine);
				mGraphics[r][c]->CallbackUpdateLineUndo(fUpdateLineUndo);
				mGraphics[r][c]->SetLineRange(m_nLineRange, m_bLineAutoRange);
				_r = r;
				_c = c; 
				return mGraphics[r][c];
			}
		}
	}
	return nullptr;
}

void ViewClass::SetLineRange(int Value, bool bAuto)
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;

	m_nLineRange = Value;
	m_bLineAutoRange = bAuto;
	for (auto r = 0; r < VIEW_ROW; r++)
	{
		for (auto c = 0; c < VIEW_COL; c++)
		{
			if (mGraphics[r][c]->GetinitWindow())
			{
				mGraphics[r][c]->SetLineRange(Value, bAuto);
			}
		}
	}
}

bool ViewClass::IsDCMfile()
{
	bool dcmfile = true;
	if (!mGraphics[View2D][LeftImage]->EmptyDCM() || !mGraphics[View2D][RightImage]->EmptyDCM())
		dcmfile = false;
	return dcmfile;
}

void ViewClass::Update2DLine(int nViewIndex, model_info::segmentation_manual_type type)
{
	windowMessageBox* masBox = nullptr;
	if (type == model_info::segmentation_manual_type::start_end_findline)
		masBox = CreateMessage("Auto Point", "Find Stenosis or Branch point ", "Close", true);//Epiline Calibration
	else
		masBox = CreateMessage("2D Boundaries", "Generating 2D Boundaries", "Close", true);

	masBox->show();
	QApplication::processEvents();

	QTime myTimer;
	myTimer.start();
	mGraphics[View2D][nViewIndex]->SetSelectWindow(false);
	mGraphics[View2D][nViewIndex]->ManualSegmentation(model_info::segmentation_model_type::none, type);
	mGraphics[View2D][nViewIndex]->draw_points();
	mGraphics[View2D][nViewIndex]->draw_lines();
	qDebug() << "2D - segmentation_manual " << myTimer.elapsed() / 1000.0f;

	masBox->hide();
	delete masBox;
}

void ViewClass::Update2DLine(int nViewIndex, model_info::segmentation_model_type Model_type, model_info::segmentation_exe_type run_type)
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;
	if (!mGraphics[View2D][LeftImage]->GetinitWindow() || !mGraphics[View2D][RightImage]->GetinitWindow())
		return;

	QTime myTimer;
	myTimer.start();

	mGraphics[View2D][nViewIndex]->SetSelectWindow(false);
	mGraphics[View2D][nViewIndex]->AutoSegmentation(Model_type, run_type);
	mGraphics[View2D][nViewIndex]->draw_points();
	mGraphics[View2D][nViewIndex]->draw_lines();
	qDebug() << "2D - segmentation_auto " << myTimer.elapsed() / 1000.0f;
}

void ViewClass::SetEndPoints(model_info::find_endpoint_type type, int view_id)
{
	int run_id = view_id;
	if (QFile::exists(QDir::currentPath().section("/", 0, -2) + QString("\\data\\EndPoints.dat")))
		QFile::remove(QDir::currentPath().section("/", 0, -2) + QString("\\data\\EndPoints.dat"));
	if (QFile::exists(QDir::currentPath().section("/", 0, -2) + QString("\\data\\EndPointIDs.dat")))
		QFile::remove(QDir::currentPath().section("/", 0, -2) + QString("\\data\\EndPointIDs.dat"));


	std::string read_path = m_strProgramPath + ("\\AI\\Images\\outputs\\");
	if (type ==  model_info::find_endpoint_type::image_open)
	{
		QFileDialog path(this);
		path.setAcceptMode(QFileDialog::AcceptOpen);
		path.setFileMode(QFileDialog::ExistingFile);
		path.setNameFilter(tr("Images (*.bmp)"));
		path.setViewMode(QFileDialog::Detail);
		QStringList strFileNameList;
		if (path.exec() != QDialog::Accepted)
			return;
		strFileNameList = path.selectedFiles();
		read_path = strFileNameList[0].toStdString();
	}

	mGraphics[View2D][run_id]->set_end_points(read_path, type); //자동 색칠 프로그램 사용시 true

}
void ViewClass::save_file(model_info::save_file_type type)
{
	if (type == model_info::save_file_type::log_file)
	{
		if (m_folderName == "")
			return;
		auto dst = m_strProgramPath + "\\data";
		for (int i = 0; i < AllImage; i++)
		{
			auto path = mGraphics[View2D][i]->GetPath();
			std::string utf8Str = convertEucKrToUtf8(path);
			std::filesystem::path filePath(utf8Str);
			std::string fileName = filePath.stem().string();
			if (i == LeftImage)
				copy_file(utf8Str, dst + "\\0_" + fileName + ".dcm");
			else
				copy_file(utf8Str, dst + "\\1_" + fileName + ".dcm");
		}
		dst = m_strProgramPath + "\\data\\case.dat";
		FILE* f = fopen(dst.c_str(), "wt");
		fprintf(f, "%d,%d\n", mGraphics[View2D][LeftImage]->GetCurrentImage() + 1, mGraphics[View2D][RightImage]->GetCurrentImage() + 1);
		fprintf(f, "%s\n", m_folderName.c_str());
		fclose(f);
	}
	else if (type == model_info::save_file_type::bpm_file)
	{
		bool is_bpm = (mGraphics[View2D][LeftImage]->get_bpm().c_str() != "" && mGraphics[View2D][RightImage]->get_bpm().c_str() != "") ? true : false;
		if (!is_bpm)
			return;
		FILE* f = fopen((m_strProgramPath + "\\data\\BPM.dat").c_str(), "wt");
		fprintf(f, "%s,%s", mGraphics[View2D][LeftImage]->get_bpm().c_str(), mGraphics[View2D][RightImage]->get_bpm().c_str());
		fclose(f);
	}
	else if(type == model_info::save_file_type::points_file)
	{
		std::string path = m_strProgramPath + ("\\data\\match_point.txt");
		FILE* f = fopen(path.c_str(), "wt");
		if (f)
		{
			vector<vector<int2>> vecPnts;
			for (int i = 0; i < AllImage; i++)
			{
				vector<int2> Pnts;
				auto point_instance = mGraphics[View2D][i]->get_points_instance();

				auto start_point = point_instance.start_point;
				auto end_point = point_instance.end_point;
				auto branch_point = point_instance.branch_point;

				Pnts.push_back(make_int2(int(start_point.x), int(start_point.y)));

				if (branch_point.size() == 2)
				{
					for (int j = 0; j < branch_point.size(); j++)
						Pnts.push_back(make_int2((branch_point[j].second.x), (branch_point[j].second.y)));
				}
				else
				{
					Pnts.push_back(make_int2(int(start_point.x), int(start_point.y)));
					Pnts.push_back(make_int2(int(end_point.x), int(end_point.y)));
				}
				Pnts.push_back(make_int2(int(end_point.x), int(end_point.y)));
				vecPnts.push_back(Pnts);
			}

			for (int j = 0; j < vecPnts.size(); j++)
			{
				vector<int2> Pnts = vecPnts[j];
				fprintf(f, QString(QString("S") + QString::number(j + 1) + QString(":	%d, %d\n")).toLocal8Bit().data(), Pnts[0].x, Pnts[0].y);
				fprintf(f, QString(QString("M1") + QString::number(j + 1) + QString(":	%d, %d\n")).toLocal8Bit().data(), Pnts[1].x, Pnts[1].y);
				fprintf(f, QString(QString("M2") + QString::number(j + 1) + QString(":	%d, %d\n")).toLocal8Bit().data(), Pnts[2].x, Pnts[2].y);
				fprintf(f, QString(QString("E") + QString::number(j + 1) + QString(":	%d, %d\n")).toLocal8Bit().data(), Pnts[3].x, Pnts[3].y);
			}
			fclose(f);
		}
	}
}

void ViewClass::SetFFR3D()
{
	if (!IsDCMfile())
		return;

	bool bAuto = false; // 0 라인 수동 , 1 라인 자동

	if (loadfile_.open_loadfile) // 저장된 데이터 불러오기
	{
		auto load_ouput_path = section(loadfile_.load_path, '/', 0, -2);
		copyDir(loadfile_.load_path, m_strProgramPath + "\\data");
		copyDir(load_ouput_path, m_strProgramPath + "\\output");
	}
	else
	{
		if (mGraphics[View2D][LeftImage]->GetModelState() == model_info::model_type::lint || mGraphics[View2D][RightImage]->GetModelState() == model_info::model_type::lint)
		{
			clear_result_line();

			if (!bAuto) //외각 센터라인 구하는거 수동
			{
				if (!mGraphics[View2D][LeftImage]->isLine())
					Update2DLine(LeftImage, model_info::segmentation_manual_type::start_end_findline);

				if (!mGraphics[View2D][RightImage]->isLine())
					Update2DLine(RightImage, model_info::segmentation_manual_type::start_end_findline);
			}
			else  //외각 센터라인 구하는거 자동
			{
				windowModel dlg(this);
				dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
				dlg.setWindowTitle("Model");
				auto size = this->rect().center().x() + m_nMoveX - dlg.geometry().width() / 2;
				dlg.move(size, this->rect().center().y());
				if (QDialog::Accepted == dlg.exec())
				{
					if (!mGraphics[View2D][LeftImage]->isLine())
						Update2DLine(LeftImage, dlg.GetModelL(), model_info::segmentation_exe_type::run_outlines);

					if (!mGraphics[View2D][RightImage]->isLine())
						Update2DLine(RightImage, dlg.GetModelR(), model_info::segmentation_exe_type::run_outlines);
				}
				save_file(model_info::save_file_type::bpm_file);
			}
			save_file(model_info::save_file_type::log_file);

			copyDir("\\data", "\\case");
			copyDir("\\output", "\\case");

			mGraphics[View2D][LeftImage]->SetSelectWindow(true);
			mGraphics[View2D][RightImage]->SetSelectWindow(false);
			return;
		}
		else
		{
			if (!bAuto)
			{
				if (mGraphics[View2D][LeftImage]->GetModelState() == model_info::model_type::branch_points
					&& mGraphics[View2D][RightImage]->GetModelState() == model_info::model_type::branch_points)
				{
					UpdateEpiline();
					findPointsOnEpipolarLine();
					save_file(model_info::save_file_type::points_file);
				}
				if (!mGraphics[View2D][LeftImage]->isLine())
					Update2DLine(LeftImage, model_info::segmentation_manual_type::find_centerline);

				if (!mGraphics[View2D][RightImage]->isLine())
					Update2DLine(RightImage, model_info::segmentation_manual_type::find_centerline);
			}
		}
	}


	if (mGraphics[View2D][LeftImage]->GetModelState() == model_info::model_type::line_2d 
		|| mGraphics[View2D][RightImage]->GetModelState() == model_info::model_type::line_2d)
		UpdataEqualLine();

	mGraphics[View2D][LeftImage]->read_data_in(m_strProgramPath.c_str());
	mGraphics[View2D][RightImage]->read_data_in(m_strProgramPath.c_str());

	copyDir("\\data", "\\case");
	copyDir("\\data\\test", "\\case");
	copyDir("\\output", "\\case");
	if (m_folderName != "") copyDir("\\Result\\", "\\case");

	auto path = m_strProgramPath + ("\\Angio_simulation\\ffr.plt");
	auto endpoint_type = model_info::find_endpoint_type::image_open;
	if (!std::filesystem::exists(path))
	{
		windowMessageBox masBox;
		masBox.setWindowFlags(masBox.windowFlags() & ~Qt::WindowContextHelpButtonHint);
		masBox.setWindowTitle("Selection");
		masBox.setModal(true);
		masBox.setText("choose left or right.. ");
		masBox.Select("Left");
		auto s = this->rect().center().x() + m_nMoveX - masBox.rect().width() / 2;
		masBox.move(s, this->rect().center().y());
		masBox.exec();
		{
			windowModel dlg(this);
			dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
			dlg.setWindowTitle("Model");
			auto size = this->rect().center().x() + m_nMoveX - dlg.geometry().width() / 2;
			dlg.move(size, this->rect().center().y());
			if (QDialog::Accepted == dlg.exec())
			{
				auto path = m_strProgramPath + ("\\image");
				if (std::filesystem::exists(path))
					removeFilesWithExtensions(path);
				if (!std::filesystem::exists(path))
					std::filesystem::create_directories(path);  // 하위 디렉터리 생성

				if (masBox.get_select() == 0)
				{
					m_nCursorView = make_int2(0, 0);
					mGraphics[View2D][LeftImage]->SetSelectWindow(true);
					mGraphics[View2D][RightImage]->SetSelectWindow(false);
					mGraphics[View2D][LeftImage]->ManualSegmentation(dlg.GetModelL(), model_info::segmentation_manual_type::run_manual_ai);
					SetEndPoints(endpoint_type, LeftImage);
				}
				else if (masBox.get_select() == 1)
				{
					m_nCursorView = make_int2(0, 1);
					mGraphics[View2D][RightImage]->SetSelectWindow(true);
					mGraphics[View2D][LeftImage]->SetSelectWindow(false);
					mGraphics[View2D][RightImage]->ManualSegmentation(dlg.GetModelR(), model_info::segmentation_manual_type::run_manual_ai);
					SetEndPoints(endpoint_type, RightImage);
				}
				save_file(model_info::save_file_type::bpm_file);
			}
		}
		Run3D();
		SetFFR();
	}
	else
	{
		SetResultView(ViewFFR, path.c_str());
	}
}

void ViewClass::RunAI(int view_id)
{
	windowModel dlg(this);
	dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
	dlg.setWindowTitle("Model");
	auto size = this->rect().center().x() + m_nMoveX - dlg.geometry().width() / 2;
	dlg.move(size, this->rect().center().y());
	if (QDialog::Accepted == dlg.exec())
	{
		auto path = m_strProgramPath + ("\\image");
		if (std::filesystem::exists(path))
			removeFilesWithExtensions(path);
		if (!std::filesystem::exists(path))
			std::filesystem::create_directories(path);  // 하위 디렉터리 생성
		if(view_id == LeftImage)
			mGraphics[View2D][view_id]->ManualSegmentation(dlg.GetModelL(), model_info::segmentation_manual_type::run_manual_ai);
		else if (view_id == RightImage)
			mGraphics[View2D][view_id]->ManualSegmentation(dlg.GetModelR(), model_info::segmentation_manual_type::run_manual_ai);
	}
	else
		return;
}

void ViewClass::Open3DFile()
{
	std::vector<std::shared_future<void>> vecFutures;
	QString strExtension = QDir::currentPath().section("/", 0, -2);
	QString FilePath = strExtension + QString("\\output");
	QString filename = FilePath + QString("\\mesh_c3r22_P_SW1.plt");
	QString ResultFilePath = QDir::currentPath().section("/", 0, -2) + "\\Result\\" + m_strFilePath.section("\\\\", -2);
	QFile::copy(filename, ResultFilePath + QString("mesh_c3r22_P_SW1.plt"));


	SetResultView(View3D, filename);
	SetResultView(ViewChart, filename);
}

void ViewClass::UpdateEpiline()
{
	if (!IsDCMfile())
		return;

	save_file(model_info::save_file_type::points_file);
	run_process("./calibrate_main.exe");

	//결과 읽어오기
	{
		auto path = m_strProgramPath + ("\\data\\intercepts.txt");
		FILE* f = fopen(path.c_str(), "r");
		if (f)
		{
		
			char ss[100];
			fgets(ss, 100, f);
			for (int i = 0; i < 2; i++)
			{
				vector<float2> INTCS;
				vector<float2> INTCE;
				vector<float2 > Intercepts_point;
				fgets(ss, 100, f);
				for (int j = 0; j < 2; j++)
				{
					fgets(ss, 100, f);

					char* ptr = strtok(ss, " ");
					ptr = strtok(NULL, " ");
					ptr = strtok(NULL, " ");

					ptr = strtok(NULL, "(,");
					if (ptr == nullptr)
						break;
					int INTC_X1 = atoi(ptr);

					ptr = strtok(NULL, " )");
					if (ptr == nullptr)
						break;
					int INTC_Y1 = atoi(ptr);
					Intercepts_point.push_back(make_float2(INTC_X1, INTC_Y1));


					ptr = strtok(NULL, ", ");
					ptr = strtok(NULL, "(,");  //INTC S X2
					int INTC_X2 = atoi(ptr);

					ptr = strtok(NULL, ")");   //INTC S Y2
					if (ptr == nullptr)
						break;
					int INTC_Y2 = atoi(ptr);
					Intercepts_point.push_back(make_float2(INTC_X2, INTC_Y2));

				}
				for (int j = 0; j < 2; j++)
				{
					INTCS.push_back(Intercepts_point[j]);
					INTCE.push_back(Intercepts_point[2 + j]);
				}

				mGraphics[View2D][i]->draw_calibration_line(INTCS,INTCE);
			}
			fclose(f); 
		}
	}
}


void ViewClass::findPointsOnEpipolarLine()
{
	vector<float2> move_points_l, move_points_r;
	auto find_point_l = mGraphics[View2D][LeftImage]->calculatePointOnEpiline(move_points_l);
	auto find_point_r = mGraphics[View2D][RightImage]->calculatePointOnEpiline(move_points_r);

	if (move_points_l.size() == move_points_r.size())
	{
		if (move_points_l.size() == 2)
		{
			mGraphics[View2D][LeftImage]->moveAlongEpiline(move_points_l);
		}
		else if (move_points_l.size() == 1)
		{
			mGraphics[View2D][LeftImage]->moveAlongEpiline(move_points_l);
			if (find_point_l == find_point_r)
				mGraphics[View2D][RightImage]->moveAlongEpiline(move_points_r);
		}
		else
			return;
	}
	else if (move_points_l.size() > move_points_r.size())
		mGraphics[View2D][LeftImage]->moveAlongEpiline(move_points_l);
	else
		mGraphics[View2D][RightImage]->moveAlongEpiline(move_points_r);

	UpdateEpiline();
}

void ViewClass::run_process(std::string program) 
{
	STARTUPINFO startupInfo;
	PROCESS_INFORMATION processInfo;
	ZeroMemory(&startupInfo, sizeof(startupInfo));
	startupInfo.cb = sizeof(startupInfo);
	ZeroMemory(&processInfo, sizeof(processInfo));

	// Create a new console and start the process
	if (CreateProcess(
		NULL,                      // No module name (use command line)
		const_cast<char*>(program.c_str()), // Command line
		NULL,                      // Process handle not inheritable
		NULL,                      // Thread handle not inheritable
		FALSE,                     // Set handle inheritance to FALSE
		CREATE_NEW_CONSOLE,        // Create a new console
		NULL,                      // Use parent's environment block
		NULL,                      // Use parent's starting directory
		&startupInfo,              // Pointer to STARTUPINFO structure
		&processInfo               // Pointer to PROCESS_INFORMATION structure
	)) {
		// Wait for the process to finish
		WaitForSingleObject(processInfo.hProcess, INFINITE);

		// Close process and thread handles
		CloseHandle(processInfo.hProcess);
		CloseHandle(processInfo.hThread);
	}
}


void ViewClass::UpdataEqualLine()
{
	QString strCoronaryType = "";
	vector<int> percent;
	auto p1 = mGraphics[View2D][LeftImage]->GetMinimumRadius();
	auto p2 = mGraphics[View2D][RightImage]->GetMinimumRadius();
	int N = 100;
	if (p1.size() == 0 && p1.size() == 0)
	{
		percent.push_back(N);
	}
	else if (p1.size() == p2.size())
	{
		int sumP = 0;
		for (int i = 0; i < p1.size(); i++)
		{
			int P = (p1[i] + p2[i]) / 2;
			if (i == p1.size() - 1)
				P = N - sumP;
			percent.push_back(P);
			sumP += P;
		}
	}
	m_vecEqualPercents.clear();

	std::vector<float2> new_cent_line_l{}, new_cent_line_r{};
	for (int N = 0; N < percent.size(); N++)
	{
		int P = percent[N];
		m_vecEqualPercents.push_back(P);
		auto new_center_points_l = mGraphics[View2D][LeftImage]->get_equilateral(P, N);
		for (int j = 0; j < new_center_points_l.size(); j++)
			new_cent_line_l .push_back(new_center_points_l[j]);

		auto new_center_points_r = mGraphics[View2D][RightImage]->get_equilateral(P, N);
		for (int j = 0; j < new_center_points_r.size(); j++)
			new_cent_line_r.push_back(new_center_points_r[j]);

	}

	mGraphics[View2D][LeftImage]->SetEquilateraLines(new_cent_line_l, LeftImage);
	mGraphics[View2D][RightImage]->SetEquilateraLines(new_cent_line_r, RightImage);
}

void ViewClass::Run3D()
{
	auto masBox = CreateMessage("3D Model", "Generating 3D Model", "Close", true);
	masBox->show();
	QApplication::processEvents();
	run_process("./C3R_main.exe");
	std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	masBox->hide();
	delete masBox;
}

void ViewClass::UpdateChartClose()
{
	
}

void ViewClass::Run_FFR()
{
	auto path = m_strProgramPath + ("\\Angio_simulation\\ffr.plt");
	std::filesystem::exists(path) == true ? SetResultView(ViewFFR, path.c_str()) : SetFFR();
}

void ViewClass::SetFFRGeneration()
{
	m_nCursorView = make_int2(0, 0);
	mGraphics[View2D][LeftImage]->SetSelectWindow(true);
	mGraphics[View2D][RightImage]->SetSelectWindow(false);

	std::string ffr_simulation_path = m_strProgramPath + "\\Angio_simulation";

	if (std::filesystem::exists(ffr_simulation_path + "\\ffr.plt"))
		std::filesystem::remove(ffr_simulation_path + "\\ffr.plt");

	if (std::filesystem::exists(ffr_simulation_path + "\\ffr.vtp"))
		std::filesystem::remove(ffr_simulation_path + "\\ffr.vtp");

	if (std::filesystem::exists(ffr_simulation_path + "\\mesh_c3r22_P_SW1.stl"))
		std::filesystem::remove(ffr_simulation_path + "\\mesh_c3r22_P_SW1.stl");

	if (std::filesystem::exists(ffr_simulation_path + "\\centerline_c3r22_P_SW1.dat"))
		std::filesystem::remove(ffr_simulation_path + "\\centerline_c3r22_P_SW1.dat");

	std::string command = ffr_simulation_path + "\\Main.exe "
						+ ffr_simulation_path + "\\mesh_c3r22_P_SW1.stl "
						+ ffr_simulation_path + "\\centerline_c3r22_P_SW1.dat";

	run_process(command);

	std::string output_path = m_strProgramPath + "\\output";
	std::string result_path = m_strProgramPath + "\\result";

	QString FilePath = QDir::currentPath().section("/", 0, -2) + "\\Result\\" + m_strFilePath.section("\\\\", -2);

	std::filesystem::copy(output_path + "\\mesh_c3r22_P_SW1.stl", ffr_simulation_path + "\\mesh_c3r22_P_SW1.stl");
	std::filesystem::copy(output_path + "\\centerline_c3r22_P_SW1.dat", ffr_simulation_path + "\\centerline_c3r22_P_SW1.dat");
	std::filesystem::copy(ffr_simulation_path + "\\ffr.plt", result_path + "\\ffr.plt");
	 
	if (mFFR3D)
	{
		mFFR3D->clear_result_view();
		delete mFFR3D;
		mFFR3D = nullptr;
	}
}

void ViewClass::SetFFR()
{
	windowBoundaryCondition dlg(this);
	dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
	auto size = this->rect().center().x() + m_nMoveX - dlg.geometry().width() / 2;
	dlg.move(size, this->rect().center().y());


	if (QDialog::Accepted == dlg.exec())
	{
		mGraphics[View2D][LeftImage]->SetSelectWindow(true);
		mGraphics[View2D][RightImage]->SetSelectWindow(false);
		m_nCursorView = make_int2(0, 0);
		SetFFRGeneration();
	}
	else
		Open3DFile();
}

void ViewClass::SetResultView(int ntype, QString filename)
{
	FILE* f = NULL;

	int r = VIEW_ROW, c = 0;
	float p0 = 0, s0 = 0, p1 = 0, s1 = 0, v = 1.0;

	if (m_nCursorView.y == 1)
	{
		p0 = mGraphics[r - 1][c]->GetPAngle();
		s0 = mGraphics[r - 1][c]->GetSAngle();
		v = mGraphics[r - 1][c]->GetScale();

		p1 = mGraphics[r - 1][c + 1]->GetPAngle();
		s1 = mGraphics[r - 1][c + 1]->GetSAngle();
	}
	if (m_vecEqualPercents.size() == 0)
		m_vecEqualPercents.push_back(100);
	if (ntype == View3D)
	{
		if (mFFR3D)
		{
			delete mFFR3D;
			mFFR3D = nullptr;
		}

		mFFR3D = new windowFFR3D(this);
		mFFR3D->setWindowTitle("3D Result");
		mFFR3D->setWindowFlags(mFFR3D->windowFlags() & ~Qt::WindowContextHelpButtonHint);
		mFFR3D->SetGridIndex(m_nCursorView.y);
		mFFR3D->SetFFR(false);
		mFFR3D->SetResultFFRView(ntype, filename);
		mFFR3D->SetMulPSAngle(p0, s0, p1, s1, v);
		std::vector<int> X_3D;
		std::vector<float> axisX_3D, axisY_3D;
		set_chart3D(X_3D, axisX_3D, axisY_3D);
		mFFR3D->SetResultChart(true, m_vecEqualPercents, axisX_3D, axisY_3D);
	}
	else if (ntype == ViewFFR)
	{
		if (!mFFR3D)
			mFFR3D = new windowFFR3D(this);

		mFFR3D->hide();
		QString FilePath = QDir::currentPath().section("/", 0, -2) + QString("\\Angio_simulation");
		mFFR3D->setWindowTitle("FFR Result");
		mFFR3D->setWindowFlags(mFFR3D->windowFlags() & ~Qt::WindowContextHelpButtonHint);
		mFFR3D->SetGridIndex(m_nCursorView.y);
		mFFR3D->SetFFR(true);
		mFFR3D->SetMulPSAngle(p0, s0, p1, s1, v);
		mFFR3D->SetResultFFRView(ntype, FilePath + QString("\\ffr.plt"));

		{//3D
			std::vector<int> X_3D, X_FFR;
			std::vector<float> axisX_3D, axisY_3D, axisX_FFR, axisY_FFR;

			if (set_chart3D(X_3D, axisX_3D, axisY_3D))
				mFFR3D->SetResultChart(true, m_vecEqualPercents, axisX_3D, axisY_3D);
			if (set_chartFFR(X_FFR, axisX_FFR, axisY_FFR))
				mFFR3D->SetResultChart(false, X_FFR, axisX_FFR, axisY_FFR);
		}
		m_bFFRShow = true;
	}

	if (mFFR3D)
	{
		mFFR3D->Show(true);
		auto sizeX = this->rect().center().x() + m_nMoveX - mFFR3D->geometry().width() / 2;

		auto sizeY = this->rect().center().y() / 2;
		mFFR3D->move(sizeX, sizeY);
		mFFR3D->show();
	}
	update();
}

void ViewClass::remove_folder(model_info::folder_type type)
{
	if (m_strProgramPath == "")
		return;

	auto path = m_strProgramPath;

	if (type == model_info::folder_type::all_path)
	{
		remove_folder(model_info::folder_type::data_path);
		remove_folder(model_info::folder_type::output_path);
		remove_folder(model_info::folder_type::result_path);
		path += ("\\data\\test");
	}
	else
	{
		if (type == model_info::folder_type::result_path)
			path += ("\\Result");
		else if (type == model_info::folder_type::data_path)
			path += ("\\data");
		else if (type == model_info::folder_type::output_path)
			path += ("\\output");
		else
			return;
	}

	if (std::filesystem::exists(path))
		removeFilesWithExtensions(path);

	if (!std::filesystem::exists(path))
		std::filesystem::create_directories(path);  // 하위 디렉터리 생성
}



void ViewClass::SetClearLine()
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;

	if (mGraphics[View2D][LeftImage]->GetinitWindow() && mGraphics[View2D][RightImage]->GetinitWindow())
	{
		windowSelect dlg(this);
		dlg.setWindowFlags(dlg.windowFlags() & ~Qt::WindowContextHelpButtonHint);
		dlg.setWindowTitle("Delete");
		auto size = this->rect().center().x() + m_nMoveX - dlg.geometry().width() / 2;
		dlg.move(size, this->rect().center().y());
		if (QDialog::Accepted == dlg.exec())
		{
			if (dlg.GetImageIndex() == AllImage)
			{
				for (int i = 0; i < AllImage; i++)
				{
					mGraphics[View2D][i]->clear_points();
					mGraphics[View2D][i]->clear_line();
				}
			}
			else
			{
				mGraphics[View2D][dlg.GetImageIndex()]->clear_points();
				mGraphics[View2D][dlg.GetImageIndex()]->clear_line();
			}
			clear_result_line();
			emit requestControlEnable(false);
		}
		else
		{
			return;
		}
	}
	else if (mGraphics[View2D][LeftImage]->GetinitWindow())
	{
		mGraphics[View2D][LeftImage]->clear_line();
		mGraphics[View2D][LeftImage]->SetSelectWindow(true);
	}
	clear_result_line();
}

void ViewClass::SetLineUndo()
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;
	if (!mGraphics[View2D][LeftImage]->GetinitWindow() && !mGraphics[View2D][RightImage]->GetinitWindow())
		return;

	bool isfail = true;
	int nViewIndx = -1;
	if (mGraphics[View2D][RightImage]->GetSelectWindow())
		nViewIndx = RightImage;
	else if (mGraphics[View2D][LeftImage]->GetSelectWindow())
		nViewIndx = LeftImage;

	isfail = mGraphics[View2D][nViewIndx]->LineUndo();

	if (!isfail)
	{
		auto message = CreateMessage("2D Line", "Line Clear?", "Run", true);
		QTimer* timer = new QTimer();
		timer->setSingleShot(true);
		QObject::connect(timer, &QTimer::timeout, [=]()
			{
				if (message)
					message->SetDialogCode(QDialog::Rejected);
				timer->deleteLater();
			});

		QMetaObject::invokeMethod(timer, "start", Qt::QueuedConnection, Q_ARG(int, 2000));
		message->exec();
		timer->deleteLater();
		if (message->GetDialogCode() == QDialog::Rejected)
		{
			if (message)
			{
				delete message;
				message = nullptr;
			}
			return;
		}
		mGraphics[View2D][LeftImage]->clear_line(model_info::clear_type::clear_calibration_line);
		mGraphics[View2D][RightImage]->clear_line(model_info::clear_type::clear_calibration_line);
		mGraphics[View2D][nViewIndx]->clear_line(model_info::clear_type::clear_line);
		if (message)
		{
			delete message;
			message = nullptr;
		}
	}
}


bool ViewClass::set_chartFFR(std::vector<int>& X, std::vector<float>& axisX, std::vector<float>& axisY)
{
	int startid = 0;
	int lastid = 0;
	{
		QString strExtension = QDir::currentPath().section("/", 0, -2);
		QString ffrFilePath = strExtension + QString("\\Angio_simulation\\ffr.plt");
		QString centerlineFilePath = strExtension + QString("\\output\\centerline_c3r22_P_SW1.dat");
		QString outputFilePath = strExtension + QString("\\output\\ffr_graph.dat");

		if (!QFile::exists(ffrFilePath) || !QFile::exists(centerlineFilePath))
			return false;

		int i, j, nc, nde, n1, n2, n3;
		float u, v, w;
		double dis, dis_min, cdis, clength_sum, ffr2, size_grid;
		char ss[100];

		FILE* fp_in1, * fp_in2, * fp_out;

		fp_in1 = fopen(centerlineFilePath.toStdString().c_str(), "r");
		fp_in2 = fopen(ffrFilePath.toStdString().c_str(), "r");
		fp_out = fopen(outputFilePath.toStdString().c_str(), "w");


		fscanf(fp_in1, "%d\n", &nc); //센터라인 데이터 읽음.

		double* xc = (double*)malloc(sizeof(double) * (nc + 1));
		double* yc = (double*)malloc(sizeof(double) * (nc + 1));
		double* zc = (double*)malloc(sizeof(double) * (nc + 1));

		for (i = 1; i <= nc; i++)
		{
			fscanf(fp_in1, "%lf %lf %lf\n", &xc[i], &yc[i], &zc[i]);
			xc[i] = xc[i] / 1000.0;
			yc[i] = yc[i] / 1000.0;
			zc[i] = zc[i] / 1000.0;
		}


		fgets(ss, 100, fp_in2);   //plt 파일 첫번째 줄 읽음.
		fgets(ss, 100, fp_in2);   //plt 파일 두번째 줄 읽음.
		char* ptr = strtok(ss, " ");
		ptr = strtok(NULL, " ");
		ptr = strtok(NULL, " ");
		ptr = strtok(NULL, " ");
		nde = atoi(ptr);         //plt 파일 두번째 줄에서 노드 갯수 숫자.

		float* x = (float*)malloc(sizeof(float) * (nde + 1));
		float* y = (float*)malloc(sizeof(float) * (nde + 1));
		float* z = (float*)malloc(sizeof(float) * (nde + 1));
		float* ffr = (float*)malloc(sizeof(float) * (nde + 1));

		for (i = 1; i <= nde; i++)
			fscanf(fp_in2, "%f %f %f %f %f %f %f\n", &x[i], &y[i], &z[i], &u, &v, &w, &ffr[i]);

		fscanf(fp_in2, "%d %d %d %d %d %d %d %d\n", &n1, &n2, &n3, &n3, &n3, &n3, &n3, &n3);


		size_grid = (x[n1] - x[n2]) * (x[n1] - x[n2])
			+ (y[n1] - y[n2]) * (y[n1] - y[n2])
			+ (z[n1] - z[n2]) * (z[n1] - z[n2]);
		size_grid = sqrt(size_grid);                      //대표적인 격자 길이.

		clength_sum = 0.0;
		fprintf(fp_out, "%lf %lf\n", clength_sum, 1.0); //센터라인 시작점 FFR값
		axisX.push_back(clength_sum * 1000.0);
		axisY.push_back(1.0);

		for (i = 2; i <= nc; i++)  //각 센터라인 점에서 가장 가까운 FFR 값 서치.
		{
			dis_min = 1000.0;
			for (j = 1; j <= nde; j++)
			{

				if (fabs(x[j] - xc[i]) < size_grid * 5.0)
					if (fabs(y[j] - yc[i]) < size_grid * 5.0)
						if (fabs(z[j] - zc[i]) < size_grid * 5.0)
						{
							dis = (x[j] - xc[i]) * (x[j] - xc[i])
								+ (y[j] - yc[i]) * (y[j] - yc[i])
								+ (z[j] - zc[i]) * (z[j] - zc[i]);

							if (dis < dis_min)
							{
								dis_min = dis;
								ffr2 = ffr[j];
							}
						}
			}

			cdis = (xc[i] - xc[i - 1]) * (xc[i] - xc[i - 1])
				+ (yc[i] - yc[i - 1]) * (yc[i] - yc[i - 1])
				+ (zc[i] - zc[i - 1]) * (zc[i] - zc[i - 1]);
			cdis = sqrt(cdis);
			clength_sum += cdis;  // 센터라인 각 점에서 길이 계산.
			fprintf(fp_out, "%f %f \n", clength_sum * 1000.0, ffr2);
			axisX.push_back(clength_sum * 1000.0);
			axisY.push_back(ffr2);

		}
		free(xc);
		free(yc);
		free(zc);

		free(x);
		free(y);
		free(z);
		free(ffr);

		fclose(fp_in1);
		fclose(fp_in2);
		fclose(fp_out);
	}


	X = mGraphics[m_nCursorView.x][m_nCursorView.y]->calculate_chart_axis();
	if (X.size() == 0)
		return false;
	return true;
}

bool ViewClass::set_chart3D(std::vector<int>& X, std::vector<float>& axisX, std::vector<float>& axisY)
{
	{
		FILE* fp_in3D;
		QString strExtension = QDir::currentPath().section("/", 0, -2);
		QString FilePath = strExtension + QString("\\output\\point_attr_c3r22_P_SW1.dat");

		if (!QFile::exists(FilePath))
			return false;
		fp_in3D = fopen(FilePath.toStdString().c_str(), "r");

		int nc = 0;
		fscanf(fp_in3D, "%d\n", &nc); //센터라인 데이터 읽음.

		double* xc = (double*)malloc(sizeof(double) * (nc + 1));
		double* yc = (double*)malloc(sizeof(double) * (nc + 1));
		for (int i = 1; i <= nc; i++)
		{
			fscanf(fp_in3D, "%lf %lf \n", &xc[i], &yc[i]);
			axisY.push_back(xc[i]);
			axisX.push_back(yc[i]);
		}
		free(xc);
		free(yc);
		fclose(fp_in3D);
	}
	

	{
		X = mGraphics[m_nCursorView.x][m_nCursorView.y]->calculate_chart_axis();
		if (X.size() == 0)
			return false;
	}
	return true;
}


void ViewClass::UpdateLine(int id, bool is_move_point)
{
	if (is_move_point)
	{
		Update2DLine(id, model_info::segmentation_manual_type::find_centerline);
		UpdateEpiline();
	}
	else
	{
		mGraphics[View2D][LeftImage]->clear_line(model_info::clear_type::clear_calibration_line);
		mGraphics[View2D][RightImage]->clear_line(model_info::clear_type::clear_calibration_line);
	}
}


void ViewClass::UpdateCursorView(int2 nIndex)
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;
	m_nCursorView = make_int2(nIndex.y, nIndex.x);

	if (nIndex.y == View2D)
	{
		auto _WC = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowCenter();
		auto _WW = mGraphics[m_nCursorView.x][m_nCursorView.y]->GetWindowWidth();
		emit requestWindowLevelControl(_WC, _WW);
		emit requestControlEnable(true);
		mGraphics[View2D][RightImage]->SetSelectWindow(false);
		mGraphics[View2D][LeftImage]->SetSelectWindow(false);
		mGraphics[View2D][m_nCursorView.y]->SetSelectWindow(true);
		if (mFFR3D)
		{
			if (mFFR3D->GetShowWindow())
			{
				if (mFFR3D->GetFFRRun())
				{
					QString exeFilePath = QDir::currentPath().section("/", 0, -2) + "\\Angio_simulation";
					if (QFile::exists(exeFilePath + QString("\\ffr.plt")))
						SetResultView(ViewFFR, exeFilePath + QString("\\ffr.plt"));
				}
				else
				{
					QString OutputFilePath = QDir::currentPath().section("/", 0, -2) + "\\output";
					if (QFile::exists(OutputFilePath + QString("\\mesh_c3r22_P_SW1.plt")))
						SetResultView(View3D, OutputFilePath + QString("\\mesh_c3r22_P_SW1.plt"));
				}
			}
		}
		emit requestSetMenuShow(true);
	}
	else if (m_nCursorView.x == View3D)
	{
		emit requestSetMenuShow(false);
	}
}

void ViewClass::setEmptyViewer(int nIdex)
{
	for (auto r = 0; r < VIEW_ROW; r++)
	{
		for (auto c = nIdex; c < VIEW_COL; c++)
		{
			if (!mGraphics[r][c])
			{
				mGraphics[r][c] = new GraphicsWindow(gridLayoutWidget_3);
				mGraphics[r][c]->SetGridIndex(r, c);
				mGraphics[r][c]->setAcceptDrops(true);
				uiGridView->addWidget(mGraphics[r][c], r, c);
			}
		}
	}
	QApplication::sendPostedEvents();
}

void ViewClass::SaveAs()
{
	if (!mGraphics[View2D][LeftImage] && !mGraphics[View2D][RightImage])
		return;
	QString strExtension = QDir::currentPath().section("/", 0, -2);

	QString dirName = QFileDialog::getExistingDirectory(this, tr("Select Directory"), strExtension, QFileDialog::ShowDirsOnly);
	if (dirName == "")
		return;

	QDir dir(dirName);
	dir.removeRecursively();
	QDir().mkdir(dirName);



	mGraphics[View2D][LeftImage]->SaveAs(dirName + "\\L");
	mGraphics[View2D][RightImage]->SaveAs(dirName + "\\R");

	//QString fileName = QFileDialog::getSaveFileName(0, tr("Save Copy As"), QFileInfo().baseName().append(".bmp"), tr("Images (*.bmp *.png)"));

	//QString strExtension = QDir::currentPath().section("/", 0, -2);

	//
	////mGraphics[View2D][LeftImage]->grab().save();
	//QPixmap image = QPixmap::grabWidget(mGraphics[View2D][LeftImage]);
	//if (!image.save(FilePath, "JPG"))
	//{
	//	QMessageBox::warning(this, "Save Image", "Error saving image.");
	//}
}

void ViewClass::pbLoad()
{
	remove_folder(model_info::folder_type::all_path);
	clear_result_line();
	clear_result_view();
	loadfile_.init();
	if (!manualImport())
		return;

	if (m_strRepository == "" || mRepository[0].children.size() <= 0)
		return;

	auto _mc = mChecker.get();

	std::string case_name = "";
	if (loadfile_.open_loadfile)
	{
		auto LoadfileName = loadfile_.load_file_path;
		auto dcm_file_name = m_strProgramPath + "";
		mRepository.clear();       
		m_strRepository = "";
		requestSetRepositoryText("");

		updateRepositoryInformation(LoadfileName, true);

		std::ifstream file(LoadfileName);
		char comma; // 쉼표를 받기 위한 임시 변수
		file >> loadfile_.frame_id_l >> comma >> loadfile_.frame_id_r;
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Ignore rest of the line
		std::getline(file, case_name);
		file.close();
	}

	for (int i = 0; i < mRepository[0].children.size(); i++)
	{
		auto dcm = std::make_unique<dcmHelper>();
		auto src = mRepository[0].children[i];
		std::replace(src.begin(), src.end(), '/', '\\');
		dcm->setFile(src.toStdString());
		*_mc++ = true;
	}

	auto fullpath = section(mRepository[0].root, '/', -1);
	m_folderName = case_name.empty() ? fullpath : case_name;

	QStringList out;
	auto _mc2 = mChecker.get();
	for (int i = 0; i < mRepository[0].children.size(); i++)
	{
		if (*_mc2++)
		{
			out << mRepository[0].children.at(i);
		}
	}
	
	emit spread(out, fullpath.c_str(), false);
}

bool ViewClass::manualImport()
{
	QFileDialog path(this);
	path.setAcceptMode(QFileDialog::AcceptOpen);
	path.setFileMode(QFileDialog::ExistingFile);
	path.setNameFilter(tr("All Files(*.*);;DICOM (*.dcm);;Images (*.png *.bmp *.jpg *.tif)"));
	path.setViewMode(QFileDialog::Detail);
	QStringList strFileNameList;
	if (path.exec())
		strFileNameList = path.selectedFiles();
	else
		return false;

	mRepository.clear();
	m_strRepository = "";
	requestSetRepositoryText("");

	QString directory = strFileNameList.at(0);
	QString strFileName = directory.section("/", 0, -2);

	auto filename = directory.section("/", -1);
	if (filename == "case.dat")
	{
		loadfile_.open_loadfile = true;
		loadfile_.load_file_path = directory.toStdString();
		loadfile_.load_path = section(loadfile_.load_file_path,'/', 0, -2);
	}

	if (!directory.isEmpty())
	{
		updateRepositoryInformation(directory.toStdString().c_str());
	}
	return true;
}

// 문자열을 특정 구분자로 나누는 함수
std::vector<std::string> ViewClass::split(const std::string& str, char delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(str);
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}

std::string ViewClass::section(const std::string& path, char delimiter, int start, int end)
{
	std::vector<std::string> parts = split(path, delimiter);
	if (end < 0) {
		end = parts.size() + end;  // 음수 값 처리 (예: -2는 뒤에서 두 번째까지)
	}
	if (end >= parts.size()) {
		end = parts.size() - 1;
	}
	if (start < 0) {
		start = parts.size() + start;  // Read from the right if start is negative
	}

	std::string result;
	for (int i = start; i <= end; ++i) {
		result += parts[i];
		if (i != end) {
			result += delimiter;  // 구분자를 추가
		}
	}
	return result;
}

void ViewClass::updateRepositoryInformation(const std::string& directoryPath,bool load_image)
{
	Repository _repo = Repository();
	_repo.root = section(directoryPath, '/', 0, -2);
	auto file_name = section(directoryPath, '/' , -1);
	
	bool ret = file_name.find('.') != std::string::npos;

	for (const auto& entry : ::filesystem::directory_iterator(_repo.root)) {
		if (!std::filesystem::is_regular_file(entry.status()))
			continue;
		std::string src = entry.path().string();
		if (load_image)
		{
			if (entry.path().has_extension())
			{
				auto strExtension = std::filesystem::path(file_name).extension().string().substr(1); // '.' 제거한 확장자
				std::string type = entry.path().extension().string().substr(1); // 확장자 가져오기
				if (type == "dcm" || type == "DCM")
					_repo.children.push_back(src.c_str());
			}
			else
			{
			//	_repo.children.push_back(src.c_str());
			}
		}
		else
		{
			if (ret)
			{
				if (entry.path().has_extension())
				{
					auto strExtension = std::filesystem::path(file_name).extension().string().substr(1); // '.' 제거한 확장자
					std::string type = entry.path().extension().string().substr(1); // 확장자 가져오기
					if (type == strExtension)
						_repo.children.push_back(src.c_str());
				}
			}
			else
			{
				_repo.children.push_back(src.c_str());
			}
		}
	}
	
	auto strSort = [=](const QString& a, const QString& b)->bool
	{
		return  a.toLower() < b.toLower();
	};
	//auto strSort = [](const std::string& a, const std::string& b) -> bool {
	//	return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
	//		[](char ac, char bc) { return std::tolower(ac) < std::tolower(bc); });
	//};
	
	std::sort(_repo.children.begin(), _repo.children.end(), strSort);
	
	if (!_repo.isEmpty()) {
		mRepository.push_back(_repo);
	}
	m_strRepository = mRepository[0].root;
	requestSetRepositoryText(m_strRepository.c_str());
}

windowMessageBox* ViewClass::CreateMessage(QString strTitle, QString strText, QString btn, bool bModal)
{
	windowMessageBox* masBox = new windowMessageBox(this);
	masBox->setWindowFlags(masBox->windowFlags() & ~Qt::WindowContextHelpButtonHint);
	masBox->setWindowTitle(strTitle);
	masBox->setModal(bModal);
	masBox->setText(strText);
	masBox->SetBtnType(btn);
	auto s = this->rect().center().x() + m_nMoveX - masBox->rect().width() / 2;
	masBox->move(s, this->rect().center().y());
	if (!bModal)
	{
		masBox->show();
		QTimer* timer = new QTimer();
		timer->setSingleShot(true);
		QObject::connect(timer, &QTimer::timeout, [=]()
			{
				masBox->SetDialogCode(QDialog::Rejected);
				masBox->hide();
				delete masBox;
				timer->deleteLater();
			});

		QMetaObject::invokeMethod(timer, "start", Qt::QueuedConnection, Q_ARG(int, 1000));
	}
	QApplication::processEvents();
	return masBox;
}


void ViewClass::copy_file(const std::filesystem::path& srcPath, const std::filesystem::path& dstPath) {
	std::ifstream src(srcPath, std::ios::binary);
	std::ofstream dst(dstPath, std::ios::binary);
	dst << src.rdbuf();
}

void ViewClass::remove_files_with_extensions(const std::filesystem::path& dir) {
	for (const auto& entry : std::filesystem::directory_iterator(dir)) {
		if (entry.is_regular_file()) {
			std::filesystem::remove(entry.path());
		}
	}
}


void ViewClass::copyDir(const std::string& srcDirPath, const std::string& dstDirPath)
{
	//if (m_folderName == "")
	//	return;

	std::filesystem::path src = std::filesystem::path(m_strProgramPath + srcDirPath);
	std::filesystem::path dst = std::filesystem::path(m_strProgramPath + dstDirPath + "\\" + m_folderName + srcDirPath);


	if (!std::filesystem::exists(src)) {
		std::cerr << "Source directory does not exist: " << src << std::endl;
		return;
	}

	if (!std::filesystem::exists(dst)) {
		if (!std::filesystem::create_directories(dst)) {
			std::cerr << "Failed to create destination directory: " << dst << std::endl;
			return;
		}
	}

	remove_files_with_extensions(dst);

	// 파일 및 디렉터리 복사
	for (const auto& entry : std::filesystem::directory_iterator(src)) 
	{
		const auto& srcPath = entry.path();
		auto dstPath = dst / srcPath.filename();

		if (entry.is_directory()) {
			std::filesystem::create_directories(dstPath);  // 하위 디렉터리 생성
			copyDir(srcPath.string(), dstPath.string()); // 하위 디렉터리 복사
		}
		else if (entry.is_regular_file()) {
			copy_file(srcPath, dstPath);  // 파일 복사
		}
	}

}

void ViewClass::removeFilesWithExtensions(const std::string& directoryPath)
{
	// QDir 객체를 생성하고 지정한 경로를 설정합니다.
	QDir dir(directoryPath.c_str());
	// 해당 디렉토리가 유효한지 확인합니다.
	if (!std::filesystem::exists(directoryPath))
	{
		std::cerr << "The directory does not exist: " << directoryPath << std::endl;
		return;
	}

	std::vector<std::string> extensions = { "bmp", "dat", "txt", "plt", "dcm", "stl", "gz", "npy" };

	for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
		if (entry.is_regular_file()) {
			std::string filePath = entry.path().string();
			std::string extension = entry.path().extension().string().substr(1); // '.'를 제거한 확장자

			// 확장자가 목록에 있는지 확인
			if (std::find(extensions.begin(), extensions.end(), extension) != extensions.end()) {
				try {
					std::filesystem::remove(entry.path());
					std::cout << "Removed: " << filePath << std::endl;
				}
				catch (const std::filesystem::filesystem_error& e) {
					std::cerr << "Failed to remove: " << filePath << " - " << e.what() << std::endl;
				}
			}
		}
	}
}


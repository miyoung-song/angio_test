#include "MainWindow.h"

MainWindow::MainWindow() : mTotalImages(0)
{
	setupUi(this);

	mSubwindowActions = new QList<QAction*>();

	setContextMenuPolicy(Qt::PreventContextMenu);

	resize(QSize(700, 480));
	move(QPoint(150, 40));

	setDockNestingEnabled(true);

	centralwidget->setAcceptDrops(true);
	centralwidget->setMouseTracking(true);

	//setCentralWidget(uiMainSplitter);

	mSeriesBrowser = new SeriesBrowserClass(uiTabWidgetLeft);
	mView = new ViewClass(uiTabWidgetCenter);

	uiTabWidgetCenter->setContextMenuPolicy(Qt::CustomContextMenu);
	
	menuContextView = new QMenu(uiTabWidgetCenter);

	actionLine3 = new QAction("Range 3", this);
	actionLine5 = new QAction("Range 5", this);
	actionLine10 = new QAction("Range 10", this);
	actionLine20 = new QAction("Range 20", this);
	actionAutoLine = new QAction("Auto ", this);

	actionLine3->setCheckable(true);
	actionLine5->setCheckable(true);
	actionLine10->setCheckable(true);
	actionLine20->setCheckable(true);
	actionAutoLine->setCheckable(true);
	actionAutoLine->setChecked(true);


	menuContextView->addAction(actionAutoLine);
	menuContextView->addAction(actionLine3);
	menuContextView->addAction(actionLine5);
	menuContextView->addAction(actionLine10);
	menuContextView->addAction(actionLine20);
	uiTabWidgetLeft->setGeometry(QRect(0, 0, 238, 517));
	uiTabWidgetCenter->setGeometry(QRect(349, 0, 617, 517));

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestViewTab,
		mView,
		&ViewClass::updateScene);

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestSetEmptyViewer,
		mView,
		&ViewClass::setEmptyViewer);

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestOneView,
		mView,
		&ViewClass::updateImageView);

	connect(mView, &ViewClass::turnOnViewTab,
		[&] {
		//	uiTabWidgetCenter->setCurrentIndex(1);
			update();
		});

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestClearView,
		mView,
		&ViewClass::clear_result_view);

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestSetViewTab,
		[&](int nIndex) {
		//	uiTabWidgetCenter->setCurrentIndex(nIndex);
			update();
		});

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestSetWindowLW,
		mView,
		&ViewClass::SetWindowLevel);

	connect(mView,
		&ViewClass::requestWindowLevelControl,
		mSeriesBrowser,
		&SeriesBrowserClass::SetWindowLevel);

	connect(mView,
		&ViewClass::spread,
		mSeriesBrowser,
		&SeriesBrowserClass::getDirs);

	connect(mSeriesBrowser,
		&SeriesBrowserClass::requestSetOrigWindowLW,
		mView,
		&ViewClass::SetOrigWindowLevel);

	connect(mView,
		&ViewClass::requestControlEnable,
		mSeriesBrowser,
		&SeriesBrowserClass::SetControlEnable);

	connect(mView,
		&ViewClass::requestSetMenuShow,
		this,
		[&](bool bshow) {
			m_bmenuShow = bshow;
		});

	connect(uiTabWidgetCenter, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(OpenContextMenu(QPoint)));

	connect(actionLine3, &QAction::triggered, this, [this]() {LineRange(3);});
	connect(actionLine5, &QAction::triggered, this, [this]() { LineRange(5); });
	connect(actionLine10, &QAction::triggered, this, [this]() { LineRange(10); });
	connect(actionLine20, &QAction::triggered, this, [this]() { LineRange(20); });
	connect(actionAutoLine, &QAction::triggered, this, [this]() { LineRange(0, true); });
	
	connect(this, &MainWindow::funcLineRange, mView, &ViewClass::SetLineRange);


#ifdef _DEBUG
	QAction* action3D;
	action3D = new QAction("3D", this);
	menuContextView->addAction(action3D);
	connect(action3D, &QAction::triggered, this, [this]() { mView->Open3DFile(); });
#endif
	/* Icon Setup */
	//QPushButton* uiActionPause = new QPushButton;
	uiPushButtonPause->setIcon(QIcon("icons/png/pause_white_48x48.png"));
	uiPushButtonPause->setIconSize(QSize(30, 30));

	uiPushButtonPlay->setIcon(QIcon("icons/png/play_arrow_white_48x48.png"));
	uiPushButtonPlay->setIconSize(QSize(30, 30));

	uiPushButtonClearLine->setIcon(QIcon("icons/png/ClearLine_white_48x48.png"));
	uiPushButtonClearLine->setIconSize(QSize(30, 30));

	uiPushButtonLineUndo->setIcon(QIcon("icons/png/LineUndo_white_48x48.png"));
	uiPushButtonLineUndo->setIconSize(QSize(30, 30));

	uiPushButtonLoad->setIcon(QIcon("icons/png/LoadFile_white_48x48.png"));
	uiPushButtonLoad->setIconSize(QSize(30 ,30));

	uiPushButtonSaveAs->setIcon(QIcon("icons/png/save_white_48x48.png"));
	uiPushButtonSaveAs->setIconSize(QSize(30, 30));

	connect(this, &MainWindow::funcPause, mView, &ViewClass::testPause);
	connect(this, &MainWindow::funcPlay, mView, &ViewClass::testPlay);
	connect(this, &MainWindow::funcLineUndo, mView, &ViewClass::SetLineUndo);
	connect(this, &MainWindow::funcClearLine, mView, &ViewClass::SetClearLine);
	connect(this, &MainWindow::funcLoad, mView, &ViewClass::pbLoad);
	connect(this, &MainWindow::funcSaveAs, mView, &ViewClass::SaveAs);

	connect(uiPushButtonPlay, &QPushButton::clicked, this, [this]() {  emit funcPlay(); });
	connect(uiPushButtonPause, &QPushButton::clicked, this, [this]() {  emit funcPause(); });
	connect(uiPushButtonLineUndo, &QPushButton::clicked, this, [this]() {  emit funcLineUndo(); });
	connect(uiPushButtonClearLine, &QPushButton::clicked, this, [this]() {  emit funcClearLine(); });
	connect(uiPushButtonLoad, &QPushButton::clicked, this, [this]() {  emit funcLoad(); });
	connect(uiPushButtonSaveAs, &QPushButton::clicked, this, [this]() {  emit funcSaveAs(); });

	connect(mView,
		&ViewClass::requestSetRepositoryText,
		[&](QString str) {
			uiRepository->setText(str);
		});
	
	showMaximized();
}

void MainWindow::resizeEvent(QResizeEvent* e)
{
	auto s = uiMainSplitter->geometry().height();
	auto s2 = uiMainSplitter->geometry();
	//auto h = uiToolBarFeatures->geometry().height();

	mSeriesBrowser->setGeometry(-15, 0, uiTabWidgetLeft->geometry().width() + 20, s + 15);

	mView->setGeometry(0, 0, uiTabWidgetCenter->geometry().width(), s);
	widowSize.setWidth(qMax(widowSize.width(), s2.width()));
	widowSize.setHeight(qMax(widowSize.height(), s2.height()));
	mView->setMoveX(uiTabWidgetLeft->geometry().width() + 15);
}

void MainWindow::closeEvent(QCloseEvent* eventConstr) {
	delete mView;
	delete mSeriesBrowser;
	QWidget::closeEvent(eventConstr);
}

template<class ...Args>
inline string MainWindow::appstr(Args&&...args)
{
	std::stringstream ss;
	(ss << ... << std::forward<Args>(args));
	return ss.str();
}

void MainWindow::OpenContextMenu(QPoint pt)
{
	//마우스 우클릭한 위치에 Popup 창으로 menu 출력
	menuContextView->popup(uiTabWidgetCenter->mapToGlobal(pt));

	if (m_bmenuShow)
	{
		menuContextView->show();
	}
	else
	{
		menuContextView->hide();
	}
	update();
	m_bmenuShow = false;
}

MainWindow::~MainWindow()
{

}

void MainWindow::LineRange(int Value ,bool bAuto)
{
	actionLine3->setChecked(Value == 3 && !bAuto);
	actionLine5->setChecked(Value == 5 && !bAuto);
	actionLine10->setChecked(Value == 10 && !bAuto);
	actionLine20->setChecked(Value == 20 && !bAuto);
	actionAutoLine->setChecked(bAuto);
	emit funcLineRange(Value,bAuto);
}


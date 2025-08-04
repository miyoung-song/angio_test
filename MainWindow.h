#pragma once
//#include "Windows.h"

//##include<qwidget.h>

#include<qmimedata.h>
#include<QtGui/QDropEvent>
#include<QtGui/QDragEnterEvent>
#include<qdrag.h>
#include<qmainwindow.h>
#include<qtoolbar.h>
#include<qtoolbutton.h>
#include<qsettings.h>
#include<qsignalmapper.h>
#include<qmdisubwindow.h>
#include<qlabel.h>
#include<qmenu.h>
#include<qspinbox.h>
#include<qdialogbuttonbox.h>
#include<qpushbutton.h>
#include<qfiledialog.h>
#include<qapplication.h>
#include<qmessagebox.h>
#include<qdesktopservices.h>
#include<qdockwidget.h>

#include"dcmHelper.h"

#include"ui_windowMain.h"

#include "SeriesBrowserClass.h"
#include"ViewClass.h"

#define MENU_DP 21
#define TOOLBAR_DP 36

#define Q_QDOC

#define SETTING SettingClass::GetInstance()

class MainWindow : public QMainWindow, public Ui::windowMain
{
	Q_OBJECT
public:
	MainWindow();
	~MainWindow();

	//int mCaptureWebcamImage;
	int mTotalImages;

	void LineRange(int Value, bool bAuto = false);

signals:
	void funcLoad();
	void funcTest();
	void funcPause();
	void funcPlay();
	void funcEdgeLine();
	void funcCenterLine();
	void funcClearLine();
	void funcLineUndo();
	void funcLineRange(int Value,bool bAuto);
	void func3D();
	void funcSaveAs();
	void funcFFR(bool bshow);

private slots:
	void OpenContextMenu(QPoint pt);

private:
	void closeEvent(QCloseEvent*);
	void resizeEvent(QResizeEvent*) override;

	template<class... Args>
	string appstr(Args&&... args);

	SeriesBrowserClass* mSeriesBrowser;
	ViewClass* mView;

	//QSignalMapper* mSignalMapper;
	enum { maxRecentFiles = 8 };
	//QAction* mActionRecentFiles[maxRecentFiles];


	QList<QAction*>* mSubwindowActions;
	
	vector<std::unique_ptr<dcmHelper>> dHelper;
	QMenu* menuContextView;

	QAction* actionAutoLine;
	QAction* actionLine3;
	QAction* actionLine5;
	QAction* actionLine10;
	QAction* actionLine20;
	QAction* actionPointMove;

	bool m_bmenuShow = true;
	QSize widowSize;
};


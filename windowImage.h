
#include<qwidget.h>
#include<qpainter.h>
#include<qscrollarea.h>
#include<qevent.h>
#include<qscrollbar.h>
#include<qgraphicsview.h>

#include "ui_windowImage.h"
#include "dcmHelper.h"
#include "Model.h"

class windowImage : public QDialog, private Ui::windowImage
{
public:
	windowImage(QWidget* mainWindow);
	~windowImage();

	void Prepare(dcmHelper* dh);
	void SetPlayState(bool b);

public:
	QImage* mImage=nullptr;
	QPixmap* mPixmap =nullptr;
	QString mImageZoom;
	QString mImageTime;
	QString mImageKeypoints;
	QString mImageDimensions;
	QString mImageSize;
	QString mImagePointer;
	QString mWindowTitle;
	QString mUid;
	QString mOriginalUid;

	int mWindowType, mFeatureType, mImageN;
	float mCurrentFactor;
	int mcurrentImage;
	int myFrame;
	bool misDhfile = true;
#ifdef DCM_U8
	QVector<unsigned char*> mRaws;
#else
	QVector<float*> mRaws;
#endif

protected:
	//void mousePressEvent(QMouseEvent* event);
	//void mouseMoveEvent(QMouseEvent* event);
	//void mouseReleaseEvent(QMouseEvent* event);
	void paintEvent(QPaintEvent* event);
	void wheelEvent(QWheelEvent* event) override;

private:
	static std::chrono::steady_clock::time_point timer_prev;
	float timer;
	bool isPlay = true;

	int mLR = 0;
	int mRA = 0;
	QPixmap mPixmapOriginal;
	QSize mOriginalSize;
	QPoint mLastPoint;
	QPointF mImageAxis;

	QLocale* mLocale;
	QPainter* mPainter;

	bool mTestActivated;
	bool mMousePressed, mWasPressed;


	bool mModified;
	int mOriginalWidth, mOriginalHeight;
	float mScaleFactorAbove100, mScaleFactorUnder100, mFactorIncrement;

	void moveScene();
	void zoomIn();
	void zoomOut();
	void scaleImage();

private slots:
	void pause();
	void play();
};


#include "windowimage.h"

std::chrono::steady_clock::time_point windowImage::timer_prev = std::chrono::high_resolution_clock::now();

windowImage::windowImage(QWidget* mainWindow)
    : QDialog::QDialog(mainWindow, Qt::Dialog)
{
    setupUi(this);

  //  mPainter = new QPainter(this);
    mLocale = new QLocale(QLocale::English);
//   setWindowFlags(Qt::FramelessWindowHint);
	uiActionPlay->setIcon(QIcon("icons/png/play_arrow_white_48x48.png"));
	uiActionPause->setIcon(QIcon("icons/png/pause_white_48x48.png"));

	connect(uiActionPause, &QPushButton::clicked, this, &windowImage::pause);
	connect(uiActionPlay, &QPushButton::clicked, this, &windowImage::play);
}

windowImage::~windowImage()
{
	for (auto i = 0; i < mRaws.size(); i++)
		SafeReleaseAry(mRaws[i]);
	mRaws.clear();
}

void windowImage::Prepare(dcmHelper* dh)
{
	misDhfile = dh->loadFile();
	if (misDhfile)
	{
		auto _ptr = dh->Data();
		mOriginalHeight = dh->getRows();
		mOriginalWidth = dh->getCols();
		size_t stride = mOriginalHeight * mOriginalWidth;
		for (auto i = 0; i < dh->getNumberOfFrames(); i++, _ptr += stride)
		{
			mRaws.append(new unsigned char[mOriginalWidth * mOriginalHeight]);
			for (int n = 0; n < mOriginalHeight * mOriginalWidth; n++)
			{
				auto val = (std::clamp(255.0 * ((_ptr[n] - (dh->getWindowCenter() - 0.5)) / (dh->getWindowWidth() - 1) + 0.5), 0.0, 255.0));
				_ptr[n] = val;
			}
			memcpy(mRaws.back(), _ptr, sizeof(unsigned char) * mOriginalHeight * mOriginalWidth);
		}
		mcurrentImage = 0;
		myFrame = dh->getNumberOfFrames();

		mLR = dh->getValuebyGE(0x0018, 0x1510).toFloat();
		mRA = dh->getValuebyGE(0x0018, 0x1511).toFloat();

		moveScene();

		mScaleFactorAbove100 = 0.5;
		mScaleFactorUnder100 = 0.25;
		mFactorIncrement = 0;
		mCurrentFactor = 1.0;

		//mOriginalSize = mImage.size();
		//mOriginalWidth = mImage.width();
		//mOriginalHeight = mImage.height();
		//
		//mImageZoom = tr("%1%").arg((int)(mCurrentFactor * 100));
		//mImageDimensions = tr("%1x%2 px").arg(mOriginalWidth).arg(mOriginalHeight);
		//float sizeInKiB = mImage.byteCount() / (float)1024;
		//if (sizeInKiB > 1024)
		//	mImageSize = mLocale->toString(sizeInKiB / (float)1024, 'f', 2).append(" MiB");
		//else mImageSize = mLocale->toString(sizeInKiB, 'f', 2).append(" KiB");
		//
		//mImagePointer = tr("%3, %3").arg(0).arg(0);
		//mImageAxis = QPointF(0, 0);
		//mMousePressed = false;
		//mWasPressed = false;
		SetPlayState(true);
	}
	else
	{
		SafeReleasePtr(mImage);
		SafeReleasePtr(mPixmap);
		mImage = (new QImage(dh->getFile()->c_str()));
		mImage->convertToFormat(QImage::Format_Grayscale8);
		mOriginalHeight = mImage->height();
		mOriginalWidth = mImage->width();
		size_t stride = mOriginalHeight * mOriginalWidth;
		mRaws.append(new unsigned char[mOriginalWidth * mOriginalHeight *3]);
		
		int r, g, b;
		auto qImage_copy = new QImage(mImage->width(), mImage->height(), QImage::Format::Format_Grayscale8);
		for (int i = 0; i < mOriginalHeight; i++)
		{
			for (int j = 0; j < mOriginalWidth; j++)
			{
				QRgb rgb = mImage->pixel(i, j);
				r = g = b = (qRed(rgb) + qGreen(rgb) + qBlue(rgb)) / 3;
				qImage_copy->setPixel(i, j, qRgb(r, g, b));
			}
		}

		int stride1 = qImage_copy->bytesPerLine();
		for (int row = 0; row < mOriginalHeight; row++)
		{
			memcpy(mRaws[0] + mOriginalWidth  * row, qImage_copy->bits() + row * stride1, mOriginalWidth ); // copy a single row, accounting for stride bytes
		}

		//memcpy(mRaws[0], mImage->bits(), sizeof(unsigned char) * mOriginalHeight * mOriginalWidth*3);
		mcurrentImage = 0;
		myFrame = 1;

		SafeReleasePtr(qImage_copy);
	}
}

void windowImage::SetPlayState(bool b)
{
	this->isPlay = b;
	timer_prev = std::chrono::high_resolution_clock::now(); 
	timer = 0;
}

//void windowImage::mousePressEvent(QMouseEvent* event)
//{
//}
//
//void windowImage::mouseMoveEvent(QMouseEvent* event)
//{
//}
//
//void windowImage::mouseReleaseEvent(QMouseEvent* event)
//{
//}

void windowImage::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	QColor fontColor = QColor(255, 255, 255, 255);
	QColor dynaColor = QColor(0, 190, 245, 255);

	painter.beginNativePainting();
	painter.endNativePainting();
	painter.setPen(fontColor);
	painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::SmoothPixmapTransform);
	painter.setFont(QFont("Arial", 10));
	painter.setPen(QColor(255, 255, 255, 255));
	painter.drawText(QPoint(int(450), int(20)), QString("Frame: %1").arg(mcurrentImage));

	if (isPlay)
	{
		auto _timer = std::chrono::high_resolution_clock::now();
		auto dd = _timer - timer_prev;
		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(dd);
		auto ggg = int(d.count() / (1000.0f / 15));
		if (ggg > timer)
		{
			mcurrentImage++;
			if (myFrame == mcurrentImage)
			{
				mcurrentImage = 0;
				isPlay = false;
			}
			else if (mcurrentImage < 0)
				mcurrentImage = myFrame - 1;
			else if (myFrame < mcurrentImage)
			{
				mcurrentImage = 0;
				isPlay = false;
			}

			moveScene();
			timer += 1;
		}
	}
	update();
}

void windowImage::wheelEvent(QWheelEvent* event)
{
	mcurrentImage++;
	if (myFrame == mcurrentImage)
	{
		mcurrentImage = 0;
		isPlay = false;
	}
	else if (mcurrentImage < 0)
		mcurrentImage = myFrame - 1;

	moveScene();
}

void windowImage::moveScene()
{
	SafeReleasePtr(mImage);
	SafeReleasePtr(mPixmap);
	mImage = (new QImage(mRaws[mcurrentImage], mOriginalWidth, mOriginalHeight, QImage::Format::Format_Grayscale8));
	mPixmap = new QPixmap();
	mPixmap->convertFromImage(*mImage);
	uiImage->setPixmap(mPixmap->scaled(uiImage->width(), uiImage->height(),Qt::KeepAspectRatio));
	//uiImage->resize(mOriginalWidth, mOriginalHeight);
	//uiImage->move(0, 0);
}

void windowImage::zoomIn()
{
	if (mCurrentFactor >= 1.0) {
		mFactorIncrement = (mCurrentFactor + mScaleFactorAbove100) / mCurrentFactor;
		mCurrentFactor += mScaleFactorAbove100;
	}
	else {
		mFactorIncrement = (mCurrentFactor + mScaleFactorUnder100) / mCurrentFactor;
		mCurrentFactor += mScaleFactorUnder100;
	}
	scaleImage();
}

void windowImage::zoomOut()
{
	if (mCurrentFactor > 1.0) {
		mFactorIncrement = (mCurrentFactor - mScaleFactorAbove100) / mCurrentFactor;
		mCurrentFactor -= mScaleFactorAbove100;
	}
	else {
		mFactorIncrement = (mCurrentFactor - mScaleFactorUnder100) / mCurrentFactor;
		mCurrentFactor -= mScaleFactorUnder100;
	}
	scaleImage();
}

void windowImage::scaleImage()
{
	mImageZoom = tr("%1%").arg((int)(mCurrentFactor * 100));
}

void windowImage::play()
{
	
	SetPlayState(true);
}

void windowImage::pause()
{
	SetPlayState(false);
}



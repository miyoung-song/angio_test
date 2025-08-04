#pragma once

//#include<QtWidgets/QStackedWidget>
#include<QtWidgets/QListWidgetItem>
#include<QtWidgets/QTableWidgetItem>
#include<QtWidgets/QListWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QHeaderView>


#include<qstring.h>
#include<qstringlist.h>

#include<qpixmap.h>
#include<qpainter.h>

#include<qevent.h>
#include<qwidget.h>

#include<qdrag.h>
#include<qmimedata.h>
#include<qdatastream.h>
#include<qpushbutton.h>

#include <qthread.h>
#include <QTextCodec>

#include "dcmHelper.h"
#include "ListWidget.h"

#include "ui_windowSeriesBrowser.h"

class Serieses
{
public:

	Serieses() {
		_dh = new dcmHelper();
		_item = new QListWidgetItem();
		_pixmap = new QPixmap();
	};
	~Serieses()
	{
		fileName.clear();
		SafeReleasePtr(_dh);// delete _dh;
		SafeReleasePtr(_item);// delete _item;
		SafeReleasePtr(_pixmap);
		SafeReleasePtr(_origin);
		//if (_origin)delete _origin;
	}
	void InitImg(bool t)
	{
		if (t)
		{
			_origin = new QImage(_dh->getData(), _dh->getCols(), _dh->getRows(), QImage::Format::Format_Grayscale8);

			float LR = _dh->getValuebyGE(0x0018, 0x1510).toFloat();
			float RA = _dh->getValuebyGE(0x0018, 0x1511).toFloat();


			_pixmap->convertFromImage(*_origin);

			QPainter painter(_pixmap);
			//int step = int(_dh->getCols() / 15);
			painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::SmoothPixmapTransform);
			painter.setFont(QFont("Arial", 38));
			painter.setPen(QColor(255, 255, 255, 255));
			painter.drawText(QPoint(int(_dh->getCols() * 0.83f), int(_dh->getRows() * 0.17f)), QString("%1").arg(_dh->getNumberOfFrames()));
			painter.drawText(QPoint(int(_dh->getCols() * 0.07f), int(_dh->getRows() * 0.9f)), QString("%1 %2 / %3 %4")
				.arg(LR > 0 ? "LAO" : "RAO").arg(abs(int(LR)))
				.arg(RA > 0 ? "CRA" : "CAU").arg(abs(int(RA))));

			QByteArray itemData;

			QDataStream ds(&itemData, QIODevice::WriteOnly);

			//QByteArray ba(_dh, size);

			//_item->setData(Qt::UserRole, QVariant::fromValue(*_dh));
			_item->setData(Qt::UserRole, QVariant::fromValue(QString::fromLocal8Bit(_dh->getFile()->c_str())));
		}
		else
		{
			QString path = QString::fromStdString(_dh->getFile()->c_str());
			_origin = new QImage(path);

			_pixmap->convertFromImage(*_origin);
			QPainter painter(_pixmap);
			//int step = int(_dh->getCols() / 15);
			painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::SmoothPixmapTransform);
			painter.setFont(QFont("Arial", 38));
			painter.setPen(QColor(255, 255, 255, 255));
		//	painter.drawText(QPoint(int(512* 0.83f), int(512 * 0.17f)), QString("%1").arg(1));
		
			QByteArray itemData;

			QDataStream ds(&itemData, QIODevice::WriteOnly);
			_item->setData(Qt::UserRole, QVariant::fromValue(QString::fromLocal8Bit(_dh->getFile()->c_str())));
		}
	}

	void setItembyScale(const int& s)
	{
		_item->setData(Qt::DecorationRole, _pixmap->scaled(s, s).copy());
	}
	void SetFileName(QString  str)
	{
		fileName.push_back(str);
	}

	void relDCMHelper() { SafeReleasePtr(_dh); }

	dcmHelper* getDCMHelper() { return _dh; };
	QListWidgetItem** getQListWidgetItem() { return &_item; };
	QString GetFileName(int i) { return fileName[i]; };
private:
	dcmHelper* _dh;
	//unique_ptr<dcmHelper> _dh;
	QListWidgetItem* _item;
	QImage* _origin;
	QPixmap* _pixmap;
	std::vector<QString> fileName;
	//mm* _m;
};

Q_DECLARE_METATYPE(Serieses)

class dcmThread : public QThread
{
	Q_OBJECT
public:
	//dcmThread() {};
	~dcmThread() override
	{
		wait(10000);
		//delete _buf;
	};
	void setObj(Serieses* dh,const int& val)
	{
		//_buf = new Serieses();
		_buf = dh;
		_bufVal = val;
		//mName = name;
	}
protected:
	void run() override {
		//dcmHelper dh(mName.toLocal8Bit().constData());
		auto t = _buf->getDCMHelper()->loadFile(0.75f);
		_buf->InitImg(t);
		_buf->setItembyScale(_bufVal);
		//dh.getData();
		
		emit resultReady(_buf);
	}
signals:
	void resultReady(Serieses*);

private:
	//QString mName;
	Serieses* _buf;
	int _bufVal;
	
};

class SeriesBrowserClass: public QWidget, Ui::windowSeriesBrowser
{
	Q_OBJECT
private:
	struct _sub
	{
		QWidget* w;
		QVBoxLayout* layout;
		_sub()
		{
			w = new QWidget();
			layout = new QVBoxLayout();
		}
		~_sub()
		{
			SafeReleasePtr(w);
			SafeReleasePtr(layout);
		}
	};
public:
	explicit SeriesBrowserClass(QWidget* parent = nullptr);
	~SeriesBrowserClass();
	void SetWindowLevel(int nwl, int nww);
	void SetControlEnable(bool bEnabled =false);

public slots:
	void getDirs(const QStringList& ql, const QString& title, bool bLoad);

	void handleDCM(Serieses*);

	void ChagnedDCM(QListWidgetItem* item);


private slots:

	void moveSlider();
	
	void SetEdit();

	void closeTab(int);

	void ClikedWL();
	void ClikedWW();
signals:
	void requestItemText(QTableWidgetItem*);
	void requestViewTab(dcmHelper*);
	void requestSetEmptyViewer(int nidex =0);
	void requestOneView(QString);
	void requestClearView();
	void requestSetViewTab(int nIndex);
	void requestSetWindowLW(int nWWVaule, int nWLVaule);
	void requestSetOrigWindowLW(int nIndex);
private:
	bool mbLoad = false;
	QList< Serieses*> mHelpers;
	QList<_sub*> mQSub;
	std::vector<int> mWW;
	std::vector<int> mWC;

};


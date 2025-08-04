#include "SeriesBrowserClass.h"


#include <QDrag>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QMimeData>

SeriesBrowserClass::SeriesBrowserClass(QWidget* parent) : QWidget(parent)
{
	setupUi(this);
	qRegisterMetaType<Serieses>();

	QIntValidator* intValidator = new QIntValidator(0, 255);
	uiEditWL->setValidator(intValidator);
	uiEditWW->setValidator(intValidator);
	connect(uiEditWL, &QLineEdit::editingFinished, this, &SeriesBrowserClass::SetEdit);
	connect(uiEditWW, &QLineEdit::textChanged, this, &SeriesBrowserClass::SetEdit);
	connect(uihorizontalSliderWL, &QAbstractSlider::valueChanged, this, &SeriesBrowserClass::moveSlider);
	connect(uihorizontalSliderWW, &QAbstractSlider::valueChanged, this, &SeriesBrowserClass::moveSlider);
	connect(uiTabWidget, &QTabWidget::tabCloseRequested, this, &SeriesBrowserClass::closeTab);
	connect(uiBtnWL, &QPushButton::clicked, this, &SeriesBrowserClass::ClikedWL);
	connect(uiBtnWW, &QPushButton::clicked, this, &SeriesBrowserClass::ClikedWW);
}

SeriesBrowserClass::~SeriesBrowserClass()
{
}

void SeriesBrowserClass::getDirs(const QStringList& ql, const QString& title, bool bLoad)
{
	emit requestClearView();
	int nCnt = uiTabWidget->count();
	for (int i = 0; i < nCnt; i++)
	{
		closeTab(uiTabWidget->count() - 1);
	}

	if (ql.size() == 0)
		return;

	mbLoad = bLoad;
	if (mbLoad)
	{
		mWC.clear();
		mWW.clear();
		emit requestSetEmptyViewer();
	}

	auto type = ql[0].section(".", -1);

	mQSub.append(new _sub);
	auto w = mQSub.back()->w;

	auto layout = mQSub.back()->layout;
	layout->addWidget(new ListWidget(190, uiTabWidget));

	w->setLayout(layout);

	uiTabWidget->insertTab(uiTabWidget->count(), w, title);
	uiTabWidget->setCurrentIndex(uiTabWidget->count() - 1);

	auto _list = static_cast<ListWidget*>(uiTabWidget->currentWidget()->layout()->itemAt(0)->widget());

	mHelpers.append(new Serieses[ql.size()]);
	auto _buffer = mHelpers.back();// vv;// new Serieses[ql.size()];
	connect(_list, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(ChagnedDCM(QListWidgetItem*)));
	for (int i = 0; i < ql.size(); i++)
	{
		dcmThread worker;
		QTextCodec* codec = QTextCodec::codecForName("eucKR");
		QString path = QString(ql.at(i)).replace('/', "\\\\");
		QString encodedString = codec->toUnicode(path.toLocal8Bit().constData());
		_buffer[i].getDCMHelper()->setFile(encodedString.toLocal8Bit().constData());
		_buffer->SetFileName(encodedString);
		_list->addItem(*_buffer[i].getQListWidgetItem());
		mWC.push_back(_buffer[i].getDCMHelper()->getWindowCenter());
		mWW.push_back(_buffer[i].getDCMHelper()->getWindowWidth());
		worker.setObj(&_buffer[i], 190);
		connect(&worker, &dcmThread::resultReady, this, &SeriesBrowserClass::handleDCM);
		connect(&worker, &dcmThread::finished, &worker, &QObject::deleteLater);
		worker.start();
	}
	if (ql.size() == 1)
		emit requestSetEmptyViewer(1);
}


void SeriesBrowserClass::handleDCM(Serieses* _ptr)
{
	if (!mbLoad)
		emit requestViewTab(_ptr->getDCMHelper());
	_ptr->relDCMHelper();
}


void SeriesBrowserClass::moveSlider()
{
	emit requestSetWindowLW(uihorizontalSliderWL->value(), uihorizontalSliderWW->value());
	uiEditWL->setText(QString::number(uihorizontalSliderWL->value()));
	uiEditWW->setText(QString::number(uihorizontalSliderWW->value()));
	update();
}

void SeriesBrowserClass::SetEdit()
{
	emit requestSetWindowLW(uiEditWL->text().toInt(), uiEditWW->text().toInt());
	uihorizontalSliderWL->setValue(uiEditWL->text().toInt());
	uihorizontalSliderWW->setValue(uiEditWW->text().toInt());
}

void SeriesBrowserClass::SetWindowLevel(int nwl, int nww)
{
	uiEditWL->setText(QString::number(nwl));
	uiEditWW->setText(QString::number(nww));
	uihorizontalSliderWL->setValue(nwl);
	uihorizontalSliderWW->setValue(nww);
	SetControlEnable(false);
}

void SeriesBrowserClass::SetControlEnable(bool bEnabled)
{
	uiEditWL->setEnabled(bEnabled);
	uiEditWW->setEnabled(bEnabled);
	uihorizontalSliderWL->setEnabled(bEnabled);
	uihorizontalSliderWW->setEnabled(bEnabled);
}

void SeriesBrowserClass::closeTab(int index)
{
	uiTabWidget->removeTab(index);
	auto qq = mHelpers[index];
	mHelpers.erase(mHelpers.begin() + index);
	mQSub.erase(mQSub.begin() + index);

	emit requestClearView();
	if (uiTabWidget->count() == 0)
		emit requestSetViewTab(0);
	delete[] qq;
}

void SeriesBrowserClass::ChagnedDCM(QListWidgetItem* item)
{
	if (mHelpers[uiTabWidget->currentIndex()])
	{
		auto _list = static_cast<ListWidget*>(uiTabWidget->currentWidget()->layout()->itemAt(0)->widget());
		int index = _list->row(item);
		auto _now = mHelpers[uiTabWidget->currentIndex()]->GetFileName(index);
		emit requestOneView(_now);
	}
};

void SeriesBrowserClass::ClikedWW()
{
	emit requestSetOrigWindowLW(dcmHelper::WindowLevel::WindowWidth);
}

void SeriesBrowserClass::ClikedWL()
{
	emit requestSetOrigWindowLW(dcmHelper::WindowLevel::WindowCenter);
}


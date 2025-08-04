#pragma once
#include<qlistwidget.h>
#include<qdrag.h>
#include<qbytearray.h>
#include<qdatastream.h>
#include<qmimedata.h>
#include<qpixmap.h>
#include<qpainter.h>

class ListWidget : public QListWidget
{
public:
	explicit ListWidget(const int& init,QWidget* parent = nullptr)
	{
		
		setParent(parent);
		
		this->setIconSize(QSize(init, init));
		this->setGridSize(QSize(init + 3, init + 3));
		//this->setSpacing(10);
		this->setViewMode(ListWidget::IconMode);
		this->setMovement(QListView::Movement::Snap);
		this->setResizeMode(ListWidget::Adjust);

		this->setDragEnabled(true);
		this->setDragDropMode(QAbstractItemView::DragOnly);
	}
	using QListWidget::QListWidget;

protected:
	void startDrag(Qt::DropActions supportedActions) {
		QDrag* drag = new QDrag(this);
		//auto idx = selectedIndexes();
		auto cur = currentIndex();
		//auto aaaaa = model()->data(cur, Qt::UserRole).value<dcmHelper>();

		QByteArray item;
		QDataStream ds(&item, QIODevice::WriteOnly);

		//auto _buf = unique_ptr<dcmHelper>();

		//auto _buf = new dcmHelper;
		//_buf->copy(aaaaa);

		auto _buf = new QString(model()->data(cur, Qt::UserRole).value<QString>());
		//auto pixmap = new model

		auto _pix = new QPixmap(model()->data(cur, Qt::DecorationRole).value<QPixmap>());

		//*_buf = aaaaa;
		//ds << model()->data(cur, Qt::UserRole);
		ds << *_buf;
		//ds << QVariantPointer<dcmHelper>::toQVariant(aaaaa);

		auto md = new QMimeData;
		md->setData("application/x-locPath", item);

		drag->setMimeData(md);
		//drag->setMimeData(model()->mimeData(idx));

		//auto q = viewport()->visibleRegion().boundingRect().size();
		//auto q1 = rectForIndex(cur).size();
		//QPixmap pixmap(viewport()->visibleRegion().boundingRect().size());
		//auto ww = rectForIndex(cur).width();
		//auto hh = rectForIndex(cur).height();
		QPixmap pixmap(visualRect(cur).x()+_pix->size().width(), visualRect(cur).y() + _pix->size().height());
		pixmap.fill(Qt::transparent);
		QPainter painter(&pixmap);
		//painter.setFont(QFont("Arial", 34));
		//painter.setPen(QColor(255, 255, 255, 255));
		//painter.drawText(QPoint(100, 100), QString("TTTTTTTTTTTTTTT"));
		//painter.drawPixmap(visualRect(cur)/*rectForIndex(cur)*/, viewport()->grab(visualRect(cur)));
		painter.drawPixmap(visualRect(cur).x(), visualRect(cur).y(), _pix->size().width(), _pix->size().height(), *_pix);
		//for (QModelIndex index : selectedIndexes()) {
		//	painter.drawPixmap(visualRect(index), viewport()->grab(visualRect(index)));
		//}
		//drag->setUserData
		drag->setPixmap(pixmap);
		drag->setHotSpot(viewport()->mapFromGlobal(QCursor::pos()));
		drag->exec(Qt::CopyAction);

		//_buf->release();
		delete _buf;
		//emit grabDH(&drag, idx);
	};

	//signals:
	//	void grabDH(QDrag**,const QModelIndexList&);
};
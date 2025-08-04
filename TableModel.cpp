#include "TableModel.h"

TableModel::TableModel(QObject* parent) : QAbstractTableModel(parent)
{
}

int TableModel::rowCount(const QModelIndex& parent) const
{
	Q_UNUSED(parent);
	return mComponent->length();
}

int TableModel::columnCount(const QModelIndex& parent) const
{
	Q_UNUSED(parent);
	return mComponentCount;
}

QVariant TableModel::data(const QModelIndex& index, int role) const
{
	if (!index.isValid() || role != Qt::DisplayRole)
	{
		return QVariant();
	}

	if(index.column()<0 || index.column()>=mComponentCount)
		return QVariant();

	return (mComponent + index.column())->at(index.row());
	
	/*switch (index.column())
	{
	case 0:
		return GE[index.row()];
	case 1:
		return Tag[index.row()];
	case 2:
		return Value[index.row()];

	default:
		break;
	}*/
	
	return QVariant();
}

QVariant TableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role == Qt::DisplayRole && orientation == Qt::Horizontal)
	{
		return mHeader.at(section);
		//switch (section)
		//{
		//case 0:
		//	return QString("(Group,Element)");
		//case 1:
		//	return QString("TAG Description");
		//case 2:
		//	return QString("Value");
		//default:
		//	break;
		//}
	}

	return QVariant();
}

void TableModel::reserve(const int& sz)
{
	for (int i = 0; i < mComponentCount; i++)
		mComponent[i].reserve(sz);
	//GE.reserve(sz);
	//Tag.reserve(sz);
	//Value.reserve(sz);
}

template<typename ...Args>
inline void TableModel::setHeader(const Args&& ...args)
{
	//QList< QString> _buffer;
	(mHeader.push_back(std::forward<Args>(args)), ...);
	mComponentCount = mHeader.size();

}

template<typename ...Args>
inline void TableModel::append(const Args&& ...args)
{
	auto ptr = mComponent;
	(ptr++->push_back(std::forward<Args>(args)), ...);
	//GE.append(_ge);
	//Tag.append(_tag);
	//Value.append(_val);
}


void TableModel::setHeader(QList<QString> aa)
{
	this->mHeader = aa;
}

void TableModel::populateData(const QList<QString>* info)
{
	auto _ii = info;
	for (int i = 0; i < mComponentCount; i++)
	{
		mComponent[i].clear();
		mComponent[i] = *_ii;
	}
	//GE.clear();
	//Tag.clear();
	//Value.clear();

	//GE = info[0];
	//Tag = info[1];
	//Value = info[2];

	//emit dataChanged(index(0, 0), index(rowCount(), columnCount()));
	return;
}

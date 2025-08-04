#pragma once
#include<string>
#include<QtCore/QAbstractTableModel>

class TableModel : public QAbstractTableModel
{
	Q_OBJECT
public:
	TableModel(QObject* parent = 0);

	int rowCount(const QModelIndex& parent = QModelIndex()) const Q_DECL_OVERRIDE;
	int columnCount(const QModelIndex& parent = QModelIndex()) const Q_DECL_OVERRIDE;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

	template<typename ...Args>
	void setHeader(const Args&&...args);
	
	void setHeader(QList<QString> aa);

	void populateData(const QList<QString>*);
	void reserve(const int&);
	template<typename ...Args>
	void append(const Args&&...args);
	
private:
	QList<QString>* mComponent;// GE, Tag, Value;
	QList<QString> mHeader;
	int mComponentCount;
};

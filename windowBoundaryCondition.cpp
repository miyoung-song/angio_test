#include "windowBoundaryCondition.h"

windowBoundaryCondition::windowBoundaryCondition(QWidget *parent) 
    :QDialog(parent,Qt::Dialog)
{
    setupUi(this);

    connect(btnStop, &QPushButton::clicked, this, &windowBoundaryCondition::BtnClickedCancel);
    connect(btnRun, &QPushButton::clicked, this, &windowBoundaryCondition::BtnClickedApply);
    this->setFixedSize(200, 100);
}

windowBoundaryCondition::~windowBoundaryCondition()
{
}


void windowBoundaryCondition::BtnClickedApply()
{
    this->accept();
}

void windowBoundaryCondition::BtnClickedCancel()
{
    this->reject();
}
#ifndef WINDOWBOUNDARYCONDITION_H
#define WINDOWBOUNDARYCONDITION_H

#include <QDialog>

#include "ui_windowBoundaryCondition.h"


class windowBoundaryCondition : public QDialog, private Ui::windowBoundaryCondition
{
    Q_OBJECT

public:
    explicit windowBoundaryCondition(QWidget *parent = nullptr);
    ~windowBoundaryCondition();

private slots:
    void BtnClickedApply();
    void BtnClickedCancel();
};

#endif // WINDOWBOUNDARYCONDITION_H

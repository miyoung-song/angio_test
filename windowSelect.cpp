#include "windowSelect.h"

windowSelect::windowSelect(QWidget* mainWindow)
    : QDialog::QDialog(mainWindow, Qt::Dialog)
{
    setupUi(this);

    connect(uiRadioButtonAll, &QRadioButton::clicked, this, &windowSelect::ClickedRadioButton);
    connect(uiRadioButtonLeft, &QRadioButton::clicked, this, &windowSelect::ClickedRadioButton);
    connect(uiRadioButtonRight, &QRadioButton::clicked, this, &windowSelect::ClickedRadioButton);
    connect(uiCancel, &QPushButton::clicked, this, &windowSelect::BtnClickedCancel);
    connect(uiApply, &QPushButton::clicked, this, &windowSelect::BtnClickedApply);
}

windowSelect::~windowSelect()
{
   // delete this;
}

void windowSelect::ClickedRadioButton()
{
    if (uiRadioButtonAll->isChecked())
        m_selectIndex = AllImage;
    else if(uiRadioButtonLeft->isChecked())
        m_selectIndex = LeftImage;
    else
        m_selectIndex = RightImage;
}

void windowSelect::BtnClickedCancel()
{
    m_selectIndex = -1;
    this->reject();
}

void windowSelect::BtnClickedApply()
{
    if (uiRadioButtonAll->isChecked())
        m_selectIndex = AllImage;
    else if (uiRadioButtonLeft->isChecked())
        m_selectIndex = LeftImage;
    else
        m_selectIndex = RightImage;
    this->accept();
}
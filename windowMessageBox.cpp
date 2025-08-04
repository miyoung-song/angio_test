#include "windowMessageBox.h"

windowMessageBox::windowMessageBox(QWidget* parent)
    :QDialog(parent, Qt::Dialog)
{
    setupUi(this);
    connect(btnNo, &QPushButton::clicked, this, &windowMessageBox::BtnClickedCancel);
    connect(btnYes, &QPushButton::clicked, this, &windowMessageBox::BtnClickedApply);
}

windowMessageBox::~windowMessageBox()
{
}

void windowMessageBox::setText(QString str)
{
    label->setText(str);
    label->setAlignment(Qt::AlignCenter);
}

void windowMessageBox::SetBtnType(QString str)
{
    if (str == "Close")
    {
        btnYes->setVisible(false);
        btnNo->setText("Close");
    }
    else if (str == "No")
    {
        btnYes->setVisible(true);
        btnNo->setText("No");
    }
    else if (str == "Run")
    {
        btnYes->setVisible(true);
        btnYes->setText("Yes");
        btnNo->setText("No");
    }
}

void windowMessageBox::Select(QString str)
{
    btnYes->setVisible(true);
    btnYes->setText("Left");
    btnNo->setText("Right");
}

void windowMessageBox::BtnClickedApply()
{
    m_nDialogCode = QDialog::Accepted;
    this->accept();
    select_ = 0;
}

void windowMessageBox::BtnClickedCancel()
{
    m_nDialogCode = QDialog::Rejected;
    this->reject();
    select_ = 1;
}

void windowMessageBox::SetDialogCode(int DialogCode)
{
    m_nDialogCode = DialogCode;
    if (m_nDialogCode == QDialog::Accepted)
        this->accept();
    else
        this->reject();
}

void windowMessageBox::closeEvent(QCloseEvent* event)
{
}

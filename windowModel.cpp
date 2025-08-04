#include "windowModel.h"

windowModel::windowModel(QWidget *parent) 
	: QDialog::QDialog(parent, Qt::Dialog)
{
    setupUi(this);
    connect(btnStop, &QPushButton::clicked, this, &windowModel::BtnClickedCancel);
    connect(btnRun, &QPushButton::clicked, this, &windowModel::BtnClickedApply);
    m_selectIndex_L = model_info::segmentation_model_type::lad;
    m_selectIndex_R = model_info::segmentation_model_type::lad;

    radio_LAD_L->setChecked(true);
    radio_LAD_R->setChecked(true);
}

windowModel::~windowModel()
{
}

void windowModel::ClickedRadioButton()
{
}

void windowModel::BtnClickedCancel()
{
    this->reject();
}

void windowModel::BtnClickedApply()
{
    if (radio_LAD_L->isChecked())
        m_selectIndex_L = model_info::segmentation_model_type::lad;
    else if (radio_LCX_L->isChecked())
        m_selectIndex_L = model_info::segmentation_model_type::lcx;
    else if(radio_RCA_L->isChecked())
        m_selectIndex_L = model_info::segmentation_model_type::rca;


    if (radio_LAD_R->isChecked())
        m_selectIndex_R = model_info::segmentation_model_type::lad;
    else if (radio_LCX_R->isChecked())
        m_selectIndex_R = model_info::segmentation_model_type::lcx;
    else if(radio_RCA_R->isChecked())
        m_selectIndex_R = model_info::segmentation_model_type::rca;
    this->accept();
}

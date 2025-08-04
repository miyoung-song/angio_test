
#include <QDialog>
#include "ui_windowmodel.h"
#include <Model_Info.h>

class windowModel : public QDialog , private Ui::windowModel
{
    Q_OBJECT

public:
    explicit windowModel(QWidget *parent = nullptr);
    ~windowModel();


    model_info::segmentation_model_type GetModelL() { return m_selectIndex_L; }
    model_info::segmentation_model_type GetModelR() { return m_selectIndex_R; }
private slots:
    void ClickedRadioButton();
    void BtnClickedCancel();
    void BtnClickedApply();
private:
    model_info::segmentation_model_type m_selectIndex_R;
    model_info::segmentation_model_type m_selectIndex_L;
};


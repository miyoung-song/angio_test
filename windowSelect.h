
#include "ui_windowSelect.h"


class windowSelect : public QDialog , private Ui::windowSelect 
{
public:
    windowSelect(QWidget* mainWindow);
    ~windowSelect();
    

    int GetImageIndex() { return m_selectIndex; }
private slots:
    void ClickedRadioButton();
    void BtnClickedCancel();
    void BtnClickedApply();
    
private:
    int m_selectIndex = -1;
    enum { LeftImage, RightImage, AllImage };

};


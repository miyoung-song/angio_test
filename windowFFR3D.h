#ifndef WINDOWFFR3D_H
#define WINDOWFFR3D_H

#include <QDialog>
#include "ui_windowFFR3D.h"
#include "GraphicsWindow.h"
#include "ChartView.h"

class windowFFR3D : public QDialog, Ui::windowFFR3D
{
    Q_OBJECT

public:
    explicit windowFFR3D(QWidget * mainWindow);
    ~windowFFR3D();

    void reject();

    void clear_result_view();
    void SetLog(FILE* (&f), QString FileName, QString str, int ntype, bool isclose = false);
    void SetResultFFRView(int ntype, QString filename);
    void SetResultChart(bool b3D, std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY);
    void SetMulPSAngle(const float& _p0, const float& _s0, const float& _p1, const float& _s1, const float& _v);

    void SetGridIndex(int viewIndex);

    void SetFFR(bool bRun);

    void Show(bool bshow);

    void closeEvent(QCloseEvent* event);
    bool GetShowWindow() { return m_bShow; };
    bool GetFFRRun() { return m_bFFRRun; };


private:
    GraphicsWindow* mGraphics= nullptr;
    ChartView* mChartForm = nullptr;
    QWidget* gridLayoutWidget_3 = nullptr;
    bool m_bShow = false;
    int m_nWidth = 0;
    int m_nHeight = 0;
    bool m_isCreate = false;
    int m_nVeiwIndex = 0;
    bool m_bFFRRun = false;
    bool isReject = false;
    vector<int> m_nMatchingPointIds;
};

#endif // WINDOWFFR3D_H

#include <QtCharts>
#include <QChart>

#include <QtCharts/QChartView>
#include <QtWidgets/QRubberBand>
#include "ChartCallout.h"
#include "Functor.h"


QT_CHARTS_USE_NAMESPACE

using namespace std;

struct ChartData
{
public:
    float lfX_Max = FLT_MIN;
    float lfX_Min = FLT_MAX;
    float lfY_Max = FLT_MIN;
    float lfY_Min = FLT_MAX;

    std::vector<float2> vecData;
 
    float GetMaxX() { return lfX_Max; }
    float GetMinX() { return lfX_Min; }
    float GetMaxY() { return lfY_Max; }
    float GetMinY() { return lfY_Min; }
};

class ChartView : public QChartView
{
public:
    ChartView(QChart *chart, QWidget *parent = 0);
    ~ChartView();
    void CreateChart();

    void Set3DChart(std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY);
    void SetFFRChart(std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY);

    void tooltip3D(QPointF point, bool state);
    void tooltipFFR(QPointF point, bool state);

protected:
    bool viewportEvent(QEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent* event) override;


    void keyPressEvent(QKeyEvent *event);
    void resizeEvent(QResizeEvent* e);

    void handleMarkerClicked();
//![2]

private:
    bool m_isTouching;
    bool m_Lbutton;

    int m_nMinId = 0;
    float2 m_LButtonPos;

    ChartData m_3D;
    ChartData m_FFR;

    ChartCallout* m_tooltip = nullptr;
    ChartCallout* m_CursorData = nullptr;

    std::function<void(int)> m_fUpdateCenterPos;
    std::function<void()> m_fUpdatChartClose;

    QValueAxis* m_axisX = nullptr;
    QSplineSeries* m_SeriesGuideLine = nullptr;
    QScatterSeries* m_ScatterGuide3Point = nullptr;
    QScatterSeries* m_ScatterGuideFFRoint = nullptr;
};


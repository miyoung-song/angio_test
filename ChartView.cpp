#include "chartview.h"
#include <QtGui/QMouseEvent>

ChartView::ChartView(QChart *chart, QWidget *parent) :
    QChartView(chart, parent),
    m_isTouching(false),
    m_Lbutton(false)
{
    CreateChart();
}

ChartView::~ChartView()
{
    if (m_tooltip)
        delete m_tooltip;

    if (m_CursorData)
        delete m_CursorData;
}

void ChartView::resizeEvent(QResizeEvent* e)
{
    auto size = QRect(0, 0, e->size().width(), e->size().height());
    chart()->setGeometry(QRect(size));
}



bool ChartView::viewportEvent(QEvent *event)
{
    if (event->type() == QEvent::TouchBegin) {
        // By default touch events are converted to mouse events. So
        // after this event we will get a mouse event also but we want
        // to handle touch events as gestures only. So we need this safeguard
        // to block mouse events that are actually generated from touch.

        // Turn off animations when handling gestures they
        // will only slow us down.
        chart()->setAnimationOptions(QChart::NoAnimation);
    }
    return QChartView::viewportEvent(event);
}

void ChartView::mousePressEvent(QMouseEvent *event)
{
    m_Lbutton = true;
    m_LButtonPos = make_float2(event->pos().x(), event->pos().y());
    chart()->setAnimationOptions(QChart::SeriesAnimations);

    //if (chart()->axisX().min() < value_at_position.x() &&  value_at_position.x() > chart()->axisX().max())
    //    x = scene_position.x();
    QChartView::mousePressEvent(event);
}

void ChartView::mouseMoveEvent(QMouseEvent *event)
{
    if (m_Lbutton)
        m_isTouching = true;
    auto  pos = event->pos();
    auto _dis = fabs(m_LButtonPos.x - event->pos().x());
    if(_dis < 5)
      return;
    QChartView::mouseMoveEvent(event);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event)
{
    if (!m_isTouching)
    {
         auto scene_position = mapToScene(event->pos());
        auto chart_position = chart()->mapFromScene(scene_position);
        auto value_at_position = chart()->mapToValue(chart_position);
        //  chart()->removeSeries(series2);
        auto axisy = chart()->axes(Qt::Vertical).first();
        QValueAxis* vaxis = 0;
        if (axisy->type() == QAbstractAxis::AxisTypeValue)
            vaxis = qobject_cast<QValueAxis*>(axisy);
        if (vaxis->min() < value_at_position.y() && vaxis->max() > value_at_position.y())
        {
            auto points = m_SeriesGuideLine->points();
            std::vector<float> _dis;
            for (int i = 0; i < points.size(); i++)
            {
                _dis.push_back(fabs(m_3D.vecData[i].x - value_at_position.x()));
                points[i].setX(value_at_position.x());
            }

            int nMinDis = min_element(_dis.begin(), _dis.end()) - _dis.begin();

            m_SeriesGuideLine->replace(points);

            {
                //3D
                auto point = m_ScatterGuide3Point->points();
                for (int i = 0; i < point.size(); i++)
                {
                    point[i].setX(value_at_position.x());
                    point[i].setY(m_3D.vecData[nMinDis].y);
                }
                m_ScatterGuide3Point->replace(point);
            }

            {
                //FFR
                auto point = m_ScatterGuideFFRoint->points();
                for (int i = 0; i < point.size(); i++)
                {
                    point[i].setX(value_at_position.x());
                    point[i].setY(m_FFR.vecData[nMinDis].y);
                }
                m_ScatterGuideFFRoint->replace(point);
            }
        }
    }
    m_Lbutton = false;
    m_isTouching = false;

    // Because we disabled animations when touch event was detected
    // we must put them back on.
    chart()->setAnimationOptions(QChart::SeriesAnimations);

     QChartView::mouseReleaseEvent(event);
}

void ChartView::wheelEvent(QWheelEvent* event)
{
    return;
}

//![1]
void ChartView::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Plus:
        chart()->zoomIn();
        break;
    case Qt::Key_Minus:
        chart()->zoomOut();
        break;
//![1]
    case Qt::Key_Left:
        chart()->scroll(-10, 0);
        break;
    case Qt::Key_Right:
        chart()->scroll(10, 0);
        break;
    case Qt::Key_Up:
        chart()->scroll(0, 10);
        break;
    case Qt::Key_Down:
        chart()->scroll(0, -10);
        break;
    case Qt::Key_Space:
        chart()->zoomReset();
        break;
    default:
        QGraphicsView::keyPressEvent(event);
        break;
    }
}

void ChartView::CreateChart()
{
    chart()->createDefaultAxes();
    chart()->setBackgroundBrush(QBrush(QColor(Qt::black)));
    chart()->setBackgroundPen(QPen(QColor(Qt::black)));
    chart()->legend()->hide();
    chart()->setAnimationOptions(QChart::SeriesAnimations);
    m_axisX = new QValueAxis();
    m_axisX->setTitleText("Vessel Length [mm]");
    m_axisX->setLabelsColor(QColor(Qt::white));
    m_axisX->setTitleBrush(QBrush(QColor(Qt::white)));
    m_axisX->setLabelsBrush(QBrush(QColor(Qt::white)));
    m_axisX->setLabelFormat("%.3f");

    m_SeriesGuideLine = new QSplineSeries();
    m_SeriesGuideLine->setPen(QPen(QColor(255, 0, 0, 200), 2));
    m_SeriesGuideLine->setPointLabelsVisible(false);

    m_ScatterGuide3Point = new QScatterSeries();
    m_ScatterGuide3Point->setName("3D");
    m_ScatterGuide3Point->setPointLabelsVisible(true);
    m_ScatterGuide3Point->setColor(QColor(0, 255, 0));
    m_ScatterGuide3Point->setPointLabelsFormat("             (@yPoint)");
    m_ScatterGuide3Point->setPointLabelsColor(QColor(Qt::white));
    m_ScatterGuide3Point->setMarkerSize(10.0);


    m_ScatterGuideFFRoint = new QScatterSeries();
    m_ScatterGuideFFRoint->setName("FFR");
    m_ScatterGuideFFRoint->setPointLabelsVisible(true);
    m_ScatterGuideFFRoint->setColor(QColor(255, 0, 255));
    m_ScatterGuideFFRoint->setPointLabelsFormat("             (@yPoint)");
    m_ScatterGuideFFRoint->setPointLabelsColor(QColor(Qt::white));
    m_ScatterGuideFFRoint->setMarkerSize(10.0);
    setRubberBand(QChartView::HorizontalRubberBand);
}

void ChartView::Set3DChart(std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY)
{
    QValueAxis* axisY = new QValueAxis;
    axisY->setTickCount(6);
    axisY->setLabelFormat("%.3f");
    axisY->setTitleText("3D Diameter [mm]");
    axisY->setLabelsColor(QColor(Qt::white));
    axisY->setTitleBrush(QBrush(QColor(Qt::white)));

    QSplineSeries* series = new QSplineSeries();
    series->setName("3D Diameter");
    series->setPen(QPen(QColor(0, 255, 0, 200), 2));
    for (int i = 0; i < vecAxisY.size(); i++)
    {
        m_3D.vecData.push_back(make_float2(vecAxisX[i], vecAxisY[i]));
        m_3D.lfX_Min = qMin(vecAxisX[i], m_3D.lfX_Min);
        m_3D.lfX_Max = qMax(vecAxisX[i], m_3D.lfX_Max);
        m_3D.lfY_Min = qMin(vecAxisY[i], m_3D.lfY_Min);
        m_3D.lfY_Max = qMax(vecAxisY[i], m_3D.lfY_Max);

        series->append(vecAxisX[i], vecAxisY[i]);
    }

    int nMinDis = min_element(vecAxisY.begin(), vecAxisY.end()) - vecAxisY.begin();
    m_nMinId = nMinDis;

    for (int i = 0; i < vecAxisX.size(); i++)
        m_SeriesGuideLine->append(vecAxisX[nMinDis], vecAxisX[i]);
    m_ScatterGuide3Point->append(vecAxisX[nMinDis], QString::number(vecAxisY[nMinDis], 'f', 5).toDouble());
    QScatterSeries* scatterM = new QScatterSeries();
    scatterM->setName("3D M1 & M2");
    scatterM->setPointLabelsVisible(true);
    scatterM->setColor(QColor(0, 84, 255));
    scatterM->setPointLabelsFormat("(@yPoint)");
    scatterM->setPointLabelsColor(QColor(Qt::white));
    scatterM->setMarkerSize(10.0);
    scatterM->setPointLabelsVisible(false);
    for (int i = 1; i < X.size() - 1; i++)
        scatterM->append(vecAxisX[X[i]], QString::number(vecAxisY[X[i]], 'f', 5).toDouble());


    chart()->addSeries(series);
    chart()->addSeries(m_SeriesGuideLine);
    chart()->addSeries(m_ScatterGuide3Point);
    chart()->addSeries(scatterM);

    chart()->addAxis(m_axisX, Qt::AlignBottom);

    series->attachAxis(m_axisX);
    m_SeriesGuideLine->attachAxis(m_axisX);
    m_ScatterGuide3Point->attachAxis(m_axisX);
    scatterM->attachAxis(m_axisX);

    chart()->addAxis(axisY, Qt::AlignLeft);

    series->attachAxis(axisY);
    m_SeriesGuideLine->attachAxis(axisY);
    m_ScatterGuide3Point->attachAxis(axisY);
    scatterM->attachAxis(axisY);


    chart()->axes(Qt::Horizontal).first()->setRange(0, m_3D.lfX_Max);
    chart()->axes(Qt::Vertical).first()->setRange(0, m_3D.lfY_Max + (m_3D.lfY_Min / 10));

    connect(series, &QLineSeries::hovered, this, &ChartView::tooltip3D);

    const auto markers = chart()->legend()->markers();
    markers[0]->setLabelBrush(QColor(255.0, 255.0, 255.0, 255.0));
    QObject::connect(markers[0], &QLegendMarker::clicked, this, &ChartView::handleMarkerClicked);

}

void ChartView::SetFFRChart(std::vector<int> X, std::vector<float> vecAxisX, std::vector<float> vecAxisY)
{
    QSplineSeries* series = new QSplineSeries();
    QScatterSeries* scatterM = new QScatterSeries();
    QValueAxis* axisX = new QValueAxis;
    QValueAxis* axisY = new QValueAxis;

    axisY->setLabelFormat("%.3f");
    axisY->setTitleText("FFR");
    axisY->setLabelsColor(QColor(Qt::white));
    axisY->setTitleBrush(QBrush(QColor(Qt::white)));
    axisY->setRange(0, 1);
    axisY->setTickCount(6);

    series->setName("FFR");
    series->setPen(QPen(QColor(255, 0, 255, 255), 2));

    for (int i = 0; i < vecAxisY.size(); i++)
    {
        m_FFR.vecData.push_back(make_float2(vecAxisX[i], vecAxisY[i]));
        m_FFR.lfX_Min = qMin(vecAxisX[i], m_FFR.lfX_Min);
        m_FFR.lfX_Max = qMax(vecAxisX[i], m_FFR.lfX_Max);
        m_FFR.lfY_Min = qMin(vecAxisY[i], m_FFR.lfY_Min);
        m_FFR.lfY_Max = qMax(vecAxisY[i], m_FFR.lfY_Max);
        series->append(vecAxisX[i], vecAxisY[i]);
    }

    scatterM->setName("FFR M1 & M2");
    scatterM->setPointLabelsVisible(true);
    scatterM->setColor(QColor(0, 84, 255));
    scatterM->setPointLabelsFormat("(@yPoint)");
    scatterM->setPointLabelsColor(QColor(Qt::white));
    scatterM->setMarkerSize(10.0);
    scatterM->setPointLabelsVisible(false);
    for (int i = 1; i < X.size() - 1; i++)
        scatterM->append(vecAxisX[X[i]], QString::number(vecAxisY[X[i]], 'f', 5).toDouble());

    m_ScatterGuideFFRoint->append(vecAxisX[m_nMinId], QString::number(vecAxisY[m_nMinId], 'f', 5).toDouble());

    chart()->addSeries(series);
    chart()->addSeries(scatterM);
    chart()->addSeries(m_ScatterGuideFFRoint);

    scatterM->attachAxis(m_axisX);
    m_ScatterGuideFFRoint->attachAxis(m_axisX);

    chart()->addAxis(axisY, Qt::AlignRight);
    series->attachAxis(axisY);
    scatterM->attachAxis(axisY);
    m_ScatterGuideFFRoint->attachAxis(axisY);

    connect(series, &QLineSeries::hovered, this, &ChartView::tooltipFFR);
    const auto markers = chart()->legend()->markers();
    QObject::connect(markers[2], &QLegendMarker::clicked, this, &ChartView::handleMarkerClicked);
    markers[2]->setLabelBrush(QColor(255.0, 255.0, 255.0, 255.0));
}

void ChartView::handleMarkerClicked()
{
    QLegendMarker* marker = qobject_cast<QLegendMarker*> (sender());
    switch (marker->type())
    {
    case QLegendMarker::LegendMarkerTypeXY:
    {
        marker->series()->setVisible(!marker->series()->isVisible());
        marker->setVisible(true);
        qreal alpha = 1.0;

        if (!marker->series()->isVisible())
            alpha = 0.5;

        QColor color;
        QBrush brush = marker->labelBrush();
        color = brush.color();
        color.setAlphaF(alpha);
        brush.setColor(color);
        marker->setLabelBrush(brush);

        brush = marker->brush();
        color = brush.color();
        color.setAlphaF(alpha);
        brush.setColor(color);
        marker->setBrush(brush);

        QPen pen = marker->pen();
        color = pen.color();
        color.setAlphaF(alpha);
        pen.setColor(color);
        marker->setPen(pen);
        break;
    }
    default:
        break;
    }
}


void ChartView::tooltip3D(QPointF point, bool state)
{
    if (!m_tooltip)
        m_tooltip = new ChartCallout(chart());
    if (point.x() < 0 || !state)
    {
        m_tooltip->hide();
        if (m_fUpdatChartClose)
            m_fUpdatChartClose();
        return;
    }
    std::vector<float> _dis;
    for (int i = 0; i < m_3D.vecData.size(); i++)
        _dis.push_back(fabs(m_3D.vecData[i].x - point.x()));

    int nMinDis = min_element(_dis.begin(), _dis.end()) - _dis.begin();

    auto str = QString("D: ") + QString::number(m_3D.vecData[nMinDis].y, 'f', 5);
    m_tooltip->setText(str);
    m_tooltip->setAnchor(point);
    m_tooltip->setZValue(11);
    m_tooltip->updateGeometry();
    m_tooltip->SetToolTip(true);
    m_tooltip->show();

  
    //if(m_fUpdateCenterPos)
    //  m_fUpdateCenterPos(X);
}

void ChartView::tooltipFFR(QPointF point, bool state)
{
    if (!m_tooltip)
        m_tooltip = new ChartCallout(chart());

    if (point.x() > 100 || !state)
    {
        m_tooltip->hide();
        if (m_fUpdatChartClose)
            m_fUpdatChartClose();
        return;
    }

    float axisY_Min = m_3D.GetMinY();
    float axisY_Max = m_3D.GetMaxY();

    auto ss = (axisY_Max + axisY_Min / 10) * (point.y());
    QPointF xy = QPointF(point.x(), ss);

    auto str = QString("D: ") + QString::number(point.y(), 'f', 5);
    m_tooltip->setText(str);
    m_tooltip->setAnchor(xy);
    m_tooltip->setZValue(11);
    m_tooltip->updateGeometry();
    m_tooltip->show();
}
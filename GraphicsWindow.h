#pragma once

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

//#include<qgl.h>
//#include<qopengl.h>

#include<cuda_runtime.h>
#include<cuda_gl_interop.h>

//#include<qwaitcondition.h>
//#include<qmutex.h>
#include<qopenglwidget.h>
#include<qopenglfunctions_4_5_core.h>
#include<qopenglvertexarrayobject.h>
#include<qopenglbuffer.h>
#include<qmatrix4x4.h>
#include<qprocess.h>

//#include<qopenglext.h>

#include<gl/GL.h>
#include<gl/GLU.h>

#include <gl/glut.h>

//#include<qdockwidget.h>
#include<qsettings.h>
#include<qpainter.h>
#include<qwidget.h>
#include<qgraphicsview.h>
#include<qevent.h>
#include <qchart.h>
#include <qsplineseries.h>

//#include"ui_windowGT.h"


#include<qstack.h>
#include <qmath.h>
#include <qopengl.h>
#include <QVector>
#include <QVector3D>
#include <qmessagebox.h>
#include <QTextCodec>

#include <QtCharts>
#include <QChartView>
#include <QChart>
#include<QtCore/QTime>

#include<utility>
#include<map>
#include<chrono>
#include<future>


#include"LineObj.h"
#include"TexObj.h"
#include"TriObj.h"
#include"AxisLineObj.h"

#include "Angio_Algorithm.h"

#include <stack>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)


class GraphicsWindow : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
    Q_OBJECT

public:
    enum TextType
    {
        TextName,
        fixTextName,
        AxisName,
    };
    enum UndoType { Lineundo = 0, burshundo };
    enum Line2D { GuideStartEndLine = 0, StenosisBranchLine, FourPointCenterLine };
    enum EipLine2D { CreatePoint = 0, MovePoint, AutoPoint };
    enum NarrowLine2D { NarrowStartPoints = 0, NarrowEndPoints, Stenosis };

    using QOpenGLWidget::QOpenGLWidget;
    explicit GraphicsWindow(QWidget* parent = 0);
    ~GraphicsWindow();

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;


    bool Parsing(const QString& s); // .plt 파일 읽어와사 3D로 뷰 화면에 출력하기 
    void Parsing(unique_ptr<dcmHelper>&); //dcm 파일 읽어와 2D로 화면에 출력하기

    void SetShowFFR(bool bShow) { m_bFFShow = bShow; }; //ffr bar 디스플레이 유무

    void Initialize();
    void Render(); 
    void ShutDown();

    /*!
    * @brief 포물선 출력
    * @param pick_point : 바꿀려는 좌표면서 포물선 결정하는 좌표 3개
    * @param point : 변경되는 점
    * @param nvessnum : 점 갯수
    * @param lfRange : 범위 ( 사실상 포물선 방향 )
    * @param nSIdex : 시작범위
    * @param nEIdex : 끝범위
    * @param bCompare : 최대범위 설정 플래그
    */

    void SetPlayState(const bool& b) { this->isPlay = b; timer_prev = std::chrono::high_resolution_clock::now(); timer = 0; rotCount = 1; }
    

    //brief dcm 정보
    float GetSID() const { return this->mDCMHelper.get()->getSourceToDetector(); }
    float GetSOD() const { return this->mDCMHelper.get()->getSourceToPatient(); }
    float GetPAngle() const { return this->mDCMHelper.get()->getPrimaryAngle(); }
    float GetSAngle() const { return this->mDCMHelper.get()->getSecondaryAngle(); }
    void SetMulPSAngle(const float& _p0, const float& _s0, const float& _p1, const float& _s1, const float& _v);
    bool EmptyDCM() const { return this->bDcmfile; };
 
    //myGrid.x 가로 myGrid.y 세로
    void SetGridIndex(const int& row, const int& col) { this->myGrid.x = col; this->myGrid.y = row; }
    int2 GetGridIndex() { return int2(this->myGrid); };
    std::string get_bpm() { return Angio_Algorithm_.get_segmentation_Instance().bpm; };
    result_info::points_instance get_points_instance() { return Angio_Algorithm_.get_points_instance(); };

    /*!
    * @brief 데이터 길이 균등하게..?
    * @param newData 새로운데이터
    * @param data 기존데이터
    * @param n 만들 데이터의 개수
    * @param nCount 기존 데이터 개수
    * @param nS 바꿀 데이터 시작 위치
    * @param nE 바꿀 데이터 끝 위치
    */
    void CreateObject(Object*& pObject, float2* _temp, int cnt, int GraphicType, int objType = Object::Shape::Line);
    void CreateObject(vector<Object*>& vecObject, float2* _temp, int cnt, int GraphicType, int objType = Object::Shape::Line);
    void ModifyObject(Object*& pObject, float2* _temp, int cnt, int GraphicType, int objType = Object::Shape::Line);
    void ModifyObject(vector<Object*>& vecObject, float2* _temp, int nIndex, int cnt, int GraphicType, int objType = Object::Shape::Line);
    void DeleteObject(vector<Object*>& vecObject);
    void DeleteObject(Object*& pObject);

    bool make_cross_object(float2 pos, model_info::points_type type);

    void ModifyLine(bool bChange);

    void save_folder(model_info::folder_type type);
    void save_dcm_file();
    void read_data_in(std::string strExtension);

    void clear_line(model_info::clear_type type = model_info::clear_type::clear_all);
    void clear_points(model_info::clear_type ntype = model_info::clear_type::clear_all);

    void SetLineRange(int Vaule, bool bAuto);

    bool isLine() { return this->m_ObjLines.size() != 0; };

    void draw_points();
    void draw_lines();
    void draw_calibration_line(vector<float2> INTCS, vector<float2> INTCE);

    void moveAlongEpiline(vector<float2> points);
    bool calculatePointOnEpiline(vector<float2>& output_points);

    void SetEquilateraLines(std::vector<float2> CenterLine, int nViewNo); // 센터라인 외각라인 정렬(100개로 균등화)

    void ManualSegmentation(model_info::segmentation_model_type model_type, model_info::segmentation_manual_type type);
    void AutoSegmentation(model_info::segmentation_model_type model_type, model_info::segmentation_exe_type run_type);

    vector<float> GetMinimumRadius();

    vector<float2> get_equilateral(int nP, int nIndex);

    bool LineUndo();

    void Get3DCenterline();

    bool isDcmFilee() { return this->bDcmfile; };

    void SetSelectWindow(bool bSelect);

    bool GetSelectWindow() { return m_bSelectGraphic; };
    bool GetinitWindow() { return m_bInit; };

    std::vector<int> calculate_chart_axis();

    void zoomFunc(bool bZoom, QRectF& RectBuffer);

    float GetScale() { return m_lfScale; };

    int GetOrigWindowCenter() { return m_nOrigWL; };
    int GetOrigWindowWidth() { return m_nOrigWW; };
    
    void SetViewSize(int nW, int nH) { m_lfHeight = nH; m_lfWidth = nW; };

    model_info::model_type GetModelState() { return m_ModelState; };

    void SetStenosisorBranchPointIds(vector<int> vecIds);
    void set_end_points(std::string fileName, model_info::find_endpoint_type type);

public slots:
    void CleanUp();
    void SetOpenImageId(int nIndex = -1);

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);

    void contextWanted();
    void sendPickRequest(const int&);

public:
    void setXTranslation(const float& step);
    void setYTranslation(const float& step);
    void setZTranslation(const float& step);

    void setXScaling(const float& step);
    void setYScaling(const float& step);
    void setZScaling(const float& step);

    void InitScreen(bool is3d);
    float3 Get3DPosition(float2 pt);
    void CallbackUpdateCursorView(std::function<void(int2)> f) { m_fUpdateCursor = f; };
    void CallbackUpdateLine(std::function<void(int,bool)> f) { m_fUpdateLine = f; };
    void CallbackUpdateLineUndo(std::function<void()> f) { m_fSetLineUndo = f; };


    //3D센터라인
    void CallbackUpdateChartClose(std::function<void()> f) { m_fSetFinish = f; };
    void CallbackUpdateCenterArrowPos(std::function<void()> f) { m_fSetFinish = f; };
    
    void CallbackUpdateChart(std::function<void(std::vector<int> , std::vector<double>)> f) { m_fUpdateChart = f; };
    
    float2 GetStartPoint() { return Angio_Algorithm_.get_points_instance().start_point; };
    float2 GetEndPoint() { return  Angio_Algorithm_.get_points_instance().end_point; };

    void SetWindowLevel(int nWC, int nWW);
    int GetWindowCenter();
    int GetWindowWidth();

    void SaveAs(QString dirName, int nStartIndex = -1, int nEndIndex = -1);

    float3 GetWinCoord(float3 pos, int nAxis);
    string GetPath() const { return this->mDCMHelper.get()->getFile()->c_str(); }
    QString GetFileName() { return m_fileName; };
    QString GetFilePath() { return m_filePath; };
    int GetNumberImage() { return numberImage; };
    int GetCurrentImage() { return currentImage; };
    
    void SetCurrentImage(int nindex);

    void DrawRectangle(QRectF& RectBuffer, float4& zf);


    void DrawBitmapText(const char* str, float3 pos, int nAxis);

    bool get_pick_point_move(model_info::points_type type, int Obj_id);

    void UpdateModel();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() override;
    void Insert3DPointName(result_info::end_point_info vecPoint3D, int nIndex);
    void resizeGL(int width, int height) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    bool eventFilter(QObject* obj, QEvent* e) override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

    float2 GetScreenToPoint(float2 ptScreen);
    float2 GetPointToScreen(float2 ptMouse);
    void NormalizeAngle(int& angle);

    virtual void keyPressEvent(QKeyEvent* event);
    void keyReleaseEvent(QKeyEvent* event) override;
private:
    void getModelMat(const float& f);
    
    void circleImage();

private:
    enum { LineStart, LineEnd };

    float2 m_MousePos;
    float2 updata_pick_point_;
    int m_MousePickIndex;

    float p0 = 0, s0 = 0, p1 = 0, s1 = 0;
    int myType = 0;
    int myFrame = 0;

    int2 myGrid;


    bool m_bTest = true;
    bool m_bInit = false;
    unique_ptr<dcmHelper> mDCMHelper;
    unique_ptr<QString[]> m_guide;
   
    float m_nWheelScale = 1.0;

    Object* m_triDim = nullptr;
    Object* m_texContainer = nullptr;

    Object* m_ObjModifyLine = nullptr;

    vector<Object*> m_ObjLines;
    vector<Object*> m_ObjEdge;

    vector<Object*> m_ObjCalibrationLines;

    vector<Object*> m_ObjStartPointCross;
    vector<Object*> m_ObjEndPointCross;
    vector<Object*> m_ObjMatchingPointsCross;

    vector<Object*> m_Objlebeling; //색칠

    vector<Object*> m_vecAxis;
    vector<Object*> m_vec3DLine;
    
    QTextCodec* m_codec;

    //bool m_bStartPointCross = false;
    //bool m_bEndPointCross = false;
    //bool m_bMatchingPointsCross = false;

    //vector<int> m_nMatchingPointIds;

   int m_nStenosisIndex = 0;

    float2 m_AxisPt;

    QMatrix4x4 m_model;
    QMatrix4x4 m_camera;
    QMatrix4x4 m_proj;
    
    QMatrix4x4 m_AxisModel;
    QMatrix4x4 m_AxisCamera;

    float m_lfWidth = 0;
    float m_lfHeight = 0;

    float m_scale=0;

    std::function<void()> m_fSetFinish;
    std::function<void()> m_fSetLineUndo;
    std::function<void(int2)> m_fUpdateCursor;
    std::function<void(int, bool) > m_fUpdateLine;
    std::function<void(std::vector<int>, std::vector<double>)> m_fUpdateChart;


    QPoint m_LastPos;

    QRectF m_textBox[4] = { QRectF(0,0,0,0), };

    QRectF mRectBuffer = QRectF(-1, -1, -1, -1); //Zoom 버퍼
    float m_zoom = 0.5;


    int move_line_index_ = -1;
    int m_nFindLineRange = 0;
    int m_nCircleSize = 5;
    
    model_info::model_type m_ModelState = model_info::model_type::lint;
    
    QString m_filePath;
    QString m_fileAllPath;
    QString m_fileName;
    QString m_datePath;

    float m_lfScale = 0.5;
    bool m_isRunning = false;
    
    int m_nProgressRange = 0;
    bool m_bAutoRange = true;
    bool m_fixMove = false;
    int m_fixIndexid = 0;

private:
    bool pressedCtrl = false;
    bool pressedlShft = false;
    bool pressedAlt = false;

    bool is_move_line_ = false;
    bool is_move_pick_point_ = false;

    bool rectDraw = false;
    bool isPlay = false;
    bool bDcmfile = true;

    bool m_Lbutton = false;
    bool m_Rbutton = false;
    bool m_Mbutton = false;

    bool m_bSelectGraphic = false;

    float m_screenRatio =0.0;
    int m_nOrigWW = WW;
    int m_nOrigWL = WL;
    int m_SeriesNumber = 0;

    int currentImage = 0;
    int numberImage = 0;

    int rotCount = 3;
    QStack<float4> m_zoomstack;
    float4 m_zoomfactor;
    float timer;
    static std::chrono::steady_clock::time_point timer_prev;

    bool m_bFFShow = false;


    static std::mutex m_mtx;
    
    float m_valuePixel = 0;

    //AngioFFR_Algorithm angioAlgorithm_;
    Angio_Algorithm Angio_Algorithm_;

    std::string program_path_;
};

class Factory
{
public:
    ~Factory() {};
public:
    GraphicsWindow* getGraphicsWindow(int key)
    {
        if (mList.find(key) == mList.end())
        {
            mList[key] = new GraphicsWindow;
        }
        return mList[key];
    }
private:
    std::map<int, GraphicsWindow*> mList;
};
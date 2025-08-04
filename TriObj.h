#pragma once

#include<qfile.h>
#include<qregexp.h>

#include "Object.h"
#include<chrono>

class TriObj :
    public Object
{
private:
    //enum TriType : std::map<int,QString> { TECPLOT = 0, UNKNOWN = 99 };
public:
    struct VertexType : public Object::VertexType
    {
        float nx=0,ny=0,nz=0,v = 0;

        static int getElementNumber() noexcept { return 7; };
        static int getBytes() noexcept { return sizeof(float) * getElementNumber(); };
    };
public:
    using Object::Object;
    ~TriObj();

    const int getSceneWidth() const override;
    const int getSceneHeight() const override;

    void initializeGL() override;
    //void Bind();
    //void Render();
    void Render(QMatrix4x4& world, const QMatrix4x4& view, const QMatrix4x4& projection) override;
    //void Release();

    //void ShutDown() override;
    //void ReleaseGLFunctions();
    void Release() override;
    void ReleaseGLFunctions() override;
    void ReleaseBuffers() override;

    void SetWidth(unsigned val);
    void SetHeight(unsigned val);
    float3 GetCenterPos() { return m_posCenter; };


    void storeData(void* data, const int& countx, const int& county = 0, const int& countz = 0) override;
    void* loadData() override;
    void moveData(void* data, int nCount , bool bChange) override;
    dm GetDistance2End(float2 end) override;
    void DataOut(const int& id) override;

    void moveScene(const int&) override;
    bool Parsing(const QString&,const bool) override;

    void SetFFR(const QString& str);
    float GetFFRMin() { return m_valueMin; };


    const QStringList getVariableName() const;

    //const QMatrix4x4 Model( QMatrix4x4& m = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)) override;


    void setFFRShow(const bool& b) { 
        bFFRShow = b; 
        timer_prev = std::chrono::high_resolution_clock::now();
        timer = 1;
    }
    bool getFFRShow() const { return bFFRShow; }

    float getMaxPlaneValue() const { return this->m_maxPlaneValue; }
    float getMinValue() const { return this->m_minVal; }
    void setCamPos(const QVector3D& qv){ m_program->setUniformValue(m_lightPosLoc, qv); }
protected:
    void allocateVertices(void*) override;

private:
    void cummulateNormalVector(VertexType&, VertexType&, VertexType&) const;
    void makeObject()  override;
    VertexType* m_pVertices = nullptr;
    VertexType* m_pRawVertices = nullptr;


    QOpenGLBuffer m_ibo;
    

    int m_facetEdge = 0;

    int m_projMatrixLoc = 0;
    int m_viewMatrixLoc = 0;
    int m_modelMatrixLoc = 0;
    int m_normalMatrixLoc = 0;
    int m_contourTypeLoc = 0;
    int m_lightPosLoc = 0;
    int m_timerLoc = 0;
    int m_valueLoc = 0;

    int m_contourType = 0;
    static std::chrono::steady_clock::time_point timer_prev;
    float timer;
    
    float m_maxPlaneValue;
    float m_minVal;
    bool bFFRShow = false;

    QStringList m_variableName;
    
    float m_value = 0;
    float m_valueMin = 0;

    unsigned int m_commonwidth = 0, m_commonheight = 0;

    //TriType tt = UNKNOWN;
    float3 m_posCenter;

};


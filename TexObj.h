#pragma once

#include<qopengltexture.h>

#include"dcmHelper.h"
#include "Object.h"

//#include<qopenglwidget.h>



QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class TexObj:
	public Object//, protected dcmHelper
{
protected:
	struct VertexType : public Object::VertexType
	{
		float u,v;
	};

public:
	using Object::Object;
	//using QOpenGLWidget::QOpenGLWidget;
	//TexObj();
	~TexObj();
	
	void initializeGL()  override;
	void Render(const QMatrix4x4& world) override;

	//void ShutDown() override;

	void Bind() override;
	void Release() override;
	void ReleaseGLFunctions() override;
	void ReleaseBuffers() override;


	unsigned char* Scene() const;
	QImage* getQImage(int nIndex) { return mImgData[nIndex]; };

	void moveScene(const int&) override;


	bool Parsing(const QString &str, const bool) override;
	void Prepare(dcmHelper*);
	const QList<QString>* getInformation() const override;

	unsigned char* getCurrentImage(int nIndex) const;
	const int& getNumberImages() const;
	const int& getCurrentImages() const;
	//const int& currentIndex();

#ifdef DCM_U8
	//unsigned char* Scene() override;
#else
	float* Scene() override;
#endif


	const int getSceneWidth() const override;
	const int getSceneHeight() const override;

	void setWindowCenter(int val);
	void setWindowWidth(int val);

	void SetWidth(unsigned val);
	void SetHeight(unsigned val);

	float getWindowCenter() const;
	float getWindowWidth() const;

	void storeData(void* data, const int& countx, const int& county = 0, const int& countz = 0) override;

	void moveData(void* data, int nCount, bool bChange) override;
	dm GetDistance2End(float2 end) override;
	void DataOut(const int& id) override;

	void* loadData() override;

	//const QMatrix4x4 Model(QMatrix4x4& m = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)) override;

	//float* testDataOut(const int& i);
	unsigned char* testDataOut(const int& i);
	
protected:
	void allocateVertices(void*)  override;
	void append(QImage*);
	void append(float*);
	void append(unsigned char*);

private:
	void makeObject() override;
	bool convertTexture(QImage*);
	bool convertTexture(float*, const int&, const int&);
	bool convertTexture(unsigned char*, const int&, const int&);

	int currentImage = -1;
	int numberImage = 0;
	
	QOpenGLTexture* mTextures[1000] = { nullptr, };
	QVector<QImage*> mImgData;

#ifdef DCM_U8
	QVector<unsigned char*> mRaws;
	QVector<unsigned short*> mRaws16bit;
#else
	QVector<float*> mRaws;
#endif

	VertexType* m_pVertices = nullptr;

	vector<std::unique_ptr<dcmHelper>> mDHelper;


	unsigned int m_commonwidth = 0, m_commonheight = 0;
	bool bDicom = true;

	float m_WindowWidth = WW, m_WindowCenter = WL;
	int mImageBitType = dcmHelper::U8;
	//float m_RescaleIntercept=0, m_RescaleSlope=0;

private:
	int m_ScaleWCTypeLoc = WL;
	int m_ScaleWWTypeLoc = WW;
	int m_ImagebitTypeLoc = dcmHelper::U8;

};


#include "TexObj.h"
#include "Model.h"

//TexObj::TexObj()
//{
//    this->Object::Object();
//    m_model.ortho(-0.5f, +0.5f, +0.5f, -0.5f, -1.0f, 1.0f);
//    m_pVertices = nullptr;
//    
//}
TexObj::~TexObj()
{
    //ShutDown();
    ////SafeReleasePointer(mDHelper);
    //m_vbo.destroy();
    //m_vao.destroy();
    ////for (auto& _tex : mTextures)
    ////{
    ////    SafeReleasePointer(_tex);
    ////}
    //delete m_program;
    ReleaseBuffers();
    ReleaseGLFunctions();
}

void TexObj::initializeGL()
{
    makeObject();

    /*m_Context->functions()->glEnable(GL_DEPTH_TEST);
    m_Context->functions()->glEnable(GL_CULL_FACE);*/

    //QOpenGLShader* vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    const char* vsrc =
        R"(
            attribute highp vec4 vertex;
            attribute mediump vec4 texCoord;
            varying mediump vec4 texc;
            uniform mediump mat4 world;
            void main(void)
            {
                gl_Position = world * vertex;
                texc = texCoord;
            }
           )";
   
    //vshader->compileSourceCode(vsrc);

    //QOpenGLShader* fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    const char* fsrc =
        R"(
            uniform sampler2D texture;
            uniform float wc;
            uniform float ww;
		    uniform int bitType;
		    uniform float ScaleWW;
		    uniform float ScaleWC;
            varying mediump vec4 texc;
            void main(void)
            {
                if(bitType == 1)
                {
                    gl_FragColor = (texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5;
                }
                else if(bitType == 3)
                {
                    gl_FragColor = ((texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5) 
                            * ((texture2D(texture, texc.st) * 1.0) / ((texture2D(texture, texc.st) * 255.0 - (ScaleWC - 0.5)) / (ScaleWW - 1.0) + 0.5));
                 }
                else
                {
                    gl_FragColor = texture2D(texture, texc.st);
                }
            }
          )";





   // *((texture2D(texture, texc.st) * 1.0) / ((texture2D(texture, texc.st) * 255.0 - (511.0 - 0.5)) / (1023.0 - 1.0) + 0.5));





//(((texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5);


  //    "uniform sampler2D texture;\n"
  //    "uniform float wc;\n"
  //    "uniform float ww;\n"
  //    "varying mediump vec4 texc;\n"
  //    "void main(void)\n"
  //    "{\n"
  //
  ////    "   gl_FragColor = 255*texture2D(texture, texc.st);\n"
  //  //  "   gl_FragColor = (out*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5;\n"
  // //   "   gl_FragColor = (texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5;\n"
  //    "   gl_FragColor = ((texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5) * ((texture2D(texture, texc.st)*1.0) / ((texture2D(texture, texc.st)*255.0 - (511.0 - 0.5)) / (1023.0 - 1.0) + 0.5))                     ;\n"
  // //   (texture2D(texture, texc.st)*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5;\n"
  //     // "   gl_FragColor = (out - (wc - 0.5)) / (ww - 1.0) + 0.5;\n"
  ////    "   gl_FragColor = (( texture2D(texture, texc.st) / 255.0) - 0.5) * (ww - 1) + (wc - 0.5);\n"
  // //    "   gl_FragColor = (gl_FragColor*255.0 - (wc - 0.5)) / (ww - 1.0) + 0.5;\n"
  //
  //
  //    "}\n";
    //fshader->compileSourceCode(fsrc);
    initializeShader(QOpenGLShader::Vertex, vsrc);
    initializeShader(QOpenGLShader::Fragment, fsrc);
    m_program->bind();
    /*m_program = new QOpenGLShaderProgram;
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vsrc);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fsrc);*/
    //m_program->addShader(vshader);
    //m_program->addShader(fshader);
    m_program->bindAttributeLocation("vertex", PROGRAM_VERTEX_ATTRIBUTE);
    m_program->bindAttributeLocation("texCoord", PROGRAM_TEXCOORD_ATTRIBUTE);
    m_program->link();
    
    

    
    m_program->setUniformValue("texture", 0);

    m_program->setUniformValue("bitType", mImageBitType);
    m_program->setUniformValue("ScaleWW", m_WindowWidth);
    m_program->setUniformValue("ScaleWC", m_WindowCenter);

    m_program->release();
    /*m_Context->functions()->glDisable(GL_DEPTH_TEST);
    m_Context->functions()->glDisable(GL_CULL_FACE);*/
    
    
}

void TexObj::Render(const QMatrix4x4& world)
{
    Bind();
    /*m_program->bind();
    m_vbo.bind();*/
    m_Context->functions()->glEnable(GL_DEPTH_TEST);
    m_Context->functions()->glEnable(GL_CULL_FACE);
    //glMatrixMode(GL_MODELVIEW);
    



    m_program->setUniformValue("world", world);
    m_program->setUniformValue("ww", m_WindowWidth);
    m_program->setUniformValue("wc", m_WindowCenter);

    //m_program->setUniformValue("view", view);
    //m_program->setUniformValue("projection", projection);

    m_program->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
    m_program->enableAttributeArray(PROGRAM_TEXCOORD_ATTRIBUTE);
    m_program->setAttributeBuffer(PROGRAM_VERTEX_ATTRIBUTE, GL_FLOAT, 0, 3, sizeof(float)*5);
    m_program->setAttributeBuffer(PROGRAM_TEXCOORD_ATTRIBUTE, GL_FLOAT, sizeof(float)*3, 2, sizeof(float)*5);
    


    //if (mTextures[currentImage]->isCreated())
    //{
    //    //mTextures[currentImage]->bind();
    //    //mTextures[currentImage]->release();
    //}
    m_Context->functions()->glDrawArrays(GL_TRIANGLE_FAN, 0, m_indexCount);
    m_Context->functions()->glDisable(GL_CULL_FACE);
    m_Context->functions()->glDisable(GL_DEPTH_TEST);
    /*m_vbo.release();
    m_program->release();*/
    //glMatrixMode()
    Release();
}

//void TexObj::ShutDown()
//{
//    Release();
//    ReleaseGLFunctions();
//    ReleaseBuffers();
//}

void TexObj::append(QImage* img)
{
    auto qImage_copy = new QImage(img->width(), img->height(), QImage::Format::Format_Grayscale8);
    int r, g, b;
    for (int i = 0; i < img->height(); i++)
    {
        for (int j = 0; j < img->width(); j++)
        {
            QRgb rgb = img->pixel(i, j);
            r = g = b = (qRed(rgb) + qGreen(rgb) + qBlue(rgb)) / 3;
            if (r > 0)
                int zz = 0;
            qImage_copy->setPixel(i, j, qRgb(r, g, b));
        }
    }
    mImgData.append(qImage_copy);
    m_commonwidth = img->width();
    m_commonheight = img->height();

    currentImage = ++numberImage - 1;
    /*if (!convertTexture(img))
    {
        mImgData.pop_back();
        --currentImage;
        --numberImage;
    }*/

}

#ifdef DCM_U8
void TexObj::append(unsigned char* raw)
{
    auto data = new unsigned char[m_commonwidth * m_commonwidth];
    memcpy_s(data, sizeof(unsigned char) * m_commonwidth * m_commonwidth, raw, sizeof(unsigned char) * m_commonwidth * m_commonwidth);
    mRaws.append(data);
    currentImage = ++numberImage - 1;
    /*if (!convertTexture(raw, width, height))
    {
        mRaws.pop_back();
        --currentImage;
        --numberImage;
    }*/
}

#else
void TexObj::append(float* raw)
{
    //mRaws.append(raw);
    mRaws.append(new float[m_commonwidth*m_commonwidth]);
    //std::reverse_copy(raw, raw + width * height, mRaws.back());
    memcpy(mRaws.back(), raw, sizeof(float) * m_commonwidth * m_commonwidth);
    
    currentImage = ++numberImage - 1;
    /*if (!convertTexture(raw, width, height))
    {
        mRaws.pop_back();
        --currentImage;
        --numberImage;
    }*/
}

#endif

void TexObj::ReleaseGLFunctions()
{
    //SafeReleaseArray(m_pVertices);
    //Object::ReleaseGLFunctions();
    m_vbo.destroy();
    m_vao.destroy();
    SafeReleasePointer(m_program);
    m_program = nullptr;
}

void TexObj::ReleaseBuffers()
{
    SafeReleaseArray(m_pVertices);
    for (auto i = 0; i < numberImage; i++)
    {
        if (mTextures[i])
        {
            mTextures[i]->destroy();
            delete mTextures[i];
            mTextures[i] = nullptr;
        }
    }
    currentImage = -1;
    numberImage = 0;
    for (auto& dat : mImgData)
        SafeReleasePointer(dat);


    for (auto i=0;i<mRaws.size();i++)
        SafeReleaseArray(mRaws[i]);
    mRaws.clear();

    for (auto i = 0; i < mRaws16bit.size(); i++)
        SafeReleaseArray(mRaws16bit[i]);
    mRaws16bit.clear();
}



void TexObj::moveScene(const int& idx)
{
    currentImage = idx;
}


//QOpenGLFunctions* TexObj::functions()
//{
//    return this->m_pFunctions;
//}

//void TexObj::Release()
//{
//     
//	for(auto& tex : mTextures)
//		SafeReleasePointer(tex);
//}

bool TexObj::Parsing(const QString& str, const bool _bDicom)
{
    //static QStringList _ext_not_DICOM ={".png",".bmp",".jpg"};
    //
    //if (str.isEmpty())
    //    return false;
    //
    //for (auto& _chker : _ext_not_DICOM)
    //    if (str.endsWith(_chker))
    //    {
    //        if (mDHelper.empty())//Not Dcm Docked
    //        {
    //            bDicom = false;
    //            break;
    //        }
    //        else//Exist Dcm Docked
    //        {
    //            return false;
    //        }
    //    }
    bDicom = _bDicom;
    if (bDicom)
    {
        mDHelper.push_back(std::make_unique<dcmHelper>());
        //mDHelper = new dcmHelper(str.toStdString());
        auto dh = mDHelper.back().get();
        
        if (dh->loadFile(str.toStdString()))
        {
            Prepare(dh);
            this->m_WindowCenter = dh->getWindowCenter();
            this->m_WindowWidth= dh->getWindowWidth();
            this->mImageBitType = dh->getTargetType();
            //this->m_RescaleSlope= dh->getRescaleSlope();
            //this->m_RescaleIntercept= dh->getRescaleIntercept();
        }
        else
            return false;
    }
    else
    {
        this->append(new QImage(str));

       //auto _ptr = dh->Data();
       //this->m_commonheight = dh->getRows();
       //this->m_commonwidth = dh->getCols();
       //this->m_WindowCenter = dh->getWindowCenter();
       //this->m_WindowWidth = dh->getWindowWidth();
       //size_t stride = m_commonwidth * m_commonheight;
       //for (auto i = 0; i < dh->getNumberOfFrames(); i++, _ptr += stride)
       //    this->append(_ptr);
    }
    return true;
    //dh
}

void TexObj::Prepare(dcmHelper* dh)
{
    auto _ptr = dh->Data();
 //   auto _ptr1 = dh->Data16bits();
    this->m_commonheight = dh->getRows();
    this->m_commonwidth = dh->getCols();
    this->m_WindowCenter = dh->getWindowCenter();
    this->m_WindowWidth = dh->getWindowWidth();
    size_t stride = m_commonwidth * m_commonheight;
    for (auto i = 0; i < dh->getNumberOfFrames(); i++, _ptr += stride)
        this->append(_ptr);
    this->mImageBitType = dh->getTargetType();
   //for (auto i = 0; i < dh->getNumberOfFrames(); i++, _ptr1 += stride)
   //{
   //    auto data = new unsigned short[m_commonwidth * m_commonwidth];
   //    std::transform(_ptr1, _ptr1 + stride, data, [&](unsigned short v)
   //        {
   //            return unsigned short(v);
   //        });
   //    mRaws16bit.append(data);
   //}
   
}

const QList<QString>* TexObj::getInformation() const
{
    ////QStringListModel* model = new QStringListModel();

    //QStringList list;
    //auto s = mDHelper.back().get()[0].Tags();
    //for (auto i = 0; i < s.size() / 3; i++)
    //    list << QString(s[i * 3].c_str()).append(s[i * 3 + 1].c_str()).append(s[i * 3 + 2].c_str());
    //qsl->setStringList(list);
    
    
    
    
    //if(mDHelper.size() >0)
    //    return mDHelper.back()->Tags();
    //else
    //{
    //    QList<QString>* aa = new QList<QString>[3];
    //    aa[0].append(QString("NO"));
    //    aa[1].append(QString("DICOM"));
    //    aa[2].append(QString("FILE"));
    //    return aa;
    //}
    return nullptr;
}

const int& TexObj::getNumberImages() const
{
    // TODO: insert return statement here
    return numberImage;
}

const int& TexObj::getCurrentImages() const
{
    return this->currentImage;
}

void TexObj::Bind()
{
    Object::Bind();
    if (currentImage > -1)
    {
        if (mTextures[currentImage])
        {
            if (mTextures[currentImage]->isCreated())
            {
                mTextures[currentImage]->bind();
            }
        }
    }
    //m_vao.bind();

}

void TexObj::Release()
{
    //m_vbo.release();
    if (numberImage > 0)
    {
        for (auto& _tex : mTextures)
        {
            if (_tex && (_tex->isCreated()))
                _tex->release();
        }
    }
    m_program->release();
    m_vbo.release();
    m_vao.release();
}

//const int& TexObj::currentIndex()
//{
//    // TODO: insert return statement here
//    return currentImage;
//}

#ifdef DCM_U8
unsigned char* TexObj::Scene() const
{
    if (mRaws.size() > 0)
    {
        return mRaws[currentImage];
    }
    else if (mImgData.size() > 0)
    {
        if (currentImage == -1)
            return false;
        unsigned char* aa = new unsigned char[mImgData[currentImage]->height() * mImgData[currentImage]->width()];
        auto qImage_copy = new QImage(mImgData[currentImage]->width(), mImgData[currentImage]->height(), QImage::Format::Format_Grayscale8);
        int r, g, b;
        for (int i = 0; i < m_commonheight; i++)
        {
            for (int j = 0; j < m_commonwidth; j++)
            {
                QRgb rgb = mImgData[currentImage]->pixel(i, j);
                r = g = b = (qRed(rgb) + qGreen(rgb) + qBlue(rgb)) / 3;
                qImage_copy->setPixel(i, j, qRgb(r, g, b));
            }
        }
        int stride1 = qImage_copy->bytesPerLine();
        auto ptr = qImage_copy->bits();
        for (auto y = 0; y < mImgData[currentImage]->height(); y++)
        {
             memcpy(aa + m_commonwidth * y, ptr + y * stride1, m_commonwidth); // copy a single row, accounting for stride bytes
        }
        SafeReleasePointer(qImage_copy);
        return aa;
    }
}
#else
float* TexObj::Scene()
{
    if (mRaws.size() > 0)
        return mRaws[currentImage];
    else if (mImgData.size() > 0)
    {
        return nullptr;
        //Error
        /*std::unique_ptr<float> aa(new float[mImgData[currentImage]->height() * mImgData[currentImage]->width()]);
        for (auto y = 0; y < mImgData[currentImage]->height(); y++)
        {
            const float* srcrow = (float*)mImgData[currentImage]->bits()*y;
            for (auto x = 0; x < mImgData[currentImage]->width(); x++)
            {
                const float* srccol = (float*)mImgData[currentImage]->scanLine(y);
                aa.get()[y * mImgData[currentImage]->width() + x] = *srccol;
            }
        }
        return aa.get();*/
    }
}
#endif


unsigned char* TexObj::getCurrentImage(int nIndex) const
{
    if (mRaws.size() > 0)
    {
        return mRaws[nIndex];
    }
    else if (mImgData.size() > 0)
    {
        unsigned char* aa = new unsigned char[mImgData[nIndex]->height() * mImgData[nIndex]->width()];
        auto qImage_copy = new QImage(mImgData[nIndex]->width(), mImgData[nIndex]->height(), QImage::Format::Format_Grayscale8);
        int r, g, b;
        for (int i = 0; i < m_commonheight; i++)
        {
            for (int j = 0; j < m_commonwidth; j++)
            {
                QRgb rgb = mImgData[nIndex]->pixel(i, j);
                r = g = b = (qRed(rgb) + qGreen(rgb) + qBlue(rgb)) / 3;
                qImage_copy->setPixel(i, j, qRgb(r, g, b));
            }
        }
        int stride1 = qImage_copy->bytesPerLine();
        auto ptr = qImage_copy->bits();
        for (auto y = 0; y < mImgData[nIndex]->height(); y++)
        {
            memcpy(aa + m_commonwidth * y, ptr + y * stride1, m_commonwidth); // copy a single row, accounting for stride bytes
        }
        SafeReleasePointer(qImage_copy);
        return aa;
    }
}

const int TexObj::getSceneWidth() const
{
    return m_commonwidth;
}

const int TexObj::getSceneHeight() const
{
    return m_commonheight;
}



void TexObj::setWindowCenter(int val)
{
    //this->m_WindowCenter += delta;
    this->m_WindowCenter = val;
}

void TexObj::setWindowWidth(int val)
{
    //this->m_WindowWidth += delta;
    this->m_WindowWidth = val;
}

void TexObj::SetWidth(unsigned val)
{
    m_lfWidth = val;
}

void TexObj::SetHeight(unsigned val)
{
    m_lfHeight = val;
}


float TexObj::getWindowCenter() const
{
    return this->m_WindowCenter;
}

float TexObj::getWindowWidth() const
{
    return this->m_WindowWidth;
}

void TexObj::storeData(void* data, const int& countx, const int& county, const int& countz)
{
}

void TexObj::moveData(void* data, int nCount, bool bChange)
{
}

Object::dm TexObj::GetDistance2End(float2 end)
{
    return dm();
}

void TexObj::DataOut(const int& id) 
{

}

void* TexObj::loadData()
{
    return nullptr;
}

//const QMatrix4x4 TexObj::Model( QMatrix4x4& model)
//{
//    if (m_Zoom)
//    {
//        m_Zoom = new float4();
//        m_Zoom->x = 1.0f;
//        m_Zoom->y = 1.0f;
//        m_Zoom->z = 1.0f;
//        m_Zoom->w = 1.0f;
//    }
//    if (m_screenRatio >= 1)
//        model.ortho(-(0.5f * m_screenRatio) / m_Zoom->x, (0.5f * m_screenRatio) / m_Zoom->y, 0.5f / m_Zoom->z, -0.5f / m_Zoom->w, -1.0f, 1.0f);
//    else
//        model.ortho(-0.5f / m_Zoom->x, 0.5f / m_Zoom->y, (0.5f / m_screenRatio) / m_Zoom->z, -(0.5f / m_screenRatio) / m_Zoom->w, -1.0f, 1.0f);
//    return model;
//   
//}


unsigned char* TexObj::testDataOut(const int& i)
{
#ifdef DCM_U8
    return this->mRaws[i];
#else
    return this->mRaws[i];
#endif
}

void TexObj::allocateVertices(void* other)
{
    std::copy((VertexType*)other, (VertexType*)other + this->m_vertexCount, (VertexType*)this->m_pVertices);
}



bool TexObj::convertTexture(QImage* img)
{
    if (currentImage == 0)
        makeObject();
    mTextures[currentImage] = new QOpenGLTexture(img->mirrored());
    return true;
}

bool TexObj::convertTexture(float* data, const int& width, const int& height)
{
    if (mTextures[currentImage])
    {
        delete mTextures[currentImage];
        mTextures[currentImage] = nullptr;
    }

    mTextures[currentImage] = new QOpenGLTexture(QOpenGLTexture::Target2D);
    mTextures[currentImage]->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    mTextures[currentImage]->create();
    mTextures[currentImage]->setAutoMipMapGenerationEnabled(true);

    mTextures[currentImage]->setSize(width,height, 1);
    mTextures[currentImage]->setFormat(QOpenGLTexture::LuminanceFormat);
    mTextures[currentImage]->allocateStorage(QOpenGLTexture::Luminance, QOpenGLTexture::Float32);
    //QOpenGLPixelTransferOptions* q = new QOpenGLPixelTransferOptions();
    //q->setSwapBytesEnabled(true);
    mTextures[currentImage]->setData(QOpenGLTexture::Luminance, QOpenGLTexture::Float32, data);
    if (mTextures[currentImage]->isStorageAllocated() & mTextures[currentImage]->isCreated())
        return true;
    return false;
}

bool TexObj::convertTexture(unsigned char* data, const int& width, const int& height)
{
    if (mTextures[currentImage])
    {
        delete mTextures[currentImage];
        mTextures[currentImage] = nullptr;
    }

    mTextures[currentImage] = new QOpenGLTexture(QOpenGLTexture::Target2D);
    mTextures[currentImage]->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    mTextures[currentImage]->create();
    mTextures[currentImage]->setAutoMipMapGenerationEnabled(true);

    mTextures[currentImage]->setSize(width, height);
    mTextures[currentImage]->setFormat(QOpenGLTexture::LuminanceFormat);
    mTextures[currentImage]->allocateStorage(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8);
    //QOpenGLPixelTransferOptions* q = new QOpenGLPixelTransferOptions();
    //q->setSwapBytesEnabled(true);
    mTextures[currentImage]->setData(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8, data);
    if (mTextures[currentImage]->isStorageAllocated() & mTextures[currentImage]->isCreated())
        return true;
    return false;
}

void TexObj::makeObject()
{
    if (!m_pVertices)
    {
        //hard coding
        static const int coords[4][3] =
        { { +1, -1, 0 }, { -1, -1, 0 }, { -1, +1, 0 }, { +1, +1, 0 } };

        //std::unique_ptr<VertexType[]> vertData(new VertexType[4]);
        SafeReleaseArray(m_pVertices);
        m_pVertices = new VertexType[4];
        m_vertexCount = 4;
        //0:qimage
        for (int j = 0; j < 4; j++)
        {
            m_pVertices[j].x = 0.5f * coords[j][0];
            m_pVertices[j].y = 0.5f * coords[j][1];
            m_pVertices[j].z = 0;
            m_pVertices[j].u = (j == 0 || j == 3);
            if (bDicom)
                m_pVertices[j].v = (j == 2 || j == 3);
            else
                m_pVertices[j].v = (j == 0 || j == 1);
        }
        m_indexCount = 4;

    }
	m_vbo.create();
	m_vbo.bind();
	m_vbo.allocate(m_pVertices, m_vertexCount * sizeof(float)*5);
    m_vbo.release();

    m_vao.create();
    m_vao.bind();

    m_vbo.bind();

    m_vbo.release();
    m_vao.release();
    //if (!mTextures)
        
    //if (numberImage != 0)
    //{//mTextures.reserve(mImgData.size());
    //    //mTextures = std::make_unique<QOpenGLTexture* []>(numberImage);
    //    //mTextures = new QOpenGLTexture * [numberImage];
    if (currentImage > mRaws.size()-1)
        currentImage = mRaws.size() - 1;

   for (auto i = 0; i < mRaws.size(); i++)
   {
     if (mTextures[i])
     {
         delete mTextures[i];
         mTextures[i] = nullptr;
     }
     ////this->convertTexture(mRaws[i], m_commonwidth, m_commonheight);
     //
      mTextures[i] = new QOpenGLTexture(QOpenGLTexture::Target2D);
      //mTextures[i]->setMinMagFilters(QOpenGLTexture::NearestMipMapNearest, QOpenGLTexture::NearestMipMapNearest);
      //mTextures[i]->setFormat(QOpenGLTexture::RGB16U);


      mTextures[i]->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
      mTextures[i]->setAutoMipMapGenerationEnabled(true);
      mTextures[i]->create();
      mTextures[i]->setSize(m_commonwidth, m_commonheight, 1);
      mTextures[i]->setFormat(QOpenGLTexture::LuminanceFormat);
      mTextures[i]->allocateStorage(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8);
      //mTextures[i]->setData(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8, mRaws[i]);
    //auto data = new unsigned char[m_commonwidth * m_commonheight];
    ////
   //    for (int n = 0; n < m_commonwidth * m_commonheight; n++)
   //    {
   //        auto v = mRaws16bit[i][n];
   //        //data[n] = (v)int(std::clamp(255.0 * ((v - (m_WindowCenter - 0.5)) / (m_WindowWidth - 1) + 0.5), 0.0, 255.0));
   //        auto a = 255.0 * ((v - (m_WindowCenter - 0.5)) / (m_WindowWidth - 1) + 0.5);
   //        auto a1 = (255 * v - (m_WindowCenter - 0.5)) / (m_WindowWidth - 1) + 0.5;
   //        auto b = std::clamp(a, 0.0, 255.0);
   //        auto c = (v * 255.0 - (m_WindowCenter - 0.5)) / (m_WindowWidth - 1) + 0.5;
   //    //((a/255.0)-0.5 *  (m_WindowWidth-1)) +(m_WindowCenter - 0.5)
   //        auto d = 0;
   //    }
    //mTextures[i]->setData(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8, data);
     
    mTextures[i]->setData(QOpenGLTexture::Luminance, QOpenGLTexture::UInt8, mRaws[i]);
   //  delete[] data;

     if (mTextures[i]->isStorageAllocated() & mTextures[i]->isCreated())
         continue;
     else
         break;
   }

    for (auto i = 0; i < mImgData.size(); i++)
    {
        mTextures[i] = new QOpenGLTexture(mImgData[i]->mirrored());
    }

   
    //}
}

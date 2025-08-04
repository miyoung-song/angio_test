#include "dcmHelper.h"

dcmHelper::dcmHelper() : mResult(EC_Normal), mNumberOfFrames(-1)
{
    mFiles = new string;
    mDcmDataset = nullptr;
    mFileFormat = nullptr;
    mDispInfo = nullptr;
    DJDecoderRegistration::registerCodecs();
    DJLSDecoderRegistration::registerCodecs();
}

dcmHelper::dcmHelper(string fname) : mResult(EC_Normal), mNumberOfFrames(-1)
{
    setFile(fname);
    mDcmDataset = nullptr;
    mFileFormat = nullptr;
    mDispInfo = nullptr;
    DJDecoderRegistration::registerCodecs();
    DJLSDecoderRegistration::registerCodecs();
}

//dcmHelper::dcmHelper(const dcmHelper& other)
//{
//   
//}

dcmHelper::~dcmHelper()
{
    //SafeReleasePtr(mFiles);
    //if(mTp)
    //    delete[] mTp;

    if (mFileFormat)
    {
        delete mFileFormat;
        mFileFormat = nullptr;
    }
    if (mDispInfo)
    {
        delete mDispInfo;
        mDispInfo = nullptr;
    }
    if (mFiles)
    {
        delete mFiles;
        mFiles = nullptr;
    }

    release();
}

const void* dcmHelper::operator[](const int& idx) const
{
    return nullptr;
}

//QDataStream& operator<<(QDataStream& out, const dcmHelper& rhs) {
//    out.writeRawData(reinterpret_cast<const char*>(&rhs), sizeof(rhs));
//    return out;
//}
//QDataStream& operator>>(QDataStream& in, dcmHelper& rhs) {
//    in.readRawData(reinterpret_cast<char*>(&rhs), sizeof(rhs));
//    return in;
//}



bool dcmHelper::loadFile(bool read)
{
    bReadString = read;
    //if (!readyForRead())
    //    return false;
    if (mFiles->empty())
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  filename is empty\n";
        return false;
    }

    DcmFileFormat ff;
    mResult = ff.loadFile(mFiles->c_str());
    if (mResult.good() != OFTrue)
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  loadFile Error\n";
        return false;
    }
    mDcmDataset = ff.getDataset();
    if (mDcmDataset == nullptr)
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  getDateset Error\n";
        return false;
    }

    if (!readOwnFile())
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  readOwnFile Error\n";
        return false;
    }

    return true;
}


bool dcmHelper::loadFile(string fname)
{
    setFile(fname);

    return loadFile();
}

bool dcmHelper::loadFile(const float& specific)
{
    bReadString = false;

    DcmFileFormat ff;
    mResult = ff.loadFile(mFiles->c_str());
    if (mResult.good() != OFTrue)
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  loadFile Error\n";
        return false;
    }

    mDcmDataset = ff.getDataset();
    if (mDcmDataset == nullptr)
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  getDateset Error\n";
        return false;
    }
    if (!readOwnFile(specific))
    {
        cerr << __FILE__ << ":" << __LINE__ << ":  readOwnFile Error\n";
        return false;
    }
    return true;
}

bool dcmHelper::readOwnFile(const float& specific)
{
    getImportInfo();

    DcmStack resultStack;


    auto iterGetElem = [&](const DcmTagKey& tagKey, DcmStack& resultStack) {
        OFCondition status = EC_TagNotFound;
        DcmStack stack;
        DcmObject* object = NULL;
        /* iterate over all elements */
        while (nextObject(stack, OFTrue).good())
        {
            /* get element */
            object = stack.top();

            /* add to result_stack */
            resultStack.push(object);
            if (object->getTag() == tagKey)
            {
                status = EC_Normal;
            }
        }
        return status;
    };

    mResult = mDcmDataset->findAndGetElements(mTarget, resultStack);


    const unsigned char* pixelData = nullptr;
    unsigned long count = 0;
    if (mDcmDataset->findAndGetUint8Array(DcmTagKey(0x0021, 0x101C), pixelData, &count).good()) {
        std::vector<unsigned short> vecOriginal(pixelData, pixelData + count);
    }
    else {
        std::cerr << "Failed to get Pixel Data \n";
    }




    //mResult = iterGetElem(mTarget, resultStack);
    if (mResult.good() != OFTrue)
    {
        if (resultStack.card() > 0)
            cerr << __FILE__ << ":" << __LINE__ << ":  missied target \n";
        else
            cerr << __FILE__ << ":" << __LINE__ << ":  check out to dcm tags\n";
        return false;
    }

    //mTp = new QList<QString>[3];
    if (bReadString)
    {
        mTp[0].reserve(resultStack.card());
        mTp[1].reserve(resultStack.card());
        mTp[2].reserve(resultStack.card());
    }
    for (auto i = 0; i < resultStack.card(); i++)
    {
        if (resultStack.elem(i)->getTag() == mTarget) //data
        {

            //            DcmPixelData* ele = OFreinterpret_cast(DcmPixelData*,OFconst_cast(unsigned char*, pixelData));
            DcmPixelData* ele = OFstatic_cast(DcmPixelData*, resultStack.elem(i));

            if (!getPixeldata(*resultStack.elem(i), ele, specific))
            {
                cerr << __FILE__ << ":" << __LINE__ << ":  Failed got a pix map\n";
                return false;
            }
        }
        else if (bReadString)
        {
            OFString buffer;
            DcmElement* _el = OFstatic_cast(DcmElement*, resultStack.elem(i));
            _el->getOFStringArray(buffer);
            //, string(DcmTag(_el->getTag()).getTagName()), string(buffer.c_str()) };

    ////0018,1110
    //        float mSourceToDetector;//SID
    //        //0018,1111
    //        float mSourceToPatient;//SOD
    //
    //        //0018,1164
    //        float mPixelSpacing = 0.368;//pixelspacing
    //
    //        //0018,1510
    //        float mPrimaryAngle;
    //        //0018,1511
    //
            mTp[0].append(QString(_el->getTag().toString().c_str()));
            mTp[1].append(QString(DcmTag(_el->getTag()).getTagName()));
            mTp[2].append(QString(buffer.c_str()));


            if (_el->getTag().getGTag() == Uint16(0x0018))
            {
                switch (_el->getTag().getETag())
                {
                case 0x1110:
                    mSourceToDetector = mTp[2].back().toFloat();
                    break;
                case 0x1111:
                    mSourceToPatient = mTp[2].back().toFloat();
                    break;
                case 0x1164:
                {
                    auto test = mTp[2].back().split('\\');
                    if (test.size() > 0)
                    {
                        mPixelSpacing[0] = test[0].toFloat();
                        if (test.size() > 1)
                        {
                            mPixelSpacing[1] = test[1].toFloat();
                        }
                        else
                        {
                            mPixelSpacing[1] = mPixelSpacing[0];
                        }
                    }
                    break;
                }
                case 0x1510:
                    mPrimaryAngle = mTp[2].back().toFloat();
                    break;
                case 0x1511:
                    mSecondaryAngle = mTp[2].back().toFloat();
                    break;
                }
            }
        }
    }


    // Essen
    if (bReadString)
    {
        if (!mDispInfo)
            mDispInfo = new displayInfo();

        for (auto& _s : mDispInfo->s)_s.reserve(256);

        OFCondition status;
        if (!mFileFormat)
        {
            mFileFormat = new DcmFileFormat();
            mFileFormat->loadFile(mFiles->c_str());
        }

        auto func = [&](QString& targ, const vRR& tag, const int& idx)
        {
            OFString result;

            float buffer = 0;
            for (auto i = 0; i < tag.size(); i++)
            {
                auto pre = tag[i].first;
                auto ge = tag[i].second;
                if (mFileFormat->getDataset()->findAndGetOFString(ge, result).good())
                {
                    if (pre.compare("#") == 0)
                    {
                        if (buffer == 0)
                            buffer = QString(result.c_str()).toFloat();
                        else
                        {
                            auto buffer2 = QString(result.c_str()).toFloat();
                            targ += QString("%1 %2 / %3 %4")
                                .arg(buffer > 0 ? "LAO" : "RAO").arg(abs(buffer))
                                .arg(buffer2 > 0 ? "CRA" : "CAU").arg(abs(buffer2));

                            this->setPrimaryAngle(buffer);
                            this->setSecondaryAngle(buffer2);
                        }
                        continue;
                    }

                    if (ge == DcmTagKey(0x0020, 0x0013))
                    {
                        this->setPerFrame(QString(result.c_str()).toInt());
                    }

                    targ += QString::fromUtf8(pre.c_str());
                    targ += result.c_str();
                    targ += "\n";
                }
                else
                {
                    targ += QString::fromUtf8(pre.c_str());
                    if (ge.getGroup() != 0xffff)
                        targ += "NO DATA\n";
                    else
                        targ += "\t\n";
                }
            }
        };

        array<vRR, 4> _avd = { mDispInfo->LeftTop,mDispInfo->RightTop,mDispInfo->RightBottom,mDispInfo->LeftBottom };
        for (auto i = 0; i < 4; i++)func(mDispInfo->s[i], _avd[i], i);


    }
    return true;


}

string* dcmHelper::getFile() const
{
    return mFiles;
}

void dcmHelper::setFile(string fname)
{
    mFiles = new string(fname);
    //*this->mFiles = fname;
}

void dcmHelper::release()
{
    SafeReleaseAry(mData16bits);
    SafeReleaseAry(mData);
    SafeReleaseAry(mPix);
    SafeReleasePtr(mFiles)
}

uchar* dcmHelper::getData()
{

#ifdef DCM_U8
    return mPix;

#else
    if (!mPix)
    {
        mPix = new uchar[mColumns * mRows * mNumberOfFrames];
        convertF32toU8H2H(mData, mPix, mColumns, mRows, mNumberOfFrames, mSamplePerPixel == 1 ? -1 : mPlanarConfiguration);
    }
    return (uchar*)mPix;
#endif // DCM_U8
}
#ifdef DCM_U8

const unsigned char* dcmHelper::constData()
{
    return mData;
}

unsigned short* dcmHelper::Data16bits()
{
    return mData16bits;
}

unsigned char* dcmHelper::Data()
{
    return mData;
}

#else

const float* dcmHelper::constData()
{
    return mData;
}

float* dcmHelper::Data()
{
    return mData;
}

#endif
const int& dcmHelper::getStep() const
{
    return mStep;
}

const int& dcmHelper::getNumberOfFrames() const
{
    return mNumberOfFrames;
}

int dcmHelper::getBitsAllocated() const
{
    return mBitsAllocated;
}

int dcmHelper::getSamplePerPixel() const
{
    return mSamplePerPixel;
}

int dcmHelper::getRows() const
{
    return mRows;
}

int dcmHelper::getSeries() const
{
    return this->mSeries;
}


int dcmHelper::getCols() const
{
    return mColumns;
}

float* const dcmHelper::getPixelSpacing()
{
    return mPixelSpacing;
}

float dcmHelper::getSourceToDetector() const
{
    return mSourceToDetector;
}

float dcmHelper::getSourceToPatient() const
{
    return mSourceToPatient;
}

double dcmHelper::getWindowCenter() const
{
    return this->mWindowCenter;
}

double dcmHelper::getWindowWidth() const
{
    return this->mWindowWidth;
}

double dcmHelper::getRescaleIntercept() const
{
    return this->mRescaleIntercept;
}

double dcmHelper::getRescaleSlope() const
{
    return this->mRescaleSlope;
}

double dcmHelper::getFrameTime() const
{
    return this->mFrameTime;
}

int dcmHelper::getIntensifierSize() const
{
    return mIntensifierSize;
}

int dcmHelper::getDistanceSourceToDetector() const
{
    return mDistanceSourceToDetector;
}

double dcmHelper::getDistanceSourceToPatient() const
{
    return mDistanceSourceToPatient;
}

double dcmHelper::getPositionerPrimaryAngle() const
{
    return mPositionerPrimaryAngle;
}

double dcmHelper::getPositionerSecondaryAngle() const
{
    return mPositionerSecondaryAngle;
}

void dcmHelper::setPrimaryAngle(const float& v)
{
    this->mPrimaryAngle = v;
}

void dcmHelper::setSecondaryAngle(const float& v)
{
    this->mSecondaryAngle = v;
}

void dcmHelper::setPerFrame(const int& v)
{
    this->mPerFrame = v;
}

const float dcmHelper::getPrimaryAngle()
{
    return this->mPrimaryAngle;
}

const float dcmHelper::getSecondaryAngle()
{
    return this->mSecondaryAngle;
}

const int dcmHelper::getPerFrame()
{
    return this->mPerFrame;
}

const QString& dcmHelper::getInformation(const int& cw) const
{
    return mDispInfo->s[cw];
}

const int dcmHelper::getTargetType()
{
    return this->tt;
}

QString dcmHelper::getValuebyGE(const Uint16& _G, const Uint16& _E)
{
    //if(!readyForRead())
    //    return QString();

    OFCondition status;
    if (!mFileFormat)
    {
        mFileFormat = new DcmFileFormat();
        mFileFormat->loadFile(mFiles->c_str());
    }
    OFString result;
    auto ge = make_unique<DcmTagKey>(_G, _E);
    if (mFileFormat->getDataset()->findAndGetOFString(*ge.get(), result).good())
        return QString(result.c_str());
    //DcmStack stack;
    //DcmObject* pObject = nullptr;
    //DcmElement* pEle = nullptr;
    ////mResult.getda
    //while (status.good())
    //{
    //    pObject = stack.top();
    //    pEle = (DcmElement*)pObject;
    //    QString tag = pEle->getTag().toString().c_str();
    //    if (tag == _comp)
    //    {
    //        OFString buffer;
    //        pEle->getOFStringArray(buffer);
    //        
    //        return QString(buffer.c_str());
    //    }
    //}
    return QString();
}

const QList<QString>* dcmHelper::Tags() const
{
    return mTp;
}

void dcmHelper::copy(dcmHelper& other)
{
    for (auto& b : other.mTp[0])this->mTp[0].append(b);
    for (auto& b : other.mTp[1])this->mTp[1].append(b);
    for (auto& b : other.mTp[2])this->mTp[2].append(b);
    this->mResult = other.mResult;

    //this->mFiles  = string(other.mFiles);
    //this->mFiles = other.mFiles;
    //copy(this->mFileFormat, this->mFileFormat + sizeof(DcmFileFormat), other.mFileFormat);
    //copy(this->mDcmDataset, this->mDcmDataset + sizeof(DcmDataset), other.mDcmDataset);
    this->mTarget = other.mTarget;
    if (other.mPix)
    {
        const int _size = other.mColumns * other.mRows * other.mNumberOfFrames;
        this->mPix = new uchar[_size];
        memcpy_s(this->mPix,
            _size,
            other.getData(),
            _size);
        //copy(other.getData(), other.getData() + other.mColumns * other.mRows * other.mNumberOfFrames, this->mPix);
    }

    if (other.mData)
    {
        int i = other.mSamplePerPixel == 1 ? -1 : other.mPlanarConfiguration;
        i = (i == -1) ? 1 : 3;
        i *= other.mColumns * other.mRows * other.mNumberOfFrames;
#ifdef DCM_U8
        this->mData = new unsigned char[i];
        memcpy_s(this->mData,
            i * sizeof(unsigned char),
            other.Data(),
            i * sizeof(unsigned char));
#else
        this->mData = new float[i];
        //copy(other.mData, other.mData + i, this->mData);
        memcpy(this->mData, other.Data(), i * sizeof(float));
#endif

    }

    this->mNumberOfFrames = other.mNumberOfFrames;
    this->mRows = other.mRows;
    this->mColumns = other.mColumns;
    this->mBitsAllocated = other.mBitsAllocated;
    this->mBitsStored = other.mBitsStored;
    this->mHighBit = other.mHighBit;
    this->mPixelRepresentation = other.mPixelRepresentation;
    this->mSamplePerPixel = other.mSamplePerPixel;
    this->mPlanarConfiguration = other.mPlanarConfiguration;
    this->mWindowCenter = other.mWindowCenter;
    this->mWindowWidth = other.mWindowWidth;
    this->mRescaleIntercept = other.mRescaleIntercept;
    this->mRescaleSlope = other.mRescaleSlope;
    this->mPhotometricInterpretation = other.mPhotometricInterpretation;
    this->mStep = other.mStep;
    this->mEXS = other.mEXS;
    this->mDseq = other.mDseq;
    this->bCompressed = other.bCompressed;
    this->bReadString = other.bReadString;

    this->mIntensifierSize = other.mIntensifierSize;
    this->mDistanceSourceToDetector = other.mDistanceSourceToDetector;
    this->mDistanceSourceToPatient = other.mDistanceSourceToPatient;
    this->mPositionerPrimaryAngle = other.mPositionerPrimaryAngle;
    this->mPositionerSecondaryAngle = other.mPositionerSecondaryAngle;
    this->mSeries = other.mSeries;

}

unsigned short dcmHelper::getValbyTagU16(const DcmTagKey& tag)
{
    unsigned short result;
    if (mDcmDataset->findAndGetUint16(tag, result).good() == OFTrue) return result;
    else return -1;
}
signed short dcmHelper::getValbyTagS16(const DcmTagKey& tag)
{
    signed short result;
    if (mDcmDataset->findAndGetSint16(tag, result).good() == OFTrue) return result;
    else return 0;
}
signed int dcmHelper::getValbyTagS32(const DcmTagKey& tag)
{
    int result;
    if (mDcmDataset->findAndGetSint32(tag, result).good() == OFTrue) return result;
    else return INT_MIN;
}
unsigned int dcmHelper::getValbyTagU32(const DcmTagKey& tag)
{
    unsigned int result;
    if (mDcmDataset->findAndGetUint32(tag, result).good() == OFTrue) return result;
    else return -1;
}
float dcmHelper::getValbyTagF32(const DcmTagKey& tag)
{
    float result;
    if (mDcmDataset->findAndGetFloat32(tag, result).good() == OFTrue) return result;
    else return 0;
}

double dcmHelper::getValbyTagF64(const DcmTagKey& tag)
{
    double result;
    if (mDcmDataset->findAndGetFloat64(tag, result).good() == OFTrue) return result;
    else return 0;
}

const char* dcmHelper::getValbyTagStr(const DcmTagKey& tag)
{
    const char* result;
    if (mDcmDataset->findAndGetString(tag, result).good() == OFTrue) return result;
    else return nullptr;
}

void dcmHelper::decodePixel()
{
    if (mBitsStored > 8)
    {
        //Decode..
    }
}

bool dcmHelper::readyForRead()
{
    if (mDcmDataset == nullptr)
    {
        if (mFiles->empty())
        {
            cerr << __FILE__ << ":" << __LINE__ << ":  filename is empty\n";
            return false;
        }

        DcmFileFormat ff;
        mResult = ff.loadFile(mFiles->c_str());
        if (mResult.good() != OFTrue)
        {
            cerr << __FILE__ << ":" << __LINE__ << ":  loadFile Error\n";
            return false;
        }

        mDcmDataset = ff.getDataset();
        if (mDcmDataset == nullptr)
        {
            cerr << __FILE__ << ":" << __LINE__ << ":  getDateset Error\n";
            return false;
        }
    }
    return true;
}

void dcmHelper::getImportInfo()
{
    //Signed Integer
    mNumberOfFrames = getValbyTagS32(DCM_NumberOfFrames);
    mSeries = getValbyTagS32(DCM_SeriesNumber);

    //Unsigned Short
    mRows = getValbyTagU16(DCM_Rows);
    mColumns = getValbyTagU16(DCM_Columns);
    mBitsAllocated = getValbyTagU16(DCM_BitsAllocated);
    mBitsStored = getValbyTagU16(DCM_BitsStored);
    mHighBit = getValbyTagU16(DCM_HighBit);
    mPixelRepresentation = getValbyTagU16(DCM_PixelRepresentation);
    mSamplePerPixel = getValbyTagU16(DCM_SamplesPerPixel);
    mPlanarConfiguration = getValbyTagU16(DCM_PlanarConfiguration);
    mLargestImagePixelValue = getValbyTagU16(DCM_LargestImagePixelValue);
    if (mLargestImagePixelValue == UINT16_MAX)
        mLargestImagePixelValue = getValbyTagS16(DCM_LargestImagePixelValue);

    mSmallestImagePixelValue = getValbyTagU16(DCM_SmallestImagePixelValue);
    if (mSmallestImagePixelValue == UINT16_MAX)
        mSmallestImagePixelValue = getValbyTagS16(DCM_SmallestImagePixelValue);

    mIntensifierSize = getValbyTagF64(DCM_IntensifierSize);
    mDistanceSourceToDetector = getValbyTagF64(DCM_DistanceSourceToDetector);
    mDistanceSourceToPatient = getValbyTagF64(DCM_DistanceSourceToPatient);
    mPositionerPrimaryAngle = getValbyTagF64(DCM_PositionerPrimaryAngle);
    mPositionerSecondaryAngle = getValbyTagF64(DCM_PositionerSecondaryAngle);
    mFrameTime = getValbyTagF64(DCM_FrameTime);


    //Float
    {
        auto _buf = getValbyTagStr(DCM_RescaleIntercept);
        mRescaleIntercept = _buf ? strtof(_buf, nullptr) : 1;
    }
    {
        auto _buf = getValbyTagStr(DCM_RescaleSlope);
        mRescaleSlope = _buf ? strtof(_buf, nullptr) : 1;
    }

    //Double
    mWindowCenter = getValbyTagF64(DCM_WindowCenter);
    mWindowWidth = getValbyTagF64(DCM_WindowWidth);

    //String
    mPhotometricInterpretation = getValbyTagStr(DCM_PhotometricInterpretation);

    auto _buf = getValbyTagStr(DcmTagKey(0x0018, 0x1164));
    if (_buf)
    {
        auto ptr = strtok((char*)_buf, "\\");
        mPixelSpacing[0] = _buf ? strtof(_buf, nullptr) : 1;
        mPixelSpacing[1] = ptr ? strtof(ptr, nullptr) : 1;
    }
    getCurrentPixelType();
}



//OFCondition DcmItem::findAndGetElements(const DcmTagKey& tagKey, DcmStack& resultStack)
//{
//    OFCondition status = EC_TagNotFound;
//    DcmStack stack;
//    DcmObject* object = NULL;
//    /* iterate over all elements */
//    while (nextObject(stack, OFTrue).good())
//    {
//        /* get element */
//        object = stack.top();
//
//        /* add to result_stack */
//        resultStack.push(object);
//        if (object->getTag() == tagKey)
//        {
//            status = EC_Normal;
//        }
//    }
//    return status;
//}




bool dcmHelper::getPixeldata(const DcmObject& obj, DcmPixelData* ele, const float& specific)
{

    checkSyntax(ele);
    //mPixelRepresentation 0: unsigned 1:signed
    void* pd = nullptr;
    int _len = 0;
    DcmPixelItem* _pix = nullptr;

    bool bSnap = (specific != 0.0f) ? true : false;
    int idx = int(specific * mNumberOfFrames) * (mSamplePerPixel == 1 ? 1 : mPlanarConfiguration) * mColumns * mRows;

    if (bCompressed)
        _len = mColumns * mRows * mNumberOfFrames;
    else
        _len = ele->getLength(mEXS);

    //mData = new float[bSnap ? _len / mNumberOfFrames : _len];
    mData = new unsigned char[bSnap ? _len / mNumberOfFrames : _len];
    switch (tt)
    {
    case dcmHelper::U8:
    {
        Uint8* _buffer = nullptr;

        if (bCompressed)
        {
            if (bSnap)
            {
                DJDecoderRegistration::registerCodecs();
                int idx = int(specific * mNumberOfFrames) * (mSamplePerPixel == 1 ? 1 : mPlanarConfiguration);
                DicomImage dicomImage(mDcmDataset, EXS_JPEGProcess14SV1, CIF_UsePartialAccessToPixelData, idx);
                if (dicomImage.getStatus() != EIS_Normal) {
                    std::cerr << "Error: Cannot load DICOM image (" << DicomImage::getString(dicomImage.getStatus()) << ")" << std::endl;
                    return 1;
                }
                const Uint8* pixelData = (const Uint8*)dicomImage.getOutputData(8);
                mPix = new uchar[mColumns * mRows];
                std::transform(pixelData, pixelData + (mColumns * mRows), mPix, [&](unsigned char v)
                    {
                        auto val = 255.0 * ((v - (mWindowCenter - 0.5)) / (mWindowWidth - 1) + 0.5);
                        return unsigned char(std::clamp(val, 0.0, 255.0));
                    });

                DJDecoderRegistration::cleanup();
            }
            else
            {
                DJDecoderRegistration::registerCodecs();
                DicomImage dicomImage(mDcmDataset, EXS_JPEGProcess14SV1, CIF_DecompressCompletePixelData);
                if (dicomImage.getStatus() != EIS_Normal) {
                    std::cerr << "Error: Cannot load DICOM image (" << DicomImage::getString(dicomImage.getStatus()) << ")" << std::endl;
                    return 1;
                }
                Uint32 totalPixelsPerFrame = mColumns * mRows;
                // 프레임 수 가져오기
                Uint16 numFrames = dicomImage.getFrameCount();

                for (Uint16 frame = 0; frame < numFrames; ++frame) {
                    // 프레임별로 8비트 픽셀 데이터 가져오기
                    const Uint8* pixelData = (const Uint8*)dicomImage.getOutputData(8, frame);
                    if (pixelData == nullptr) {
                        std::cerr << "Error: Cannot extract pixel data for frame " << frame << std::endl;
                        return false;
                    }

                    // 해당 프레임의 데이터를 전체 배열에 복사
                    std::memcpy(mData + frame * totalPixelsPerFrame, pixelData, totalPixelsPerFrame);
                }
                DJDecoderRegistration::cleanup();
            }
        }
        else
        {
            ele->getUint8Array(_buffer);
            if (bSnap)
            {
                mPix = new uchar[mColumns * mRows];
                std::transform(_buffer + idx, _buffer + idx + (mColumns * mRows), mPix, [&](unsigned char v)
                    {
                        auto val = 255.0 * ((v - (mWindowCenter - 0.5)) / (mWindowWidth - 1) + 0.5);
                        return unsigned char(std::clamp(val, 0.0, 255.0));
                    });
            }
            else
                std::copy_n(_buffer, _len, mData);
        }
        break;
    }
    case dcmHelper::S8:
    {
        return false;
    }
    case dcmHelper::U16:
    {
        Uint16* _buffer = nullptr;
        if (bCompressed)
            break;
        mResult = bCompressed ? _pix->getUint16Array(_buffer) : ele->getUint16Array(_buffer);
        if (bSnap)
        {
            mPix = new uchar[mColumns * mRows];
            // std::copy_n(_buffer + idx, mColumns * mRows, mPix);
            std::transform(_buffer + idx, _buffer + idx + (mColumns * mRows), mPix, [&](unsigned short v)
                {
                    return unsigned char(std::clamp(255.0 * ((v - (mWindowCenter - 0.5)) / (mWindowWidth - 1) + 0.5), 0.0, 255.0));
                });

        }
        else
        {
            if (_len > mNumberOfFrames * mColumns * mRows)
                _len = mNumberOfFrames * mColumns * mRows;

            std::transform(_buffer, _buffer + _len, mData, [&](unsigned short v)
                {
                    return unsigned char(std::clamp(255.0 * ((v - (mWindowCenter - 0.5)) / (mWindowWidth - 1) + 0.5), 0.0, 255.0));
                });
        }

        break;
    }
    default:
        return false;
    }

    DJDecoderRegistration::cleanup();
    return true;
}


void dcmHelper::getCurrentPixelType()
{
    switch (mBitsAllocated)
    {
    case 8:
        mTarget = DCM_PixelData; tt = mPixelRepresentation == 0 ? U8 : S8; break;
    case 16:
        mTarget = DCM_PixelData; tt = mPixelRepresentation == 0 ? U16 : S16; break;
        //case 32:
        //    mTarget = DCM_FloatPixelData; tt = F32; break;
        //case 64:
        //    mTarget = DCM_DoubleFloatPixelData; tt = F64; break;
    default:
        break;
    }
}


void dcmHelper::checkSyntax(DcmPixelData* elem)
{
    //Encalsulated test
    //DcmPixelData* elem = OFstatic_cast(DcmPixelData*, obj);
    if (mDseq)
        delete mDseq;
    mDseq = nullptr;

    mEXS = EXS_Unknown;
    const DcmRepresentationParameter* rep = nullptr;
    if (elem)
        elem->getOriginalRepresentationKey(mEXS, rep);

    bCompressed = DcmXfer(mEXS).isEncapsulated();
    //Non EC varify
}

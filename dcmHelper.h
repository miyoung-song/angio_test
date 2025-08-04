#pragma once


#include"dcmtk\config\osconfig.h"
#include"dcmtk\dcmdata\dctk.h"
#include"dcmtk\dcmdata\dcddirif.h"
#include"dcmtk\dcmdata\dcpxitem.h"
#include"dcmtk\dcmimgle\dimosct.h"
#include"dcmtk\dcmimgle\dimo2img.h"
#include"dcmtk\ofstd\oftypes.h"
#include "dcmtk\dcmimgle\dcmimage.h"
#include <dcmtk/dcmjpeg/djdecode.h>
#include <dcmtk/dcmjpls/djdecode.h>

#pragma comment(lib,"iphlpapi.lib")
#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"netapi32.lib")
#pragma comment(lib,"wsock32.lib")

#pragma comment(lib,"cmr.lib")
#pragma comment(lib,"dcmdata.lib")
#pragma comment(lib,"dcmdsig.lib")
#pragma comment(lib,"dcmect.lib")
#pragma comment(lib,"dcmfg.lib")
#pragma comment(lib,"dcmimage.lib")
#pragma comment(lib,"dcmimgle.lib")
#pragma comment(lib,"dcmiod.lib")
#pragma comment(lib,"dcmjpeg.lib")
#pragma comment(lib,"dcmjpls.lib")
#pragma comment(lib,"dcmnet.lib")
#pragma comment(lib,"dcmpmap.lib")
#pragma comment(lib,"dcmpstat.lib")
#pragma comment(lib,"dcmqrdb.lib")
#pragma comment(lib,"dcmrt.lib")
#pragma comment(lib,"dcmseg.lib")
#pragma comment(lib,"dcmsr.lib")
#pragma comment(lib,"dcmtkcharls.lib")
#pragma comment(lib,"dcmtls.lib")
#pragma comment(lib,"dcmtract.lib")
#pragma comment(lib,"dcmwlm.lib")
#pragma comment(lib,"i2d.lib")
#pragma comment(lib,"ijg12.lib")
#pragma comment(lib,"ijg16.lib")
#pragma comment(lib,"ijg8.lib")
#pragma comment(lib,"oflog.lib")
#pragma comment(lib,"ofstd.lib")

#include"Functor.h"

//#include"TableModel.h"
#include<qdatastream.h>

#include<cstring>
#include<vector>
#include<map>

#define DCM_U8
#define WL 120
#define WW 255 

using namespace std;

/// <summary>
/// Distance to source to detector		//0018,1110
/// Distence source to patient			//0018,1111
/// IntensifierSize -> * root(2) / 
/// 
/// 
/// 	//0018,1162
/// Positioner Primary angle			//0018,1510
/// Positioner Secondary angle			//0018,1511
/// µî°£°Ý
/// </summary>


#define GE(a,b) {Uint16((a)) << 2 + Uint16((b));}
#define SafeReleasePtr(x) {if(x){delete x;x=nullptr;}}
#define SafeReleaseAry(x) {if(x){delete[] x;x=nullptr;}}
class dcmHelper : private DcmItem
{
private:
	using RR = pair<string, DcmTagKey>;
	using vRR = vector<RR>;
	struct displayInfo
	{
		const vRR LeftTop = {
			RR("",DcmTagKey(0x0010,0x0010)),
			RR("",DcmTagKey(0x0010,0x0040)),
			RR("",DcmTagKey(0x0010,0x0020)),
			RR("",DcmTagKey(0x0008,0x0022))
		};
		const vRR RightTop = {
			RR("",DcmTagKey(0x0008,0x0080))
		};
		const vRR RightBottom = {
			//RR("#",DcmTagKey(0x0028,0x1050)),
			//RR("WL",DcmTagKey(0x0028,0x1051))
			RR("WL",DcmTagKey())
		};
		const vRR LeftBottom = {
			RR("Frame: ",DcmTagKey()),
			RR("Series no.: ",DcmTagKey(0x0020,0x0011)),
			RR("Instance no.: ",DcmTagKey(0x0020,0x0013)),
			RR("Acq. speed: ",DcmTagKey(0x0018,0x0040)),
			RR("Cal. fac.: ",DcmTagKey(0x0028,0x0030)),
			RR("#",DcmTagKey(0x0018,0x1510)),
			RR("#",DcmTagKey(0x0018,0x1511))
		};

		array<QString, 4> s;
	};
public:
	enum targetType : int { U8 = 1, S8 = 2, U16 = 3, S16 = 4, F32 = 5, F64 = 6 };
	enum WindowLevel :int { WindowWidth = 0, WindowCenter };
public:
	dcmHelper();
	dcmHelper(string fname);
	//dcmHelper(const dcmHelper& );
	~dcmHelper();
	const void* operator[] (const int& idx) const;

	bool loadFile(bool read = false);
	bool loadFile(string fname);
	bool loadFile(const float& r);

	void release();

	string* getFile() const;
	void setFile(string fname);

	uchar* getData();
#ifdef DCM_U8
	const unsigned char* constData();
	unsigned char* Data();
#else

	const float* constData();
	float* Data();
#endif

	unsigned short* Data16bits();

	const int& getStep() const;
	const int& getNumberOfFrames() const;
	int getBitsAllocated() const;
	int getSamplePerPixel() const;
	int getRows() const;
	int getSeries() const;
	int getCols() const;


	float* const getPixelSpacing();
	float getSourceToDetector() const;
	float getSourceToPatient() const;

	double getWindowCenter() const;
	double getWindowWidth() const;
	double getRescaleIntercept() const;
	double getRescaleSlope() const;

	double getFrameTime() const;

	int getIntensifierSize() const;
	int getDistanceSourceToDetector() const;

	double getDistanceSourceToPatient() const;
	double getPositionerPrimaryAngle() const;
	double getPositionerSecondaryAngle() const;

	void setPrimaryAngle(const float& v);
	void setSecondaryAngle(const float& v);
	void setPerFrame(const int& v);


	const float getPrimaryAngle();
	const float getSecondaryAngle();
	const int getPerFrame();

	const QString& getInformation(const int& cw) const;

	QString getValuebyGE(const Uint16&, const Uint16&);

	const int getTargetType();

	const QList<QString>* Tags() const;

	//friend QDataStream& operator<<(QDataStream& out, const char*& rhs);
	//friend QDataStream& operator>>(QDataStream& in, char*& rhs);

	void copy(dcmHelper& other);
protected:

	//OFCondition findAndGetElements(const DcmTagKey& tagKey, DcmStack& resultStack);

private:
	bool readyForRead();
	//T tagCollection(const DcmTagKey&);
	void getImportInfo();

	bool getPixeldata(const DcmObject& obj, DcmPixelData* ele, const float& specific);

	void getCurrentPixelType();
	void checkSyntax(DcmPixelData* elem);

	bool readOwnFile(const float& specific = 0.0f);

	unsigned short getValbyTagU16(const DcmTagKey&);
	signed short getValbyTagS16(const DcmTagKey&);
	signed int getValbyTagS32(const DcmTagKey&);
	unsigned int getValbyTagU32(const DcmTagKey&);
	float getValbyTagF32(const DcmTagKey&);
	double getValbyTagF64(const DcmTagKey&);
	const char* getValbyTagStr(const DcmTagKey&);

	void decodePixel();

private:

	QList<QString> mTp[3];

	OFCondition mResult;
	string* mFiles = nullptr;

	displayInfo* mDispInfo = nullptr;

	DcmFileFormat* mFileFormat = nullptr;

	DcmDataset* mDcmDataset = nullptr;
	DcmTagKey mTarget;

	uchar* mPix = nullptr;
#ifdef DCM_U8
	unsigned char* mData = nullptr;
#else
	float* mData = nullptr;
#endif // DCM_U8

	targetType tt;

	unsigned short* mData16bits = nullptr;


	int mNumberOfFrames, mLargestImagePixelValue, mSmallestImagePixelValue;
	unsigned short mRows = 512, mColumns = 512, mBitsAllocated, mBitsStored, mHighBit, mPixelRepresentation, mSamplePerPixel, mPlanarConfiguration;
	int mIntensifierSize, mDistanceSourceToDetector;
	double mDistanceSourceToPatient, mPositionerPrimaryAngle, mPositionerSecondaryAngle, mFrameTime;
	double mWindowCenter = WL, mWindowWidth = WW, mRescaleIntercept, mRescaleSlope;
	string mPhotometricInterpretation;


	//0018,1110
	float mSourceToDetector;//SID
	//0018,1111
	float mSourceToPatient;//SOD

	//0018,1164
	float mPixelSpacing[2] = { 0.368,0.368 };//pixelspacing

	//0018,1510
	float mPrimaryAngle;
	//0018,1511
	float mSecondaryAngle;
	int mPerFrame;

	int mStep;

	E_TransferSyntax mEXS;
	DcmPixelSequence* mDseq = nullptr;
	bool bCompressed;

	bool bReadString;

	int mSeries;
};

//
//QDataStream& operator<<(QDataStream& out, dcmHelper* const& rhs) {
//	out.writeRawData(reinterpret_cast<const char*>(&rhs), sizeof(rhs));
//	return out;
//}
//
//QDataStream& operator >> (QDataStream& in, dcmHelper& rhs) {
//	in.readRawData(reinterpret_cast<char*>(&rhs), sizeof(rhs));
//	return in;
//}

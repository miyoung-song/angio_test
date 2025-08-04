#pragma once
#include<qevent.h>

//#include<qcursor.h>
#include<qmouseevent.h>

#include<qtimer.h>
#include <qwidget.h>
#include <qmimedata.h>
#include <qdatastream.h>
#include <future>
#include <qfiledialog.h>
#include <codecvt>

#include "GraphicsWindow.h"
#include "dcmHelper.h"
#include "TexObj.h"
#include "TaskManager/Manager.h"


#include "ui_windowGT.h"
#include "ui_windowselect.h"

#include "windowSelect.h"
#include "windowImage.h"
#include "windowFFR3D.h"
#include "windowMessageBox.h"
#include "windowBoundaryCondition.h"
#include "windowModel.h"


#define VIEW_COL 2
#define VIEW_ROW 1

class string_section {
private:
    std::string str;  // ���� ���ڿ�
    std::string delimiter;  // ������

    // ���ڿ��� �����ڷ� ������ �Լ� (private �޼���)
    std::vector<std::string> split() const {
        std::vector<std::string> tokens;
        size_t start = 0, end = 0;

        // �����ڷ� ���ڿ� �и�
        while ((end = str.find(delimiter, start)) != std::string::npos) {
            tokens.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
        }
        tokens.push_back(str.substr(start));  // ������ �κ� �߰�

        return tokens;
    }

public:
    // ������: ���ڿ��� �����ڸ� �Է¹���
    string_section(const std::string& input_str, const std::string& delim)
        : str(input_str), delimiter(delim) {}

    // section �Լ�: ���ڿ� ���� ����
    std::string get_section(int start, int end) const {
        std::vector<std::string> tokens = split();

        // �ε����� ������ �Է¹����� �ڿ������� ��
        if (start < 0) start = tokens.size() + start;
        if (end < 0) end = tokens.size() + end;

        // start�� end�� ��ȿ���� �˻�
        if (start >= tokens.size() || end < start) return "";

        std::string result;
        for (int i = start; i <= end && i < tokens.size(); ++i) {
            if (i > start) result += delimiter;  // ������ �߰�
            result += tokens[i];
        }

        return result;
    }
};

class ViewClass :
    public QWidget, Ui::windowGT
{
    Q_OBJECT

public:
    enum ModelType { EndDiastolic = 0, Segmentation_2D, Segmentation_3D, FFR };
    explicit ViewClass(QWidget* parent = 0);
    ~ViewClass();

    QString FromUnicode(QString str); // �ѱ� �����ڵ� ��ȯ
    std::string convertEucKrToUtf8(const std::string& eucKrStr);

    void updateRepositoryInformation(const std::string& directoryPath, bool load_image = false);

    void setEmptyViewer(int nIdex); // ȭ�� ���÷���

    bool IsDCMfile();

    void Update2DLine(int nViewIndex, model_info::segmentation_manual_type type);

    void Update2DLine(int nViewIndex, model_info::segmentation_model_type Model_type, model_info::segmentation_exe_type run_type);

    void SetEndPoints(model_info::find_endpoint_type type, int view_id);

    void save_file(model_info::save_file_type type);

    void Run3D(); // Line �����ͷ� 3D ������ ���
    void Open3DFile();
    void UpdateEpiline();
    void findPointsOnEpipolarLine();
    void run_process(std::string program);
    // plt ���� �ҷ�����

    void UpdataEqualLine();

    void SetFFR(); // 3D ������ ffr ó��
    void SetFFRGeneration();// �ܺ� ���α׷� ffr ����ϴ� exe ����

    void clear_result_view(); // view ȭ�� �ʱ�ȭ

    bool set_chartFFR(std::vector<int>& X, std::vector<float>& axisX, std::vector<float>& axisY);

    bool set_chart3D(std::vector<int>& X, std::vector<float>& axisX, std::vector<float>& axisY);

    void UpdateLine(int id, bool is_move_point);

    void UpdateCursorView(int2 nIndex); // ���õ� view �ε���

    void SetResultView(int ntype, QString filename); //ffr ��� ������ ȭ�� ���


    void remove_folder(model_info::folder_type type);

    void clear_result_line();  //ffr ��� ������ ȭ�� �ʱ�ȭ


    void UpdateChartClose(); //�̻��

    void Run_FFR();

    windowMessageBox* CreateMessage(QString strTitle, QString strText, QString btn = "Close", bool bModal = false);

    void copy_file(const std::filesystem::path& srcPath, const std::filesystem::path& dstPath);

    void remove_files_with_extensions(const std::filesystem::path& dir);

    void setMoveX(int x) { m_nMoveX = x; }; // view ȭ�� ������ġ

    void ProcessFinished(int exitCode, QProcess::ExitStatus exitStatus); // �ܺ����α׷� ���� �Լ�

    void copyDir(const std::string& srcDirPath, const std::string& dstDirPath);
    void removeFilesWithExtensions(const std::string& directoryPath);

public slots:
    void updateScene(dcmHelper*);
    void updateImageView(QString str);
    void SetOrigWindowLevel(int nIndex);
    void SetWindowLevel(int nWC, int nWW);
    void SetLineRange(int Value, bool bAuto);

    void pbLoad(); // �̹��� �ε�
    void SaveAs();

    void testPause(); // �������
    void testPlay(); // ������
    void SetClearLine(); // ȭ�� �ʱ�ȭ
    void SetLineUndo(); // ���� ���� ���

    void SetFFR3D(); // 2D -> 3D -> FFR

    void RunAI(int view_id);

signals:
    void turnOnViewTab();
    void closed(ViewClass*);

    void requestWindowLevelControl(int nWL, int nWW);
    void requestControlEnable(bool bEnabled = false);

    void requestSetRepositoryText(QString str);
    void requestSetMenuShow(bool bshow);
    void spread(const QStringList&, const QString& _title, bool bLoad);

protected:
    GraphicsWindow* getEmptyViewer(int&, int&);
    void SetViewer(dcmHelper* dh);
    bool manualImport();
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string section(const std::string& path, char delimiter, int start, int end = -1);
    void dragEnterEvent(QDragEnterEvent*) override;
    void dropEvent(QDropEvent* e)  override;

    struct Repository 
    {
        std::string root;
        QStringList children;

        Repository() : root(""), children() {}

        bool isEmpty() const {
            return root.empty() && children.empty();
        }

        std::string path(const int& index) const {
            if (index < 0 || index >= children.size()) {
                return ""; // ������ ��� ��� �� ���ڿ� ��ȯ
            }
            return root + '/' + (children.at(index)).toStdString();
        }
    };

    struct Load_file_Instance
    {
        bool open_loadfile = false;
        std::string load_path;
        std::string load_file_path;
        int frame_id_l = 0;
        int frame_id_r = 0;
        void init()
        {
            open_loadfile = false;
            load_path = "";
            load_file_path = "";
            frame_id_l = 0;
            frame_id_r = 0;
        }
    };

private:
    GraphicsWindow* mGraphics[2][2] = { nullptr, };
    windowFFR3D* mFFR3D = nullptr;
    windowFFR3D* mtestChart = nullptr;
    std::string m_strProgramPath;
    std::string m_folderName;

    enum { View2D, View3D, ViewFFR, ViewChart };
    enum { LeftImage, RightImage, AllImage };

    int2 m_nCursorView;
    int m_nLineRange = 0;
    bool m_bLineAutoRange = true;

    int m_WindowWidth = WW, m_WindowCenter = WL;
    std::vector<Repository> mRepository;
    unique_ptr<bool[]> mChecker;
    bool m_bFFRShow = false;
    bool m_b3DShow = false;
    std::string m_strRepository;
    int m_nReferenceImageIndex = 0;
    QString m_strFilePath;
    int m_nMoveX = 0;
    QTime mtestTimer;
    bool m_testing = false; //�ÿ� 
    std::vector<int> m_vecEqualPercents;
    Load_file_Instance loadfile_;
};

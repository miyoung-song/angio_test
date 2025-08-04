#ifndef WINDOWMESSAGEBOX_H
#define WINDOWMESSAGEBOX_H

#include <QDialog>

#include "ui_windowMessageBox.h"


class windowMessageBox : public QDialog, private Ui::windowMessageBox
{
    Q_OBJECT

public:
    explicit windowMessageBox(QWidget* parent = nullptr);
    ~windowMessageBox();

    void setText(QString str);

    void SetBtnType(QString str);

    void Select(QString str);

    int get_select() { return select_; };
    bool GetDialogCode() { return m_nDialogCode; };
    void SetDialogCode(int DialogCode);

    void SetClose() { this->reject(); };
    void closeEvent(QCloseEvent* event);

private slots:
    void BtnClickedApply();
    void BtnClickedCancel();

public:
    int m_nDialogCode = QDialog::Rejected;
    int select_ = -1;
};

#endif // WINDOWMESSAGEBOX_H

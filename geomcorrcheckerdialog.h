#ifndef GEOMCORRCHECKERDIALOG_H
#define GEOMCORRCHECKERDIALOG_H

#include <QDialog>
#include <QStringList>
#include <QDir>
#include <image.h>
#include "geomcorr.h"
#include <QApplication>
#include <gaincorr.h>



namespace Ui {
class geomCorrCheckerDialog;
}

class geomCorrCheckerDialog : public QDialog
{
    Q_OBJECT

public:
    explicit geomCorrCheckerDialog(QWidget *parent = 0);
    ~geomCorrCheckerDialog();
    void getFileList();

private:
    QStringList fileList;
    QDir dir;
    QDir outDir;
    Ui::geomCorrCheckerDialog *ui;
    QLabel* activeLabel;
    int firstIndex;
    int lastIndex;
    int n;
    int u;
    Geomcorr geomcorr;
    Gaincorr gaincorr;



private slots:
    void setDir();
    void displayImage(int i);
    void chooseRotatedImage();
    void setCounterLabel(int i);
    void validateResult();
    void validateResult(int i); // slot when first and last index are read and number of circles changed.

    void reset();
    void calculate();


};

#endif // GEOMCORRCHECKERDIALOG_H

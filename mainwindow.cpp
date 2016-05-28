#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QChar>
#include <QDir>
#include "gaincorr.h"
#include "geomcorr.h"
#include <QTime> // debug reasons.
#include<iostream>
#include "directorystructureconverter.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->pushButton_export->setVisible(false);



}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::resetGui()
{
    ui->pushButton_export->setVisible(false);
}


void MainWindow::on_button_choosefile_clicked()
{
    QString filename = QFileDialog::getOpenFileName(0,"Fájl megnyitása", "C:\\"); //Debug.
    if(filename.endsWith(QChar('f')))
    {
        image.readfromfloatfile(filename.toStdString());
    }
    else
    image.readfromfile(filename.toStdString());
    image.drawimage(ui->label);
    image.calculate_meanvalue_on_GPU();
    image.writedetailstoscreen(ui->textEdit);
    std::cout << "max on GPU:" << image.getmax() <<std::endl<<
                 "min on GPU: " << image.getmin() <<std::endl;

    std::cout <<"correlating with itsetlf : " << image.correlateWith(image) << std::endl;
    ui->pushButton_export->setVisible(true);

}

void MainWindow::on_pushButton_clicked()
{
    resetGui();
    gc.readAndCalculateGain();
}
// end of pushbutton_clicked

void MainWindow::on_pushButton_2_clicked()
{
    resetGui();
    gc.readgainfactors();


}

void MainWindow::on_pushButton_3_clicked()
{
    resetGui();
    image.clear();
    QString filename = QFileDialog::getOpenFileName(0,"Fájl megnyitása", "C:\\"); //Debug.
    if(filename.endsWith(QChar('f')))
    {
        image.readfromfloatfile(filename.toStdString());
        image.readinfo();
    }
    else
    {
     image.readfromfile(filename.toStdString());
     image.readinfo();
    }
    image.drawimage(ui->label);
    image.writedetailstoscreen(ui->textEdit);


    std::cout <<"corrigating... " <<std::endl;
    gc.gaincorrigateimage(image);
    std::cout <<"calculating... " <<std::endl;

    image.calculate_meanvalue_on_GPU();
    image.writedetailstoscreen(ui->textEdit);


    image.writetofloatfile(image.getfilename());
    std::cout <<"Image written to " << image.getfilename() <<std::endl;
    std::cout <<"kesz." << std::endl;
}

void MainWindow::on_pushButton_4_clicked()
{
    resetGui();
    gc.readAndCalculateOffset();
}

void MainWindow::on_pushButton_5_clicked()
{
    resetGui();
    gc.readoffsetfactors();
}

void MainWindow::on_pushButton_6_clicked()
{
    resetGui();
    image.clear();
    QString filename = QFileDialog::getOpenFileName(0,"Fájl megnyitása", "C:\\"); //Debug.
    if(filename.endsWith(QChar('f')))
    {
        image.readfromfloatfile(filename.toStdString());
    }
    else
    image.readfromfile(filename.toStdString());
    gc.offsetcorrigateimage(image);
    image.drawimage(ui->label);
    image.calculate_meanvalue_on_GPU();
    image.writedetailstoscreen(ui->textEdit);
    image.writetofloatfile(filename.toStdString() +"f");
}

void MainWindow::on_pushButton_7_clicked()
{
    resetGui();
    Geomcorr geomcor;

    geomcor.readAndCalculateGeom();
    geomcor.exportText("C:/awing/ellipses.txt");
    std::cout <<"finito" << std::endl;






}

void MainWindow::on_pushButton_8_clicked()
{
    resetGui();
    DirectoryStructureConverter dSC;
    dSC.copyDirAsImage();
}

void MainWindow::on_pushButton_9_clicked()
{
    resetGui();
    Gaincorr gaincorr;
    gaincorr.readgainfactors();
    gaincorr.readoffsetfactors();

    //Asking for input and output  directories.
       QString inputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which projection images"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

       QString outputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to save corrigated projection images"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

    QDir inputDirectory(inputDir);
    QDir outputDirectory(outputDir);

    QString inputPath = inputDirectory.absolutePath();
    QString outputPath = outputDirectory.absolutePath();
    QStringList nameFilter("*.bin"); //name filter.
    QStringList fileList = inputDirectory.entryList(nameFilter); //File list of .bin files in the actual subdirectory.

    std::cout << "Gain corrigating " << fileList.size() << " number of images from " << inputPath.toStdString() << " to " << outputPath.toStdString() << std::endl;

    if(!QDir(outputDir + "/jpegs").exists())
    QDir().mkdir(outputDir + "/jpegs");


    for(int i=0; i< fileList.size(); i++)
    {
        Image image;
        image.readfromfile( QString ( inputPath + "/" +  fileList.at(i) ).toStdString());
        std::cout << "image " << i << " with voltage: " << image.getvoltage() << " and amperage : " << image.getamperage() << " and mean intensity: " << image.getmean() << std::endl;
        gaincorr.offsetcorrigateimage(image);
        gaincorr.gaincorrigateimage(image);
        image.writetofile( QString ( outputPath + "/"  + fileList.at(i) ).toStdString() );
        image.saveAsJPEG(QString ( outputPath + "/jpegs/"  + fileList.at(i) ).toStdString() + ".jpg");

    }


}

void MainWindow::on_pushButton_export_clicked()
{
    QString outputDir = QFileDialog::getSaveFileName(0,"Save image as...", "C:\\",".jpg");
    image.saveAsJPEG(outputDir.toStdString());
}

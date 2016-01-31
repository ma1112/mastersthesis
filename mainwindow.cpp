#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QChar>
#include <QDir>
#include "gaincorr.h"
#include <QTime> // debug reasons.
#include<iostream>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);



}

MainWindow::~MainWindow()
{
    delete ui;
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



}

void MainWindow::on_pushButton_clicked()
{
    gc.readAndCalculateGain();



}
// end of pushbutton_clicked

void MainWindow::on_pushButton_2_clicked()
{
    //gc.readgianfactors();
    gc.readgainfactors();


}

void MainWindow::on_pushButton_3_clicked()
{
    image.clear();
    QString filename = QFileDialog::getOpenFileName(0,"Fájl megnyitása", "C:\\"); //Debug.
    if(filename.endsWith(QChar('f')))
    {
        image.readfromfloatfile(filename.toStdString());
        readinfo();
    }
    else
    image.readfromfile(filename.toStdString());
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
    gc.readAndCalculateOffset();
}

void MainWindow::on_pushButton_5_clicked()
{
    gc.readoffsetfactors();
}

void MainWindow::on_pushButton_6_clicked()
{
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

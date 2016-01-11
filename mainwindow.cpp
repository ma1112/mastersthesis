#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QChar>
#include <QDir>
#include "gaincorr.h"
#include <QTime> // debug reasons.


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
    image.calculate_meanvalue_on_CPU();
    image.writedetailstoscreen(ui->textEdit);


}

void MainWindow::on_pushButton_clicked()
{
    gc.readimages();
    gc.calculate();



}
// end of pushbutton_clicked

void MainWindow::on_pushButton_2_clicked()
{
    gc.readfactors();
}

void MainWindow::on_pushButton_3_clicked()
{
    image.clear();
    QString filename = QFileDialog::getOpenFileName(0,"Fájl megnyitása", "C:\\"); //Debug.
    if(filename.endsWith(QChar('f')))
    {
        image.readfromfloatfile(filename.toStdString());
    }
    else
    image.readfromfile(filename.toStdString());
    image.drawimage(ui->label);
    image.writedetailstoscreen(ui->textEdit);


    std::cout <<"corrigating... " <<std::endl;
    gc.corrigateimage(image);
    std::cout <<"calculating... " <<std::endl;

    image.calculate_meanvalue_on_GPU();
    image.writedetailstoscreen(ui->textEdit);


    image.writetofloatfile(image.getfilename());
    std::cout <<"kesz." << std::endl;
}

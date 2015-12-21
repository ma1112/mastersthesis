#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>

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
    image.readfromfile(filename);
    image.drawimage(ui->label);
    image.writedetailstoscreen(ui->textEdit);

}

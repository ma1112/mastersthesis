#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QFileDialog>
#include <QString>
#include <QStringList>
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
    image.readfromfile(filename.toStdString());
    image.drawimage(ui->label);
    image.writedetailstoscreen(ui->textEdit);
    QTime timer;
    timer.start();
    image.copy_to_GPU();
    std::cout<<"Copy to GPU: " << timer.restart()  <<std::endl;
    image.calculate_meanvalue_on_GPU();
    std::cout<<"Mean on GPU " << timer.restart() <<"value = " << image.getmean()  <<std::endl;

    image.calculate_meanvalue_on_CPU();
    std::cout<<"Mean on CPU: " << timer.restart()  <<"value = " << image.getmean()<<std::endl;
}

void MainWindow::on_pushButton_clicked()
{
    Gaincorr gc;
    std::cout << "Launching gc.read()" <<std::endl;
    gc.read();
    std::cout <<"Exited from read()" <<std::endl;
    gc.calculate();

    /*
    double meanVoltage = 0;
    double meanAmperage =0;
    double meanExptime=0;
    double meanintensity = 0;
    QString dir = QFileDialog::getExistingDirectory(this, tr("Mappa kiválasztása---"),
                                                    "C:\\",
                                                     QFileDialog::DontResolveSymlinks);
    QStringList nameFilter("*.bin");
    QDir directory(dir);
    QStringList filelist = directory.entryList(nameFilter);
    images.reserve(filelist.size());
    for(int i=0;i<filelist.length();i++)
    {
        image.readfromfile(directory.absoluteFilePath(filelist.at(i)));
        images.push_back(image);
       image.writedetailstoscreen(ui->textEdit);
        meanVoltage += (images.back().getvoltage()) / filelist.length();
        meanAmperage += (images.back().getamperage()) / filelist.length();
        meanExptime += (images.back().getexptime()) / filelist.length();
        meanintensity += (images.back().getmean()) / filelist.length();
    }
    ui->textEdit->append("Mean voltage =");
    ui->textEdit->append(QString::number(meanVoltage));

    ui->textEdit->append("meanAmperage =");
    ui->textEdit->append(QString::number(meanAmperage));

    ui->textEdit->append("meanExptime =");
    ui->textEdit->append(QString::number(meanExptime));

    ui->textEdit->append("meanintensity =");
    ui->textEdit->append(QString::number(meanintensity));


    std::vector<Image>::iterator iter;
    ui->textEdit->append("Valami nem stimmelt a kovetkezo kepekkel:");


image.clear();
    for(iter = images.begin(); iter != images.end();)
    {
        if     (
              abs(iter->getmean() - meanintensity) > meanintensity * 0.1f ||
             abs(iter->getvoltage() - meanVoltage) > meanVoltage * 0.1f ||
             abs(iter->getamperage() - meanAmperage) > meanAmperage * 0.1f ||
             abs(iter->getexptime() - meanExptime) > meanExptime * 0.1f
                )
        {
           ui->textEdit->append(QString::fromStdString(iter->getid()));
           iter =  images.erase(iter);

        }
        else
            {
             image+=*iter;
            ++iter;
            }
    }
    image.drawimage(ui->label);
 */

}
// end of pushbutton_clicked

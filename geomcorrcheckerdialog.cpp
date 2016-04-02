#include "geomcorrcheckerdialog.h"
#include "ui_geomcorrcheckerdialog.h"
#include <iostream>
#include <QFileDialog>
geomCorrCheckerDialog::geomCorrCheckerDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::geomCorrCheckerDialog)
{
    ui->setupUi(this);


    gaincorr.readgainfactors();
    gaincorr.readoffsetfactors();



    activeLabel = NULL;
    ui->slider->setMinimum(0);
    ui->slider->setMaximum(0);
    firstIndex = 50000;
    lastIndex = 0;
    n=2;
    u=0;

    ui->rightButton->setDisabled(true);
    ui->rightButton->setVisible(true);
    ui->leftButton->setDisabled(true);
    ui->rightButton->setVisible(true);
    ui->rightButton->setDisabled(true);
    ui->resetButton->setEnabled(true);
    ui->resetButton->setVisible(true);
    ui->goodButton->setVisible(true);
    ui->goodButton->setDisabled(true);
    connect(ui->resetButton, SIGNAL (clicked()), this, SLOT(reset()));
    connect(ui->inputDirButton, SIGNAL (clicked()), this, SLOT(setDir()));
    connect(ui->leftButton, SIGNAL (clicked()), this, SLOT(chooseRotatedImage()));
    connect(ui->goodButton, SIGNAL (clicked()), this, SLOT(calculate()));






}

geomCorrCheckerDialog::~geomCorrCheckerDialog()
{
    delete ui;
}


void geomCorrCheckerDialog::getFileList()
{
    QStringList nameFilter("*.bin"); //name filter.
    fileList = dir.entryList(nameFilter); //File list of .bin files
    if( fileList.length()<=1)
    {
        ui->inputDirLabel->setText("Invalid Directory");
        reset();
        return;
    }

    ui->slider->setMinimum(0);
    ui->slider->setMaximum(fileList.length()-1);
    ui->slider->setValue(0);
    activeLabel = ui->leftLabel;
    if( dir.exists() && fileList.length() > 0)
    {
        displayImage(0);
        setCounterLabel(0);
        connect(ui->slider, SIGNAL (sliderMoved(int)), this, SLOT(displayImage(int)));
        connect(ui->slider, SIGNAL (sliderMoved(int)), this, SLOT(setCounterLabel(int)));

        ui->leftButton->setEnabled(true);
    }
}

void geomCorrCheckerDialog::setDir ()
{
    reset();

    QString directory = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which contains folders of images to calculate geometry correction data"),
                                                           "C:\\",
                                                            QFileDialog::DontResolveSymlinks);
    dir = QDir(directory);
    if(dir.exists())
    {
        ui->inputDirLabel->setText(directory);
        getFileList();
    }
    else
    {
        ui->inputDirLabel->setText("Invalid.");
    }
}



void geomCorrCheckerDialog::displayImage(int i)
{
    if(fileList.length()==0)
    {
        std::cout << "ERROR! File list is not loaded." << std::endl;
        return;
    }

    Image image;
    image.readfromfile(dir.absoluteFilePath(fileList.at(i)).toStdString());
    gaincorr.offsetcorrigateimage(image);
    gaincorr.gaincorrigateimage(image);
    image.drawimage(activeLabel);
}

void geomCorrCheckerDialog::chooseRotatedImage()
{
    firstIndex = ui->slider->value();
    activeLabel = ui->rightLabel;
    ui->topLabel->setText("Choose an image with a rotation of ca. 360 degrees.");
    ui->leftButton->setDisabled(true);
    ui->leftButton->setText(QString::number(firstIndex));
    ui->rightButton->setVisible(true);
    ui->rightButton->setEnabled(true);
    connect(ui->rightButton, SIGNAL (clicked()), this, SLOT(validateResult()));

}

void geomCorrCheckerDialog::setCounterLabel(int i)
{
    ui->counterLabel->setText(QString::number(i) + "/" + QString::number(fileList.length()));
}


void geomCorrCheckerDialog::validateResult()
{
    connect(ui->spinBox, SIGNAL(valueChanged(int)) , this, SLOT(validateResult(int)));
    ui->topLabel->setText("Please Wait...");
    ui->leftButton->setDisabled(true);
    ui->rightButton->setDisabled(true);
    ui->slider->setDisabled(true);
    n=ui->spinBox->value();
    u=ui->spinBox_2->value();
    qApp->processEvents(); // refresh GUI


    Image firstImage;
    firstImage.readfromfile(dir.absoluteFilePath(fileList.at(firstIndex)).toStdString());
    gaincorr.offsetcorrigateimage(firstImage);
    gaincorr.gaincorrigateimage(firstImage);




    Image current;




        lastIndex = ui->slider->value();

        int correlationRadius = round((lastIndex  - firstIndex) * 0.125f);
        float maxCorrelation = -1;
        int maxCorrelationIndex = firstIndex;


        for(int j =  lastIndex - correlationRadius; j <= std::min(lastIndex + correlationRadius, fileList.length()-1) ; j++)
        {
            current.readfromfile(dir.absoluteFilePath(fileList.at(j)).toStdString());
            gaincorr.offsetcorrigateimage(current);
            gaincorr.gaincorrigateimage(current);
            float result = firstImage.correlateWith(current);
            if(result >maxCorrelation )
            {
                maxCorrelation = result;
                maxCorrelationIndex = j;
            }

        }
        lastIndex = maxCorrelationIndex;

        geomcorr.initializeDeviceVector(n, lastIndex - firstIndex, u);

    geomcorr.extractCoordinates(firstImage,true, true);
    firstImage.calculate_meanvalue_on_GPU();
    firstImage.drawimage(ui->leftLabel);
    current.readfromfile(dir.absoluteFilePath(fileList.at(lastIndex)).toStdString());
    gaincorr.offsetcorrigateimage(current);
    gaincorr.gaincorrigateimage(current);
    geomcorr.extractCoordinates(current,true,true);
    current.calculate_meanvalue_on_GPU();
    current.drawimage(ui->rightLabel);
    ui->rightButton->setText(QString::number(lastIndex));
    ui->slider->setValue(lastIndex);
    setCounterLabel(lastIndex);
    activeLabel = NULL;
    ui->goodButton->setEnabled(true);
    ui->topLabel->setText("Adjust number of circles, then click on 'Looks Good' to continue, 'Reset' to start over again.");
}


void geomCorrCheckerDialog::validateResult(int i)
{
    ui->topLabel->setText("Please Wait.");
    ui->spinBox->setDisabled(true);
    qApp->processEvents(); // refresh GUI




    n=i;
    u=ui->spinBox_2->value();
    Image firstImage;
    firstImage.readfromfile(dir.absoluteFilePath(fileList.at(firstIndex)).toStdString());
    gaincorr.offsetcorrigateimage(firstImage);
    gaincorr.gaincorrigateimage(firstImage);
    geomcorr.initializeDeviceVector(n, lastIndex - firstIndex,u);
    Image current;


    geomcorr.extractCoordinates(firstImage,true,true);
    firstImage.calculate_meanvalue_on_GPU();
    firstImage.drawimage(ui->leftLabel);
    current.readfromfile(dir.absoluteFilePath(fileList.at(lastIndex)).toStdString());
    gaincorr.offsetcorrigateimage(current);
    gaincorr.gaincorrigateimage(current);
    geomcorr.extractCoordinates(current,true,true);
    current.calculate_meanvalue_on_GPU();
    current.drawimage(ui->rightLabel);
    ui->topLabel->setText("Adjust number of circles, then click on 'Looks Good' to continue, 'Reset' to start over again.");
    ui->spinBox->setEnabled(true);
}

void geomCorrCheckerDialog::reset()
{
    ui->spinBox->setEnabled(true);
    disconnect(ui->spinBox, SIGNAL (valueChanged(int)) , this, SLOT(validateResult(int)));


    ui->slider->setMinimum(0);
    ui->slider->setMaximum(0);
    firstIndex = 50000;
    lastIndex = 0;

    ui->rightButton->setDisabled(true);
    ui->rightButton->setVisible(true);
    ui->leftButton->setDisabled(true);
    ui->rightButton->setVisible(true);
    ui->rightButton->setDisabled(true);
    ui->resetButton->setEnabled(true);
    ui->resetButton->setVisible(true);
    ui->goodButton->setVisible(true);
    ui->goodButton->setDisabled(true);
    activeLabel = ui->leftLabel;
    ui->rightLabel->clear();
    ui->leftLabel->clear();

    if( dir.exists() && fileList.length() > 1)
    {
        displayImage(0);
        setCounterLabel(0);
        connect(ui->slider, SIGNAL (sliderMoved(int)), this, SLOT(displayImage(int)));
        connect(ui->slider, SIGNAL (sliderMoved(int)), this, SLOT(setCounterLabel(int)));

        ui->leftButton->setEnabled(true);
        ui->slider->setEnabled(true);
        ui->slider->setMinimum(0);
        ui->slider->setMaximum(fileList.length()-1);
        ui->leftButton->setText("Choose");
        ui->rightButton->setText("Choose");
    }


}

void geomCorrCheckerDialog::calculate()
{
    ui->goodButton->setDisabled(true);
    ui->resetButton->setDisabled(true);
    ui->spinBox->setDisabled(true);
    ui->spinBox_2->setDisabled(true);
    qApp->processEvents(); // refresh GUI

    if( !(dir.exists()) || fileList.length() <2 || firstIndex == 50000 || firstIndex >= fileList.length()
            || firstIndex < 0 || lastIndex <=0 || lastIndex >= fileList.length() || firstIndex >= lastIndex)
    {
        reset();
        return;
    }

    Image image;
    for(int i = firstIndex; i<lastIndex; i++)
    {
        //Process each images.
       std::cout << "Processing image " << i<<" withing range of "<< firstIndex << " - " << lastIndex<<". (" << (100.0f*(i - firstIndex)) / (float)(lastIndex - firstIndex)<<"%.)" << std::endl;
        image.readfromfile(dir.absoluteFilePath(fileList.at(i)).toStdString());
        gaincorr.offsetcorrigateimage(image);
        gaincorr.gaincorrigateimage(image);
        if( !(image.getamperage() > 0 && image.getexptime() >0 && image.getvoltage() > 0))
        {
            std::cout << "WARNING: Image at " << image.getfilename() << ", with id " << image.getid()
                      << "Is not valid for geom corectoin due to it is empty." << std::endl;
            continue;
        }
        image.calculate_meanvalue_on_GPU();
        if(image.getmax() < 2* image.getmin())
        {
            std::cout << "WARNING: Image at " << image.getfilename() << ", with id " << image.getid()
                      << "Is not valid for geom corectoin due to it lacks contrast." << std::endl;
            continue;
        }





        if(i==firstIndex) // at the first step, extract only N number of circles.
            geomcorr.extractCoordinates(image,false,true);
        else
            geomcorr.extractCoordinates(image);
        geomcorr.addCoordinates();

        //Cinema:
        image.drawimage(ui->leftLabel);
        qApp->processEvents(); // refresh GUI



    }

    std::cout << "Fitting ellipse." << std::endl;

    for(int i=0; i<n; i++)
    {


        float a,b,c,u,v,error;
        a=b=c=u=v=error=0.0f;
        geomcorr.fitEllipse(i, &a,&b, &c, &u, &v, &error);

    }
    geomcorr.exportText("C:/awing/ellipsesmarc15.txt");
    std::cout << "fit success." << std::endl;




}


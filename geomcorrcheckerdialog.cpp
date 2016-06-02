#include "geomcorrcheckerdialog.h"
#include "ui_geomcorrcheckerdialog.h"
#include <iostream>
#include <QFileDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <math.h>       /* round, floor, ceil, trunc */
#include "coordinatedialog.h"

geomCorrCheckerDialog::geomCorrCheckerDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::geomCorrCheckerDialog)
{
    ui->setupUi(this);


    D_estimated = 1.0;
    gaincorr.readgainfactors();
    gaincorr.readoffsetfactors();



    activeLabel = NULL;
    ui->progressBar->setVisible(false);
    ui->progressBar->setRange(0,100);


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


    ui->progressBar->setVisible(true);
    ui->progressBar->setRange(lastIndex - correlationRadius , std::min(lastIndex + correlationRadius, fileList.length()-1));

    for(int j =  lastIndex - correlationRadius; j <= std::min(lastIndex + correlationRadius, fileList.length()-1) ; j++)
    {
        ui->progressBar->setValue(j);
        qApp->processEvents(); // refresh GUI



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
    ui->progressBar->setValue(0);
    ui->progressBar->setRange(0,100);
    ui->progressBar->setVisible(false);
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
    if(maxCorrelation < 0.9)
    {
        ui->topLabel->setText("WARNING: Coorelation is too low: " + QString::number(maxCorrelation) +", rotated index is possibly false.");
    }
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

    ui->progressBar->setVisible(false);
    ui->progressBar->setValue(0);
    ui->spinBox_D->setDisabled(false);


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

/********************************************************************
 * *****************************************************************
 *                        C A L C U L A T E
 * *************************************************************
 */

void geomCorrCheckerDialog::calculate()
{




    ui->goodButton->setDisabled(true);
    ui->resetButton->setDisabled(true);
    ui->spinBox->setDisabled(true);
    ui->spinBox_2->setDisabled(true);
    D_estimated = (long double ) ui->spinBox_D->value() / 0.0748;
    ui->spinBox_D->setDisabled(true);
    qApp->processEvents(); // refresh GUI

    if( !(dir.exists()) || fileList.length() <2 || firstIndex == 50000 || firstIndex >= fileList.length()
            || firstIndex < 0 || lastIndex <=0 || lastIndex >= fileList.length() || firstIndex >= lastIndex)
    {
        reset();
        return;
    }
    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(true);


    //Asking for distances.
        int *x_balls = new int[n]();
        int *y_balls = new int[n]();

    int distanceReadFromFiles = 0;

    //Reding distances from file if it exists.
    QFile distanceFile(dir.absoluteFilePath("distance.txt"));
    if(distanceFile.open(QIODevice::ReadOnly))
    {

        foreach (QString i,QString(distanceFile.readAll()).split(QRegExp("[\r\n]"),QString::SkipEmptyParts)){
           if( distanceReadFromFiles >= n )
           {
               std::cout << "Too much lines in distance. txt. Considerint only the first " << n << "." << std::endl;
               continue;
           }

            x_balls[distanceReadFromFiles] = i.section(",",0,0).toInt();
            y_balls[distanceReadFromFiles] = i.section(",",1,1).toInt();
            distanceReadFromFiles++;
            std::cout << "Ball " << distanceReadFromFiles << " coorcinates from file: x: "<< x_balls[distanceReadFromFiles-1] <<" and y = " << y_balls[distanceReadFromFiles-1] << std::endl;
        }
        distanceFile.close();
    }


    if(distanceReadFromFiles < n )
    {
        std::cout << "Distances could not be loaded from distance text file. Requesting it manually." << std::endl;



    for(int i=0; i< n; i++)
    {
        CoordinateDialog coordinateDialog;
        coordinateDialog.setXDestination(x_balls + i);
        coordinateDialog.setYDestination(y_balls + i);
        coordinateDialog.setLabel("Input of the coordinates of the " + QString::number(i +1) + "-th ball on the plastic lattice:");
        if(i > 0)
        {
            coordinateDialog.setX(x_balls[i-1]);
            coordinateDialog.setX(y_balls[i-1]);

        }
        coordinateDialog.exec();
        std::cout << "Coordinate for ball " << i << ": x:" << x_balls[i] << ", y:" << y_balls[i] << std::endl;
    }
    }


//TODO: Save distance to file if it is aquired form the user.

    std::cout << std::endl;
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
        {
            geomcorr.extractCoordinates(image,false,true);
            geomcorr.addCoordinates(image);
        }
        else
        {
            geomcorr.extractCoordinates(image);
            geomcorr.addCoordinates(image);

        }

        ui->progressBar->setValue((100.0f*(i - firstIndex)) / (float)(lastIndex - firstIndex));
        qApp->processEvents(); // refresh GUI
    }

    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(false);



    std::cout << "Calcuating eta "<< std::endl;
    geomcorr.calculateEta();
    std::cout << "Fitting ellipse." << std::endl;


    float* a = new float[n]();
    float* b = new float[n]();
    float* c = new float[n]();
    float* u = new float[n]();
    float* v = new float[n]();
    double* error = new double[n* 5]();


    for(int i=0; i<n; i++)
    {
        geomcorr.fitEllipse(i, a+ i,b +i, c +i, u +i, v +i , error +5*i);
    }
    //DEBUG
    std::cout << "fit success. " << std::endl;
    if(     geomcorr.exportText("C:/awing/ellipses.txt"))
        std::cout << "data saved to file : c:/awing/ellipses.txt" << std::endl;


    //Checking which ellipses are valid... ( paramter is not infinity)
    std::cout << "validEllipses:";
    bool *validEllipse = new bool[n];
    for(int i=0; i<n; i++)
    {
        if(a[i] != a[i] || b[i] != b[i] || c[i] != c[i] || u[i] != u[i] || v[i] != v[i] ||
                b[i] < 0 || a[i] < 0)
            validEllipse[i] = false;
        else
            validEllipse[i] = true;
        if(validEllipse[i])
            std::cout << i <<":T  ";
        else
            std::cout << i <<":F " ;
    }
    std::cout << std::endl;


//Checking order of balls and determining z0 plane.

    int* orderOfBalls = new int[n](); // balls are not in up to down order. Determining sorted list of indexes.
    float minV = 50000;
    int minIndex = 0;
    for(int i=0; i<n; i++)
    {
        if(v[i] < minV)
        {
            minIndex = i;
            minV = v[i];
        }

    }
    orderOfBalls[0] = minIndex;
    float minVLastRound = minV;

    for(int i=1; i<n;i++)
    {
        float minVthisRound = 5000;
        int minIndex = 0;

        for(int j=0;j<n;j++)
        {
            if( v[j] < minVthisRound && v[j] > minVLastRound + 1 )
            {
                minIndex = j;
                minVthisRound = v[j];
            }
        }
        orderOfBalls[i] = minIndex;
        minVLastRound = minVthisRound;
    }

    std::cout << "Order of balls:" << std::endl;
    for(int i=0; i<n ; i++)
    {
        std::cout << orderOfBalls[i] << " " ;
    }
    std::cout << std::endl;



    //Calculation of scanner parameters


    /* TBOTONDS METHOD - INGORED....

    //for every pair of ellipses:
    int pairs = 0;
    //Representing TBOTOND's (21) equation as
    // y = v0 + d^2 * x
    //than you have to only calculate the slope and intercept of this line.

    double xmean = 0;
    double ymean = 0;
    double xymean = 0;
    double x2mean = 0;



   for(int i=0; i<n; i++) // parameter 1 as k
   {
       for(int j=0; j<n; j++) // parameter 2 as k'
       {
           if(j <=i)
           {
               continue;
           }
           pairs++;
           double y  = 0.5 * (v[i] + v[j]) - 0.5 / (v[i] - v[j])  * (1/b[i] - 1/b[j]);
           double x = 0.5 / (v[i] - v[j] ) * (a[i] / b[i] - a[j] / b[j]);
           xmean +=x;
           ymean +=y;
           xymean += x*y;
           x2mean +=x*x;
       }
   }

   xmean /=pairs;
   ymean /= pairs;
   xymean /= pairs;
   x2mean /= pairs;

   double D2 = (xymean - xmean * ymean) / (x2mean - xmean * xmean);
   double D = sqrt(D2);
   double v0star = ymean - D2 * xmean;

   std::cout << "D= " << sqrt(D2) <<std::endl;
   std::cout << "v0*" << v0star << std::endl;





   double umean = 0;
   double* phases = new double[n]();
   for(int i=0; i<n ; i++)
   {
       umean += u[i];
   }
   umean/=n;

   double* rho = new double[n];
   double* xsi = new double[n];

   for(int i=0; i<n; i++)
   {
       phases[i] =geomcorr.calculatePhase(i,umean);
       rho[i] = 1 / sqrt(b[i]  *  ( v[i] - v0star) * ( v[i] - v0star)  );
       xsi[i] = ( v[i] - v0star) * (1- rho[i]*rho[i]) / D;
   }

   int* realDistancex = new int[n](); //millimeters
   int* realDistancey = new int[n]();

   int* orderOfBalls = new int[n](); // balls are not in up to down order. Determining sorted list of indexes.
   float minV = 50000;
   int minIndex = 0;
   for(int i=0; i<n; i++)
   {
       if(v[i] < minV)
       {
           minIndex = i;
           minV = v[i];
       }

   }
   orderOfBalls[0] = minIndex;
   float minVLastRound = minV;

   for(int i=1; i<n;i++)
   {
       float minVthisRound = 5000;
       int minIndex = 0;

       for(int j=0;j<n;j++)
       {
           if( v[j] < minVthisRound && v[j] > minVLastRound + 1 )
           {
               minIndex = j;
               minVthisRound = v[j];
           }
       }
       orderOfBalls[i] = minIndex;
       minVLastRound = minVthisRound;
   }

   std::cout << "Order of balls:" << std::endl;
   for(int i=0; i<n ; i++)
   {
       std::cout << orderOfBalls[i] << " " ;
   }
   std::cout << std::endl;



   for(int i=1; i<n; i++)
   {
       QDialog dialog;
       QFormLayout form(&dialog);
       form.addRow(new QLabel("Distance of ball " + QString::number(i) + " from ball 0 in the x and y direction in millimeters" ));
       QSpinBox *xSpinBox = new QSpinBox(&dialog);
       form.addRow(xSpinBox);
       QSpinBox *ySpinBox = new QSpinBox(&dialog);
       form.addRow(ySpinBox);
       xSpinBox->setMinimum(0);
       ySpinBox->setMinimum(0);
       xSpinBox->setValue(realDistancex[orderOfBalls[i-1]]);
       ySpinBox->setValue(realDistancey[orderOfBalls[i-1]]);
       ySpinBox->setMinimum(ySpinBox->value());



       QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                  Qt::Horizontal, &dialog);
       form.addRow(&buttonBox);
       QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
       QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));
       do
       {
           std::cout << "Asking for ball " << i << std::endl;

       } while(dialog.exec() != QDialog::Accepted);

            realDistancex[ orderOfBalls[i]] =  xSpinBox->value();
            std::cout << "ball " <<i << " : " <<  xSpinBox->value() << "  " << ySpinBox->value() << std::endl;
            realDistancey[orderOfBalls[i]] = ySpinBox->value();
   }


// for every pair:
   double R = 0.0;
   pairs = 0;

   for(int i=0; i<n;i++)
   {
       for( int j=0; j<n; j++)
       {
           if( j<=i)
           {
               continue;
           }
           pairs++;
           double realDistance  =sqrt ( pow( realDistancex[i] - realDistancex[j] ,2) +
                                    pow(realDistancey[i] - realDistancey[j],2) );

           R+=realDistance / sqrt( pow( xsi[i] - xsi[j],2) + rho[i] * rho[i] + rho[j] * rho[j] - 2.0 * rho[i] * rho[j] * cos (phases[i] - phases[j]) );


       }
   }
   R /= pairs;
std::cout << "R : " << R << std::endl;

*/


    //calculate Z=0 plane
    float minb = 50000;
    int minbIndex =-1;
    for(int i=0; i<n; i++)
    {
        if( ! validEllipse[i]) // an ellipse tend to be invalid when b = 0
        {
            if(0 < minb)
            {
                minb=0;
                minbIndex = i;
                continue;
            }
        }
        float thisb = sqrt ( 1 / b[i]);
        //debug:
        std::cout << "B in pixels at exllipse " << i << " is " << thisb <<std::endl;
        if(thisb < minb)
        {
            minb = thisb;
            minbIndex = i;
        }
    }

    int minbOrder = 0;
    for(int i=0; i<n; i++)
    {
        if(orderOfBalls[i] == minbIndex)
        {
            minbOrder = i;
            break;
        }
    }
//DEBUG
    std::cout << "Detected minb is " << minb << " for ellipse "<< minbIndex << " that is the " <<  minbOrder<< "-th ellipse in order from up to down. ( 0 to n-1)" << std::endl;



    if(minbOrder==0 || minbOrder== n-1 )
    {
        std::cout << "Warning! All circles are on one side of the z=0 plane. ( minbIndex = ) " << minbIndex
                  << std::endl ;

    }

    float z0 = 0;

    if(minbOrder == 0)
    {
        z0 = -1;
    }
    else if ( minbOrder == n-1)
    {
        z0 = 2000;
    }
    else
    {



    //looking for b of the ellipse that is above that:
    int indexabove = orderOfBalls[minbOrder-1];
    float bAbove;

    bAbove = 1  / sqrt(b[indexabove]);
    if (bAbove != bAbove )
    {
        bAbove =0;
    }

    //under that:
    int indexbelov =  orderOfBalls[minbOrder+1];
    float bBelove;

    bBelove =  1 / sqrt(b[indexbelov]);
    if( bBelove != bBelove)
    {
        bBelove =0;
    }

    std::cout << "minbIndex : " << minbIndex<< std::endl;
    std::cout << "indexbelov: " << indexbelov << std::endl;
    std::cout << "indexabove: " << indexabove << std::endl;
    std::cout << "bBelove : " << bBelove << std::endl;
    std::cout << " bAbove : " << bAbove << std::endl;

    if( bBelove > bAbove )
    {
        z0 = float (v[minbIndex] + v [indexabove ]) * 0.5f;
    }
    else
    {
        z0 = float (v[minbIndex] + v [indexbelov ]) * 0.5f;
    }

    }


    std::cout << "z0 = " << z0 <<std::endl;


    //calculate D


    int pairs = 0;
    long double Dmean = 0.0;
    long double Dnorm = 0.0;
    long double dDmean = 0.0;
    for(int i=0; i<n; i++)
    {
        if(! validEllipse[i])
        {
            continue;
        }

        for(int j=0; j<n; j++)
        {
            if(! validEllipse[j])
            {
                continue;
            }

            if( j <= i)
            {
                continue;
            }

            int z1 = v[i]<z0?-1:+1;
            int z2 = v[j]<z0?-1:+1;
            /*
            if(z1 * z2 > 0)
            {
                continue;
            }
            */

            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            long double D2  = 0.0;
            long double m1 = sqrt(b[j] - c[j] * c[j] / a[j]) / sqrt(b[i] - c[i] * c[i] / a[i]);
            long double m0 = (v[j] - v[i] ) * sqrt(b[j]  - c[j] * c[j]  / a[j]);
            long double n0 =(1.0-m0 * m0 - m1 * m1)  / ( 2.0 * m0 * m1);
            long double n1 = (a[j] - a[i] * m1 * m1)  / (2.0 * m0 * m1);
            long double a1 =a[i];
            long double a2 = a[j];
            long double b1 = b[i];
            long double b2 = b[j];
            long double c1 = c[i];
            long double c2 = c[j];

            //du =
            long double du1 = error[i*5];
            long double du2 = error[j*5];
            long double dv1 = error[i*5 + 1];
            long double dv2 = error[j*5 +1];
            long double da1 = error[i*5 + 2];
            long double da2 = error[j*5 + 2];
            long double db1 = error[i*5 + 3];
            long double db2 = error[j*5 + 3];
            long double dc1 = error[i*5 + 4];
            long double dc2 = error[j*5 + 4];

            //double dm0 = sqrt( pow(dv1 * () ,2) +    );

            std::cout << "v[i]= " <<v[i] << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "v[j]= " <<v[j] << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "m1= " << m1 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "m0= " << m0 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "n0= " << n0 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "n1= " << n1 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "a[i]= " << a[i] << " and i= " << i << " j=" << j <<std::endl;

            double Dminus =0.0;
            double Dplus = 0.0;
            int eps = -1;

            // alternative version:

            D2 = ((a[i] - 2.0 * n0 * n1) -eps *  sqrt(a[i] * a[i] + 4.0 * n1 * n1 - 4.0 * n0 * n1 * a[i])) / ( 2.0 * n1 * n1);
            Dminus = sqrt(D2);
            std::cout << "Alternative D= " << Dminus << " and i= " << i << " j=" << j <<std::endl;


            //opposite version:

            D2 = ((a[i] - 2.0 * n0 * n1) + eps *  sqrt(a[i] * a[i] + 4.0 * n1 * n1 - 4.0 * n0 * n1 * a[i])) / ( 2.0 * n1 * n1);
            Dplus = sqrt(D2);
            std::cout << "D= " << Dplus << " and i= " << i << " j=" << j <<std::endl;

            double D;
            if( abs ( Dplus - D_estimated) < abs(Dminus - D_estimated))
            {
                D= Dplus;
                eps = +1;
                std::cout << "Choosing " << Dplus << " as D. ( Eps = 1)" << std::endl;
            }
            else
            {
                D=Dminus;
                eps = -1;
           std::cout << "Choosing " << Dminus << " as D. ( eps = -1)"<< std::endl;

            }

            //Error of D:

           long double dm1 = sqrt(
                        pow(da1 * m1 /2.0 / ( b1 - c1 * c1 / a1  ) * (c1 * c1 / a1 / a1),2)
                        + pow(db1 * m1 / 2.0 / (b1 - c1 * c1 / a1) ,2)
                        + pow(dc1 * m1 / 2.0 / (b1 - c1 * c1 / a1) * 2.0 * c1 / a1 ,2)
                        + pow( da2 * m1 / 2.0 / (b2 - c2 * c2 / a2) * c2 * c2 / a2 / a2,2)
                        + pow ( db2 * m1 / 2.0 / (b2 - c2 * c2 / a2) ,2 )
                        + pow(dc2 * m1 / 2.0 / (b2 - c2 * c2 / a2) * 2.0 * c2 / a2 ,2)
                        );

           std::cout <<std::endl << "Error of m1 = " << dm1 << "that is " << dm1 / m1 * 100.0 << " percent" <<std::endl << std::endl;

           long double dm0 = sqrt(
                        pow(dv2 * sqrt(b2 - c2 * c2  / a2),2)
                        + pow(dv1 *sqrt(b2 - c2 * c2  / a2),2 )
                        + pow( da2 * m0 / 2.0 /(b2 - c2 * c2  / a2) * c2 * c2 / a2 / a2,2 )
                        + pow( db2  * m0 / 2.0/ (b2 - c2 * c2  / a2),2 )
                        + pow ( dc2 * m0 / 2.0/ (b2 - c2 * c2  / a2) * 2 * c2 / a2,2)
                        );

           std::cout <<std::endl << "Error of m0 = " << dm0 << "that is " << dm0 / m0 * 100.0 << " percent" <<std::endl << std::endl;


           long double dn0  = sqrt(
                        pow((dm0 *( -2.0 * m0 * 2.0 * m0 * m1 - (1.0 - m0 * m0 - m1 * m1) * 2.0 * m1) / pow(2.0 * m0* m1,2)),2)
                        + pow ((dm1 *( -2.0 * m1 * 2 * m1 * m0 - (1.0 - m1 * m1 - m0 * m0) * 2.0 * m0) / pow(2.0 * m1* m0,2)),2  )
                        );

           std::cout <<std::endl << "Error of n0 = " << dn0 << "that is " << dn0 / n0 * 100.0 << " percent" <<std::endl << std::endl;



           long double dn1 = sqrt(
                        pow(da2 / (2.0 * m0 * m1),2)
                        + pow(da1 * m1  /(2.0 * m0 ),2)
                        + pow(dm0 * n1 / m0,2 )
                        + pow(dm1 * (-2.0 * a1 * m1 * 2.0 * m0 * m1 - (a2 - a1 * m1 * m1) * 2 * m0) / pow(2.0*m0*m1,2),2)
                        );

           std::cout <<std::endl << "Error of n1 = " << dn1 << "that is " << dn1 / n1 * 100.0 << " percent" <<std::endl << std::endl;



           long double dD2 = sqrt(
                        pow(da1 /  2.0 / n1 / n1 *(1.0 - eps * 0.5 / sqrt(a1*a1+4.0*n1*n1-4.0*n0*n1*a1) *( 2.0 * a1  - 4.0 * n0*n1)  ),2 )
                        + pow(dn0 / 2.0 / n1 / n1 * (-2.0 * n1 - eps * 0.5 / sqrt(a1*a1+4.0*n1*n1-4.0*n0*n1*a1) * 4*n1*a1 ),2)
                       + pow(dn1 * (((-2.0 * n0 - eps * 0.5 / sqrt (a1 * a1 + 4.0 * n1 * n1 - 4.0 * n0 * n1 * a1)* (8.0 * n1 - 4.0 * n0 * a1))* 2.0 * n1 * n1) - D2 * 8.0 * n1 * n1 * n1) / (4 * pow(n1,4)),2)
                        );

           std::cout <<std::endl << "Error of D2 = " << dD2 << "that is " << dD2 / D2 * 100.0 << " percent" <<std::endl << std::endl;


           long double dD = 0.5 * dD2 / D;

           std::cout <<std::endl << "Error of D = " << dD << "that is " << dD / D * 100.0 << " percent" <<std::endl << std::endl;





            double v0star = v[i] - z1 * sqrt(a[i] + a[i] * a[i] * D2) / sqrt(a[i] * b[i] - c[i] * c[i]);
            double u0star = 0.5* u[i] + 0.5 * u[j] + 0.5 * c[i] / a[i] * (v[i] - v0star ) + 0.5 * c[j] / a[j]  * ( v[j] - v0star);

            std::cout << "v0star and un0star with not avaraged D: " <<std::endl;
            std::cout << "v0star: " << v0star << std::endl;
            std::cout << "u0star : " << u0star << std::endl;

            if(!(D!= D) && ! (dD != dD)) // if D is a number
            {
                Dmean += D / pow(dD,2);
                Dnorm += 1.0/ pow(dD,2);
                dDmean += pow(1.0 / dD,2);

                pairs++;
            }
        }

    }

    if(pairs ==0)
    {

        delete[] a;
        delete[] b;
        delete[] c;
        delete[]u;
        delete[] v;
        delete[] error;
        delete[] validEllipse;
        //delete[] phases;
        //delete[] rho;
        // delete[] xsi;
        delete[] x_balls;
        delete[] y_balls;
        std::cout << "ERROR: no valid pairs found while calculating D." << std::endl;
        std::cout << " geometry collection is not avaiable." << std::endl;
        return;

    }


    long double D = Dmean / Dnorm;
    dDmean = sqrt(dDmean);
    dDmean /= Dnorm;
    long double D2 = D*D;


    std::cout << "D = " << D <<" pixels  with  error of " << dDmean << ". That is  "<< dDmean / D * 100.0 << " percent.  Estimated from " << pairs << " pairs of ellipses."<<std::endl;
    std::cout << "In mm: D= " << D *0.0748 << " mm." << std::endl;

    long double* xsi = new long double[n]();
    long double* dxsi = new long double[n]();
    //double rho = Rho is not used ever...

    long double v0starMean = 0.0;
    long double v0starNorm = 0.0;
    long double dv0starMean = 0.0;

    pairs =0;
    //calculating other parameters.
    for(int i=0; i<n; i++)
    {
        if(! validEllipse[i])
        {
            continue;
        }
        long double du1 = error[i*5];
        long double dv1 = error[i*5 + 1];
        long double da1 = error[i*5 + 2];
        long double db1 = error[i*5 + 3];
        long double dc1 = error[i*5 + 4];


        int z1 = v[i]<z0?-1:+1;
        xsi[i] = D * z1* a[i] * sqrt(a[i]) / sqrt(a[i]*b[i] + a[i] * a[i] * b[i] * D2 -c[i] * c[i] );
        dxsi[i]= sqrt(
                    pow(dDmean * (pow (a[i],3) * D / xsi[i] - xsi[i] * a[i] * a[i] * b[i] * D) / (a[i] * b[i] + a[i] * a[i] * b[i] * D2 -c[i] * c[i]),2)
                    + pow(da1* (1.5 * D2 * a[i] * a[i] / xsi[i] - xsi[i] * 0.5 * (b[i] + 2 * a[i] * b[i] * D2) / (a[i] * b[i] + a[i] * a[i] *b[i] * D2 - c[i] * c[i])),2)
                    + pow(db1 * 0.5 * xsi[i] / (a[i] * b[i] + a[i] * a[i] * b[i] * D2 - c[i] * c[i]) * (a[i] + a[i] * a[i] * D2),2)
                    + pow(dc1 *0.5 * xsi[i] / (a[i] * b[i] + a[i] * a[i] * b[i] * D2 - c[i] * c[i]) * (-2.0 - c[i]),2 )
                    );

        std::cout << "xsi["<< i<<"] is "  << xsi[i] << ". Error is " << dxsi[i] << " and that is " << 100.0*  dxsi[i] / xsi[i] << " percent." << std::endl;




        long double v0star = v[i] - z1 * sqrt(a[i] + a[i] * a[i] * D2) / sqrt(a[i] * b[i] - c[i] * c[i]);
        long double d2v0star =

                    pow(dv1 * 1.0,2)
                +pow(da1 * -1.0 * z1 * ((0.5 / sqrt(a[i] + a[i] * a[i] * D2) * sqrt(a[i]*b[i]-c[i]*c[i])* (1 + 2 * a[i] * D2)) - (sqrt(a[i]+ a[i]*a[i]*D2) * 0.5 / sqrt(a[i]*b[i]-c[i]*c[i]) * b[i]) )/ (a[i] * b[i] - c[i] * c[i]) ,2)
                   // + pow(da1 * z1 * 0.5 / (a[i]*b[i] - c[i] * c[i]) * (z1 / v0star * (1.0 + 2.0 * a[i] * D2) - v0star / z1 * b[i]),2)
                   // + pow (db1 * -0.5 * v0star * a[i] / (a[i] * b[i] - c[i] * c[i]),2 )
                + pow(db1 * -1.0 * z1 * sqrt(a[i]+a[i]*a[i]*D2) * -0.5 * sqrt(a[i]*b[i]-c[i]*c[i]) / (a[i]*b[i]-c[i]*c[i]) * a[i]  ,2)
                  //  + pow ( dc1 * -0.5 * v0star / (a[i] * b[i] - c[i] * c[i]) * -2.0 * c[i],2);
                + pow(dc1 * -1.0 * z1 * sqrt(a[i]+a[i]*a[i]*D2) * -0.5 * sqrt(a[i]*b[i]-c[i]*c[i]) / (a[i]*b[i]-c[i]*c[i]) * -2.0 * c[i],2);

        long double dv0star = sqrt(d2v0star);

        std::cout << "v0star from ellipse "  << i << " is "<< v0star << " with an error of " << dv0star << ". that is " << 100.0 * dv0star / v0star << " percent.  " << std::endl;
        //DEBUG
        std::cout << "v0star calculated from vi=" << v[i] << ", z1=" << z1 << ", a1 =" << a[i] << "D: " << sqrt(D2) << ", c1 = " << c[i]<<std::endl << std::endl;
        if(! (v0star != v0star) && !(d2v0star != d2v0star) ) // valid numbers
        {
            v0starMean += v0star / d2v0star;
            v0starNorm += 1.0 / d2v0star;
            dv0starMean += 1.0 / d2v0star;
            pairs++;
        }

    }

    if(pairs == 0)
    {
        delete[] a;
        delete[] b;
        delete[] c;
        delete[]u;
        delete[] v;
        delete[] error;
        delete[] validEllipse;
        //delete[] phases;
        //delete[] rho;
        delete[] xsi;
        delete[]dxsi;
        delete[] x_balls;
        delete[] y_balls;

        std::cout << " ERROR Could not determine v0star." << std::endl;
        return;
    }

    long double v0star = v0starMean / v0starNorm;
    dv0starMean = sqrt(dv0starMean);
    long double dv0star = dv0starMean / v0starNorm;

    std::cout <<std::endl << std::endl<< "Averaged v0star is " << v0star << " with error of " << dv0star << " that is " << 100.0 * dv0star / v0star << " percent. "<<std::endl << std::endl << std::endl;


    long double u0starMean = 0.0;
    long double u0starNorm = 0.0;
    long double du0starMean = 0.0;


    long double fiMean = 0.0;
    long double fiNorm = 0.0;
    long double dfiMean = 0.0;

    pairs =0;



    for(int i=0; i<n; i++)
    {

        if(! validEllipse[i])
        {
            continue;
        }



        for(int j=0; j<n; j++)
        {
            if(! validEllipse[j])
            {
                continue;
            }


            if( j <=i)
            {
                continue;
            }

            int z1 = v[i]<z0?-1:+1;
            int z2 = v[j]<z0?-1:+1;


            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            long double du1 = error[i*5];
            long double du2 = error[j*5];
            long double dv1 = error[i*5 + 1];
            long double dv2 = error[j*5 +1];
            long double da1 = error[i*5 + 2];
            long double da2 = error[j*5 + 2];
            long double db1 = error[i*5 + 3];
            long double db2 = error[j*5 + 3];
            long double dc1 = error[i*5 + 4];
            long double dc2 = error[j*5 + 4];

            long double u0star = 0.5* u[i] + 0.5 * u[j] + 0.5 * c[i] / a[i] * (v[i] - v0star ) + 0.5 * c[j] / a[j]  * ( v[j] - v0star);
            long double du0star = sqrt(
                        pow(0.5 * du1,2)
                        + pow(0.5 * du2 ,2)
                        + pow(dc1 * (v[i] - v0star) * 0.5 / a[i] ,2 )
                        + pow( da1 * c[i] * 0.5 / a[i] / a[i]  * (v[i] - v0star) , 2)
                        + pow( dv1 * c[i] / a[i] * 0.5 ,2)
                        + pow(dv0star * (c[i] * 0.5 / a[i] + c[j]  * 0.5 / a[j]),2)
                        + pow(dc2 * 0.5 * a[j]  * (v[j] - v0star) ,2)
                        + pow(da2 * c[j] * 0.5 / a[j] / a[j] * (v[j] - v0star) ,2)
                        + pow(dv2 * c[j] * 0.5 / a[j] ,2)
                        );

            std::cout << "u0star from ellipse " << i << " and " << j << " is " << u0star << " with an error of " << du0star << ". that is "<< du0star  * 100.0 / u0star << " percent. " <<std::endl;

            long double sinfi = -0.5*c[i] / a[i] * xsi[i] - 0.5 * c[j] / a[j]  * xsi[j] ;
            long double fi = asin(sinfi );
            long double dsinfi = sqrt(
                        pow(dxsi[i] * -0.5 * c[i] / a[i] ,2)
                        + pow(dc1 * xsi[i] * 0.5 /a[i],2)
                        + pow( da1 * c[i] * 0.5 / a[i] / a[i] * xsi[i],2)
                        +pow(dxsi[j] * -0.5 * c[j] / a[j] ,2)
                        + pow(dc2 * xsi[j] * 0.5 /a[j],2)
                        + pow( da2 * c[j] * 0.5 / a[j] / a[j] * xsi[j],2)
                        );
            long double dfi = 1 / sqrt(1 - sinfi * sinfi) * dsinfi;

            std::cout << "fi from ellipse " << i << " and " << j << " is " << fi << " with an error of " << dfi << ". that is " << dfi * 100.0 / fi << " percent. " << std::endl;



            if( u0star == u0star && du0star == du0star && fi == fi && dfi == dfi)
            {
                u0starMean += u0star *  1 / pow(du0star,2);
                u0starNorm +=  1.0 / pow(du0star,2);
                du0starMean += 1.0 / pow(du0star,2);

                fiMean += fi * 1/ pow(dfi,2);
                fiNorm +=  1/ pow(dfi,2);
                dfiMean +=  1/ pow(dfi,2);

                pairs++;

            }

        }
    }

    if(pairs == 0)
    {
        delete[] a;
        delete[] b;
        delete[] c;
        delete[]u;
        delete[] v;
        delete[] error;
        delete[] validEllipse;
        //delete[] phases;
        //delete[] rho;
        delete[] xsi;
        delete[]dxsi;
        delete[] x_balls;
        delete[] y_balls;

        std::cout << " ERROR Could not determine u0star and fi." << std::endl;
        return;
    }

    long double u0star = u0starMean  / u0starNorm;
    long double du0star = sqrt(du0starMean) / u0starNorm;
    long double fi = fiMean / fiNorm;
    long double dfi = sqrt(dfiMean) / fiNorm;

    std::cout << " Averaged u0* = " << u0star << "with  error of " << du0star << " that is " << 100.0 * du0star / u0star << " percent." << std::endl;

    std::cout << "Averaged  fi = " << fi << " radian = " << fi / 2.0 / 3.14159265358979323846  * 360.0 << " degree. Error is " <<  dfi << "radian that is " << dfi* 100.0 / fi << " percent." << std::endl;


    //calculating R:

    long double RMean   =0.0;
    long double RNorm   =0.0;
    long double dRMean   =0.0;





    pairs =0;




    for(int i=0; i<n; i++)
    {

        if(! validEllipse[i])
        {
            continue;
        }

        int* u1 = new int[lastIndex - firstIndex]();
        int* v1 = new int[lastIndex - firstIndex]();

        geomcorr.coordinatesToCPU(u1,v1,i);



        for(int j=0; j<n; j++)
        {
            if(! validEllipse[j])
            {
                continue;
            }


            if( j <=i)
            {
                continue;
            }

            int z1 = v[i]<z0?-1:+1;
            int z2 = v[j]<z0?-1:+1;


            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            int* u2 = new int[lastIndex - firstIndex]();
            int* v2 = new int[lastIndex - firstIndex]();

            geomcorr.coordinatesToCPU(u2,v2,j);
            long double doverR2 =0.0;
            long double d2doverR2 = 0.0;


            //Debug:
            long double d2doverR2Part1 =0.0;
            long double d2doverR2Part2 =0.0;
            long double d2doverR2Part3 =0.0;


            for(int k=0; k< lastIndex - firstIndex; k++)
            {

                //Corrigating by eta:

                double eta = geomcorr.getEta();
                double u0 = u1[k];
                double v0  =v1[k];
                u1[k] = u0 * cos(eta) - v0* sin(eta);
                v1[k] = u0 * sin(eta) + v0 * cos(eta);
                u0 = u2[k];
                v0 = v2[k];
                u2[k] =  u0 * cos(eta) - v0* sin(eta);
                v2[k] = u0 * sin(eta) + v0 * cos(eta);


                doverR2 +=pow(xsi[i] * (u1[k] - u0star) / (v1[k] - v0star) - xsi[j] * (u2[k] - u0star) / (v2[k] - v0star) ,2 ) + pow((D*xsi[i] / (v1[k] - v0star)  - D * xsi[j] / (v2[k] - v0star)) ,2) + pow(xsi[i] - xsi[j],2);


                d2doverR2Part1 +=pow(xsi[i] * (u1[k] - u0star) / (v1[k] - v0star) - xsi[j] * (u2[k] - u0star) / (v2[k] - v0star) ,2 );
                d2doverR2Part2 += pow((D*xsi[i] / (v1[k] - v0star)  - D * xsi[j] / (v2[k] - v0star)) ,2);
                    d2doverR2Part3 += pow(xsi[i] - xsi[j],2);

                d2doverR2 += (
                            //Todo: Fill not xsi related stuff.... i do not want.
                            pow(dxsi[i] * ((2 * (xsi[i] * (u1[k]- u0star) / (v1[k] - v0star) - xsi[j] * (u2[k] - u0star) / (v2[k] - v0star)  ) *(u1[k]- u0star) / (v1[k] - v0star) ) + (2 *( D*xsi[i] / (v1[k] - v0star)  - D * xsi[j] / (v2[k] - v0star))) * D / (v1[k] - v0star) + 2 * (xsi[i] - xsi[j] )  )  ,2 )
                           +  pow(dxsi[j] * ((2 * (xsi[i] * (u1[k]- u0star) / (v1[k] - v0star) - xsi[j] * (u2[k] - u0star) / (v2[k] - v0star)  ) * -1.0 * (u2[k]- u0star) / (v2[k] - v0star) ) + (2 * (D*xsi[i] / (v1[k] - v0star)  - D * xsi[j] / (v2[k] - v0star))) * -1.0 * D / (v2[k] - v0star) - 2.0 * (xsi[i] - xsi[j] )  )  ,2 )
                            ) ;
            }
            doverR2 /= (lastIndex - firstIndex);

            long double ddoverR2 = sqrt(d2doverR2) / (lastIndex - firstIndex);

            std::cout << "Ellipse " << i << " and " << j << ": doverR2 is" << doverR2 << " with error: " << ddoverR2 << " that is " << ddoverR2 *100.0 / doverR2 << " percent."<<std::endl;

            std::cout << " d2doverR2Part1=" << d2doverR2Part1 / (lastIndex - firstIndex) << " d2doverR2Part2=" << d2doverR2Part2 / (lastIndex - firstIndex) << " d2doverR2Part3=" << d2doverR2Part3 / (lastIndex - firstIndex)  << std::endl;
            //search for the i-t ball:

            int ball1Index = 0;
            int ball2Index = 0;

            for(int b1 = 0; b1 < n ; b1 ++)
            {
                if(orderOfBalls[b1] == i)
                {
                    ball1Index = b1;
                    break;
                }
            }

            for(int b2 = 0; b2 < n ; b2 ++)
            {
                if(orderOfBalls[b2] == j)
                {
                    ball2Index = b2;
                    break;
                }
            }

            long double x1 = x_balls[ball1Index]  * 2   ; // distance of two holes is 2 mm.
            long double x2 = x_balls[ball2Index]* 2 ;

            long double y1 = y_balls[ball1Index]* 2 ;
            long double y2 = y_balls[ball2Index]* 2 ;

            long double d2 = pow(x1 - x2,2) + pow(y1 - y2,2);
            long double d = sqrt(d2);
            std:: cout << " distance of balls "<< i << " and " << j << " is " << d << " mm."<< std::endl;


            long double R = sqrt ( 1.0 /doverR2 ) * d;
            long double dR= pow(sqrt(1.0 /doverR2 ),3) * 0.5  * d * ddoverR2;

            std::cout << " R ( in mm) is " << R << " with an error of " <<dR << " pixels that is " << 100.0 * dR / R << " percent . "  << " from ellipse " << i << " and "<< j << std::endl;

            if( R == R && dR == dR)
            {
            RMean += R * 1.0 / pow(dR,2);
            RNorm += 1.0 / pow(dR,2);
            dRMean += 1.0 / pow(dR,2);
            }

            delete[] u2;
            delete[] v2;

        }

        delete[] u1;
        delete[] v1;
    }

    long double R = RMean / RNorm;
    long double dR = sqrt(dRMean) / RNorm;

    std::cout << " Averaged R is " << R << " mm with an error of " << dR << " mm that is " << 100.0 * dR / R << " percent. " << std::endl;




    long double eta = geomcorr.getEta();
    long double u0 = u0star * cos(eta) + v0star * sin(eta);
    long double v0 = u0star * -1.0 * sin(eta) + v0star * cos(eta);

    std::cout << "u0: " << u0 << std::endl
               <<" v0 : " << v0 << std::endl;


    std::cout << " ------" << std::endl  <<std::endl << " Summary:"
              << "Eta: " <<  eta * 0.5 / 3.1415926535897932384626433 * 360.0 <<" degree"<< std::endl
              << " u0: " << u0 << " pixel" << std::endl
              << " v0: "  << v0 << " pixel"<< std::endl
              << "D: "<< D *0.0748  << " mm" << std::endl
              << "R: " << R << " mm " << std::endl;

/*****************************************************************


                        WU METHOD


******************************************************************/
//Wu method::::

    std::cout << std::endl << "Omg here is Wu method" << std::endl;

    float D_wu=0.0f; float v0_wu = 0.0f;
    geomcorr.dAndVWithWu( a,  b, v, &D_wu, &v0_wu );

    std::cout << " Wu method:" << std::endl
              << "D: " << D_wu << " pixel that is " << D_wu * 0.0748 << " mm"<< std::endl
              << " V0: " << v0_wu << std::endl;


    //Calculating with error:
//See http://fizipedia.bme.hu/images/9/92/Hibaszamitas.pdf for details

    long double S = 0.0;
    long double Sx = 0.0;
    long double Sy = 0.0;
    long double Sxx = 0.0;
    long double Sxy = 0.0;
    long double delta = 0.0;


    for(int i=0; i<n; i++)
    {

        if(! validEllipse[i])
        {
            continue;
        }


        for(int j=0; j<n; j++)
        {
            if(! validEllipse[j])
            {
                continue;
            }


            if( j <=i)
            {
                continue;
            }


            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            long double y = 0.5 * (v[i] + v[j] ) - 0.5 /(v[i] - v[j] ) * (1/b[i] - 1/b[j]);

            long double dv1 = error[i*5 + 1];
            long double dv2 = error[j*5 +1];
            long double db1 = error[i*5 + 3];
            long double db2 = error[j*5 + 3];

            long double dy2 =
                        pow(dv1 * (0.5 - 0.5  *(1/b[i] - 1/b[j]) /(v[i] - v[j] )) ,2)+
                        pow(dv2 * (0.5 - 0.5  *(1/b[i] - 1/b[j]) /(v[i] - v[j] )) ,2)+
                        pow( db1 * 0.5 * (v[i] + v[j] ) - 0.5 /(v[i] - v[j] ) * (1/b[i] / b[i] ),2 )+
                        pow( db2 * 0.5 * (v[i] + v[j] ) - 0.5 /(v[i] - v[j] ) * (1/b[j] / b[j] ),2 )
                        ;
            long double x = 0.5 / (v[i] - v[j]) * (a[i] / b[i] - a[j] / b[j]);

            S += 1 / dy2;
            Sx += x / dy2;
            Sy += y / dy2;
            Sxx += x * x / dy2;
            Sxy += x *y /dy2;
        }
    }
    delta = S *  Sxx - Sx * Sx;

    long double v0_wu_witherror = (Sxx * Sy - Sx * Sxy) / delta;
    long double D2_wu_witherror =( S * Sxy - Sx * Sy ) / delta;

    long double dv0_wu =  sqrt ( Sxx / delta);
    long double dD2_wu = sqrt( S / delta);

    long double D_wu_witherror = sqrt ( D2_wu_witherror);
    long double dD_wu = 0.5  / D_wu_witherror *  dD2_wu;

    std::cout << " Fitting with error:" << std::endl;
    std:: cout << " D = " << D_wu_witherror << " pixel with error of " << dD_wu << " that is " << 100.0 * dD_wu / D_wu_witherror << " percent. " << std::endl;
    std::cout << " D in mm = " <<D_wu_witherror * 0.0748 << " with error of " << dD_wu  * 0.0748 << " mm." << std::endl;
    std::cout << " v0 = " << v0_wu_witherror << " with error of " << dv0_wu << " that is " << 100.0 * dv0_wu / v0_wu_witherror << " percent. " << std::endl;


    long double u0_wu_mean = 0.0;
    long double u0_wu_norm = 0.0;
    long double u0_wu_error = 0.0;
    for(int i=0; i< n; i++)
    {
        long double du = error[i*5];

        if( u[i] == u[i] && du == du) // valid number
        {
            u0_wu_mean+= u[i] / du / du;
            u0_wu_norm += 1.0 / du / du;
            u0_wu_error += 1.0 / du / du;
        }
    }

    long double u0_wu = u0_wu_mean / u0_wu_norm;
    long double du0_wu = sqrt(u0_wu_error) / u0_wu_norm;


    std::cout<< " U0: " << u0_wu << " with error of " << du0_wu << " that is " << 100.0 * du0_wu / u0_wu << " percent." << std::endl;

    //Calculatin R:

    long double RWuMean   =0.0;
    long double RWuNorm   =0.0;
    long double dRWuMean   =0.0;



    for(int i=0; i<n; i++)
    {

        if(! validEllipse[i])
        {
            continue;
        }


        for(int j=0; j<n; j++)
        {
            if(! validEllipse[j])
            {
                continue;
            }


            if( j <=i)
            {
                continue;
            }


            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            long double ro1 = 1.0 / abs(v[i]-v0_wu_witherror) / sqrt(b[i]);
            long double ro2 = 1.0 / abs(v[j]-v0_wu_witherror) / sqrt(b[j]);

            long double dv1 = error[i*5 + 1];
            long double dv2 = error[j*5 +1];
            long double db1 = error[i*5 + 3];
            long double db2 = error[j*5 + 3];

            //TODO: FILL v and D error...

            long double dro1  = sqrt (
                        pow (dv1  * ro1 / (v[i]-v0_wu_witherror) ,2) +
                        pow(db1 * ro1 * 0.5 / b[i] ,2)+
                        pow(dv0_wu  *ro1 / (v[i]-v0_wu_witherror),2 )
                        )
                    ;
            long double dro2  = sqrt (
                        pow (dv2  * ro2 / (v[j]-v0_wu_witherror) ,2) +
                        pow(db2 * ro2 * 0.5 / b[j] ,2)+
                        pow (dv0_wu  * ro2 / (v[j]-v0_wu_witherror) ,2)
                        );

            long double zeta1 = (v[i]-v0_wu_witherror) * (1-ro1 * ro1 ) / D_wu_witherror;
            long double zeta2 = (v[j]-v0_wu_witherror) * (1-ro2 * ro2 ) / D_wu_witherror;

            long double dzeta1 = sqrt(
                        pow(dv1 * (1-ro1 * ro1 ) / D_wu_witherror,2 )+
                        pow(dv0_wu * (1-ro1 * ro1 ) / D_wu_witherror,2 )+
                        pow(dro1 *  (v[i]-v0_wu_witherror) * (- 2.0 * ro1 ) / D_wu_witherror,2)
                        + pow (dD_wu * zeta1 /D_wu_witherror,2  )
                        );

            long double dzeta2 = sqrt(
                        pow(dv2 * (1-ro2 * ro2 ) / D_wu_witherror,2 )+
                        pow(dv0_wu * (1-ro2 * ro2 ) / D_wu_witherror,2 )+
                        pow(dro2 *  (v[j]-v0_wu_witherror) * (- 2.0 * ro2 ) / D_wu_witherror,2)
                        + pow (dD_wu * zeta2 /D_wu_witherror,2  )
                        );
            //Determining phase ( 1 or 180)

           bool ball1Left =  geomcorr.isBallOnLeftSide(i,u0_wu);
           bool ball2Left =  geomcorr.isBallOnLeftSide(j,u0_wu);
           int cosDelta;
           if(ball1Left && ball2Left || !ball1Left && !ball2Left )
               cosDelta = 1;
           else
               cosDelta = -1;



            long double dOverR2= pow(zeta1 - zeta2,2) + ro1 * ro1 + ro2 * ro2 - 2 *ro1 * ro2 * cosDelta;


            long double ddOverR2 =
                    sqrt(
                            pow(dzeta1 * 2.0 * (zeta1 - zeta2),2) +
                            pow(dzeta2 *(2.0 * (zeta1 - zeta2)),2)+
                            pow( dro1 * (2.0 * ro1 - 2.0 * ro2 * cosDelta  ),2)+
                            pow( dro2 * (2.0 * ro2 - 2.0 * ro1 * cosDelta  ),2)
                        );





            //search for the i-t ball:

            int ball1Index = 0;
            int ball2Index = 0;

            for(int b1 = 0; b1 < n ; b1++)
            {
                if(orderOfBalls[b1] == i)
                {
                    ball1Index = b1;
                    break;
                }
            }

            for(int b2 = 0; b2 < n ; b2++)
            {
                if(orderOfBalls[b2] == j)
                {
                    ball2Index = b2;
                    break;
                }
            }

            long double x1 = x_balls[ball1Index]  * 2   ; // distance of two holes is 2 mm.
            long double x2 = x_balls[ball2Index]* 2 ;

            long double y1 = y_balls[ball1Index]* 2 ;
            long double y2 = y_balls[ball2Index]* 2 ;

            long double d2 = pow(x1 - x2,2) + pow(y1 - y2,2);
            long double d = sqrt(d2);

            long double R = sqrt( 1.0 / dOverR2 * d2 );

            long double dR = ddOverR2 * d * 0.5 /dOverR2 / sqrt(dOverR2)  ;

            std::cout << " WU: d IS " <<d << "from ellipse " << i << " and " <<  j << " that is ball " <<ball1Index << " , " << ball2Index  << std::endl;

            std::cout << "Wu ro1 is " << ro1 << "with relative error " << 100.0 * dro1 / ro1 << "from ellipse " << i << " and " <<  j << std::endl;
            std::cout << "Wu ro2is " << ro2 << "with relative error " << 100.0 * dro2 / ro2 << "from ellipse " << i << " and " <<  j << std::endl;

            std::cout << "Wu zeta 1  " << zeta1 << "with relative error " << 100.0 * dzeta1 / zeta1 << "from ellipse " << i << " and " <<  j << std::endl;
            std::cout << "Wu zeta 2  " << zeta2 << "with relative error " << 100.0 * dzeta2 / zeta2 << "from ellipse " << i << " and " <<  j << std::endl;


            std::cout << "Wu dOverR2 is " << dOverR2 << "with relative error " << 100.0 * ddOverR2 / dOverR2 << "from ellipse " << i << " and " <<  j << std::endl;

            std::cout << " WU: R IS " <<R << " with relative error " << 100.0 * dR / R << "% " << "from ellipse " << i << " and " <<  j << std::endl;

            RWuMean += R / (dR * dR);
            RWuNorm += 1.0  / (dR * dR);
            dRWuMean += 1.0  / (dR * dR);
        }
    }

    long double RWu = RWuMean / RWuNorm;
    long double dRWu = sqrt(dRWuMean) / RWuNorm;

    std::cout << "Averaged R with Wu method: " << RWu << " mm with error of " << dRWu << " mm that is  " << 100.0 * dRWu / RWu << " percent. " << std::endl;









    delete[] a;
    delete[] b;
    delete[] c;
    delete[]u;
    delete[] v;
    delete[] error;
    delete[] validEllipse;
    //delete[] phases;
    //delete[] rho;
    delete[] xsi;
    delete[] dxsi;
    delete[] x_balls;
    delete[] y_balls;

}


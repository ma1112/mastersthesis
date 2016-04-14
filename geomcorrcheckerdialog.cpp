#include "geomcorrcheckerdialog.h"
#include "ui_geomcorrcheckerdialog.h"
#include <iostream>
#include <QFileDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <math.h>       /* round, floor, ceil, trunc */

geomCorrCheckerDialog::geomCorrCheckerDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::geomCorrCheckerDialog)
{
    ui->setupUi(this);


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
    ui->progressBar->setValue(0);
    ui->progressBar->setVisible(true);

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
    float* error = new float[n]();

    for(int i=0; i<n; i++)
    {
        geomcorr.fitEllipse(i, a+ i,b +i, c +i, u +i, v +i , error +i);
    }
    geomcorr.exportText("C:/awing/ellipses.txt");
    std::cout << "fit success." << std::endl;


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


    //debug:
      for(int i=0; i< n; i++)
      {
          std::cout << "b[" <<i <<"] = " << 1 / sqrt(b[i]) <<std::endl;
      }



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

    if(minbOrder==0 || minbOrder== n-1 )
    {
        std::cout << "ERROR! All circles are on one side of the z=0 plane. ( minbIndex = ) " << minbIndex
                  << std::endl << " Geometry correction is not avaiable. "<< std::endl;
        delete[] a;
        delete[] b;
        delete[] c;
        delete[]u;
        delete[] v;
        delete[] error;
        delete[] validEllipse;
        return;
    }

    float z0 = 0;

    //looking for b of the ellipse that is above that:
    int indexabove = 0;
    float bAbove = 0;
    for(int i=0; i<n; i++)
    {

        if(orderOfBalls[i] == orderOfBalls[minbIndex] -1 )
        {
            indexabove = i;
            break;
        }
    }
    bAbove = 1  / sqrt(b[indexabove]);
    if (bAbove != bAbove )
    {
        bAbove =0;
    }

    //under that:
    int indexbelov = 0;
    float bBelove = 0;
    for(int i=0; i<n; i++)
    {

        if(orderOfBalls[i] == orderOfBalls[minbIndex] +1 )
        {
            indexbelov = i;
            break;
        }
    }
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


    std::cout << "z0 = " << z0 <<std::endl;


    //calculate D

    // for every pair where z1z2 < 0

    int pairs = 0;
    double Dmean = 0.0;
    double D2mean = 0.0;
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

            if( j <i)
            {
                continue;
            }
            int z1 = v[i]<z0?1:-1;
            int z2 = v[j]<z0?1:-1;
            if(z1 * z2 > 0)
            {
                continue;
            }

            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }

            double D2  = 0.0;
            double m1 = sqrt(b[j] - c[j] * c[j] / a[j]) / sqrt(b[i] - c[i] * c[i] / a[i]);
            double m0 = (v[j] - v[i] ) * sqrt(b[j]  - c[j] * c[j]  / a[j]);
            double n0 =(1.0-m0 * m0 - m1 * m1)  / ( 2 * m0 * m1);
            double n1 = (a[j] - a[i] * m1 * m1)  / (2.0 * m0 * m1);
            std::cout << "v[i]= " <<v[i] << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "v[j]= " <<v[j] << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "m1= " << m1 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "m0= " << m0 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "n0= " << n0 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "n1= " << n1 << " and i= " << i << " j=" << j <<std::endl;
            std::cout << "a[i]= " << a[i] << " and i= " << i << " j=" << j <<std::endl;

            double D =0.0;

            // alternative version:

            D2 = ((a[i] - 2.0 * n0 * n1) + sqrt(a[i] * a[i] + 4 * n1 * n1 - 4 * n0 * n1 * a[i])) / ( 2 * n1 * n1);
            D = sqrt(D2);
            std::cout << "Alternative D= " << D << " and i= " << i << " j=" << j <<std::endl;


            //opposite version:

                D2 = ((a[i] - 2.0 * n0 * n1) - sqrt(a[i] * a[i] + 4 * n1 * n1 - 4 * n0 * n1 * a[i])) / ( 2 * n1 * n1);
            D = sqrt(D2);
            std::cout << "D= " << D << " and i= " << i << " j=" << j <<std::endl;
            double v0star = v[i] - z1 * sqrt(a[i] + a[i] * a[i] * D2) / sqrt(a[i] * b[i] - c[i] * c[i]);
            double u0star = 0.5* u[i] + 0.5 * u[j] + 0.5 * c[i] / a[i] * (v[i] - v0star ) + 0.5 * c[j] / a[j]  * ( v[j] - v0star);

            std::cout << "v0star and un0star with not avaraged D: " <<std::endl;
            std::cout << "v0star: " << v0star << std::endl;
            std::cout << "u0star : " << u0star << std::endl;

            if(!(D!= D)) // if D is a number
            {
                Dmean += D;
                D2mean += D2;
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
        std::cout << "ERROR: no valid pairs found while calculating D." << std::endl;
        std::cout << " geometry collection is not avaiable." << std::endl;
        return;

    }


    double D = Dmean / pairs;
    double D2 = D2mean / pairs;
    std::cout << "D = " << D <<" with statictical error of " << sqrt(D2 - D*D) << ". Estimated from " << pairs << " pairs of ellipses."<<std::endl;

    double* xsi = new double[n];
    double v0starMean = 0.0;
    double v0star2Mean = 0.0;
    double u0starMean = 0.0;
    double u0star2Mean = 0.0;
    double fimean = 0.0;
    pairs =0;

    //calculating other parameters.
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

            if( j <i)
            {
                continue;
            }
            int z1 = v[i]<z0?1:-1;
            int z2 = v[j]<z0?1:-1;
            if(z1 * z2 > 0)
            {
                continue;
            }
            if( a[i] != a[i] || b[i] != b[i] || c[i] != c[i] ||
                    a[j] != a[j] || b[j] != b[j] || c[j] != c[j] ) // either is not a number
            {
                continue;
            }


            xsi[i] = D * z1* a[i] * sqrt(a[i]) / sqrt(a[i]*b[i] + a[i] * a[i] * b[i] * D2 -c[i] * c[i] );
            xsi[j] = D * z2* a[j] * sqrt(a[j]) / sqrt(a[j]*b[j] + a[j] * a[j] * b[j] * D2 -c[j] * c[j] );
            double v0star = v[i] - z1 * sqrt(a[i] + a[i] * a[i] * D2) / sqrt(a[i] * b[i] - c[i] * c[i]);
            double u0star = 0.5* u[i] + 0.5 * u[j] + 0.5 * c[i] / a[i] * (v[i] - v0star ) + 0.5 * c[j] / a[j]  * ( v[j] - v0star);
            double fi = asin(-0.5*c[i] / a[i] * xsi[i] - 0.5 * c[j] / a[j]  * xsi[j] );
            //double rho = Rho is not used ever...

            if( v0star == v0star && u0star == u0star && fi == fi)
            {
                std::cout << "v0star = " << v0star << " when i = " << i<< " and j = " << j << std::endl;
                v0starMean += v0star;
                v0star2Mean += v0star * v0star;
                u0starMean += u0star ;
                u0star2Mean+= u0star * u0star;
                fimean += fi;
                pairs++;
            }

        }
    }
    v0starMean/=pairs;
    u0starMean/=pairs;
    fimean /=pairs;

    std::cout << " v0* = " << v0starMean << "with statistical error of " << sqrt(v0star2Mean - v0starMean * v0starMean) << std::endl;
    std::cout << " u0* = " << u0starMean << "with statistical error of " << sqrt(u0star2Mean - u0starMean * u0starMean) << std::endl;
    std::cout << " fi = " << fimean << " radian = " << fimean / 2.0 / 3.14159265358979323846  * 360.0 << " degree" << std::endl;














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

}


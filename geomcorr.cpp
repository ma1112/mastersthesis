#include "geomcorr.h"
#include"QDir"
#include"QString"
#include"QStringList"
#include"image_cuda_compatible.h"
#include"QFileDialog"
#include"iostream"
#include "image.h"


void Geomcorr::readAndCalculateGeom(int n)
{
    //Asking for input and output  directories.
       QString dir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which contains folders of images to calculate geometry correction data"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

       QString outdir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to save geometry correction factors"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

gc.readgainfactors();
gc.readoffsetfactors();

       QDir directory (dir);
       QStringList nameFilter("*.bin"); //name filter.
       QStringList filelist = directory.entryList(nameFilter); //File list of .bin files


       Image image;
       if(filelist.size() >0)
       {
           initializeDeviceVector(n,filelist.size());
       }

       for(int i = 0; i<filelist.size(); i++)
       {
           //Process each images.
           image.readfromfile(directory.absoluteFilePath(filelist.at(i)).toStdString());
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

           gc.offsetcorrigateimage(image);
           gc.gaincorrigateimage(image);

           extractCoordinates(image);
           //image.saveAsJPEG("C:/awing/hough/" + filelist.at(i).toStdString() + ".jpg");
           //image.writetofloatfile("C:/awing/hough/" + filelist.at(i).toStdString() + "f");

           addCoordinates();


       }
}

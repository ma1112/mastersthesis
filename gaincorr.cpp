#include "gaincorr.h"
#include <QTime> //debug
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QDir>
#include <math.h>       /* round*/


Gaincorr::Gaincorr()
{

}


int Gaincorr::read()
{
    std::cout << "gancorr::read()" << std::endl; //Debug.

    //Asking for directory.
    QString dir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Mappa kiválasztása---"),
                                                    "C:\\",
                                                     QFileDialog::DontResolveSymlinks);
    //looking for subdirs:

    QDir directory(dir);
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();


    images.reserve(subdirs.size()); // vector for the final, averaged images.



    for(int i=0;i<subdirs.size();i++) // opening every subdir for reading images...
    {

        //Looking for .bin files
       QStringList nameFilter("*.bin"); //name filter.
       QDir subdirectory(directory.absoluteFilePath(subdirs.at(i))); //Qdir for storing actual subdirectory.

       QStringList filelist = subdirectory.entryList(nameFilter); //File list of .bin files in the actual subdirectory.
       images_temp.reserve(filelist.size()); //images_temp for reading images from one file. MAY NOT BE USED IN THE FUTURE.
       images_temp.clear();


       //Variables to calculate mean values of voltage, amperage, exptime, intensity of images in a subdirectory.
       double meanVoltage =0;
       double meanAmperage =0;
       double meanExptime = 0;
       double meanIntensity =0;

       for(int j=0;j<filelist.length();j++)       //Opening every file in the given subdirectory
           //Note: subdirectory is indexed by i, file is indexed by j.

       {
          // calculating mean values of the images.
           //image is a temporary Image, that reads from files.


           image.readfromfile(subdirectory.absoluteFilePath(filelist.at(j)).toStdString());

           //Note: images with 0 voltage or amperage (or other parameters) are ignored. X-ray tube was probably off...
           if(image.getvoltage() > 1 && image.getexptime() > 1 && image.getmean() > 1 && image.getvoltage() > 1)
               {

               images_temp.push_back(image); //loading images from one subdir to images_temp vecor.

               meanVoltage += ( (images_temp.back().getvoltage()) / (double) filelist.length());
               meanAmperage += ((images_temp.back().getamperage()) / (double) filelist.length());
               meanExptime += ((images_temp.back().getexptime()) /  (double) filelist.length());
               meanIntensity += ((images_temp.back().getmean()) /  (double) filelist.length());
               }
       }





       image.clear(); // I'll sum the good images to this variable.


       int count = 0; //counts good images in a subfolder.


       //Deleting images that differs more than 10 percent from the mean. Recalculating mean values.
           for(std::vector<Image_cuda_compatible>::iterator iter = images_temp.begin(); iter != images_temp.end();)
           {
               //Checking for every image if they parameters are near the mean of the parameters.
               if     (
                     abs(iter->getmean() - meanIntensity) > meanIntensity * 0.1f ||
                    abs(iter->getvoltage() - meanVoltage) > meanVoltage * 0.1f ||
                    abs(iter->getamperage() - meanAmperage) > meanAmperage * 0.1f ||
                    abs(iter->getexptime() - meanExptime) > meanExptime * 0.1f
                       )
               { //if the image is corrupted, delete it. Also put som info to the console.
                   std::cout << "Bad image: " << iter->getid() <<" in folder " << subdirectory.absolutePath().toStdString()<<std::endl;
                   std::cout << "meanIntenstiy = " << meanIntensity << "getmean =" <<iter->getmean()
                             <<"meanVoltage = " << meanVoltage << " getvoltage =" <<iter->getvoltage()
                            <<"meanAmperage = " << meanAmperage << "getamperage = " << iter->getamperage()
                           <<"meanExptime = " << meanExptime << "getexptime = " << iter ->getexptime()
                          <<"Filelist.length()" << filelist.length()
                         <<std::endl<<std::endl;
                  // no reason to wate time on eraseing. iter =  images_temp.erase(iter);
                   ++iter;//

               }
               else // image is good.
                   {

                   count +=1;

                    image+=*iter; //Summing images at the image variable.
                   ++iter;
                   }
           }











           //Dividing image parameters and pixel values by count
         image/=count;

           //Image processing finished in the current subdir. If there was some good image in that folder, add the averaged image to the averaged images' vector.
           if( images_temp.size() > 1)
               {
               //Rounding voltage to multiply of 5
               int voltage = round(image.getvoltage());
               int remainder  = voltage %5;

               if(remainder != 0)

               {
                   voltage = voltage + 5 - remainder;
               }
                   imagemap[voltage].reserve(50);
                   imagemap[voltage].push_back(image);
               //images.push_back(image);
               std::cout <<"Images loaded from directory " << subdirectory.absolutePath().toStdString()
                        <<"With mean: " << image.getmean()
                       << "voltage: "  <<image.getvoltage()
                       <<" amperage: " << image.getamperage()
                      <<"exptime: " << image.getexptime()<<std::endl;
               std::cout<<std::endl << "Images with voltage " <<voltage << ": " << imagemap[voltage].size() << std::endl;
               }
           else
              {
               std::cout <<"Not enough good images in directory" <<subdirectory.absolutePath().toStdString()<<std::endl;
               std::cout << "meanIntensity = "<< meanIntensity <<std::endl;
           }




    } // end of for (every subdirectory)

 std::cout << "Exiting read();" <<std::endl;
return 0;


} // end of read() function.

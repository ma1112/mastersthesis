#include "gaincorr.h"
#include <QTime> //debug
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QDir>
#include <math.h>       /* round*/
#include <set>
#include <QStringRef>



Gaincorr::Gaincorr()
{
    gcfolder = "";

}

//! Reads images to calculate gain correction data.

//! The functions asks for an input folder. In this folder, images sould be in subfolders.
//! Every subfolder sould contain one ore more image with the same settings (Voltage, Exp time, amperage), with an info file.
//! Images are only loaded from the subfolders of the user given directory.
//! Images are stored in the gc_im_container class.

int Gaincorr::readimages()
{

 //Asking for input and output  directories.
    QString dir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which contains folders of images to calculate gain correction data"),
                                                    "C:\\",
                                                     QFileDialog::DontResolveSymlinks);

    QString Qgcfolder = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to save gain correction factors (slope and intercept)"),
                                                    "C:\\",
                                                     QFileDialog::DontResolveSymlinks);
    gcfolder = Qgcfolder.toStdString();


 //looking for subdirs:

    QDir directory(dir);
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();


    images.reserve(subdirs.size()); // vector for the final, averaged images.

    if(subdirs.size() == 0)
    {
        std::cout << "ERROR : No subfolders in folder" << dir.toStdString() << "Gain correction halted." <<std::endl;
        return 0;
    }

 // opening every subdir for reading images...

    for(int i=0;i<subdirs.size();i++)
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

       if(filelist.length() == 0)
       {
           std::cout<<"Warning: No .bin file in subfolder" << directory.absoluteFilePath(subdirs.at(i)).toStdString() <<std::endl;
       }

       else
       {
//Opening every file in the given subdirectory


       for(int j=0;j<filelist.length();j++)
           //Note: subdirectory is indexed by i, files are indexed by j.
       {
          // calculating mean values of the images.
           //image is a temporary Image, that reads from files.
           //DEBUG: Could be improved....


           image.readfromfile(subdirectory.absoluteFilePath(filelist.at(j)).toStdString());

 //Note: images with 0 voltage or amperage (or other parameters) are ignored. X-ray tube was probably off...
           if(image.getvoltage() > 1 && image.getexptime() > 1 && image.getmean() > 1 && image.getvoltage() > 1)
               {

               images_temp.push_back(image); //loading images from one subdir to images_temp vecor.

               meanVoltage += ( (images_temp.back().getvoltage()) );
               meanAmperage += ((images_temp.back().getamperage()));
               meanExptime += ((images_temp.back().getexptime()));
               meanIntensity += ((images_temp.back().getmean()) );
               }
       }
       //if there is any non-blank images, calculate mean values!
       if(images_temp.size() > 0)
       {
           meanVoltage/=images_temp.size();
           meanAmperage/=images_temp.size();
           meanExptime/=images_temp.size();
           meanIntensity/=images_temp.size();
       }





       image.clear(); // I'll sum the good images to this variable.
       //DEBUG Could be improved. E.g. sum at the GPU memory.



       int count = 0; //counts good images in a subfolder.



       //Ignoring images that differs more than 10 percent from the mean. Recalculating mean values.
           for(std::vector<Image_cuda_compatible>::iterator iter = images_temp.begin(); iter != images_temp.end();)
           {
               //Checking for every image if they parameters are near the mean of the parameters.
               //(10% difference is allowed.)
               //DEBUG: Is that OK?
               if     (
                     abs(iter->getmean() - meanIntensity) > meanIntensity * 0.1f ||
                    abs(iter->getvoltage() - meanVoltage) > meanVoltage * 0.1f ||
                    abs(iter->getamperage() - meanAmperage) > meanAmperage * 0.1f ||
                    abs(iter->getexptime() - meanExptime) > meanExptime * 0.1f
                       )
               { //if the image is corrupted, ignore it. Also put som info to the console.
                   std::cout << "Bad image: " << iter->getid() <<" in folder " << subdirectory.absolutePath().toStdString()<<std::endl;
                   std::cout << "meanIntenstiy = " << meanIntensity << "getmean =" <<iter->getmean()
                             <<"meanVoltage = " << meanVoltage << " getvoltage =" <<iter->getvoltage()
                            <<"meanAmperage = " << meanAmperage << "getamperage = " << iter->getamperage()
                           <<"meanExptime = " << meanExptime << "getexptime = " << iter ->getexptime()
                          <<"Filelist.length()" << filelist.length()
                         <<std::endl<<std::endl;
                   ++iter;//

               }
               else // image is good.
                   {

                   count +=1;

                    image+=*iter; //Summing images at the image variable.
                    //DEBUG: Should sum at the GPU.
                   ++iter;
                   }
           }



           //Image processing finished in the current subdir. If there was some good image in that folder, add the averaged image to the averaged images' vector.
           if( count > 1)
               {
               //Dividing image parameters and pixel values by count
               //(count is the total number of good images in the subdirectory.)

               image/=count;

               //Rounding voltage to multiply of 5          //
               int voltage = round(image.getvoltage());     //
               int remainder  = voltage %5;                 //
                                                            //
               if(remainder != 0)                           //
                                                            //
               {                                            //
                   voltage = voltage + 5 - remainder;       //
               }                                            //


               //Searching for saturation exposition.
               if(image.maxintensity() > 16382)  // saturation at 16383
                   //DEBUG: Possibly need to replace this part to the offset correction,
                   //once it is written.
               {
                   int satvalue = (int) (image.getexptime() * image.getamperage());
                   saturation.insert(std::pair<int,int> (voltage, satvalue ));
                   if( (int) ((saturation.find(voltage))->second) > satvalue)
                   {
                       saturation[voltage] = satvalue;
                   }
               }


               //Copying sum  of good images in the current subfolder to the imagemap class.
               //DEBUG: Wastes a lot of memory, should count exact space need before doing anything.
               std::cout <<"reserving memory for " << subdirs.size() << "with " << voltage <<" kv.  Needed RAM: " << image.size*sizeof(float)*subdirs.size() / 1024/1024 << " Mb." <<std::endl;
                   imagemap[voltage].reserveIfEmpty(subdirs.size());


                   imagemap[voltage].add(image);


               std::cout <<"Images loaded from directory " << subdirectory.absolutePath().toStdString()
                        <<"With mean: " << image.getmean()
                       << "voltage: "  <<image.getvoltage()
                       <<" amperage: " << image.getamperage()
                      <<"exptime: " << image.getexptime()<<std::endl;
               std::cout<<std::endl << "Images with voltage " <<voltage << ": " << imagemap[voltage].getimageno() << std::endl;
               }
           else //(if count <=1)
              {
               std::cout <<"Not enough good images in directory" <<subdirectory.absolutePath().toStdString()<<std::endl;
               std::cout << "meanIntensity = "<< meanIntensity <<std::endl;
           }


     } //    end of    if(filelist.length() == 0) {} else {


    } // end of for (every subdirectory)

 std::cout << "Exiting read();" <<std::endl;
return 0;


} // end of readimages() function.



//! Reads factors for gain correction from files.

//!  Reads slope and intercept files for gain correction. Asks the user to choose a folder.
//! In this folder there should be the files containing slope and intercept data for gain correction.
//! Name format: intercept<voltage>.binf and slope<voltage>.binf. Reads every file with the given syntax,
//! and stores them, if both an intercept and a slope file is there for a given voltage.
//DEBUG file names will possibly change onceoffset correction is written.



void Gaincorr::readfactors()
{

    QString Qgcfolder = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to load gain correction factors from (slope and intercept)"),
                                                    "C:\\",
                                                     QFileDialog::ReadOnly );
    gcfolder = Qgcfolder.toStdString();

    //Looking for .binf files
   QStringList slopeNameFilter("slope*.binf");
   QStringList interceptNameFilter("intercept*.binf");

   QDir dir (Qgcfolder);

   QStringList slopelist = dir.entryList(slopeNameFilter);
   QStringList interceptlist = dir.entryList(interceptNameFilter);

   if(slopelist.size() == 0 || interceptlist.size() == 0)
   {
       std::cout <<"Error: No gain correction factor files were found in foler " << gcfolder <<std::endl;
       std::cout << "Correction factor files are names slope<Voltage>.binf and intercept<Voltage>.binf." <<std::endl;
       return;
   }

   //Checkig input files's voltage infos:
   std::set<std::string> slopevoltages;
   std::set<std::string>interceptvoltages;
//Extracking voltage informations from slope files.
   for(int i=0; i< slopelist.size();i++)
   {
       QFileInfo info = QFileInfo(slopelist.at(i));
       QString name = info.baseName();
       QString  voltage = name.mid( name.lastIndexOf("slope") +5 , name.lastIndexOf(".binf") - 1  );
       slopevoltages.insert(voltage.toStdString());
   }

   //Extracking voltage informations from intercept files.

   for(int i=0; i< interceptlist.size();i++)
   {
       QFileInfo info = QFileInfo(interceptlist.at(i));
       QString name = info.baseName();
       QString  voltage = name.mid( name.lastIndexOf("intercept") +9 , name.lastIndexOf(".binf") - 1  );
       interceptvoltages.insert(voltage.toStdString());
   }

   //Reading slope and intercept with a common voltage setting
   for(std::set<std::string>::iterator iter = slopevoltages.begin(); iter!=slopevoltages.end();)
   {
       std::string thisvoltage = *iter; //voltage of the current slope file

       if(interceptvoltages.find(thisvoltage) != interceptvoltages.end()) // if both slope and intercept is aviable with that voltage...
       {
           //Read them...
           std::string filename = gcfolder;
           filename.append("/intercept");
           filename.append(thisvoltage);
           filename.append(".binf");
           intercept.readfromfloatfile(filename);
           intercept.calculate_meanvalue_on_CPU();

           filename = gcfolder;
           filename.append("/slope");
           filename.append(thisvoltage);
           filename.append(".binf");
           slope.readfromfloatfile(filename);
           slope.calculate_meanvalue_on_CPU();
           //Check if they are not blank...
           //DEBUG: Could check if every number is 0...

           if ( (intercept.getmean() <0 || intercept.getmean() >0 ) && ((slope.getmean() <0 || slope.getmean() >0)))
           {
               //Store them at the map...
           intercepts[atoi((thisvoltage).c_str())] = intercept;
           slopes[atoi((thisvoltage).c_str())] = slope;

           std::cout <<"Read gaincorr factors for " << thisvoltage << " voltage." << std::endl;
            }
            else
           {
               std::cout <<"Bad gain correction factor file for voltage " << thisvoltage << std::endl;
           }
           interceptvoltages.erase(interceptvoltages.find(thisvoltage));
           ++iter;


       }
       else // if there is a slope file for thisvoltage, but there is no intercept file
       {
           std::cout <<"Warning! No intecept file found for voltage "  << thisvoltage;
           ++iter;
       }

   }


   //listing voltage values that has either only one intercept, ot only one slope file:


//Printing out voltages with given intercept file but with a missing slope file.
   for(std::set<std::string>::iterator iter = interceptvoltages.begin(); iter!=interceptvoltages.end();iter++)
   {
       std::cout <<"Warning! No slope file found for voltage "  << *iter;
   }


   intercept.clear();
   slope.clear();
   image.clear();




} //end of void readfactors()







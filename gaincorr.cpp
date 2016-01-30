#include "gaincorr.h"
#include <QTime> //debug
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QDir>
#include <math.h>       /* round*/
#include <set>
#include <QStringRef>
#include <iostream>
#include <fstream>
#include <vector>
#include <fstream>      // std::ofstream




Gaincorr::Gaincorr()
{
    gcfolder = "";


}



//! Reads image files for osset calibration, caluclates and saves ofset corrigation data.

//! Asks the user for the folder that contains offset correction image set, and another folder
//! to save the offset correction factors.
//! Images should be in the subfolders of the given input folder. Every subolder should contain
//! images with the same exposition time settings.
//! Images are read and then offset corrections factors are determined by linear fit to everry pixel
//! in the function of the exposition time.
//! They are saved in float type binary image files in the output folder,
//! with names offsetslope.binf and offsetintercept.binf


void Gaincorr::readAndCalculateOffset()
{
    // ------------------------------------------------------------
    //
    //               Looking for valid subfolders
    //
    //--------------------------------------------------------------

     QStringList goodFolderList;



    //Asking for input and output  directories.
       QString dir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which contains folders of images to calculate offset correction data"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

       QString Qgcfolder = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to save offset correction factors (slope and intercept)"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);
       gcfolder = Qgcfolder.toStdString();

    QDir directory(dir);
    QString path = directory.absolutePath();
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();
    for(int i=0; i<subdirs.size(); i++) //for every subdirectory
    {
        //Readinf info files to determine number of valid folders.

        std::ifstream myfile;
        myfile.open(path.toStdString() + "/" +subdirs.at(i).toStdString() +"/info.txt" );
        if (! (myfile.is_open()))
        {
            std::cout<< "Warning: There is no info.txt file in folder "
                     << path.toStdString() + "/" +subdirs.at(i).toStdString()
                     << ", or info file is not readable. Folder is ignored."
                     <<std::endl;

            continue; // jump to the next subfolder.
        }
        std::string line;
        //reading header
        do
        {
            getline (myfile,line);


        } while(line.find("*****************" ) == std::string::npos && !(myfile.eof()) );



        if(myfile.eof())
        {
            std::cout << "WARNING There is no info in the info file in folder "
                      << path.toStdString() + "/" +subdirs.at(i).toStdString()
                      <<"." << std::endl;
            continue;
        }


        //reading info and calculating mean exposition time and checking if the
        //images are made without operating the X-ray source.
        int count = 0;

        while(getline (myfile,line) && line.length() > 2)
        {

            std::stringstream ss(line);
            std::string temp;
            int thisVoltage =0;
            int thisAmperage =0;
            int thisExptime=0;

            ss >> temp; // id
            ss >> temp; // rotation
            ss >> thisVoltage;
            ss >> thisAmperage;
            ss >> thisExptime;



                if((thisVoltage ==0 || thisAmperage ==0 ) && thisExptime >0 )
                {
                    count++;
                }

        } //every line is read.

        myfile.close();


            if(count > 2)
            {
                goodFolderList << subdirs.at(i);
            }


    } //every subdir is processed.



        if (goodFolderList.size() < 3)
        {
            goodFolderList.clear();
            std::cout << "ERROR Too few directories for offset correction. " <<std::endl;
            return;
        }

    //List is ready.

        //-------------------------------------------------------
        //
        //                Loading files
        //
        //-------------------------------------------------------
        gc_im_container  im_container;


        im_container.inicialize(goodFolderList.size());

        for(int i=0; i< goodFolderList.size();i++)
        {
            std::cout << "Loading files from " << goodFolderList.at(i).toStdString() << std::endl;


            //Looking for .bin files
           QStringList nameFilter("*.bin"); //name filter.
           QDir subdirectory(directory.absoluteFilePath(goodFolderList.at(i))); //Qdir for storing actual subdirectory.

           QStringList filelist = subdirectory.entryList(nameFilter); //File list of .bin files in the actual subdirectory.
           images_temp.reserve(filelist.size()); //images_temp for reading images from one file. MAY NOT BE USED IN THE FUTURE.
           images_temp.clear();

           if(filelist.size() == 0)
           {
               std::cout<<"Warning: No .bin file in subfolder" << directory.absoluteFilePath(goodFolderList.at(i)).toStdString() <<std::endl;
               continue;
           }
           double meanIntensity =0.0f;
           double meanExptime = 0.0f;



           //Opening every file in the given subdirectory


           for(int j=0;j<filelist.length();j++)
           //Note: subdirectory is indexed by i, files are indexed by j.
           {
               //std::cout << "Processing file " << subdirectory.absoluteFilePath(filelist.at(j)).toStdString() << std::endl;
               image.readfromfile(subdirectory.absoluteFilePath(filelist.at(j)).toStdString());
               image.copy_to_GPU();
               image.calculate_meanvalue_on_GPU();


     //Note: images with 0 voltage or amperage (or other parameters) are ignored. X-ray tube was probably off...
               if      (
                       ((!(image.getvoltage() > 0) && !(image.getvoltage() <0))
                       || (!(image.getamperage() > 0) && !(image.getamperage() <0 ) ) )
                       && image.getexptime() > 1 &&  image.getmean() > 1
                       )
                   {

                   images_temp.push_back(image); //loading images from one subdir to images_temp vecor.

                   meanExptime += ((images_temp.back().getexptime()));
                   meanIntensity += ((images_temp.back().getmean()) );
                   }
           } //end of for( every image in current subfolder)

           //if there is any non-blank images, calculate mean values!
           if(images_temp.size() > 0)
           {
               meanExptime/=images_temp.size();
               meanIntensity/=images_temp.size();
           }
           else
           {
               std::cout << "There is not enough good images at  folder " << goodFolderList.at(i).toStdString() << std::endl;
               continue;
           }

           image.clear(); // I'll sum the good images to this variable.
           int count = 0; //counts good images in a subfolder.

           //Ignoring images that differs more than 10 percent from the mean. Recalculating mean values.
               for(int k=0; k <  images_temp.size(); k++)
               {
                   //Checking for every image if they parameters are near the mean of the parameters.
                   //(10% difference is allowed.)
                   //DEBUG: Is that OK?
                   if     (
                        abs(images_temp.at(k).getmean() - meanIntensity) > meanIntensity * 0.1f ||
                        abs(images_temp.at(k).getexptime() - meanExptime) > meanExptime * 0.1f
                           )
                   { //if the image is corrupted, ignore it. Also put som info to the console.
                       std::cout << "Bad image: " << subdirectory.absolutePath().toStdString()
                                 <<"id:" << images_temp.at(k).getid() <<std::endl;
                       std::cout << "meanIntenstiy = " << meanIntensity << "getmean =" <<images_temp.at(k).getmean()
                                 << " getvoltage =" <<images_temp.at(k).getvoltage()
                                 << "getamperage = " << images_temp.at(k).getamperage()
                               <<"meanExptime = " << meanExptime << "getexptime = " << images_temp.at(k).getexptime()
                             <<std::endl<<std::endl;

                   }
                   else // image is good.
                       {

                       count +=1;

                        image+=images_temp.at(k); //Summing images at the image variable.

                       }
               }



               //Image processing finished in the current subdir. If there was some good image in that folder, add the averaged image to the averaged images' vector.
               if( count > 1)
                   {
                   //Dividing image parameters and pixel values by count
                   //(count is the total number of good images in the subdirectory.)

                   image/=count;
                   image.setamperage(1.0f);








                       im_container.add(image);






                   std::cout <<"Images loaded from directory " << subdirectory.absolutePath().toStdString()
                            <<"With mean: " << image.getmean()
                           << "voltage: "  <<image.getvoltage()
                           <<" amperage: " << image.getamperage()
                          <<"exptime: " << image.getexptime()<<std::endl;
                   }
               else //(if count <=1)
                  {
                   std::cout <<"Not enough good images in directory" <<subdirectory.absolutePath().toStdString()<<std::endl;
                   std::cout << "meanIntensity = "<< meanIntensity <<std::endl;
               }
               } // end of loading files.


               //----------------------------------------------------------------------------
               //
               //                      CALCULATING AND SAVING DATA
               //
               //----------------------------------------------------------------------------

               Image_cuda_compatible slope;
               Image_cuda_compatible intercept;

               im_container.calculate(slope,intercept);



               std::string slopefile = gcfolder + "/" + "offsetslope"  + ".binf";
               std::string interceptfile = gcfolder + "/" + "offsetintercept"  + ".binf";
               slope.writetofloatfile(slopefile);
               slope.clear();
               intercept.writetofloatfile(interceptfile);
               intercept.clear();

              slopes[0] =slope;
              intercepts[0] = intercept;
}



//! Reads images to calculate gain correction data.

//! The functions asks for an input folder and an output folder. In the input folder,
//! images sould be in subfolders.
//! Every subfolder sould contain one ore more image with the same settings
//! (Voltage, Exp time, amperage), with an info file.
//! Images are only loaded from the subfolders of the user given directory.
//! Images are stored in the gc_im_container class, then gain correction data are calculated
//! by linear fitting to offset corrected pixel values in the function of the product of
//! exposition time and amperage.


void Gaincorr::readAndCalculateGain()
{
    if(slopes.find(0) == slopes.end() || intercepts.find(0)==intercepts.end())
    {
        std::cout <<"ERROR: No offset calibration data."
                 <<" Load or calculate offset calibration data first."
                <<std::endl;
        return;
    }


    // ------------------------------------------------------------
    //
    //               Looking for valid subfolders
    //
    //--------------------------------------------------------------

    std::map<int, QStringList> goodFolderMap;



    //Asking for input and output  directories.
       QString dir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, which contains folders of images to calculate gain correction data"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);

       QString Qgcfolder = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to save gain correction factors (slope and intercept)"),
                                                       "C:\\",
                                                        QFileDialog::DontResolveSymlinks);
       gcfolder = Qgcfolder.toStdString();

    QDir directory(dir);
    QString path = directory.absolutePath();
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();
    for(int i=0; i<subdirs.size(); i++) // for every subfolder
    {

        std::ifstream myfile;
        myfile.open(path.toStdString() + "/" +subdirs.at(i).toStdString() +"/info.txt" );
        if (! (myfile.is_open()))
        {
            std::cout<< "Warning: There is no info.txt file in folder "
                     << path.toStdString() + "/" +subdirs.at(i).toStdString()
                     << ", or info file is not readable."
                     <<std::endl;

            continue; // jump to the next subfolder.
        }
        std::string line;
        //reading header
        do
        {
            getline (myfile,line);


        } while(line.find("*****************" ) == std::string::npos && !(myfile.eof()) );



        if(myfile.eof())
        {
            std::cout << "WARNING There is no info in the info file in folder "
                      << path.toStdString() + "/" +subdirs.at(i).toStdString()
                      <<"." << std::endl;
            continue;
        }


        //reading infos
        std::vector <int> voltagesInThisFolder;
        int count = 0;

        while(getline (myfile,line) && line.length() > 2)
        {

            std::stringstream ss(line);
            std::string temp;
            int thisVoltage =0;
            int thisAmperage =0;
            int thisExptime=0;

            ss >> temp; // id
            ss >> temp; // rotation
            ss >> thisVoltage;
            ss >> thisAmperage;
            ss >> thisExptime;


                //at gain orrection we would like to avoid pictures with 0 voltage or amperage or exptime

                if(thisVoltage >0 && thisAmperage >0  && thisExptime >0 )
                {
                    voltagesInThisFolder.push_back(thisVoltage);
                    count++;
                }


        } //every line is read.

        myfile.close();




            if(count > 2)
            {
                float meanVoltageInThisFolder = 0;
                //Calculate mean voltage
                for(std::vector<int>::iterator iter = voltagesInThisFolder.begin();
                        iter != voltagesInThisFolder.end();
                        iter++)
                {
                    meanVoltageInThisFolder += *iter;
                }
                meanVoltageInThisFolder /= voltagesInThisFolder.size();

                //Remove outliers.

                for(std::vector<int>::iterator iter = voltagesInThisFolder.begin();
                        iter != voltagesInThisFolder.end();)
                {
                    if( abs(*iter -meanVoltageInThisFolder ) > meanVoltageInThisFolder*0.1f)
                    {
                        iter=voltagesInThisFolder.erase(iter);
                    }
                    else
                    {
                        ++iter;
                    }
                }

                //Recalculate mean voltage;
                meanVoltageInThisFolder =0.0f;
                if( voltagesInThisFolder.size() > 2)
                {

                    for(std::vector<int>::iterator iter = voltagesInThisFolder.begin();
                            iter != voltagesInThisFolder.end();
                            iter++)
                    {
                        meanVoltageInThisFolder += *iter;
                    }
                    meanVoltageInThisFolder /= voltagesInThisFolder.size();


                    //Rounding meanVoltageInThisFolder to multiply of 5

                    int meanVoltageInteger = round(meanVoltageInThisFolder);
                    int remainder  = meanVoltageInteger %5;

                    if(remainder != 0)

                    {
                        meanVoltageInteger = meanVoltageInteger + 5 - remainder;
                    }
                    goodFolderMap.insert(std::pair<int, QStringList> (meanVoltageInteger, QStringList()));
                    goodFolderMap[meanVoltageInteger].append(path + "/" + subdirs.at(i));
                }
                else
                {
                    std::cout << "Not enoguh good images in folder " << subdirs.at(i).toStdString()
                              <<"(High deviation in values.)" << std::endl;
                }
            } // end of if (count >2)
            else
            {
                std::cout << "Not enoguh good images in folder " << subdirs.at(i).toStdString()
                          <<"(Too few images in the folder.)" << std::endl;
            }
    } //every subdir is processed.


        for(std::map<int, QStringList>::iterator iter = goodFolderMap.begin();
                iter != goodFolderMap.end();)
        {
            if(iter->second.size() < 8)
            {
                std::cout <<"Not enough subfolders with voltage" << iter->first<<std::endl;
                goodFolderMap.erase(iter++);
            }
            else
            {
                ++iter;
            }
        }


    //Map is ready.

    //-------------------------------------------------------
    //
    //                Loading files
    //
    //-------------------------------------------------------




 for( std::map<int,QStringList>::iterator iter = goodFolderMap.begin();
      iter != goodFolderMap.end();
      ++iter)
 {
     if(iter->second.size()  < 5)
     {
         std::cout << "Too few subfolders with good images at voltage " << iter->first << std::endl;
         continue;
     }
     gc_im_container  im_container;

     im_container.inicialize(iter->second.size());


     for(int i = 0; i< iter->second.size(); i++)
     {
         std::cout << "Loading files from " << iter->second.at(i).toStdString() << std::endl;


         //Looking for .bin files
        QStringList nameFilter("*.bin"); //name filter.
        QDir subdirectory(directory.absoluteFilePath(iter->second.at(i))); //Qdir for storing actual subdirectory.

        QStringList filelist = subdirectory.entryList(nameFilter); //File list of .bin files in the actual subdirectory.
        images_temp.reserve(filelist.size()); //images_temp for reading images from one file. MAY NOT BE USED IN THE FUTURE.
        images_temp.clear();

        if(filelist.size() == 0)
        {
            std::cout<<"Warning: No .bin file in subfolder" << directory.absoluteFilePath(iter->second.at(i)).toStdString() <<std::endl;
            continue;
        }
        double meanIntensity =0.0f;
        double meanExptime = 0.0f;
        double meanAmperage = 0.0f;



        //Opening every file in the given subdirectory


        for(int j=0;j<filelist.length();j++)
        //Note: subdirectory is indexed by i, files are indexed by j.
        {
            //std::cout << "Processing file " << subdirectory.absoluteFilePath(filelist.at(j)).toStdString() << std::endl;
            image.readfromfile(subdirectory.absoluteFilePath(filelist.at(j)).toStdString());
            image.copy_to_GPU();
            image.calculate_meanvalue_on_GPU();


  //Note: images with 0 voltage or amperage (or other parameters) are ignored. X-ray tube was probably off...
            if( abs(image.getvoltage() - iter->first) < iter->first * 0.1f && image.getexptime() > 1 && image.getmean() > 1 && image.getamperage() > 1)
                {

                images_temp.push_back(image); //loading images from one subdir to images_temp vecor.

                meanAmperage += ((images_temp.back().getamperage()));
                meanExptime += ((images_temp.back().getexptime()));
                meanIntensity += ((images_temp.back().getmean()) );
                }
        } //end of for( every image in current subfolder)

        //if there is any non-blank images, calculate mean values!
        if(images_temp.size() > 0)
        {
            meanAmperage/=images_temp.size();
            meanExptime/=images_temp.size();
            meanIntensity/=images_temp.size();
        }
        else
        {
            std::cout << "There is not enough good images at  folder " << iter->second.at(i).toStdString() << std::endl;
            continue;
        }

        image.clear(); // I'll sum the good images to this variable.
        int count = 0; //counts good images in a subfolder.

        //Ignoring images that differs more than 10 percent from the mean. Recalculating mean values.
            for(int k=0; k <  images_temp.size(); k++)
            {
                //Checking for every image if they parameters are near the mean of the parameters.
                //(10% difference is allowed.)
                //DEBUG: Is that OK?
                if     (
                     abs(images_temp.at(k).getmean() - meanIntensity) > meanIntensity * 0.1f ||
                     abs(images_temp.at(k).getamperage() - meanAmperage) > meanAmperage * 0.1f ||
                     abs(images_temp.at(k).getexptime() - meanExptime) > meanExptime * 0.1f
                        )
                { //if the image is corrupted, ignore it. Also put som info to the console.
                    std::cout << "Bad image: " << subdirectory.absolutePath().toStdString()
                              <<"id:" << images_temp.at(k).getid() <<std::endl;
                    std::cout << "meanIntenstiy = " << meanIntensity << "getmean =" <<images_temp.at(k).getmean()
                              <<"Voltage = " << iter->first << " getvoltage =" <<images_temp.at(k).getvoltage()
                             <<"meanAmperage = " << meanAmperage << "getamperage = " << images_temp.at(k).getamperage()
                            <<"meanExptime = " << meanExptime << "getexptime = " << images_temp.at(k).getexptime()
                          <<std::endl<<std::endl;

                }
                else // image is good.
                    {

                    count +=1;

                     image+=images_temp.at(k); //Summing images at the image variable.

                    }
            }



            //Image processing finished in the current subdir. If there was some good image in that folder, add the averaged image to the averaged images' vector.
            if( count > 1)
                {
                //Dividing image parameters and pixel values by count
                //(count is the total number of good images in the subdirectory.)

                image/=count;


                //Searching for saturation exposition.
                if(image.maxintensity() > 16382)  // saturation at 16383

                {
                    int satvalue = (int) round((image.getexptime() * image.getamperage()));
                    saturation.insert(std::pair<int,int> (iter->first, satvalue ));
                    if( (int) ((saturation.find(iter->first))->second) > satvalue)
                    {
                        saturation[iter->first] = satvalue;
                    }
                }



                offsetcorrigateimage(image);




                    im_container.add(image);
                    //DEBUG

                    std::stringstream ss;
                    std::string v,a,e;
                    ss << image.getexptime();
                    e = ss.str();
                    ss.clear();
                    ss<<image.getvoltage();
                    v = ss.str();
                    ss.clear();
                    ss<<image.getamperage();
                    a = ss.str();
                    ss.clear();

                  // image.writetofloatfile("C:\\awing\\gaintemp\\" + v + "_" + a + "_" + e  + ".binf");



                std::cout <<"Images loaded from directory " << subdirectory.absolutePath().toStdString()
                         <<"With mean: " << image.getmean()
                        << "voltage: "  <<image.getvoltage()
                        <<" amperage: " << image.getamperage()
                       <<"exptime: " << image.getexptime()<<std::endl;
                }
            else //(if count <=1)
               {
                std::cout <<"Not enough good images in directory" <<subdirectory.absolutePath().toStdString()<<std::endl;
                std::cout << "meanIntensity = "<< meanIntensity <<std::endl;
            }






     } // end of for(every subdir at given voltage




     //----------------------------------------------------------------------------
     //
     //                      CALCULATING
     //
     //----------------------------------------------------------------------------

     Image_cuda_compatible slope;
     Image_cuda_compatible intercept;

     im_container.calculate(slope,intercept);

     std::ostringstream oss;
     oss << iter->first; // voltage

     std::string ending = oss.str();
     std::string slopefile = gcfolder + "/" + "slope" + ending + ".binf";
     std::string interceptfile = gcfolder + "/" + "intercept" + ending + ".binf";
     slope.writetofloatfile(slopefile);
     slope.clear();
     intercept.writetofloatfile(interceptfile);
     intercept.clear();

    slopes[iter->first] =slope;
    intercepts[iter->first] = intercept;



 } // end of for( iterate through voltages)

image.clear();
images_temp.clear();


//Write saturation to file and final check:

if(saturation.empty() )
{
    std::cout << "ERROR: There is no saturation data. (I do not know what failed.)"
              << "gain correction calculation is unsuccessfull."
              <<std::endl;
    slopes.clear();
    intercepts.clear();
    return;
}

for(std::map<int, int>::iterator iter = saturation.begin();
        iter!= saturation.end();)
{
    int voltage = iter->first;
    if( slopes.find(voltage) == slopes.end() ||
            intercepts.find(voltage) == intercepts.end())
    {
        std::cout <<"ERROR: NO saturation data for voltage " << voltage
                 <<" (I do not know what failed.)" <<std::endl
                <<"Slope and saturation data for gain correction at this voltage are erased."
               <<std::endl;

        saturation.erase(iter++);
        if(slopes.find(voltage) != slopes.end())
        {
            slopes.erase(slopes.find(voltage));
        }
        if(intercepts.find(voltage) != intercepts.end())
        {
            intercepts.erase(intercepts.find(voltage));
        }

    }
    else
    {
        ++iter;
    }
}

//write slope data to file
std::ofstream myfile ( gcfolder + "/saturation.txt", std::ios_base::trunc);
if(!(myfile.is_open()))
{
    std::cout <<"Warnig! text file to write slope data could not be opened at "
             <<  gcfolder + "/saturation.txt" <<std::endl;
    return;
}

for(std::map<int,int>::iterator iter=saturation.begin();
    iter!=saturation.end();
    ++iter)
{
    myfile <<iter->first <<'\n';
    myfile << iter->second <<'\n';
}
myfile.close();

} // end of readimages() function.


//! Reads offset correction data.

//!  Reads slope and intercept files for offset correction. Asks the user to choose a folder.
//! In this folder there should be the files containing slope and intercept data for offset correction.
//! Name format: offsetintercept.binf and offsetslope.binf.
void Gaincorr::readoffsetfactors()
{
    //Removes old factors.
    if(slopes.find(0) != slopes.end())
    {
        slopes.erase(slopes.find(0));
    }
    if(intercepts.find(0) != intercepts.end())
    {
        intercepts.erase(intercepts.find(0));
    }

    QString Qgcfolder = QFileDialog::getExistingDirectory(0, QString::fromStdString("Folder, to load gain correction factors from (slope and intercept)"),
                                                    "C:\\",
                                                     QFileDialog::ReadOnly );
    gcfolder = Qgcfolder.toStdString();

    QFileInfo checkslope( QString::fromStdString(gcfolder + "/" + "offsetslope"  + ".binf"));
    QFileInfo checkintercept( QString::fromStdString(gcfolder + "/" + "offsetintercept"  + ".binf"));
    Image_cuda_compatible slope;
    Image_cuda_compatible intercept;

    if(checkslope.exists() && checkslope.isFile())
    {
        slope.readfromfloatfile(gcfolder + "/" + "offsetslope"  + ".binf");
    }
    else
    {
        std::cout << "ERROR! offsetslope.binf file not found in folder" << gcfolder
                  <<std::endl;
    }
    if(checkintercept.exists() && checkintercept.isFile())
    {
        intercept.readfromfloatfile(gcfolder + "/" + "offsetintercept"  + ".binf");
    }
    else
    {
        std::cout << "ERROR! offsetintercept.binf not found in folder " << gcfolder
                  <<std::endl;
    }
    intercept.calculate_meanvalue_on_GPU();
    slope.calculate_meanvalue_on_GPU();
    if(intercept.getmean() > 0 && slope.getmean() > 0)
    {
        slopes[0] = slope;
        intercepts[0] = intercept;
        std::cout <<"Offset data read succesfull." << std::endl;
    }
    else
    {
        std::cout << "ERROR : There were errors while reading offset data."
                  <<std::endl;
    }



}


//! Reads factors for gain correction from files.

//!  Reads slope and intercept files for gain correction. Asks the user to choose a folder.
//! In this folder there should be the files containing slope and intercept data for gain correction.
//! Name format: intercept<voltage>.binf and slope<voltage>.binf. Reads every file with the given syntax,
//! and stores them, if both an intercept and a slope file is there for a given voltage.




void Gaincorr::readgainfactors()
{
    saturation.clear();
    for(std::map<int, Image_cuda_compatible>::iterator iter = slopes.begin();
        iter != slopes.end();)
    {
        if(iter -> first != 0)
        {
            slopes.erase(iter++);
        }
        else
        {
            ++iter;
        }
    }




        for(std::map<int, Image_cuda_compatible>::iterator iter = intercepts.begin();
            iter != intercepts.end();)
        {
            if(iter -> first != 0)
            {
                intercepts.erase(iter++);
            }
            else
            {
                ++iter;
            }
        }



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
           Image_cuda_compatible slope;
           Image_cuda_compatible intercept;

           std::string filename = gcfolder +"/intercept" + thisvoltage + ".binf" ;

           intercept.readfromfloatfile(filename);
           intercept.calculate_meanvalue_on_CPU();

           filename = gcfolder + "/slope" + thisvoltage + ".binf" ;
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


   std::ifstream myfile( gcfolder + "/saturation.txt");
   if(! (myfile.is_open()))
   {
       slopes.clear();
       intercepts.clear();
       slopes.clear();
       std::cout << "ERROR! Slope sata file is not readable at "
                 <<gcfolder + "/saturation.txt" <<std::endl
                <<"Gain correction data read is unsucesfull." <<std::endl;
       return;
   }
   std::string line;
   do
   {
       if( (getline(myfile,line)))
       {

           int voltage = std::atoi(line.c_str());
           getline(myfile,line);
           int sat = std::atoi(line.c_str());
           if(voltage > 0 && sat > 0)
           {
               saturation.insert(std::pair<int,int>(voltage,sat));
           }
           else
           {
               std::cout << "WARNING! saturation file is corrupted. (Invalid data)." << std::endl;
           }
       }
   } while( !(myfile.eof()));

   //final check:
    for(std::map<int,int>::iterator iter = saturation.begin();
        iter!=saturation.end();)
    {
        int voltage = iter -> first;
        if(slopes.find(voltage) == slopes.end() || intercepts.find(voltage) == intercepts.end() )
        {
            std::cout << "WARNING: no slope or intercept for voltage " << voltage
            << " with valid saturation data." <<std::endl;
            saturation.erase(iter++);
        }
        else
        {
            ++iter;
        }
    }


    for(std::map<int,Image_cuda_compatible>::iterator iter = slopes.begin();
            iter != slopes.end();)
    {
        int voltage = iter->first;
        if(voltage !=0)
        {


            if(saturation.find(voltage ) == saturation.end() ||
                    intercepts.find(voltage) == intercepts.end())
            {
                std::cout << "WARNING: no saturation or intercept for voltage " << voltage
                << " with valid slope data." <<std::endl;
                slopes.erase(iter++);
            }
            else
            {
                ++iter;
            }
        }
        else
        {
            ++iter;
        }
    }


    for(std::map<int,Image_cuda_compatible>::iterator iter = intercepts.begin();
            iter != intercepts.end();)
    {
        int voltage = iter->first;
        if(voltage !=0)
        {

            if(saturation.find(voltage ) == saturation.end() ||
                    slopes.find(voltage) == slopes.end())
            {
                std::cout << "WARNING: no saturation or slope for voltage " << voltage
                << " with valid intercept data." <<std::endl;
                intercepts.erase(iter++);
            }
            else
            {
                ++iter;
            }
        }
        else
        {
            ++iter;
        }
    }

    std::cout<<"Readed gain correction data for voltages:" <<std::endl;
   for( std::map<int,int>::iterator iter = saturation.begin();
                iter != saturation.end();
                ++iter)
   {
       std::cout << iter->first << std::endl;
   }








} //end of void readfactors()

//! Offset corrigates the given image.

//! Image is offset corrigated and owerwritten in the memory.



void Gaincorr::offsetcorrigateimage(Image_cuda_compatible &image)

{
    float expTime  = image.getexptime();
    if( !(expTime > 0))
    {

            std::cout << "ERROR : There is no info or info is invalid for image"<<std::endl
                      <<"Id: " << image.getid() << std::endl
                      <<"Voltage: " << image.getvoltage() << std::endl
                      <<"Amperage:" << image.getamperage()<<std::endl
                      <<"filename: " << image.getfilename() << std::endl;
            std::cout<<"I can't offse corrigat that." <<std::endl <<std::endl;
            return;
    }
    if(slopes.find(0)== slopes.end() || intercepts.find(0) == intercepts.end()) // do not contain.
    {
        std::cout << "ERROR: There is no data to do offset corrigation." << std::endl;
        return;
    }
    if( !(slopes[0].getmean() > 0) || !(intercepts[0].getmean() > 0) )
    {
        std::cout << "ERROR: offset correction picture is empty. (Probably not read?)"
                  <<std::endl;
        return;
    }

    image.copy_to_GPU();
    image.remove_from_CPU();
    slopes[0].copy_to_GPU();
    intercepts[0].copy_to_GPU();

    //could write operator*, but it would not be faster ...
    Image_cuda_compatible slopeTimesExp = slopes[0];
    slopeTimesExp *= image.getexptime();
    image-=intercepts[0];
    image-=slopeTimesExp;
}






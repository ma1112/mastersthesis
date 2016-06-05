#include "directorystructureconverter.h"
#include <QApplication>
#include <QMessageBox>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::stringbuf
#include <QDir>
#include <QString>
#include <QStringList>
#include <QFile>
#include <QFileDialog>
#include <fstream>
#include <QTextStream>
#include "image_cuda_compatible.h"

DirectoryStructureConverter::DirectoryStructureConverter()
{

}


void DirectoryStructureConverter::copyAll()
{
    // ------------------------------------------------------------
    //
    //               Looking for valid subfolders
    //
    //--------------------------------------------------------------

     QStringList goodFolderList;



     //Asking for input and output  directories.
     QString inputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select folder, which contains folders of images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);

     QString outputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select Folder, to save all images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);
     //Checking if output folder is empty:
     if(QDir(outputDir).entryInfoList(QDir::NoDotAndDotDot|QDir::AllEntries).count() >= 0)
     {
         QMessageBox::StandardButton reply;
         reply = QMessageBox::question(0, "Continue?" , "This operation will delete all images and info file from the non empty output directory " + outputDir
                                       );
         if( reply == QMessageBox::No)
         {
             return;
         }

         QDir dir(outputDir);
         dir.setNameFilters(QStringList() << "*.bin" << "*.binf" << "info.txt");
         dir.setFilter(QDir::Files);
         foreach(QString dirFile, dir.entryList())
         {
             dir.remove(dirFile);
         }
     }

     // copying image files.

     long id = 1;

    QDir directory(inputDir);
    QString path = directory.absolutePath();
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();
    for(int i=0; i<subdirs.size(); i++) //for every subdirectory
    {
        //Reading info files to determine number of valid folders.

        std::ifstream inputInfoFile;
        inputInfoFile.open((path.toStdString() + "/" +subdirs.at(i).toStdString() +"/info.txt").c_str() );
        if (! (inputInfoFile.is_open()))
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
            getline (inputInfoFile,line);


        } while(line.find("*****************" ) == std::string::npos && !(inputInfoFile.eof()) );



        if(inputInfoFile.eof())
        {
            std::cout << "WARNING There is no info in the info file in folder "
                      << path.toStdString() + "/" +subdirs.at(i).toStdString()
                      <<"." << std::endl;
            inputInfoFile.close();
            continue;
        }


        //reading info
        while(getline (inputInfoFile,line) && line.length() > 2)
        {

            std::stringstream ss(line);
            std::string temp;
            int currentId =-1;
            ss >> currentId; // id
            // check if file exists:
            QString thisFilename =  path + "/" +subdirs.at(i) + "/" +  QString("%1").arg(currentId,4,10,QChar('0')) + ".bin";
            QFile thisImageFile(thisFilename);
            if(!thisImageFile.exists())
            {
                std::cout << "Warning: could not find image for info at " << thisFilename.toStdString() << std::endl;
                continue;
            }
            else
            {
                QFile outputInfoFile(outputDir+ "/info.txt");
                if(outputInfoFile.open(QIODevice::Append))
                {
                    QFile::copy(thisFilename, outputDir + "/" +  QString("%1").arg(id,4,10,QChar('0')) + ".bin");
                    QTextStream fileStream( &outputInfoFile );
                    fileStream << id << " " << QString::fromStdString(ss.str()) << "\n";
                    outputInfoFile.close();
                    id++;
                }
                else
                {
                    std::cout << "ERROR: could not open outout info file for writing. at " << outputDir.toStdString() << "/info.txt" << std::endl;
                    return;
                }
            }
        } //every line is read.

        inputInfoFile.close();
    } //every subdir is processed.
}

//! Copies images from subfolders to one folder.
//! Images from a single subfolder are averaged and bads-are-ignored.
// Hard code copy. Sorry.
void DirectoryStructureConverter::copyDirAsImage()
{
    // ------------------------------------------------------------
    //
    //               Looking for valid subfolders
    //
    //--------------------------------------------------------------




     //Asking for input and output  directories.
     QString inputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select folder, which contains folders of images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);

     QString outputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select Folder, to save all images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);
     //Checking if output folder is empty:
     if(QDir(outputDir).entryInfoList(QDir::NoDotAndDotDot|QDir::AllEntries).count() >= 0)
     {
         QMessageBox::StandardButton reply;
         reply = QMessageBox::question(0, "Continue?" , "This operation will delete all images and info file from the non empty output directory " + outputDir
                                       );
         if( reply == QMessageBox::No)
         {
             return;
         }

         QDir dir(outputDir);
         dir.setNameFilters(QStringList() << "*.bin" << "*.binf" << "info.txt");
         dir.setFilter(QDir::Files);
         foreach(QString dirFile, dir.entryList())
         {
             dir.remove(dirFile);
         }
     }

     // copying image files.

     long id = 1;

    QDir directory(inputDir);
    QString path = directory.absolutePath();
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();
    for(int i=0; i<subdirs.size(); i++) //for every subdirectory
    {
        //Reading info files to determine number of valid folders.

        std::ifstream inputInfoFile;
        inputInfoFile.open((path.toStdString() + "/" +subdirs.at(i).toStdString() +"/info.txt").c_str() );
        if (! (inputInfoFile.is_open()))
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
            getline (inputInfoFile,line);


        } while(line.find("*****************" ) == std::string::npos && !(inputInfoFile.eof()) );



        if(inputInfoFile.eof())
        {
            std::cout << "WARNING There is no info in the info file in folder "
                      << path.toStdString() + "/" +subdirs.at(i).toStdString()
                      <<"." << std::endl;
            inputInfoFile.close();
            continue;
        }


        Image_cuda_compatible imageInThisFolder;
        imageInThisFolder.reserve_on_GPU();
        int validImagesInThisFolder =0;

        //reading info
        while(getline (inputInfoFile,line) && line.length() > 2)
        {


            std::stringstream ss(line);
            int currentId =-1;
            ss >> currentId; // id
            // check if file exists:
            QString thisFilename =  path + "/" +subdirs.at(i) + "/" +  QString("%1").arg(currentId,4,10,QChar('0')) + ".bin";
            QFile thisImageFile(thisFilename);
            if(!thisImageFile.exists())
            {
                std::cout << "Warning: could not find image for info at " << thisFilename.toStdString() << std::endl;
                continue;
            }
            else
            {
                Image_cuda_compatible thisImage;
                thisImage.readfromfile(thisFilename.toStdString());
               imageInThisFolder += thisImage;
               validImagesInThisFolder++;

            }
        } //every line is read.

        inputInfoFile.close();

        QFile outputInfoFile(outputDir+ "/info.txt");
        if(outputInfoFile.open(QIODevice::Append))
        {
            QString outputImageFileName = outputDir + "/" +  QString("%1").arg(id,4,10,QChar('0')) + ".bin";
            imageInThisFolder/=validImagesInThisFolder;
            imageInThisFolder.writetofile(outputImageFileName.toStdString());

            QTextStream fileStream( &outputInfoFile );
            fileStream << id << " 0 " << imageInThisFolder.getvoltage() << " " << imageInThisFolder.getamperage() << " " << imageInThisFolder.getexptime() <<  " 9 10 0 15 15 0 1 1 1\n" ;
            outputInfoFile.close();
            id++;
        }
        else
        {
            std::cout << "ERROR: could not open outout info file for writing. at " << outputDir.toStdString() << "/info.txt" << std::endl;
            return;
        }
    } //every subdir is processed.
}






/*


//! Copies images from subfolders to one folder. As is.
// Hard code copy. Sorry.
void DirectoryStructureConverter::copyDirAsIs()
{
    // ------------------------------------------------------------
    //
    //               Looking for valid subfolders
    //
    //--------------------------------------------------------------




     //Asking for input and output  directories.
     QString inputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select folder, which contains folders of images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);

     QString outputDir = QFileDialog::getExistingDirectory(0, QString::fromStdString("Select Folder, to save all images."),
                                                          "C:\\",
                                                          QFileDialog::DontResolveSymlinks);
     //Checking if output folder is empty:
     if(QDir(outputDir).entryInfoList(QDir::NoDotAndDotDot|QDir::AllEntries).count() >= 0)
     {
         QMessageBox::StandardButton reply;
         reply = QMessageBox::question(0, "Continue?" , "This operation will delete all images and info file from the non empty output directory " + outputDir
                                       );
         if( reply == QMessageBox::No)
         {
             return;
         }

         QDir dir(outputDir);
         dir.setNameFilters(QStringList() << "*.bin" << "*.binf" << "info.txt");
         dir.setFilter(QDir::Files);
         foreach(QString dirFile, dir.entryList())
         {
             dir.remove(dirFile);
         }
     }

     // copying image files.

     long id = 1;

    QDir directory(inputDir);
    QString path = directory.absolutePath();
    directory.setFilter(QDir::AllDirs | QDir::NoDotAndDotDot);
    QStringList subdirs  = directory.entryList();
    for(int i=0; i<subdirs.size(); i++) //for every subdirectory
    {
        //Reading info files to determine number of valid folders.

        std::ifstream inputInfoFile;
        inputInfoFile.open((path.toStdString() + "/" +subdirs.at(i).toStdString() +"/info.txt").c_str() );
        if (! (inputInfoFile.is_open()))
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
            getline (inputInfoFile,line);


        } while(line.find("*****************" ) == std::string::npos && !(inputInfoFile.eof()) );



        if(inputInfoFile.eof())
        {
            std::cout << "WARNING There is no info in the info file in folder "
                      << path.toStdString() + "/" +subdirs.at(i).toStdString()
                      <<"." << std::endl;
            inputInfoFile.close();
            continue;
        }


        Image_cuda_compatible imageInThisFolder;
        imageInThisFolder.reserve_on_GPU();
        int validImagesInThisFolder =0;

        //reading info
        while(getline (inputInfoFile,line) && line.length() > 2)
        {


            std::stringstream ss(line);
            int currentId =-1;
            ss >> currentId; // id
            // check if file exists:
            QString thisFilename =  path + "/" +subdirs.at(i) + "/" +  QString("%1").arg(currentId,4,10,QChar('0')) + ".bin";
            QFile thisImageFile(thisFilename);
            if(!thisImageFile.exists())
            {
                std::cout << "Warning: could not find image for info at " << thisFilename.toStdString() << std::endl;
                continue;
            }
            else
            {
                Image_cuda_compatible thisImage;
                thisImage.readfromfile(thisFilename.toStdString());
                QFile outputInfoFile(outputDir+ "/info.txt");
                if(outputInfoFile.open(QIODevice::Append))
                {
                    QString outputImageFileName = outputDir + "/" +  QString("%1").arg(id,4,10,QChar('0')) + ".bin";
                    thisImage.writetofile(outputImageFileName.toStdString());

                    QTextStream fileStream( &outputInfoFile );
                    fileStream << id << " 0 " << imageInThisFolder.getvoltage() << " " << imageInThisFolder.getamperage() << " " << imageInThisFolder.getexptime() <<  " 9 10 0 15 15 0 1 1 1\n" ;
                    outputInfoFile.close();
                    id++;
                }
                else
                {
                    std::cout << "ERROR: could not open outout info file for writing. at " << outputDir.toStdString() << "/info.txt" << std::endl;
                    return;
                }

            }
        } //every line is read.

        inputInfoFile.close();


    } //every subdir is processed.
}
*/

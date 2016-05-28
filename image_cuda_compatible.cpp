#include "image_cuda_compatible.h"

#include <QTime>
#include <QFileInfo>
#include <QImage>
#include<QPixmap>
#include<QString>
#include<iostream>
#include<fstream>
#include<sstream>


//! Constructor.
Image_cuda_compatible::Image_cuda_compatible()  {
initialize();
}

//! Destructor.
Image_cuda_compatible::~Image_cuda_compatible() {
    //std::cout << "DESTROYING "<<filename<<std::endl;
    remove_from_GPU(); }




//! Copy Constructor.

Image_cuda_compatible::Image_cuda_compatible(const Image_cuda_compatible& other)
{



    min = other.min;
    max = other.max;


    voltage = other.voltage;
    amperage= other.amperage;
    exptime = other.exptime;
     mean=other.mean;
     stdev = other.stdev;
     filename = other.filename;
     directory = other.directory;
     id = other.id;

     gpu_im = NULL;


     if(other.gpu_im != NULL)
     {
         reserve_on_GPU();
         copy_GPU_image(other.gpu_im);
     }


     return;
}


Image_cuda_compatible& Image_cuda_compatible::operator=(const Image_cuda_compatible& other)
 {


    if(this != &other)
    {
     voltage = other.voltage;
     amperage= other.amperage;
     exptime = other.exptime;
      mean=other.mean;
      stdev = other.stdev;

         filename = other.filename;
         directory = other.directory;

          id = other.id;
          min = other.min;
          max = other.max;




          if(other.gpu_im != NULL)
          {
              reserve_on_GPU();
              copy_GPU_image(other.gpu_im);
          }
          else
          {
              gpu_im = NULL;
          }

     }
    return *this;


 }

//! Operator for addition.
//! Adds another image's attributes to this. Also adds pixel values on the GPU.
 Image_cuda_compatible&  Image_cuda_compatible::operator+=(Image_cuda_compatible &other)
 {
     reserve_on_GPU();
     if(other.gpu_im == NULL)
     {
         std::cout << "Warning: adding empty image " << other.getfilename() << " to image " << this->getfilename() << std::endl;
         return *this;
     }

    mean+=0.0f;
    stdev = 0.0f;
    voltage +=other.voltage;
    amperage +=other.amperage;
    exptime +=other.exptime;
    add_on_GPU(other);

    return *this;
}


 //! Subtracts another image's attributes to this. Also adds pixel values on the GPU.
  Image_cuda_compatible&  Image_cuda_compatible::operator-=(Image_cuda_compatible &other)
  {



     mean-=other.mean;
     stdev = 0.0f;
     voltage -=other.voltage;
     amperage -=other.amperage;
     exptime -=other.exptime;
     subtract_on_GPU(other);

     return *this;
 }





 //! Divides the image with the divisor, on the GPU.
 Image_cuda_compatible&  Image_cuda_compatible::operator/=(float n)
 {
     if(n > 0 || n < 0)
        {
        divide_on_GPU(n);
        voltage /=n;
        amperage /=n;
        exptime /=n;
        mean /=n;
        stdev = 0.0f;
        }
     else
        {
         std::cout << "WARNING! Trying to divide by 0. (Nothing happened.)"
                   <<std::endl;
        }
     return *this;

 }

 Image_cuda_compatible&  Image_cuda_compatible::operator*=(float n)
 {
    multiply_on_GPU(n);
    voltage *=n;
    amperage *=n;
    exptime *=n;
    mean *=n;
    stdev = 0.0f;
    return *this;
}






//! Initializes the image. Set's everything to 0. Memory is not assigned. Use only in ctor!
void Image_cuda_compatible::initialize()
{
    filename ="";
    directory = "";
    id = "";
    gpu_im = NULL;
    voltage = 0;
    amperage = 0;
    exptime = 0;
    mean = 0;
    min = 0.0f;
    max = 1e30f;
    stdev = 0.0f;
    return;
}




//! Sets every variable to default and removes the image from the GPU & the CPU
void Image_cuda_compatible::clear()
{
    remove_from_GPU();
    initialize();
}



//! Reads an image from a binary file, containing pixel values as unsigned integers. Also reads info file data.

void  Image_cuda_compatible::readfromfile( std::string filename )
{

    remove_from_GPU();
    reserve_on_GPU();

    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.is_open())
    {
        std::cout<<"Could not open image file at " << filename<<std::endl;
        return;

    }

    file.close();
    cudaReadFromFile(filename.c_str());


    QString qfilename = QString::fromStdString(filename);
    QFileInfo info = QFileInfo(qfilename);
    id = info.baseName().toStdString();
    while(id.at(0) == '0')
    {
        id.erase(id.begin());
    }

    directory = info.path().toStdString();
    this->filename = filename;
    readinfo();


return;
}



//! Reads data from the info file for the specified image.
void Image_cuda_compatible::readinfo()
{
   // std::cout  << "Reading info for file " << filename << "\n";
    std::ifstream file;
    std::string infofilename = directory;
    infofilename.append("/info.txt");
    //std::cout << "Info file found at " << infofilename << "\n";
    file.open(infofilename.c_str());
    std::string line;



    while (std::getline( file, line) ) //Read a line
       {
          std::stringstream ss(line);
          std::string temp;

          ss >> temp; // get Image_cuda_compatible id of the current row
          if(! temp.compare(id)) // temp == id
          {
              ss>> temp;
              ss>> voltage;
              ss>> amperage;
              ss >> exptime;
              file.close();
            //s  std::cout<<" Voltage: " <<voltage<<"\n Amperage: " << amperage << "\nExptime: "<< exptime<<std::endl << std::endl;
              return;

          }


       }
    std::cout << "There were no info for image with ID"
              << id << "in info file" << infofilename
              <<std::endl;
    file.close();
    return;


}




//! Writes image values to a binary file, with FLOAT values.

void Image_cuda_compatible::writetofloatfile(std::string filename)
{

    float *image = new float[size];
    cudaGetArrayToHost(image);


    FILE * file;

    if(NULL == (file = fopen(filename.c_str(), "wb")))
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(image, sizeof(float), size, file );
    fclose(file);
    delete[] image;

}


//! Reads an image from a binary file, containing pixel values as floats. Also reads info file data.

//! Image is put to the CPU memory.

void Image_cuda_compatible::readfromfloatfile(std::string fname)
{
    reserve_on_GPU();
    FILE *file = NULL;
    if(NULL == (file = fopen(fname.c_str(), "rb")))
        {
            std::cout << "Failed to open file" << fname <<" for reading" << std::endl;
        }
    fclose(file);
    cudaReadFromFloatFile(fname.c_str());

    QString qfilename = QString::fromStdString(fname);
    QFileInfo info = QFileInfo(qfilename);
    id = info.baseName().toStdString();
    while(id.at(0) == '0')
    {
        id.erase(id.begin());
    }

    directory = info.path().toStdString();
    filename = fname;

}






//! Returns the voltage of the image.
float Image_cuda_compatible::getvoltage()
{
    return voltage;
}

//! Returns the amperage of the image.
float Image_cuda_compatible::getamperage()
{
    return amperage;
}

//! Returns the exptime of the image.
float Image_cuda_compatible::getexptime()
{
    return exptime;
}

//! Returns the mean of the image.
//! If the mean value is not yet calculated ( or it is 0), (re)calculates it.
float Image_cuda_compatible::getmean()
{
    if(! (mean >0))
    {
        calculate_meanvalue_on_GPU();
    }
    return mean;
}
//! Returns the minimum intensity of the image.
//! If the minimum intensity value is not yet calculated ( or it is 0), (re)calculates it.
float Image_cuda_compatible::getmin()
{
    if(min ==0.0f)
    {
        calculate_meanvalue_on_GPU();
    }
    return min;
}

//! Returns the maximum intensity of the image.
//! If the maximum intensity value is not yet calculated,, calculates it.
float Image_cuda_compatible::getmax()
{
    if(max == 1e30f)
    {
        calculate_meanvalue_on_GPU();
    }
    return max;
}
float Image_cuda_compatible::getstdev()
{
    if( stdev ==0.0f)
    {
        calculate_meanvalue_on_GPU();
    }
    return stdev;
}

//! Returns the ID of the image.
std::string Image_cuda_compatible::getid()
{
    return id;
}

//! Sets the voltage value of the image.
void Image_cuda_compatible::setvoltage(float f)
{
    voltage = f;
    return;
}

//! Sets the amperage value of the image.
void Image_cuda_compatible::setamperage(float f)
{
    amperage = f;
    return;
}
//! Sets the exptime value of the image.
void Image_cuda_compatible::setexptime(float f)
{
    exptime = f;
    return;
}

//! Returns with the filanem of the image.
//! If the image was not read from  a file, returns "".
std::string Image_cuda_compatible::getfilename()
{
    return filename;
}


//! Saves the image as a Jpeg.

//! It can actually save any Qt supported format, filename extension determines
//! the image type.
void Image_cuda_compatible::saveAsJPEG(std::string filename)
{
    unsigned char *im_8_bit = new unsigned char[size];
    cudaGetUCArrayToHost(im_8_bit);
   QImage im_Pixmap (im_8_bit, width, height, QImage::Format_Indexed8);
   QVector<QRgb> my_table;
   for(int i = 0; i < 256; i++) my_table.push_back(qRgb(i,i,i));
   im_Pixmap.setColorTable(my_table);
   im_Pixmap.save(QString::fromStdString( filename));
   delete[] im_8_bit;
}






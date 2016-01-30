#include "image_cuda_compatible.h"

#include <QTime>
#include <QFileInfo>
#include<QString>
Image_cuda_compatible::Image_cuda_compatible()  {
initialize();
}





//! Creates image object by reading it's pixel values from an array.

Image_cuda_compatible::Image_cuda_compatible (float* array)
{
    initialize();
    readfromarray(array);
}


Image_cuda_compatible::~Image_cuda_compatible() { remove_from_CPU(); remove_from_GPU(); }


//! Reads pixel values from a float array.
void Image_cuda_compatible::readfromarray(float* array)

{
    reserve_on_CPU();

    double meandouble = 0;
    for(int i = 0 ; i < size ; i++)
        {
        im[i] = array[i];
        meandouble += im[i];
        //TODO: Error handling.
    }
    mean = float(meandouble / (double) size);

}

//! Copy Constructor.

Image_cuda_compatible::Image_cuda_compatible(const Image_cuda_compatible& other)
{
    voltage = other.voltage;
    amperage= other.amperage;
    exptime = other.exptime;
     mean=other.mean;
     filename = other.filename;
     directory = other.directory;
     id = other.id;
     if(other.im != NULL)
     {
        im = new float[size];
       // std::cout << "Reserved memory on CPU @" <<im << "for image " << id <<std::endl;

        memcpy(im, other.im, size * sizeof (float));
     }
     else
     {
         im = NULL;
     }
     gpu_im = NULL;

     if(other.gpu_im != NULL)
     {
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

         filename = other.filename;
         directory = other.directory;

          id = other.id;

          if(other.im != NULL)
          {
             im = new float[size];
             //std::cout << "Reserved memory on CPU @" <<im << "for image " << id <<std::endl;

             memcpy(im, other.im, size * sizeof (float));
          }
          else
          {
              im = NULL;
          }
          if(other.gpu_im != NULL)
          {
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

    other.copy_to_GPU();

    copy_to_GPU();

    mean+=other.mean;
    voltage +=other.voltage;
    amperage +=other.amperage;
    exptime +=other.exptime;
    add_on_GPU(other);

    return *this;
}


 //! Subtracts another image's attributes to this. Also adds pixel values on the GPU.
  Image_cuda_compatible&  Image_cuda_compatible::operator-=(Image_cuda_compatible &other)
  {

     other.copy_to_GPU();

     copy_to_GPU();

     mean-=other.mean;
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
    return *this;
}



//! Initializes the image. Set's everything to 0. Memory is not assigned. Use only in ctor!
void Image_cuda_compatible::initialize()
{
    im = NULL;
    filename ="";
    directory = "";
    id = "";
    gpu_im = NULL;
    voltage = 0;
    amperage = 0;
    exptime = 0;
    mean = 0;
    min = 0;
    max = 1e30f;
    return;
}




//! Sets every variable to default and removes the image from the GPU & the CPU
void Image_cuda_compatible::clear()
{
    remove_from_CPU();
    remove_from_GPU();
    filename="";
    directory="";
    mean=0;
    voltage =0;
    amperage=0;
    exptime = 0;
    min = 0;
    max = 1e30f;
}



//! Reads an image from a binary file, containing pixel values as unsigned integers. Also reads info file data.

//! Image is put to the CPU memory.
void  Image_cuda_compatible::readfromfile( std::string filename )
{
    reserve_on_CPU();
    remove_from_GPU();

    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.is_open())
    {
        std::cout<<"Could not open image file at " << filename<<std::endl;
        return;

    }
    unsigned char bytes[2];





    for(int i=0;i<size;i++)
    {
        unsigned short value;
      file.read( (char*)bytes, 2 );  // read 2 bytes from the file
      value = bytes[0] | (bytes[1] << 8);  // construct the 16-bit value from those bytes
       im[i] = value ;
     }
    file.close();


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
    return;


}

//! Writes image values to a binary file, with unsigned int values.

void Image_cuda_compatible::writetofile(std::string filename)
{
    copy_from_GPU();
    unsigned short* sh_im = new unsigned short[size];
    for(int i=0;i<size;i++)
    {
        sh_im[i] = (unsigned short) ( round(im[i]));
    }

    FILE * file;

    if(NULL == (file = fopen(filename.c_str(), "wb")))
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(sh_im, sizeof(unsigned short), size, file );
    delete[] sh_im;

}



//! Writes image values to a binary file, with FLOAT values.

void Image_cuda_compatible::writetofloatfile(std::string filename)
{


    copy_from_GPU();
    FILE * file;

    if(NULL == (file = fopen(filename.c_str(), "wb")))
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(im, sizeof(float), size, file );
    fclose(file);

}


//! Reads an image from a binary file, containing pixel values as floats. Also reads info file data.

//! Image is put to the CPU memory.

void Image_cuda_compatible::readfromfloatfile(std::string fname)
{
    reserve_on_CPU();
    remove_from_GPU();
    FILE *file = NULL;
    if(NULL == (file = fopen(fname.c_str(), "rb")))
        {
            std::cout << "Failed to open file" << fname <<" for reading" << std::endl;
        }
    fread(im, sizeof(float), size, file);
    fclose(file);

    QString qfilename = QString::fromStdString(fname);
    QFileInfo info = QFileInfo(qfilename);
    id = info.baseName().toStdString();
    while(id.at(0) == '0')
    {
        id.erase(id.begin());
    }

    directory = info.path().toStdString();
    filename = fname;
    readinfo();

}





//Working feature but it may not be used in the future.
//! Calculates image mean on the CPU. May not be used in the future.
void  Image_cuda_compatible::calculate_meanvalue_on_CPU()

{
    copy_from_GPU();
    double  meanvalue = 0.0; // double for higher precision when summing awful lot of numbers.
    for(int i=0; i<size;i++)
    {
        meanvalue +=  im[i];
    }
    meanvalue = meanvalue / (double) size;

mean = (float) meanvalue;
}

//Getter-setters:

float Image_cuda_compatible::getvoltage()
{
    return voltage;
}

float Image_cuda_compatible::getamperage()
{
    return amperage;
}

float Image_cuda_compatible::getexptime()
{
    return exptime;
}

float Image_cuda_compatible::getmean()
{
    return mean;
}
float Image_cuda_compatible::getmin()
{
    return min;
}
float Image_cuda_compatible::getmax()
{
    return max;
}

std::string Image_cuda_compatible::getid()
{
    return id;
}


void Image_cuda_compatible::setvoltage(float f)
{
    voltage = f;
    return;
}

void Image_cuda_compatible::setamperage(float f)
{
    amperage = f;
    return;
}

void Image_cuda_compatible::setexptime(float f)
{
    exptime = f;
    return;
}

//!Calculates the minimum intensity on the image on the CPU and returns it's value.
float Image_cuda_compatible::minintensity()
{
    copy_from_GPU();
float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] < intensity ) intensity = im[i];
}
return intensity;
}
//!Calculates the maximum intensity on the image on the CPU and returns it's value.

float Image_cuda_compatible::maxintensity()
{
    copy_from_GPU();

float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] > intensity ) intensity = im[i];
}
return intensity;
}


std::string Image_cuda_compatible::getfilename()
{
    return filename;
}



//! Assigns memory for the image on the CPU. Sets pixel values to 0.
float* Image_cuda_compatible::reserve_on_CPU()
{
    if(im == NULL)
    {
        im = new float[size]();
       // std::cout << "Reserved memory on CPU @" <<im << "for image " << id <<std::endl;
    }
    return im;
}

//! If there is memory assigned for the image on the CPU, it frees the memory.
void Image_cuda_compatible::remove_from_CPU()
{
    if(im != NULL)
    {
        delete[] im;
        im = NULL;
    }
}





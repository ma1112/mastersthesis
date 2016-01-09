#include "image_cuda_compatible.h"

#include <QTime>
#include <QFileInfo>
#include<QString>
Image_cuda_compatible::Image_cuda_compatible()  {
    im = NULL;
initialize();
}







Image_cuda_compatible::Image_cuda_compatible (float* array)
{
    initialize();
    readfromarray(array);
}


Image_cuda_compatible::~Image_cuda_compatible() { delete[] im; remove_from_GPU(); }



void Image_cuda_compatible::readfromarray(float* array)
{
    double meandouble = 0;
    for(int i = 0 ; i < size ; i++)
        {
        im[i] = array[i];
        meandouble += im[i];
        //TODO: Error handling.
    }
    mean = float(meandouble / (double) size);

}


//copy constructor
Image_cuda_compatible::Image_cuda_compatible(const Image_cuda_compatible& other)
{
    im = new float[size];
    voltage = other.voltage;
    amperage= other.amperage;
    exptime = other.exptime;
     mean=other.mean;
     filename = other.filename;
     directory = other.directory;
     id = other.id;
     memcpy(im, other.im, size * sizeof (float));
     return;
}


Image_cuda_compatible& Image_cuda_compatible::operator=(const Image_cuda_compatible& other)
 {
    std::cout<<"Callling operator= of image_cuda_compatible"<<std::endl;


    if(this != &other)
    {
     voltage = other.voltage;
     amperage= other.amperage;
     exptime = other.exptime;
      mean=other.mean;

         filename = other.filename;
         directory = other.directory;

          id = other.id;
     memcpy(im, other.im, size * sizeof (float));


     }
    return *this;


 }

 Image_cuda_compatible&  Image_cuda_compatible::operator+=(const Image_cuda_compatible &other)
 {

    for(int i = 0; i< size; i++)
        {
        im[i] += other.im[i];

    }
    mean+=other.mean;
    voltage +=other.voltage;
    amperage +=other.amperage;
    exptime +=other.exptime;

    return *this;
}

 Image_cuda_compatible&  Image_cuda_compatible::operator/=(int n)
 {
    for(int i = 0; i< size; i++)
    {
        im[i] /= n;
    }
    voltage /=n;
    amperage /=n;
    exptime /=n;
    mean /=n;
    return *this;
}




void Image_cuda_compatible::initialize()
{
    std::cout << "initialize() " << std::endl;
    im = new float[size];
    filename ="";
    directory = "";
    id = "";
    gpu_im = NULL;
    voltage = 0;
    amperage = 0;
    exptime = 0;
    mean = 0;
    return;
}




//! Sets every variable to default and removes the image from the GPU.
void Image_cuda_compatible::clear()
{
    if(im != NULL)
    {
    delete[] im;
    }
    im = new float[size]();
    remove_from_GPU();
    filename="";
    directory="";
    mean=0;
    voltage =0;
    amperage=0;
    exptime = 0;
}




void  Image_cuda_compatible::readfromfile( std::string filename )
{

    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.is_open())
    {
        return;
       //TODO: Error Handling.
    }
    double meandouble = 0;
    unsigned char bytes[2];





    for(int i=0;i<size;i++)
    {
        unsigned short value;
      file.read( (char*)bytes, 2 );  // read 2 bytes from the file
      value = bytes[0] | (bytes[1] << 8);  // construct the 16-bit value from those bytes
       im[i] = value ;
      meandouble += im[i];
     }
    mean = float(meandouble / (double) size);
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
    //TODO: Error happened.
    return;


}



void Image_cuda_compatible::writetofile(std::string filename)
{
    unsigned short sh_im[size];
    for(int i=0;i<size;i++)
    {
        sh_im[i] = (unsigned short) (im[i]);
    }

    FILE * file;

    if(NULL == (file = fopen(filename.c_str(), "wb")))
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(sh_im, sizeof(unsigned short), size, file );

}


void Image_cuda_compatible::writetofloatfile(std::string filename)
{


    FILE * file;

    if(NULL == (file = fopen(filename.c_str(), "wb")))
    {
            std::cout << "Failed to open file " << filename << "for writing."<< std::endl;
            return;
    }
    fwrite(im, sizeof(float), size, file );

}


void Image_cuda_compatible::readfromfloatfile(std::string filename)
{
    FILE *file = NULL;
    if(NULL == (file = fopen(filename.c_str(), "rb")))
        {
            std::cout << "Failed to open file" << filename <<" for reading" << std::endl;
        }
    fread(im, sizeof(float), size, file);
}





//Working feature but it may not be used in the future.
void  Image_cuda_compatible::calculate_meanvalue_on_CPU()

{
    double  meanvalue = 0.0; // double for higher precision when summing awful lot of numbers.
    for(int i=0; i<size;i++)
    {
        meanvalue +=  im[i];
    }
    meanvalue = meanvalue / (double) size;

mean = (float) meanvalue;
}


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

float Image_cuda_compatible::minintensity()
{
float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] < intensity ) intensity = im[i];
}
return intensity;
}

float Image_cuda_compatible::maxintensity()
{
float intensity =im[0];
for(int i=1;i<size;i++)
{
    if(im[i] > intensity ) intensity = im[i];
}
return intensity;
}

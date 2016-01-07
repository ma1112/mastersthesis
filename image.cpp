
#include "image.h"


Image::Image() : Image_cuda_compatible() {}
Image::Image( float* array) : Image_cuda_compatible(array){}


Image::~Image()
{
  //  delete[] im;

}


//copy constructor
 Image::Image (Image const& image)
 {
    // std::cout << "Calling copy constructor for Image az &" <<this<<std::endl;
     *this = image;

 }

 Image& Image::operator+=(const Image& other)
  {
     Image_cuda_compatible::operator+=(other);
     return *this;
 }





Image::Image(std::string filename) : Image_cuda_compatible()
{
    //TODO: Error handling.
   // im = new float[size];
    readfromfile(filename);
}



Image::Image(QString filename) : Image_cuda_compatible()
{
    //TODO: Error handling.
   // im = new float[size];
    readfromfile(filename);

}



void  Image::readfromfile( std::string filename )
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





//Reads the image from .bin file. Overload: filename can be either Qstring or std::string.
void  Image::readfromfile( QString filename )
{
    Image::readfromfile( filename.toStdString());
}
//Draws image to QLabel. Image must be read.
void Image::drawimage (QLabel* label)
{
    //16 bit to 8 bit
         unsigned char *im_8_bit = new unsigned char[size];
         unsigned char temp;
         for(int i=0;i<size;i++)
         {

             temp = abs( ((short)im[i])>>8);
             im_8_bit[i] = temp;

         }


             QImage im_Pixmap (im_8_bit, width, height, QImage::Format_Indexed8);

             QVector<QRgb> my_table;
             for(int i = 0; i < 256; i++) my_table.push_back(qRgb(i,i,i));
             im_Pixmap.setColorTable(my_table);

             label->setScaledContents(true);
             label->setPixmap(QPixmap::fromImage(im_Pixmap));
             label->show();

             delete[] im_8_bit;

}





void Image::readinfo()
{
   // std::cout  << "Reading info for file " << filename << "\n";
    std::ifstream file;
    std::string infofilename = directory;
    infofilename.append("\\info.txt");
   // std::cout << "Info file found at " << infofilename << "\n";
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






void Image::writedetailstoscreen(QTextEdit* textEdit)
{
    textEdit->clear();
    textEdit->append("Voltage: " + QString::number(voltage) );
    textEdit->append("Amperage: " + QString::number(amperage) );
    textEdit->append("Exposition time: " + QString::number(exptime) );
    textEdit->append("Mean:"+ QString::number(mean));




}







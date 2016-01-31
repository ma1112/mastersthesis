
#include "image.h"


Image::Image() : Image_cuda_compatible() {}


Image::~Image()
{

}


//copy constructor
 Image::Image (Image const& image)
 {
    // std::cout << "Calling copy constructor for Image az &" <<this<<std::endl;
     *this = image;

 }















//! Reads the image from .bin file.
void  Image::Qreadfromfile( QString filename )
{
    Image::readfromfile( filename.toStdString());
}

//! Draws image to QLabel. Image must be read before.
void Image::drawimage (QLabel* label)
{
  /*  float min = minintensity();
    float max = maxintensity();
    //16 bit to 8 bit
         unsigned char *im_8_bit = new unsigned char[size];
         float temp;
         for(int i=0;i<size;i++)
         {

             temp = 256 * ( im[i] - min) / max ;
             im_8_bit[i] = (unsigned char) ( round(temp));

         }


             QImage im_Pixmap (im_8_bit, width, height, QImage::Format_Indexed8);

             QVector<QRgb> my_table;
             for(int i = 0; i < 256; i++) my_table.push_back(qRgb(i,i,i));
             im_Pixmap.setColorTable(my_table);

             label->setScaledContents(true);
             label->setPixmap(QPixmap::fromImage(im_Pixmap));
             label->show();

             delete[] im_8_bit;
             */

}











void Image::writedetailstoscreen(QTextEdit* textEdit)
{
    textEdit->clear();
    textEdit->append("Voltage: " + QString::number(voltage) );
    textEdit->append("Amperage: " + QString::number(amperage) );
    textEdit->append("Exposition time: " + QString::number(exptime) );
    textEdit->append("Mean:"+ QString::number(mean));
    textEdit->append("Min:"+ QString::number(min));
    textEdit->append("Max:"+ QString::number(max));






}








#include "image.h"
#include"qstring.h"
#include "iostream"


Image::Image() {initialize();}


Image::~Image()
{

}


//!copy constructor
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

         unsigned char *im_8_bit = new unsigned char[size];
         cudaGetUCArrayToHost(im_8_bit);

             QImage im_Pixmap (im_8_bit, width, height, QImage::Format_Indexed8);

             QVector<QRgb> my_table;
             for(int i = 0; i < 256; i++) my_table.push_back(qRgb(i,i,i));
             im_Pixmap.setColorTable(my_table);

             label->setScaledContents(true);
             label->setPixmap(QPixmap::fromImage(im_Pixmap));
             label->show();

             delete[] im_8_bit;

}

//! Writes the image details to the given QTextEdit. Used to debug reasons at the moment.
void Image::writedetailstoscreen(QTextEdit* textEdit)
{
    textEdit->clear();
    textEdit->append("Id: " + QString::fromStdString(id) );
    textEdit->append("Filename: " + QString::fromStdString(filename) );
    textEdit->append("Voltage: " + QString::number(voltage) );
    textEdit->append("Amperage: " + QString::number(amperage) );
    textEdit->append("Exposition time: " + QString::number(exptime) );
    textEdit->append("Mean:"+ QString::number(mean));
    textEdit->append("StDev:"+ QString::number(stdev));
    textEdit->append("Min:"+ QString::number(min));
    textEdit->append("Max:"+ QString::number(max));






}







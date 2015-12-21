#ifndef IMAGE_H
#define IMAGE_H

#include "image_cuda_compatible.h"
#include <QMainWindow>
#include <QLabel>
#include <QString>
#include <QFileInfo>
#include <sstream>
#include <QTextEdit>
//#include "cuda_kernel_calls.h"


class Image : public Image_cuda_compatible
{
public:

    Image();
    Image( unsigned short* array) ;
    Image( std::string filename);
    Image(QString filename);



    //~Image();

    void readfromfile(std::string filename );

    void readfromfile(QString filename);
    void drawimage (QLabel* label);
    void writedetailstoscreen(QTextEdit* textEdit);
    void Image::readinfo();
};



#endif // IMAGE_H

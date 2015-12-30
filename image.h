#ifndef IMAGE_H
#define IMAGE_H

#include "image_cuda_compatible.h"
#include <QMainWindow>
#include <QLabel>
#include <QString>
#include <QFileInfo>
#include <sstream>
#include <QTextEdit>


//! \class Image
//! \brief: Image class that uses QT elements.
//!
//! Inherited from image_cuda_compatible, this class is capable of storing image and it's data
//! as well as it can use QT related functions. It can draw itself and can read itself and it's info from file.

class Image : public Image_cuda_compatible
{
public:

    Image(); //!<Default constructor
    Image( unsigned short* array) ; //!<Constructor that copies image from a this.size long array
    Image( std::string filename); //!< Constructor that reads the image from the file.
    Image(QString filename); //!< Constructor that reads the image from the file. QString version.



    //~Image();

    void readfromfile(std::string filename ); //!< Reads image data from file.

    void readfromfile(QString filename); //!< Reads image data from file. QString version.
    void drawimage (QLabel* label); //!< Draws the image to the QLabel.
    void writedetailstoscreen(QTextEdit* textEdit); //!< Writes technical information to the QTextEdit.
    void Image::readinfo(); //!< Reads image info from the info file. Info file must be in the same folder as the image.
};



#endif // IMAGE_H

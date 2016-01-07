#ifndef GAINCORR_H
#define GAINCORR_H
#include "image.h"
#include<QFileDialog>
#include <QString>
#include <QStringList>
#include <QDir>
#include <vector>

//! \class Gaincorr
//! \brief: Calculates, contains and reads Gein correction data
//!
//! <<insert long description here >>


class Gaincorr
{
public:
    Gaincorr();
    int read();



private:
    Image slope;
    Image intercept;
    Image image; // for temporary holding reasons
    std::vector<Image> images_temp;
    std::vector<Image> images;
};

#endif // GAINCORR_H

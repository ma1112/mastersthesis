#ifndef GAINCORR_H
#define GAINCORR_H
#include "image_cuda_compatible.h"
#include <vector>
#include <map>

//! \class Gaincorr
//! \brief: Calculates, contains and reads Gain correction data
//!
//! <<insert long description here >>


class Gaincorr
{
public:
    Gaincorr();
    int read();
    void calculate();



private:
    Image_cuda_compatible slope;
    Image_cuda_compatible intercept;
    Image_cuda_compatible image; // for temporary holding reasons
    std::vector<Image_cuda_compatible> images_temp;
    std::vector<Image_cuda_compatible> images;
    std::map<int, std::vector<Image_cuda_compatible> > imagemap;
};

#endif // GAINCORR_H

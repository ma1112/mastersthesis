#ifndef GAINCORR_H
#define GAINCORR_H
#include "image_cuda_compatible.h"
#include <vector>
#include <map>
#include "gc_im_container.h"

//! \class Gaincorr
//! \brief: Calculates, contains and reads Gain correction data
//!
//! <<insert long description here >>


class Gaincorr
{
public:
    Gaincorr();
    int readimages();
    void calculate();
    void readfactors();
    void corrigateimage(Image_cuda_compatible &image);



private:
    Image_cuda_compatible slope;
    Image_cuda_compatible intercept;


    std::map<int,Image_cuda_compatible> slopes;
    std::map<int,Image_cuda_compatible> intercepts;
    std::map<int, int> saturation;


    Image_cuda_compatible image; // for temporary holding reasons
    std::vector<Image_cuda_compatible> images_temp;
    std::vector<Image_cuda_compatible> images;
    std::map<int, gc_im_container > imagemap;
    std::string gcfolder; //!< Folder where gc data (slope and intercept) are saved or readen from.



};

#endif // GAINCORR_H

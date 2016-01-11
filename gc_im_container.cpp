#include "gc_im_container.h"

gc_im_container::gc_im_container()
{
h_images = NULL;
d_images = NULL;
images = 0;
size = 0;
h_settings = NULL;
d_settings = NULL;
}


void gc_im_container::add(Image_cuda_compatible &im)
{
    std::cout <<"Adding image " <<std::endl;
    if(size <=0)
    {
        std::cout<<"ERROR: gc_im_container is not inicialized. " <<std::endl;
        return;
    }


    if(images == size)
    {
        std::cout<<"ERROR: Can't add more images to gc_im_container. It already has" << images <<"images." <<std::endl;
        return;
    }


    for(int i = 0; i< im.size; i++)
    {
        h_images[i*size + images] =im.im[i];
    }
    h_settings[images]= im.amperage*im.exptime;
    images++;
    std::cout <<"image added " <<std::endl;

    return;
}

int gc_im_container::getimageno()
{
    return images;
}



int gc_im_container::getsize()
{
    return size;
}


float gc_im_container::calculateXmean()
{
    float mean = 0;
    for(int i=0;i<images;i++)
    {
        mean += h_settings[i];
    }
    return mean;

}

float gc_im_container::calculateX2mean()
{
    float mean = 0;
    for(int i=0;i<images;i++)
    {
        mean += h_settings[i] * h_settings[i];
    }
    return mean;

}






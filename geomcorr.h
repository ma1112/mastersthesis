#ifndef GEOMCORR_H
#define GEOMCORR_H
#include "image_cuda_compatible.h"
#include"thrust/device_vector.h"

#include"gaincorr.h"
class Geomcorr
{
    friend class Gaincorr;
public:
    Geomcorr();
    ~Geomcorr();
    void extractCoordinates(Image_cuda_compatible &image);
    void readAndCalculateGeom(int n);
    void initializeDeviceVector(int n, int size);
    void addCoordinates();
    void exportText(std::string filename);

private:
    Gaincorr gc;
    float* d_filter;
    const int filterWidth = 9;
    Gaincorr *gaincorr;
    int* d_coordinates;
    int addedCoordinates;
    int n;
    int size;
    int* d_coordinatesFromThatImage;


};

#endif // GEOMCORR_H

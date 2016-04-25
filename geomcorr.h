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
    void extractCoordinates(Image_cuda_compatible &image, bool drawOnly = false, bool onlyN = false);
    void readAndCalculateGeom();
    void initializeDeviceVector(int n, int size, int u);
    void addCoordinates();
    void addCoordinates(Image_cuda_compatible &);
    bool exportText(std::string filename);
    void calculateEta();
    void fitEllipse(int i, float* a, float* b, float* c, float* u, float* v, double *error);
    int getn();
    double calculatePhase(int i, float u);
    bool coordinatesToCPU(int* h_x, int* h_y, int n);


private:
    float* d_filter;
   int filterWidth ;
    Gaincorr *gaincorr;
    int* d_coordinates;
    int* d_addedCoordinates;
    int n; // number of circles visible on all images
    int u; // unseen circles ( on first image)
    int size;
    int* d_coordinatesFromThatImage;
    double eta;


};

#endif // GEOMCORR_H

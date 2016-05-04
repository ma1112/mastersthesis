#include "geomcorr.h"
#include"QDir"
#include"QString"
#include"QStringList"
#include"image_cuda_compatible.h"
#include"QFileDialog"
#include"iostream"
#include "image.h"
#include "geomcorrcheckerdialog.h"


void Geomcorr::readAndCalculateGeom()
{


    geomCorrCheckerDialog gccDialog;
    gccDialog.setModal(true);
    gccDialog.exec();

}

int Geomcorr::getn()
{
    return n;
}

double Geomcorr::getEta()
{
    return eta;
}

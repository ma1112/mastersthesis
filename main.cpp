#include "mainwindow.h"
#include <QApplication>
#ifndef _MSC_VER
#include "initcuda.cuh"
#endif
#include <QtGlobal>

int main(int argc, char *argv[])
{

#ifndef _MSC_VER
    initCuda();
#endif


    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

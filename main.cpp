#include "mainwindow.h"
#include <QApplication>
#ifndef _MSC_VER
#include "initcuda.cuh"
#endif
#include <QtGlobal>
#include <iostream>

int main(int argc, char *argv[])
{

#ifndef _MSC_VER

    std::cout << "Runing on linux: Using semaphore. " << std::endl;
    initCuda();
#endif


    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

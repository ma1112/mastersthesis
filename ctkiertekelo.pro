#-------------------------------------------------
#
# Project created by QtCreator 2015-12-06T11:36:50
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ctkiertekelo
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    image.cpp \
    image_cuda_compatible.cpp \
    gaincorr.cpp \
    gc_im_container.cpp \
    geomcorr.cpp \
    geomcorrcheckerdialog.cpp \
    directorystructureconverter.cpp

HEADERS  += mainwindow.h
HEADERS  +=    directorystructureconverter.h
HEADERS  +=    geomcorrcheckerdialog.h
HEADERS  += geomcorr.h
HEADERS  +=  book.cuh
HEADERS +=  gc_im_container.h

HEADERS +=   gaincorr.h

 HEADERS  +=   book.h
HEADERS  +=    image_cuda_compatible.h
HEADERS  +=     image.h


FORMS    += mainwindow.ui \
    geomcorrcheckerdialog.ui






INCLUDEPATH += $$PWD/
DEPENDPATH += $$PWD/

DESTDIR = debug
OBJECTS_DIR = debug/obj           # directory where .obj files will be saved
CUDA_OBJECTS_DIR = debug/obj      # directory where .obj  of cuda file will be saved
# This makes the .cu files appear in your project

OTHER_FILES +=      # cuda_code.cu # this is my cu file need to compile

# CUDA settings <-- may change depending on your system

CUDA_SOURCES += cuda_image_kernel_calls.cu     # let NVCC know which file you want to compile CUDA NVCC
CUDA_SOURCES += cuda_gaincorr_kernel_calls.cu
CUDA_SOURCES += cuda_gc_im_conitainer_functions.cu
CUDA_SOURCES += cuda_geomcorr_kernel_calls.cu
CUDA_SOURCES += book.cu



CUDA_DIR =  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5"
SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_20           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS += --use_fast_math # default setting

# include paths

INCLUDEPATH += $$CUDA_DIR/include\


# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/x64\


# Add the necessary libraries
CUDA_LIBS= -lcuda -lcudart -lcusolver -lcusparse
#add quotation for those directories contain space (Windows required)
CUDA_INC +=$$join(INCLUDEPATH,'" -I"','-I"','"')

LIBS += $$CUDA_LIBS
#nvcc config
# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CONFIG(debug, debug|release) {
    #Debug settings
    # Debug mode
    cuda_d.input    = CUDA_SOURCES
    cuda_d.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN64 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
     # Release settings
     cuda.input    = CUDA_SOURCES
     cuda.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
     cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN64 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
     cuda.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda
}




INCLUDEPATH += $$PWD/lib
DEPENDPATH += $$PWD/lib

DISTFILES += \
    cuda_reduce_kernels.cuh \
    cuda_image_kernel_calls.cu \
    cuda_gaincorr_kernel_calls.cu \
    cuda_gc_im_conitainer_functions.cu \
    cuda_geomcorr_kernel_calls.cu \
    book.cu

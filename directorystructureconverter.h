//! Class to convert directory structure from "every subdir contains images with different settings"
//! to " a single directory contains every image."
#ifndef DIRECTORYSTRUCTURECONVERTER_H
#define DIRECTORYSTRUCTURECONVERTER_H



class DirectoryStructureConverter
{
public:
    DirectoryStructureConverter();
    void copyAll();
    void copyDirAsImage();
};

#endif // DIRECTORYSTRUCTURECONVERTER_H

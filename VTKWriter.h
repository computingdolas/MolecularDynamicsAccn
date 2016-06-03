#ifndef VTKWRITER_H
#define VTKWRITER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "Type.h"
#include "cudaDeviceBuffer.h"

class VTKWriter{
private:
    std::string file_base;

    int file_number;
public:
    VTKWriter(std::string file_base);
    void writeVTKOutput(const cudaDeviceBuffer<real_d> &mass, const cudaDeviceBuffer<real_d> &position, const cudaDeviceBuffer<real_d> &velocity, int num_particles);

};
#endif // VTKWRITER_H

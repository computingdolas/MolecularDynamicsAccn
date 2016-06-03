#include "VTKWriter.h"
#include <iomanip>
//Constructor
VTKWriter::VTKWriter(std::string file_base){
    this->file_base = file_base;
    file_number = 0;
}

//Output file writer
void VTKWriter::writeVTKOutput(const cudaDeviceBuffer<real_d> &mass, const cudaDeviceBuffer<real_d> &position, const cudaDeviceBuffer<real_d> &velocity, int num_particles){
    std::ofstream outfile;
    std::ostringstream file_number_str;
    file_number_str<<this->file_number;
    std::string file_name = this->file_base+file_number_str.str()+".vtk";
    this->file_number++;

    outfile.open("vtk/"+file_name);
    if(!outfile.is_open()){
        std::cerr<<"Could not write vtk output to file "<<file_name<<std::endl;
        exit(-1);
    }

    outfile<<"# vtk DataFile Version 4.0"<<std::endl;
    outfile<<"hesp visualization file"<<std::endl;
    outfile<<"ASCII"<<std::endl;

    //Output position as an unstructured grid
    outfile<<"DATASET UNSTRUCTURED_GRID"<<std::endl;
    outfile<<"POINTS "<<num_particles<<" double"<<std::endl;
    for(int i=0;i<num_particles;i++){
        outfile<<std::setprecision(8)<<position[i*3]<<" "<<position[i*3+1]<<" "<<position[i*3+2]<<std::endl;
    }

    //Output mass
    outfile<<"CELLS 0 0"<<std::endl;
    outfile<<"CELL_TYPES 0"<<std::endl;
    outfile<<"POINT_DATA "<<num_particles<<std::endl;
    outfile<<"SCALARS m double"<<std::endl;
    outfile<<"LOOKUP_TABLE default"<<std::endl;
    for(int i=0;i<num_particles;i++){
        outfile<<std::setprecision(8)<<mass[i]<<std::endl;
    }

    //Output velocity
    outfile<<"VECTORS v double"<<std::endl;
    for(int i=0;i<num_particles;i++){
        outfile<<std::setprecision(8)<<velocity[i*3]<<" "<<velocity[i*3+1]<<" "<<velocity[i*3+2]<<std::endl;
    }
}

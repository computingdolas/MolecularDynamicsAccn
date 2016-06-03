#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <fstream>

#include "Type.h"
#include "cudaDeviceBuffer.h"

class Parser{
public:
    int num_params;
    int num_particles;
    std::string filename;
    std::map<std::string,std::string> params;

    std::vector<real_d> mass;
    std::vector<real_d> pos;
    std::vector<real_d> vel;

    //Constructor
    Parser(std::string filename){
        this->num_params = 0;
        this->filename = filename;
    }

    //Parse the parameters
    void readParameters();

    //Read input configuration
    void readInputConfiguration();

    // Fill the cudaDeviceBuffers
    void fillBuffers(cudaDeviceBuffer<real_d> &mass, cudaDeviceBuffer<real_d> &velocity, cudaDeviceBuffer<real_d> &position ) ;

};
#endif // PARSER_H

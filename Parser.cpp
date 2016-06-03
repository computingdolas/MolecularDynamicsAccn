#include "Parser.h"

//Read parameter names, their values and store in in a string to string map
void Parser::readParameters(){
    std::string param_name;
    std::string param_value;

    std::ifstream inputfile;
    inputfile.open(this->filename);

    if(!inputfile.is_open()){
        std::cerr<<"Could not open "<<this->filename<<std::endl;
    }

    ;
    while(true){
        inputfile>>param_name>>param_value;
        params[param_name] = param_value;

        this->num_params++;

        if(inputfile.eof()){
            break;
        }
    }
    inputfile.close();
}

//Read input configuration and return the number of particles
void Parser::readInputConfiguration(){
    std::string config_filename;

    real_d mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z;

    config_filename = this->params["part_input_file"];

    std::ifstream input_file;

    input_file.open(config_filename);

    if(!input_file.is_open()){
        std::cerr<<"Could not open file "<<config_filename;
    }
    else{
        input_file>>this->num_particles;
    }

    for(int i=0;i<this->num_particles;i++){
        input_file>>mass>>pos_x>>pos_y>>pos_z>>vel_x>>vel_y>>vel_z;
        (this->mass).push_back(mass);
        (this->pos).push_back(pos_x);
        (this->pos).push_back(pos_y);
        (this->pos).push_back(pos_z);
        (this->vel).push_back(vel_x);
        (this->vel).push_back(vel_y);
        (this->vel).push_back(vel_z);
    }

    input_file.close();
}

void Parser::fillBuffers(cudaDeviceBuffer<real_d> &mass,
                         cudaDeviceBuffer<real_d> &velocity,
                         cudaDeviceBuffer<real_d> &position) {

    for (real_l i =0 ; i < this->num_particles ; ++i ){

        real_l vidx  = i * 3 ;
        mass[i] = this->mass[i] ;
        position[vidx]    = this->pos[vidx] ;
        position[vidx +1] = this->pos[vidx+1];
        position[vidx +2] = this->pos[vidx+2];
        velocity[vidx]    = this->vel[vidx] ;
        velocity[vidx +1] = this->vel[vidx+1] ;
        velocity[vidx +2] = this->vel[vidx+2];
    }
}

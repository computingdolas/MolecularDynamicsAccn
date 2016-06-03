#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>
#include "Parser.h"
#include "PhysicalVariable.h"
#include "Type.h"
#include "kernel.cuh"
#include <string>
#include "VTKWriter.h"
#include <iomanip>
#include "Time.hpp"

int main(int argc, const char * argv[]) {
    // Reading from file
    Parser p(argv[1]);
    p.readParameters();
    p.readInputConfiguration();

    // Parameters from the file
    real_d time_end = std::stod(p.params["time_end"]) ;
    real_d timestep_length = std::stod(p.params["timestep_length"]) ;
    real_d epsilon = std::stod(p.params["epsilon"]) ;
    real_d sigma = std::stod(p.params["sigma"]) ;
    real_d r_cut = std::stod(p.params["r_cut"]);
    real_l vtk_out_freq = std::stol(p.params["vtk_out_freq"]) ;
    real_l threads_per_blocks = std::stol(p.params["cl_workgroup_1dsize"]) ;
    std::string vtk_name = p.params["vtk_out_name_base"] ;
    real_d xmin = std::stod(p.params["x_min"]);
    real_d xmax = std::stod(p.params["x_max"]);
    real_d ymin = std::stod(p.params["y_min"]);
    real_d ymax = std::stod(p.params["y_max"]);
    real_d zmin = std::stod(p.params["z_min"]);
    real_d zmax = std::stod(p.params["z_max"]);
    real_l xn = std::stol(p.params["x_n"]);
    real_l yn = std::stol(p.params["y_n"]);
    real_l zn = std::stol(p.params["z_n"]);

    real_d len_x = (xmax-xmin)/xn;
    real_d len_y = (ymax-ymin)/yn;
    real_d len_z = (zmax-zmin)/zn;

    // number of Particles
    const real_l numparticles = p.num_particles ;
    const real_l numcells = xn*yn*zn;

    // Creating the device Buffers
    cudaDeviceBuffer<real_d> mass(numparticles,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> position(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> velocity(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forceold(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forcenew(numparticles,PhysicalQuantity::Vector) ;
    //cudaDeviceBuffer<real_l> cell_list(numcells,PhysicalQuantity::Scalar);
    //cudaDeviceBuffer<real_l> particle_list(numparticles,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<real_d> const_args(9,PhysicalQuantity::Scalar);
    //cudaDeviceBuffer<real_l> num_cells(3,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<real_l> neighbour_list(numparticles*numparticles,PhysicalQuantity::Scalar);

    //Initiliazing the buffers for mass,velocity and position
    p.fillBuffers(mass,velocity,position);

    //Filling in the host data for the constant arguements
    const_args[0] = xmin;
    const_args[1] = xmax;
    const_args[2] = ymin;
    const_args[3] = ymax;
    const_args[4] = zmin;
    const_args[5] = zmax;
    const_args[6] = len_x;
    const_args[7] = len_y;
    const_args[8] = len_z;

    //Number of cells per dimension
    //num_cells[0] = xn;
    //num_cells[1] = yn;
    //num_cells[2] = zn;

    // Allocating memory on Device
    mass.allocateOnDevice();
    position.allocateOnDevice();
    velocity.allocateOnDevice();
    forceold.allocateOnDevice();
    forcenew.allocateOnDevice();
   // cell_list.allocateOnDevice();
   // particle_list.allocateOnDevice();
    const_args.allocateOnDevice();
   // num_cells.allocateOnDevice();
    neighbour_list.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    forceold.copyToDevice();
    forcenew.copyToDevice();
    //cell_list.copyToDevice();
    //particle_list.copyToDevice();
    const_args.copyToDevice();
    //num_cells.copyToDevice();
    neighbour_list.copyToDevice();

    VTKWriter writer(vtk_name) ;

    //Calculate the number of blocks
    real_l num_blocks ;

    if(numparticles % threads_per_blocks ==0) num_blocks = numparticles / threads_per_blocks ;
    else num_blocks = (numparticles / threads_per_blocks) + 1;

    std::cout<<"The number of blocks and threads/block resp are: "<<num_blocks<<" "<<threads_per_blocks<<std::endl;
    real_d time_taken = 0.0 ;

    HESPA::Timer time ;
    // Algorithm to follow
    {

        real_l iter = 0 ;
        //calculate Initial forces
        formNeighbourList<<<num_blocks,threads_per_blocks>>>(neighbour_list.devicePtr,position.devicePtr,const_args.devicePtr,numparticles,r_cut);
       	 calcForcesNeighbourList<<<num_blocks ,threads_per_blocks>>>(forcenew.devicePtr,neighbour_list.devicePtr,numparticles, const_args.devicePtr,position.devicePtr,sigma,epsilon);
	
        for(real_d t =0.0 ; t < time_end ; t+= timestep_length ) {
            time.reset();

            if(iter % vtk_out_freq == 0){
                // copy to host back
                forcenew.copyToHost();
                forceold.copyToHost();
                position.copyToHost();
                velocity.copyToHost();
                writer.writeVTKOutput(mass,position,velocity,numparticles);
            }


            // Update the Position
            updatePosition<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,velocity.devicePtr,mass.devicePtr,numparticles,timestep_length,const_args.devicePtr);

            // Copy the forces
            copyForces<<<num_blocks,threads_per_blocks>>>(forceold.devicePtr,forcenew.devicePtr, numparticles);


            // Calculate New forces
            setToZero<<<1,numparticles>>>(forcenew.devicePtr,numparticles);
            formNeighbourList<<<num_blocks,threads_per_blocks>>>(neighbour_list.devicePtr,position.devicePtr,const_args.devicePtr,numparticles,r_cut);
            calcForcesNeighbourList<<<num_blocks ,threads_per_blocks>>>(forcenew.devicePtr,neighbour_list.devicePtr,numparticles,\
                                                                        const_args.devicePtr,position.devicePtr,sigma,epsilon);

            // Update the velocity
            updateVelocity<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,forceold.devicePtr,velocity.devicePtr,mass.devicePtr,numparticles,timestep_length);

            time_taken += time.elapsed();

            // Iterator count
            ++iter ;
        }
	
    }	
    std::cout<<"The time taken for "<<numparticles<<" is:= "<<time_taken<<std::endl ;

    return 0;
}

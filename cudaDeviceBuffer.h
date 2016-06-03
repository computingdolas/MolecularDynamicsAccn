//
//  cudaDeviceBuffer.hpp
//  Particle_Simulation
//
//  Created by Sagar Dolas on 11/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#ifndef cudaDeviceBuffer_hpp
#define cudaDeviceBuffer_hpp

#include <stdio.h>
#include <vector>
#include "Type.h"
#include "PhysicalVariable.h"
#include <cuda_runtime.h>
#include <iostream>

#define DIM 3  //..dimension of the system we ar working on ..//
template <typename type>
class cudaDeviceBuffer {
    
private:
    
    PhysicalQuantity phyVar ;
    std::vector<type> data ; // Data on the host ...//
    real_l actualSize ; // Actual size to be allocated ...//
    real_l numBytes_ ;
    
public:
    
    
    type *devicePtr ; // Pointer on the Device ..///

    // Constructing the Buffer......//
    explicit cudaDeviceBuffer(real_l numParticles_,const PhysicalQuantity phyVar_) ;
    ~cudaDeviceBuffer() ;
    
    // Accessing the data.....//
    const type& operator[](real_l index_) const ;
    type& operator[](real_l index_) ;
    
    // Memory Operations....//
    void copyToHost() ;
    void copyToDevice() ;
    
    // Allocate and Deallocate Memory on device....//
    void allocateOnDevice() ;
    void freeMemoryOnDevice() ;
    
    //Calculate Memory...//
    void bytesToAllocate() ;
    
    //Reseting the data array....//
    void reset() ;
    
    // Check Error....//
    void checkError(const cudaError_t err) ;
};

template<typename type>
cudaDeviceBuffer<type>::cudaDeviceBuffer(real_l _numParticles,const PhysicalQuantity _phyVar ) : phyVar(_phyVar) {

    if (phyVar == PhysicalQuantity::Scalar) {
        actualSize = _numParticles ;
    }
    else
        actualSize = _numParticles * DIM;

    // Allocating the data
    data.resize(actualSize,0.0) ; // May be problem here

    // Calculating the Number of Bytes
    bytesToAllocate() ;

}
template <typename type>
cudaDeviceBuffer<type>::~cudaDeviceBuffer<type>(){

    freeMemoryOnDevice() ; // Do not call this function untill and unless you are sure that you have allocated memory in the device

}
template<typename type>
const type& cudaDeviceBuffer<type>::operator[](real_l index_) const{

    return data[index_] ;
}

template<typename type>
type& cudaDeviceBuffer<type>::operator[](real_l index_){

    return data[index_] ;
}

template<typename type>
void cudaDeviceBuffer<type>::copyToDevice(){

    checkError(cudaMemcpy(devicePtr, &data[0], numBytes_, cudaMemcpyHostToDevice)) ;
}
template<typename type >
void cudaDeviceBuffer<type>::copyToHost(){

    checkError(cudaMemcpy(&data[0], devicePtr, numBytes_, cudaMemcpyDeviceToHost)) ;
}

template<typename type>
void cudaDeviceBuffer<type>::allocateOnDevice(){

    checkError(cudaMalloc(&devicePtr, numBytes_));
}

template<typename type>
void cudaDeviceBuffer<type>::freeMemoryOnDevice(){

    checkError(cudaFree(devicePtr)) ;
}

template<typename type>
void cudaDeviceBuffer<type>::bytesToAllocate(){

    //calculate number of bytes
    numBytes_ = actualSize * sizeof(type) ;
}

template<typename type>
void cudaDeviceBuffer<type>::reset(){

    // Not defined yet
}

template<typename type>
void cudaDeviceBuffer<type>::checkError(const cudaError_t err){

    if(err!= cudaSuccess){
        std::cout<<cudaGetErrorString(err)<<std::endl ;
        exit(-1) ;
    }
}

#endif /* cudaDeviceBuffer_hpp */

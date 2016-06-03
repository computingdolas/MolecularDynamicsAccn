//
//  Time.cpp
//  Assignment_1
//
//  Created by Sagar Dolas on 02/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#include "Time.hpp"

using namespace std::chrono;

HESPA::Timer::Timer():start_(Clock_t::now()) {

}

void HESPA::Timer::reset(){
    start_ = Clock_t::now() ;
}

double HESPA::Timer::elapsed() const {
    
    return duration<double,std::milli>(Clock_t::now() -start_).count() ;
}


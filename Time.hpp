//
//  Time.hpp
//  Assignment_1
//
//  Created by Sagar Dolas on 02/05/16.
//  Copyright © 2016 Sagar Dolas. All rights reserved.
//

#ifndef Time_hpp
#define Time_hpp

#include <stdio.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock_t;

namespace HESPA {
    
    class Timer {
        
    private:
        Clock_t::time_point start_ ;
    public:
        
        Timer() ;
        void reset() ;
        double elapsed() const ;
        
    };
}

#endif /* Time_hpp */

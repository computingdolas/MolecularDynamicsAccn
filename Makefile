#Compiler FLags 
CFLAGS = -c -std=c++11 -O3 
CCg = g++ 

#Compiler flags for Cuda 
CFLAGScuda = -std=c++11 -arch sm_20 -O3 
CC = nvcc

all	:particle cell par neighboure

particle	:Parser.o VTKWriter.o Time.o 
		$(CC) $(CFLAGScuda) Parser.o VTKWriter.o Time.o Simulation.cu -o brute

cell		:Parser.o VTKWriter.o Time.o
		$(CC) $(CFLAGScuda) Parser.o VTKWriter.o Time.o SimulationCP.cu -o cell

par		:Parser.o VTKWriter.o Time.o
		$(CC) $(CFLAGScuda) Parser.o VTKWriter.o Time.o SimulationPP.cu -o parp

neighboure	:Parser.o VTKWriter.o Time.o
		$(CC) $(CFLAGScuda) Parser.o VTKWriter.o Time.o SimulationNL.cu -o neighbl

Parser.o	:Parser.cpp
		$(CCg) $(CFLAGS) Parser.cpp

VTKWriter.o	:VTKWriter.cpp
		$(CCg) $(CFLAGS) VTKWriter.cpp

Time.o		:Time.cpp
		$(CCg) $(CFLAGS) Time.cpp

clean		:
		rm -rf *.o *.out  

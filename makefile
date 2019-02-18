#Important:
#see the params.h for some of the parameters
#need to do a make clean first before executing the program



SOURCES = import_dataset.cpp main.cu GPU.cu kernel.cu 
OBJECTS = import_dataset.o 
CUDAOBJECTS = GPU.o kernel.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++11 -O3 -Xcompiler -fopenmp -arch=compute_50 -code=sm_50 -lcuda -lineinfo
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES




all: $(EXECUTABLE)



.cpp.o:
	$(CC) $(CFLAGS) $(FLAGS) $(SEARCHMODE) $(PARAMS) $<



main.o: main.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) main.cu 

kernel.o: kernel.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(FLAGS) $(CFLAGS) $(SEARCHMODE) $(PARAMS) GPU.cu	



$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm main

copy:
	scp kernel.cu GPU.cu main.cu kernel.h params.h benoit@egr33.egr.nau.edu:~/paper_code/


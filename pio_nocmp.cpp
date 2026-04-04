#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include "mpi.h"
#include "rw.h"


int main(int argc, char* argv[]) {
    srand(time(0));
    char* cfgFile;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   
    double * absbound;

    //Begin modify this part

    char* file_folder = "/public/share/acnnprvuzd/data/sdrbench/Hurricane";
    int num_vars = atoi(argv[1]);

    //Begin modify this part
    int hurricane_num_vars = 13;
/*    char hurricane_file[13][50] = {"Uf48_truncated.bin.f32", "Vf48_truncated.bin.f32", "Wf48_truncated.bin.f32",
                                   "TCf48_truncated.bin.f32", "Pf48_truncated.bin.f32", "QVAPORf48_truncated.bin.f32",
                                   "CLOUDf48_log10_truncated.bin.f32", "QCLOUDf48_log10_truncated.bin.f32", "QICEf48_log10_truncated.bin.f32",
                                   "QRAINf48_log10_truncated.bin.f32", "QSNOWf48_log10_truncated.bin.f32", "QGRAUPf48_log10_truncated.bin.f32",
                                   "PRECIPf48_log10_truncated.bin.f32"};*/
    char hurricane_file[13][50] = {"Uf48.bin.f32", "Vf48.bin.f32", "Wf48.bin.f32",
                                   "TCf48.bin.f32", "Pf48.bin.f32", "QVAPORf48.bin.f32",
                                   "CLOUDf48.log10.bin.f32", "QCLOUDf48.log10.bin.f32", "QICEf48.log10.bin.f32",
                                   "QRAINf48.log10.bin.f32", "QSNOWf48.log10.bin.f32", "QGRAUPf48.log10.bin.f32",
                                   "PRECIPf48.log10.bin.f32"};    
//double hurricane_abs_bound[13] = {1.851616, 1.874008, 0.331488, 2.12402, 132.72278, 0.0004084144, 0.20304, 0.7021114, 0.373662, 0.714628, 0.695594, 0.715312, 0.716028};
double hurricane_abs_bound[13] = {9.258076e-02,9.370042e-02,1.657436e-02,1.062011e-01,6.636139e+00,2.042072e-05,1.015203e-02,3.510567e-02,1.868306e-02,3.573139e-02,3.477967e-02,3.576564e-02,3.580139e-02};    
// miranda
    int miranda_num_vars = 7;
    char miranda_file[7][50] = {"velocityy_truncated.bin.f32", "velocityx_truncated.bin.f32", "density_truncated.bin.f32",
                                "pressure_truncated.bin.f32", "velocityz_truncated.bin.f32", "viscocity_truncated.bin.f32",
                                "diffusivity_truncated.bin.f32"};
    double miranda_abs_bound[7] = {0.1317327, 0.135345, 0.06, 0.1324284, 0.4246152, 0.0786399, 0.07845450000000001};

    int scale_num_vars = 12;
    char scale_file[12][50] = {"PRES-98x1200x1200.f32", "QC-98x1200x1200.log10.f32", "QG-98x1200x1200.log10.f32",
                                   "QI-98x1200x1200.log10.f32", "QR-98x1200x1200.log10.f32", "QS-98x1200x1200.log10.f32",
                                   "QV-98x1200x1200.log10.f32", "RH-98x1200x1200.f32", "T-98x1200x1200.f32",
                                   "U-98x1200x1200.f32", "V-98x1200x1200.f32", "W-98x1200x1200.f32",
                                  };
    double scale_abs_bound[12] ={1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};

    int aramco_num_vars = 60;
    char aramco_file[60][50];
    double aramco_abs_bound[60];
    for (int i=0;i<60;i++){
        //char name[50];
        sprintf(aramco_file[i],"aramco-snapshot-%d.f32",1000+10*i);
        //salt_file[i]=name;
        aramco_abs_bound[i]=1e-3;

    }





    char file[100][50];
    
    
    if (num_vars == hurricane_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], hurricane_file[i]);
        absbound = hurricane_abs_bound;
    } else if (num_vars == miranda_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], miranda_file[i]);
        absbound = miranda_abs_bound;
    } 
    else if (num_vars == scale_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], scale_file[i]);
        absbound = scale_abs_bound;
    }
    else if (num_vars == aramco_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], aramco_file[i]);
        absbound = aramco_abs_bound;
    }
    else {
        printf("No such variable, exit\n");
//        SZ_Finalize();
        MPI_Finalize();
        return 0;
    }


    size_t  r1,r2,r3,r4,r5; //384 384 256, 500 500 100
    r1 = atof(argv[2]);
    r2 = atof(argv[3]);
    r3= atof(argv[4]);
    r4=0;
    r5=0;
   
    //End modify this part
    //End modify this part

    if (world_rank == 0) printf("Start parallel compressing ... \n");
    if (world_rank == 0) printf("size: %d\n", world_size);
    double start, end;
    double costReadOri = 0.0, costReadZip = 0.0, costWriteZip = 0.0, costWriteOut = 0.0, costComp = 0.0, costDecomp = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    size_t inSize;
    size_t nbEle;
    int status;
    float* dataIn;

    size_t est_size = r1 * r2 * r3 * sizeof(float) * num_vars;
    unsigned char* output_buffer = (unsigned char*)malloc(est_size);
    unsigned char* output_buffer_pos = output_buffer;
    int folder_index = world_rank;
    for (int i = 0; i < num_vars; i++) {
        char filename[100];
        sprintf(filename, "%s/%s", file_folder, file[i]);
	// std::cout<<filename<<std::endl;
        // Read Input Data
        if (world_rank == 0) {
            start = MPI_Wtime();
            dataIn = readFloatData(filename, &nbEle, &status);
            end = MPI_Wtime();
            // printf("file %s read time: %.2f\n", filename, end - start);
            start = MPI_Wtime();
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
            end = MPI_Wtime();
            // printf("broadcast time: %.2f\n", end - start);
        } else {
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            dataIn = (float*)malloc(nbEle * sizeof(float));
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            costReadOri += end - start;
        }

        size_t raw_size = nbEle * sizeof(float);
        memcpy(output_buffer_pos, dataIn, raw_size);
        output_buffer_pos += raw_size;
        free(dataIn);
    }
    struct stat st = {0};
    if (stat("/public/share/acnnprvuzd/data/out", &st) == -1) {
        mkdir("/public/share/acnnprvuzd/data/out", 0777);
    }
    char raw_filename[100];
    sprintf(raw_filename, "%s/raw_%d_%d.out", "/public/share/acnnprvuzd/data/out", folder_index, rand());  // Write Raw Data

    size_t total_size = output_buffer_pos - output_buffer;
    // Write Raw Data
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
    writeByteData(output_buffer, total_size, raw_filename, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end = MPI_Wtime();
        costWriteZip += end - start;
    }
    free(output_buffer);

    // Read Raw Data
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
    output_buffer = readByteData(raw_filename, &inSize, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end = MPI_Wtime();
        costReadZip += end - start;
    }
    if (inSize != total_size) {
        printf("ERROR! Broken file : %s", raw_filename);
    } else {
        remove(raw_filename);
    }
    free(output_buffer);

    if (world_rank == 0) {
        printf("Finish parallel IO bandwidth test\n");
        printf("Timecost of writing original files = %.4f seconds\n", costWriteZip);
        printf("Throughput of writing original files = %.4f GB/s\n", total_size * world_size / 1000.0 / 1000 / 1000 / costWriteZip);
        printf("Timecost of reading original files = %.4f seconds\n", costReadZip);
        printf("Throughput of reading original files = %.4f GB/s\n", total_size * world_size / 1000.0 / 1000 / 1000 / costReadZip);
    }

    MPI_Finalize();

    return 0;
}

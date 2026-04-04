#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "SZ3/api/sz.hpp"
#include "rw.h"
#include "mpi.h"

// USAGE:
// mpirun -np 16 parallel sz.config folder_num r3 r2 r1
int main(int argc, char * argv[])
{
    srand(time(0));
    size_t r5=0,r4=0,r3=0,r2=0,r1=0;
    char *cfgFile;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);    
    
    if(argc < 3)
    {
        printf("Test case: parallel_sz3 [config_file] num_vars [dimension sizes...]\n");
        printf("Example: parallel_sz3 sz.config 7 384 384 256\n");
        exit(0);
    }

    //cfgFile=argv[1];
    cfgFile=NULL;
    double eb=atof(argv[2]);
    
    if(argc>=5)
      r1 = atoi(argv[4]); //8
    if(argc>=6)
      r2 = atoi(argv[5]); //8
    if(argc>=7)
      r3 = atoi(argv[6]); //128
    if(argc>=8)
      r4 = atoi(argv[7]);
    if(argc>=9)
      r5 = atoi(argv[8]);
    
//    SZ_Init(NULL);

    if (world_rank == 0) printf ("Start parallel compressing ... \n");
    if (world_rank == 0) printf("size: %d\n", world_size);
    double start, end;
    double costReadOri = 0.0, costReadZip = 0.0, costWriteZip = 0.0, costWriteOut = 0.0, costComp = 0.0, costDecomp = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    int num_vars = atoi(argv[3]);
   

    int qmcpack8h_num_vars = 2;
    char qmcpack8h_file[2][50] = {"spin_0_truncated.bin.dat", "spin_1_truncated.bin.dat"};
    double qmcpack8h_rel_bound[2] = {1e-6, 1e-6};

    // qmacpack6k
    int qmcpack6k_num_vars = 20;
    char qmacpack6k_file[20][50] = {"s2700l300_truncated.bin.dat", "s4500l300_truncated.bin.dat", "s1200l300_truncated.bin.dat",
                                    "s300l300_truncated.bin.dat", "s4200l300_truncated.bin.dat", "s5400l300_truncated.bin.dat",
                                    "s1800l300_truncated.bin.dat", "s5700l300_truncated.bin.dat", "s4800l300_truncated.bin.dat",
                                    "s3300l300_truncated.bin.dat", "s5100l300_truncated.bin.dat", "s1500l300_truncated.bin.dat",
                                    "s600l300_truncated.bin.dat", "s0l300_truncated.bin.dat", "s3600l300_truncated.bin.dat",
                                    "s900l300_truncated.bin.dat", "s3900l300_truncated.bin.dat", "s3000l300_truncated.bin.dat",
                                    "s2100l300_truncated.bin.dat", "s2400l300_truncated.bin.dat"};
    double qmacpack6k_rel_bound[20] = {1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
                                       1e-6, 1e-6, 1e-6, 1e-6, 1e-6};

    // Hurricane
    int hurricane_num_vars = 13;
    /*char hurricane_file[13][50] = {"Uf48.bin.dat", "Vf48.bin.dat", "Wf48.bin.dat",
                                   "TCf48.bin.dat", "Pf48.bin.dat", "QVAPORf48.bin.dat",
                                   "CLOUDf48_log10.bin.dat", "QCLOUDf48_log10.bin.dat", "QICEf48_log10.bin.dat",
                                   "QRAINf48_log10.bin.dat", "QSNOWf48_log10.bin.dat", "QGRAUPf48_log10.bin.dat",
                                   "PRECIPf48_log10.bin.dat"};*/
    char hurricane_file[13][50] = {"Uf48.bin.f32", "Vf48.bin.f32", "Wf48.bin.f32",
                                   "TCf48.bin.f32", "Pf48.bin.f32", "QVAPORf48.bin.f32",
                                   "CLOUDf48.log10.bin.f32", "QCLOUDf48.log10.bin.f32", "QICEf48.log10.bin.f32",
                                   "QRAINf48.log10.bin.f32", "QSNOWf48.log10.bin.f32", "QGRAUPf48.log10.bin.f32",
                                   "PRECIPf48.log10.bin.f32"};    
double hurricane_rel_bound[13] ={7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4,7e-4, 7e-4, 7e-4, 7e-4, 7e-4};
    // miranda
    int miranda_num_vars = 7;
    char miranda_file[7][50] = {"velocityy_truncated.bin.dat", "velocityx_truncated.bin.dat", "density_truncated.bin.dat",
                                "pressure_truncated.bin.dat", "velocityz_truncated.bin.dat", "viscocity_truncated.bin.dat",
                                "diffusivity_truncated.bin.dat"};
    double miranda_rel_bound[7] = {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};

     int scale_num_vars = 12;
    char scale_file[12][50] = {"PRES-98x1200x1200.dat", "QC-98x1200x1200.log10.dat", "QG-98x1200x1200.log10.dat",
                                   "QI-98x1200x1200.log10.dat", "QR-98x1200x1200.log10.dat", "QS-98x1200x1200.log10.dat",
                                   "QV-98x1200x1200.log10.dat", "RH-98x1200x1200.dat", "T-98x1200x1200.dat",
                                   "U-98x1200x1200.dat", "V-98x1200x1200.dat", "W-98x1200x1200.dat",
                                  };
    double scale_rel_bound[12] ={1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};
    int nyx_num_vars = 6;
    char nyx_file[6][50] = {"temperature.f32", "velocity_x.f32", "velocity_y.f32",
                                "velocity_z.f32", "dark_matter_density.f32", "baryon_density.f32"};
    double nyx_rel_bound[7] = {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};
    int aramco_num_vars = 50;
    char aramco_file[50][50];
    double aramco_rel_bound[50];
    for (int i=0;i<50;i++){
        //char name[50];
        sprintf(aramco_file[i],"aramco-snapshot-%d.f32",1000+10*i);
        //salt_file[i]=name;
        aramco_rel_bound[i]=1e-3;

    }

    // assignment
    char file[100][50];
    double *rel_bound;
    //std::cout << "num_vars" << num_vars <<std::endl;
    if (num_vars == qmcpack6k_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], qmacpack6k_file[i]);
        rel_bound = qmacpack6k_rel_bound;
    } else if (num_vars == qmcpack8h_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], qmcpack8h_file[i]);
        rel_bound = qmcpack8h_rel_bound;
    } else if (num_vars == hurricane_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], hurricane_file[i]);
        rel_bound = hurricane_rel_bound;
    } else if (num_vars == miranda_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], miranda_file[i]);
        rel_bound = miranda_rel_bound;
    } else if (num_vars == nyx_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], nyx_file[i]);
        rel_bound = nyx_rel_bound;
    }
    else if (num_vars == scale_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], scale_file[i]);
        rel_bound = scale_rel_bound;
    }
    else if (num_vars == aramco_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], aramco_file[i]);
        rel_bound = aramco_rel_bound;
    }
    else {
        printf("No such variable, exit\n");
//        SZ_Finalize();
        MPI_Finalize();
        return 0;
    }
    size_t compressed_size[20];

    char folder[100] = "/home/x-jliu18/Jinyang/datasets/Hurricane";
    char filename[100];
    char zip_filename[100];
    // char out_filename[100];
    size_t inSize, outSize; 
    size_t nbEle;
    int status;
    float * dataIn;

    size_t est_compressed_size = r1 * r2 * r3 * sizeof(float) * num_vars / 5;
    unsigned char * compressed_output = (unsigned char *) malloc(est_compressed_size);
    unsigned char * compressed_output_pos = compressed_output;
    int folder_index = world_rank;
    
    SZ3::Config conf;
    if (r2 == 0) {
        conf = SZ3::Config(r1);
    } else if (r3 == 0) {
        conf = SZ3::Config(r2, r1);
    } else if (r4 == 0) {
        conf = SZ3::Config(r3, r2, r1);
    } else {
        conf = SZ3::Config(r4, r3, r2, r1);
    }
    if (cfgFile!=NULL) {
        conf.loadcfg(cfgFile);
    }
    conf.errorBoundMode=SZ3::EB_REL;
    for(int i=0; i<num_vars; i++){
        sprintf(filename, "%s/%s", folder, file[i]);
        //sprintf(filename, "%s/%d/%s", folder, folder_index, file[i]);
        //std::cout << filename << std::endl;
        //conf.relErrorBound = rel_bound[i];
        conf.relErrorBound = eb;
        // Read Input Data
        if(world_rank == 0){
            start = MPI_Wtime();
            dataIn = readFloatData(filename, &nbEle, &status);
            end = MPI_Wtime();
            //printf("file %s read time: %.2f\n", filename, end - start);
            start = MPI_Wtime();
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
            end = MPI_Wtime();
            //printf("broadcast time: %.2f\n", end - start);

        }
        else{
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            dataIn = (float *) malloc(nbEle * sizeof(float));
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costReadOri += end - start;
        }
        
        // Compress Input Data
        size_t out_size;

        //if (world_rank == 0) printf ("Compressing %s\n", filename);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();

        char *bytesOut = SZ_compress<float>(conf, dataIn, compressed_size[i]);
       // printf ("Compressing %d end.\n", world_rank);
//        unsigned char *bytesOut = SZ_compress_args(SZ_FLOAT, dataIn, &compressed_size[i], REL, 0, rel_bound[i], 0, r5, r4, r3, r2, r1);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costComp += end - start;
        }
        free (dataIn);
        memcpy(compressed_output_pos, bytesOut, compressed_size[i]);
       // printf ("memcpy %d end.\n", world_rank);
      
        compressed_output_pos += compressed_size[i];
        //free(bytesOut);
      
    }
    //printf ("total %d end.\n", world_rank);
    struct stat st = {0};
    if (stat("/home/x-jliu18/Jinyang/datasets/tmp", &st) == -1) {
        mkdir("/home/x-jliu18/Jinyang/datasets/tmp", 0777);
    }
    sprintf(zip_filename, "%s/sz3_%d_%d.out", "/home/x-jliu18/Jinyang/datasets/tmp", folder_index, rand());    // Write Compressed Data
    size_t total_size = compressed_output_pos - compressed_output;
  
    // Write Compressed Data
    MPI_Barrier(MPI_COMM_WORLD);
    //if (world_rank == 0) printf("write compressed file to disk %s \n", zip_filename);
    if(world_rank == 0) start = MPI_Wtime();
    writeByteData(compressed_output, total_size, zip_filename, &status);
    //printf ("write %d end.\n", world_rank);
  
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        end = MPI_Wtime();
        costWriteZip += end - start;
    }
    free(compressed_output);
    // Read Compressed Data
    MPI_Barrier(MPI_COMM_WORLD);
    //if (world_rank == 0) printf("read compressed file from disk %s \n", zip_filename);
    if(world_rank == 0) start = MPI_Wtime();
    compressed_output = readByteData(zip_filename, &inSize, &status);
   // printf ("read %d end.\n", world_rank);
    if (inSize != total_size) {
        printf("ERROR! Broken file : %s", zip_filename);
    } else {
        remove(zip_filename);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        end = MPI_Wtime();
        costReadZip += end - start;
    }
    compressed_output_pos = compressed_output;



    for(int i=0; i<num_vars; i++){
        // Decompress Compressed Data
        MPI_Barrier(MPI_COMM_WORLD);
        //if (world_rank == 0) printf("decompress %d-th field\n", i);
        if(world_rank == 0) start = MPI_Wtime();
        float *dataOut = SZ_decompress<float>(conf,(char*)compressed_output_pos, compressed_size[i]);
        //printf ("decomp %d end.\n", world_rank);
    
//        float *dataOut = SZ_decomprescs(SZ_FLOAT, compressed_output_pos, compressed_size[i], r5, r4, r3, r2, r1);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costDecomp += end - start; 
        }
        compressed_output_pos += compressed_size[i];
        //free(dataOut);
    }
    free(compressed_output);

    if (world_rank == 0)
    {
        printf ("SZ3 Finish parallel compressing, total compression ratio %.4g.\n", 1.0*r1*r2*r3*sizeof(float)*num_vars / total_size);
        printf("Separate ratios: ");
        for(int i=0; i<num_vars; i++){
            printf("%.4g ", 1.0*r1*r2*r3*sizeof(float) / compressed_size[i]);
        }
        printf("\n");
        printf ("Timecost of reading original files = %.4f seconds\n", costReadOri);
        printf ("Timecost of compressing using %d processes = %.4f seconds\n", world_size, costComp);
        printf ("Timecost of writing compressed files = %.4f seconds\n", costWriteZip);
        printf ("Timecost of reading compressed files = %.4f seconds\n", costReadZip);
        printf ("Timecost of decompressing using %d processes = %.4f seconds\n\n", world_size, costDecomp);

        
        printf ("Timecost of writing decompressed files = %.4f seconds\n", costWriteOut);
        
        
    }


//    SZ_Finalize();

    MPI_Finalize();

    return 0;
}

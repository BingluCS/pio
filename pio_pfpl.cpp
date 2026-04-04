#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

#include "pfpl_f32_noa_cpu_api.hpp"
#include "mpi.h"

static const char* INPUT_DIR = "/public/share/acnnprvuzd/data/sdrbench/NYX";
static const char* OUTPUT_DIR = "/public/share/acnnprvuzd/data/out";

static int ensure_dir(const char* dir)
{
    struct stat st;
    if (stat(dir, &st) == 0) {
        if (S_ISDIR(st.st_mode))
            return 0;
        errno = ENOTDIR;
        return -1;
    }
    if (mkdir(dir, 0777) == 0)
        return 0;
    if (errno == EEXIST)
        return 0;
    return -1;
}

static void build_tmp_filename(char* filename, size_t filename_len, const char* prefix, int world_rank)
{
    const char* jobid = getenv("SLURM_JOB_ID");
    if (jobid == NULL || jobid[0] == '\0')
        jobid = getenv("PBS_JOBID");
    if (jobid == NULL || jobid[0] == '\0')
        jobid = "nojid";
    snprintf(filename, filename_len, "%s/%s_%s_%d_%d_%ld.out",
             OUTPUT_DIR, prefix, jobid, world_rank, (int)getpid(), (long)time(NULL));
}

static int write_all_bytes(const unsigned char* bytes, size_t byte_length, const char* filename)
{
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    size_t written = 0;
    if (fd < 0)
        return -1;
    while (written < byte_length) {
        ssize_t count = write(fd, bytes + written, byte_length - written);
        if (count < 0) {
            if (errno == EINTR)
                continue;
            close(fd);
            return -1;
        }
        if (count == 0) {
            close(fd);
            errno = EIO;
            return -1;
        }
        written += (size_t)count;
    }
    if (close(fd) != 0)
        return -1;
    return 0;
}

static unsigned char* read_all_bytes(const char* filename, size_t* byte_length)
{
    int fd = open(filename, O_RDONLY);
    struct stat st;
    unsigned char* buf;
    size_t total = 0;
    if (fd < 0)
        return NULL;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    if (st.st_size < 0) {
        close(fd);
        errno = EIO;
        return NULL;
    }
    *byte_length = (size_t)st.st_size;
    buf = (unsigned char*)malloc((*byte_length == 0) ? 1 : *byte_length);
    if (buf == NULL) {
        close(fd);
        errno = ENOMEM;
        return NULL;
    }
    while (total < *byte_length) {
        ssize_t count = read(fd, buf + total, *byte_length - total);
        if (count < 0) {
            if (errno == EINTR)
                continue;
            free(buf);
            close(fd);
            return NULL;
        }
        if (count == 0) {
            free(buf);
            close(fd);
            errno = EIO;
            return NULL;
        }
        total += (size_t)count;
    }
    if (close(fd) != 0) {
        free(buf);
        return NULL;
    }
    return buf;
}


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

    if(argc < 4)
    {
        printf("Test case: parallel_pfpl [unused] error_bound num_vars [dimension sizes...]\n");
        printf("Example: parallel_pfpl none 1e-3 7 384 384 256\n");
        MPI_Finalize();
        return 0;
    }

    cfgFile = NULL;
    double eb = atof(argv[2]);

    if(argc>=5)
      r1 = atoi(argv[4]);
    if(argc>=6)
      r2 = atoi(argv[5]);
    if(argc>=7)
      r3 = atoi(argv[6]);
    if(argc>=8)
      r4 = atoi(argv[7]);
    if(argc>=9)
      r5 = atoi(argv[8]);

    if (world_rank == 0) printf ("Start parallel compressing ... \n");
    if (world_rank == 0) printf("size: %d\n", world_size);
    double start, end;
    double costReadOri = 0.0, costReadZip = 0.0, costWriteZip = 0.0, costWriteOut = 0.0, costComp = 0.0, costDecomp = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    int num_vars = atoi(argv[3]);

    int qmcpack8h_num_vars = 2;
    char qmcpack8h_file[2][50] = {"spin_0_truncated.bin.dat", "spin_1_truncated.bin.dat"};
    double qmcpack8h_rel_bound[2] = {1e-6, 1e-6};

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

    int hurricane_num_vars = 13;
    char hurricane_file[13][50] = {"Uf48.bin.f32", "Vf48.bin.f32", "Wf48.bin.f32",
                                   "TCf48.bin.f32", "Pf48.bin.f32", "QVAPORf48.bin.f32",
                                   "CLOUDf48.log10.bin.f32", "QCLOUDf48.log10.bin.f32", "QICEf48.log10.bin.f32",
                                   "QRAINf48.log10.bin.f32", "QSNOWf48.log10.bin.f32", "QGRAUPf48.log10.bin.f32",
                                   "PRECIPf48.log10.bin.f32"};
    double hurricane_rel_bound[13] ={7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4, 7e-4,7e-4, 7e-4, 7e-4, 7e-4, 7e-4};

    int miranda_num_vars = 7;
    char miranda_file[7][50] = {"velocityy_truncated.bin.dat", "velocityx_truncated.bin.dat", "density_truncated.bin.dat",
                                "pressure_truncated.bin.dat", "velocityz_truncated.bin.dat", "viscocity_truncated.bin.dat",
                                "diffusivity_truncated.bin.dat"};
    double miranda_rel_bound[7] = {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};

    int scale_num_vars = 12;
    char scale_file[12][50] = {"PRES-98x1200x1200.dat", "QC-98x1200x1200.log10.dat", "QG-98x1200x1200.log10.dat",
                               "QI-98x1200x1200.log10.dat", "QR-98x1200x1200.log10.dat", "QS-98x1200x1200.log10.dat",
                               "QV-98x1200x1200.log10.dat", "RH-98x1200x1200.dat", "T-98x1200x1200.dat",
                               "U-98x1200x1200.dat", "V-98x1200x1200.dat", "W-98x1200x1200.dat"};
    double scale_rel_bound[12] ={1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};

    int nyx_num_vars = 6;
    char nyx_file[6][50] = {"temperature.f32","velocity_y.f32",
                            "velocity_z.f32", "velocity_x.f32"};
    double nyx_rel_bound[7] = {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};

    int aramco_num_vars = 50;
    char aramco_file[50][50];
    double aramco_rel_bound[50];
    for (int i=0;i<50;i++){
        sprintf(aramco_file[i],"aramco-snapshot-%d.f32",1000+10*i);
        aramco_rel_bound[i]=1e-3;
    }

    char file[100][50];
    double *rel_bound;
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
    } else if (num_vars == scale_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], scale_file[i]);
        rel_bound = scale_rel_bound;
    } else if (num_vars == aramco_num_vars) {
        for (int i = 0; i < num_vars; i++) strcpy(file[i], aramco_file[i]);
        rel_bound = aramco_rel_bound;
    } else {
        printf("No such variable, exit\n");
        MPI_Finalize();
        return 0;
    }

    size_t compressed_size[100];
    size_t original_size[100];

    const char *folder = INPUT_DIR;
    char filename[512];
    char zip_filename[512];
    size_t inSize;
    size_t nbEle;
    size_t total_original_size = 0;
    size_t total_size = 0;
    int status;
    float * dataIn;
    float * dataOut;
    unsigned char * compressed_output = NULL;
    unsigned char * compressed_output_pos;
    unsigned char * bytesOut;
    int folder_index = world_rank;
    float threshold = INFINITY;

    if (cfgFile != NULL) {
        threshold = (float)atof(cfgFile);
    }
    num_vars = 4;
    for(int i=0; i<num_vars; i++){
        sprintf(filename, "%s/%s", folder, file[i]);

        if(world_rank == 0){
            start = MPI_Wtime();
            dataIn = readFloatData(filename, &nbEle, &status);
            start = MPI_Wtime();
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
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

        original_size[i] = nbEle * sizeof(float);
        total_original_size += original_size[i];

        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();
        bytesOut = pfpl_f32_noa_compress(dataIn, nbEle, (float)eb, threshold, &compressed_size[i]);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costComp += end - start;
        }
        free(dataIn);

        if (bytesOut == NULL || compressed_size[i] == 0) {
            printf("PFPL compression failed for %s\n", filename);
            if (compressed_output != NULL) free(compressed_output);
            MPI_Finalize();
            return 1;
        }

        compressed_output = (unsigned char *)realloc(compressed_output, total_size + compressed_size[i]);
        memcpy(compressed_output + total_size, bytesOut, compressed_size[i]);
        total_size += compressed_size[i];
        free(bytesOut);
    }

    if (ensure_dir(OUTPUT_DIR) != 0) {
        printf("ERROR! Failed to prepare output dir %s: %s\n", OUTPUT_DIR, strerror(errno));
        free(compressed_output);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    build_tmp_filename(zip_filename, sizeof(zip_filename), "pfpl", folder_index);

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0) start = MPI_Wtime();
    if (write_all_bytes(compressed_output, total_size, zip_filename) != 0) {
        printf("ERROR! Failed to write compressed file %s: %s\n", zip_filename, strerror(errno));
        free(compressed_output);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        end = MPI_Wtime();
        costWriteZip += end - start;
    }
    free(compressed_output);

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0) start = MPI_Wtime();
    compressed_output = read_all_bytes(zip_filename, &inSize);
    if (compressed_output == NULL) {
        printf("ERROR! Failed to read compressed file %s: %s\n", zip_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (inSize != total_size) {
        printf("ERROR! Broken file : %s (expected %zu bytes, got %zu bytes)\n", zip_filename, total_size, inSize);
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
        size_t dec_nbEle = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();
        dataOut = pfpl_f32_noa_decompress(compressed_output_pos, compressed_size[i], &dec_nbEle);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costDecomp += end - start;
        }
        compressed_output_pos += compressed_size[i];
        if (dataOut != NULL) free(dataOut);
    }
    free(compressed_output);

    if (world_rank == 0)
    {
        printf ("PFPL Finish parallel compressing, total compression ratio %.4g.\n", 1.0 * total_original_size / total_size);
        printf("Separate ratios: ");
        for(int i=0; i<num_vars; i++){
            printf("%.4g ", 1.0 * original_size[i] / compressed_size[i]);
        }
        printf("\n");
        printf ("Timecost of reading original files = %.4f seconds\n", costReadOri);
        printf ("Timecost of compressing using %d processes = %.4f seconds\n", world_size, costComp);
        printf ("Timecost of writing compressed files = %.4f seconds\n", costWriteZip);
        printf ("Timecost of reading compressed files = %.4f seconds\n", costReadZip);
        printf ("Timecost of decompressing using %d processes = %.4f seconds\n\n", world_size, costDecomp);
        printf ("Timecost of writing decompressed files = %.4f seconds\n", costWriteOut);
    }

    MPI_Finalize();
    return 0;
}

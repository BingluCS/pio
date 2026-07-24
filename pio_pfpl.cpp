#include <iostream>
#include <stdio.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>

#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif

#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#include "pfpl_f32_noa_cpu_api.hpp"
#include "mpi.h"

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

static int writeData(const unsigned char* bytes, size_t byte_length, const char* filename)
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) return -1;
    fout.write(reinterpret_cast<const char*>(bytes), byte_length);
    if (!fout.good()) return -1;
    return 0;
}


unsigned char* readData(char *srcFilePath, size_t *nbEle, int *status)
{
    size_t inSize;
    std::ifstream inFile(srcFilePath, std::ios::binary);
    inFile.seekg(0, std::ios::end);
    inSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    unsigned char *daBuf = (unsigned char *)malloc(inSize);
    if (!inFile.read(reinterpret_cast<char*>(daBuf), inSize)) {
        std::cerr << "Failed to read file" << std::endl;
        free(daBuf);
        return NULL;
    }
    *nbEle = inSize / 4;
    return daBuf;
}

static unsigned char* readBytes(const char *srcFilePath, size_t *byteSize)
{
    std::ifstream inFile(srcFilePath, std::ios::binary);
    if (!inFile.good()) return NULL;
    inFile.seekg(0, std::ios::end);
    std::streamoff inSize = inFile.tellg();
    if (inSize <= 0) return NULL;
    inFile.seekg(0, std::ios::beg);

    unsigned char *daBuf = (unsigned char *)malloc(static_cast<size_t>(inSize));
    if (daBuf == NULL) return NULL;
    if (!inFile.read(reinterpret_cast<char*>(daBuf), inSize)) {
        free(daBuf);
        return NULL;
    }
    *byteSize = static_cast<size_t>(inSize);
    return daBuf;
}

static double dataRange(const float *data, size_t nbEle)
{
    if (nbEle == 0) return 0.0;
    float minVal = data[0];
    float maxVal = data[0];
    for (size_t i = 1; i < nbEle; i++) {
        const float v = data[i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    return static_cast<double>(maxVal - minVal);
}

static size_t pfpl_f32_abs_compress_bound(size_t nbEle)
{
    return pfpl_f32_noa_max_compressed_size(nbEle) + 64;
}

static size_t pfpl_f32_abs_compress_into(const float *data, size_t nbEle, float absErrorBound,
                                         float threshold, unsigned char *encoded, size_t encodedCapacity)
{
    if (data == NULL || encoded == NULL || nbEle == 0 ||
        nbEle * sizeof(float) >= 2147221529ULL ||
        encodedCapacity < pfpl_f32_abs_compress_bound(nbEle)) return 0;

    int encodedSize = 0;
    pfpl_f32_noa_encode(reinterpret_cast<const byte*>(data),
                        static_cast<int>(nbEle * sizeof(float)),
                        encoded, &encodedSize, absErrorBound, threshold, 1.0f);
    if (encodedSize <= 0) {
        return 0;
    }
    return static_cast<size_t>(encodedSize);
}

static float* pfpl_f32_abs_decompress(const unsigned char *input, size_t inputSize, size_t *nbEle)
{
    return pfpl_f32_noa_decompress(input, inputSize, nbEle);
}

void usage() {
    printf("Test case: pio_pfpl -e error_bound -d dataset_name -i list_file -o output_dir -n nums [-z [parts]] [-1 r1 | -2 r1 r2 | -3 r1 r2 r3 | -4 r1 r2 r3 r4]\n");
    printf("Example: pio_pfpl -e 1e-3 -d nyx -i /path/to/list.txt -o /path/to/out -n 4 -z 4 -3 r1 r2 r3\n");
}
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(argc < 6)
    {
        if (world_rank == 0) usage();
        MPI_Finalize();
        return 0;
    }
    size_t r4 = 1;
    size_t r3 = 1;
    size_t r2 = 1;
    size_t r1 = 1;

    double eb = 1e-3;
    char *dataset_name = NULL;
    char *folder = NULL;
    const char *output_dir = "./out";
    int num_vars = 0;
    bool split_z = false;
    size_t z_parts = 1;

    for (int i = 1; i < argc; i++) {
        switch (argv[i][1]) {
            case '1':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1) usage();
                break;
            case '2':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r2) != 1)
                    usage();
                break;
            case '3':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                    sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1)
                    usage();
                break;
            case '4':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                    sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1 ||
                    ++i == argc || sscanf(argv[i], "%zu", &r4) != 1)
                    usage();
                break;
            case 'e':
                if (++i == argc) usage();
                eb = atof(argv[i]);
                break;
            case 'i':
                if (++i == argc) usage();
                folder = argv[i];
                break;
            case 'o':
                if (++i == argc) usage();
                output_dir = argv[i];
                break;
            case 'n':
                if (++i == argc || sscanf(argv[i], "%d", &num_vars) != 1) usage();
                break;
            case 'd':
                if (++i == argc) usage();
                dataset_name = argv[i];
                break;
            case 'z':
                split_z = true;
                z_parts = 2;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    if (sscanf(argv[++i], "%zu", &z_parts) != 1) usage();
                }
                break;
            default:
                usage();
                break;
        }
    }

    if (dataset_name == NULL || folder == NULL || num_vars <= 0) {
        if (world_rank == 0) {
            printf("ERROR: Missing required arguments: -d (dataset_name), -i (list file), or -n (num_vars).\n");
            usage();
        }
        MPI_Finalize();
        return 1;
    }
    if (split_z && (r4 != 1 || r3 <= 1 || z_parts < 2 || z_parts > r3)) {
        if (world_rank == 0) printf("ERROR: -z currently supports only 3D data with z dimension > 1 and 2 <= parts <= z dimension.\n");
        MPI_Finalize();
        return 1;
    }

    std::ifstream list_file(folder);
    if (!list_file.good()) {
        if (world_rank == 0) printf("ERROR! Input information folder %s does not exist or is not accessible.\n", folder);
        MPI_Finalize();
        return 1;
    }


    std::string line;
    std::string input_dir;
    char file[100][50];
    bool found_dataset = false;
    int i = 0;
    while (std::getline(list_file, line)) {
        if (!found_dataset) {
            if (strncasecmp(line.c_str(), dataset_name, strlen(dataset_name)) == 0) {
                found_dataset = true;
                std::istringstream iss(line);
                std::string dummy;
                iss >> dummy >> input_dir;
            }
        } else {
            sscanf(line.c_str(), "%s", file[i]);
            i++;
            if (i >= num_vars) break;
        }
    }
    list_file.close();

    if (world_rank == 0) printf("Start parallel compressing ... \n");
    if (world_rank == 0) printf("size: %d\n", world_size);

    MPI_Barrier(MPI_COMM_WORLD);

    size_t compressed_size[100] = {0};
    size_t original_size[100] = {0};
    size_t total_original_size = 0;
    size_t total_size = 0;
    std::vector<unsigned char> compressed_output;
    float *dataIn;
    const size_t active_parts = split_z ? z_parts : 1;
    if (num_vars * active_parts > 200) {
        if (world_rank == 0) printf("ERROR: num_vars * z parts must be <= 200.\n");
        MPI_Finalize();
        return 1;
    }
    std::vector<size_t> part_compressed_size(num_vars * active_parts, 0);
    std::vector<unsigned char*> part_output(num_vars * active_parts, NULL);
    std::vector<size_t> z_offsets(active_parts + 1, 0);
    for (size_t p = 0; p <= active_parts; p++) {
        z_offsets[p] = split_z ? (r3 * p / active_parts) : (p == 0 ? 0 : r3);
    }
    if (world_rank == 0 && split_z) {
        printf("z-axis split into %zu slabs:", active_parts);
        for (size_t p = 0; p < active_parts; p++) printf(" %zu", z_offsets[p + 1] - z_offsets[p]);
        printf("\n");
    }

    const size_t expected_nbEle = r1 * r2 * r3 * r4;
    size_t cmpBufferCap = pfpl_f32_abs_compress_bound(expected_nbEle);
    if (split_z) {
        cmpBufferCap = 0;
        const size_t xy = r1 * r2;
        for (size_t p = 0; p < active_parts; p++) {
            const size_t slab_elements = (z_offsets[p + 1] - z_offsets[p]) * xy;
            cmpBufferCap = std::max(cmpBufferCap, pfpl_f32_abs_compress_bound(slab_elements));
        }
    }
    for (size_t idx = 0; idx < part_output.size(); idx++) {
        part_output[idx] = static_cast<unsigned char *>(malloc(cmpBufferCap));
        if (part_output[idx] == NULL) {
            printf("ERROR! Failed to allocate PFPL output buffer on rank %d\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    double start, end;
    double costReadOri = 0.0, costComp = 0.0;
    double costBoundPrep = 0.0;
    double costReadZip = 0.0, costWriteZip = 0.0, costDecomp = 0.0;
    double localCompTotal = 0.0;
    double compMin[100] = {0.0};
    double compMax[100] = {0.0};
    double compAvg[100] = {0.0};

    size_t nbEle;
    for(int i = 0; i < num_vars; i++) {
        std::string filename = input_dir + "/" + file[i];
        if(world_rank == 0){
            start = MPI_Wtime();
            int readStatus = 0;
            dataIn = reinterpret_cast<float*>(readData(const_cast<char *>(filename.c_str()), &nbEle, &readStatus));
            if (dataIn == NULL || nbEle == 0) {
                printf("ERROR! Failed to read input file %s\n", filename.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            if (nbEle != expected_nbEle) {
                printf("ERROR! Dimension mismatch for %s: file has %zu elements, but dataset %s expects %zu (%zu x %zu x %zu x %zu)\n",
                       filename.c_str(), nbEle, dataset_name, expected_nbEle, r1, r2, r3, r4);
                free(dataIn);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        else{
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            dataIn = (float *) malloc(nbEle * sizeof(float));
            if (dataIn == NULL) {
                printf("ERROR! Failed to allocate input buffer on rank %d\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
#if defined(__linux__)
            uintptr_t pageBase = reinterpret_cast<uintptr_t>(dataIn) & ~static_cast<uintptr_t>(4095);
            madvise(reinterpret_cast<void *>(pageBase),
                    nbEle * sizeof(float) + (reinterpret_cast<uintptr_t>(dataIn) - pageBase),
                    MADV_HUGEPAGE);
#endif
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costReadOri += end - start;
        }

        // accumulate original size for compression ratio calculation
        original_size[i] = nbEle * sizeof(float);
        total_original_size += original_size[i];

        double absErrorBound = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();
        if(world_rank == 0) {
            absErrorBound = eb * dataRange(dataIn, nbEle);
            if (absErrorBound <= 0.0) absErrorBound = eb;
        }
        MPI_Bcast(&absErrorBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costBoundPrep += end - start;
        }

        if(world_rank == 0) start = MPI_Wtime();
        double localCompStart = MPI_Wtime();
        compressed_size[i] = 0;
        const size_t xy = r1 * r2;
        if (split_z) {
            for (size_t p = 0; p < active_parts; p++) {
                const size_t idx = i * active_parts + p;
                const size_t slab_elements = (z_offsets[p + 1] - z_offsets[p]) * xy;
                part_compressed_size[idx] = pfpl_f32_abs_compress_into(
                    dataIn + z_offsets[p] * xy, slab_elements, static_cast<float>(absErrorBound),
                    INFINITY, part_output[idx], cmpBufferCap);
                compressed_size[i] += part_compressed_size[idx];
            }
        } else {
            const size_t idx = i;
            part_compressed_size[idx] = pfpl_f32_abs_compress_into(
                dataIn, nbEle, static_cast<float>(absErrorBound), INFINITY,
                part_output[idx], cmpBufferCap);
            compressed_size[i] = part_compressed_size[idx];
        }
        double localCompTime = MPI_Wtime() - localCompStart;
        localCompTotal += localCompTime;
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costComp += end - start;
        }

        double compSum = 0.0;
        MPI_Reduce(&localCompTime, &compMin[i], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localCompTime, &compMax[i], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localCompTime, &compSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(world_rank == 0) compAvg[i] = compSum / world_size;
        free(dataIn);

        if (compressed_size[i] == 0) {
            printf("PFPL compression failed for %s\n", filename.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        total_size += compressed_size[i];
    }

    for (size_t idx = 0; idx < part_output.size(); idx++) {
        compressed_output.insert(compressed_output.end(), part_output[idx],
                                 part_output[idx] + part_compressed_size[idx]);
        free(part_output[idx]);
    }

    char zip_filename[1024];
    if (ensure_dir(output_dir) != 0) {
        printf("ERROR! Failed to prepare output dir %s: %s\n", output_dir, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    snprintf(zip_filename, sizeof(zip_filename), "%s/%s_%d_%d_%ld.out",
             output_dir, "pfpl", world_rank, (int)getpid(), (long)time(NULL));
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0) start = MPI_Wtime();
    if (writeData(compressed_output.data(), compressed_output.size(), zip_filename) != 0) {
        printf("ERROR! Failed to write compressed file %s: %s\n", zip_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        end = MPI_Wtime();
        costWriteZip += end - start;
    }
    compressed_output.clear();
    compressed_output.shrink_to_fit();

    size_t inSize = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0) start = MPI_Wtime();
    unsigned char *compressed_input = readBytes(zip_filename, &inSize);
    if (compressed_input == NULL) {
        printf("ERROR! Failed to read compressed file %s: %s\n", zip_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (inSize != total_size) {
        printf("ERROR! Broken file : %s (expected %zu bytes, got %zu bytes)\n", zip_filename, total_size, inSize);
        free(compressed_input);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    remove(zip_filename);
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank == 0){
        end = MPI_Wtime();
        costReadZip += end - start;
    }

    unsigned char *compressed_input_pos = compressed_input;
    float *dataOutArr[200] = {nullptr};
    for(int i = 0; i < num_vars; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();
        bool decomp_ok = true;
        if (split_z) {
            const size_t xy = r1 * r2;
            for (size_t p = 0; p < active_parts; p++) {
                const size_t idx = i * active_parts + p;
                size_t dec_nbEle = 0;
                dataOutArr[idx] = pfpl_f32_abs_decompress(compressed_input_pos, part_compressed_size[idx], &dec_nbEle);
                compressed_input_pos += part_compressed_size[idx];
                const size_t expected_part = (z_offsets[p + 1] - z_offsets[p]) * xy;
                if (dataOutArr[idx] == NULL || dec_nbEle != expected_part) decomp_ok = false;
            }
        } else {
            size_t dec_nbEle = 0;
            dataOutArr[i] = pfpl_f32_abs_decompress(compressed_input_pos, part_compressed_size[i], &dec_nbEle);
            compressed_input_pos += part_compressed_size[i];
            if (dataOutArr[i] == NULL || dec_nbEle != expected_nbEle) decomp_ok = false;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costDecomp += end - start;
        }
        if (!decomp_ok) {
            printf("PFPL decompression failed for field %d\n", i);
            free(compressed_input);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    for (size_t idx = 0; idx < part_output.size(); idx++) free(dataOutArr[idx]);
    free(compressed_input);

    double totalCompMin = 0.0;
    double totalCompMax = 0.0;
    double totalCompSum = 0.0;
    MPI_Reduce(&localCompTotal, &totalCompMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localCompTotal, &totalCompMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localCompTotal, &totalCompSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        printf("PFPL Finish parallel compressing on %s, total compression ratio %.4g.\n", dataset_name, 1.0 * total_original_size / total_size);
        printf("Separate ratios: ");
        for(int i = 0; i < num_vars; i++){
            printf("%.4g ", 1.0 * original_size[i] / compressed_size[i]);
        }
        printf("\n");
        printf("Timecost of reading original files = %.4f seconds\n", costReadOri);
        printf("Timecost of preparing relative error bounds = %.4f seconds\n", costBoundPrep);
        printf("Timecost of compressing using %d processes = %.4f seconds\n", world_size, costComp);
        printf("Local compression time total min/max/avg = %.4f %.4f %.4f seconds\n",
               totalCompMin, totalCompMax, totalCompSum / world_size);
        printf("Local compression time per variable min/max/avg:\n");
        for(int i = 0; i < num_vars; i++){
            printf("  %s: %.4f %.4f %.4f seconds\n", file[i], compMin[i], compMax[i], compAvg[i]);
        }
        printf("Timecost of writing compressed files = %.4f seconds\n", costWriteZip);
        printf("Timecost of reading compressed files = %.4f seconds\n", costReadZip);
        printf("Timecost of decompressing using %d processes = %.4f seconds\n", world_size, costDecomp);
    }

    MPI_Finalize();
    return 0;
}

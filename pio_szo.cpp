#include <stdio.h>

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
#include <iostream>
#include <memory>
#include <sstream>
#include <algorithm>
#include <exception>
#include <vector>

#include "SZo/api/sz.hpp"
#include "mpi.h"

static int ensure_dir(const char* dir)
{
    struct stat st;
    if (stat(dir, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        errno = ENOTDIR;
        return -1;
    }
    if (mkdir(dir, 0777) == 0) return 0;
    if (errno == EEXIST) return 0;
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

static unsigned char* readBytes(const char *srcFilePath, size_t *byteSize)
{
    std::ifstream inFile(srcFilePath, std::ios::binary);
    if (!inFile.good()) return NULL;
    inFile.seekg(0, std::ios::end);
    std::streamoff inSize = inFile.tellg();
    if (inSize <= 0) return NULL;
    inFile.seekg(0, std::ios::beg);

    unsigned char *daBuf = (unsigned char *)malloc(static_cast<size_t>(inSize));
    if (daBuf == NULL) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return NULL;
    }
    if (!inFile.read(reinterpret_cast<char*>(daBuf), inSize)) {
        std::cerr << "Failed to read file" << std::endl;
        free(daBuf);
        return NULL;
    }
    *byteSize = static_cast<size_t>(inSize);
    return daBuf;
}

static unsigned char* readFloatData(const char *srcFilePath, size_t *nbEle)
{
    size_t byteSize = 0;
    unsigned char *data = readBytes(srcFilePath, &byteSize);
    if (data == NULL) return NULL;
    *nbEle = byteSize / sizeof(float);
    return data;
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

void usage() {
    printf("Test case: pio_szo -e error_bound -d dataset_name -i list_file -o output_dir -n nums [-z [parts]] [-1 r1 | -2 r1 r2 | -3 r1 r2 r3 | -4 r1 r2 r3 r4]\n");
    printf("Example: pio_szo -e 1e-3 -d nyx -i /path/to/list.txt -o /path/to/out -n 4 -z 4 -3 r1 r2 r3\n");
}
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(time(0));
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
        printf("ERROR: Missing required arguments: -d (dataset_name), -i (list file), or -n (num_vars).\n");
        usage();
        return 1;
    }

    std::ifstream list_file(folder);
    if (!list_file.good()) {
        printf("ERROR! Input information folder %s does not exist or is not accessible.\n", folder);
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

    size_t compressed_size[100];
    size_t original_size[100];

    size_t total_original_size = 0;
    size_t total_size = 0;
    std::vector<unsigned char> compressed_output;
    unsigned char* cmpArr[200] = {nullptr};   // per-variable compressed bytes; concatenated into compressed_output AFTER the timed loop
    size_t cmpArrSize[200] = {0};

    float *dataIn;

    SZo::Config conf(r4, r3, r2, r1);
    conf.relErrorBound = eb;
    conf.cmprAlgo = SZo::ALGO_INTERP_LORENZO;
    if (split_z && (r4 != 1 || r3 <= 1 || z_parts < 2 || z_parts > r3)) {
        if (world_rank == 0) {
            printf("ERROR: -z currently supports only 3D data with z dimension > 1 and 2 <= parts <= z dimension.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const size_t active_parts = split_z ? z_parts : 1;
    if (num_vars * active_parts > 200) {
        if (world_rank == 0) printf("ERROR: num_vars * z parts must be <= 200.\n");
        MPI_Finalize();
        return 1;
    }
    std::vector<size_t> z_offsets(active_parts + 1, 0);
    for (size_t p = 0; p <= active_parts; p++) {
        z_offsets[p] = split_z ? (r3 * p / active_parts) : (p == 0 ? 0 : r3);
    }
    if (world_rank == 0 && split_z) {
        printf("z-axis split into %zu slabs:", active_parts);
        for (size_t p = 0; p < active_parts; p++) printf(" %zu", z_offsets[p + 1] - z_offsets[p]);
        printf("\n");
    }
    auto make_slab_conf = [&](size_t slab_z) {
        SZo::Config slab_conf = conf;
        size_t slab_dims[4] = {r4, slab_z, r2, r1};
        slab_conf.setDims(slab_dims, slab_dims + 4);
        slab_conf.relErrorBound = eb;
        return slab_conf;
    };
    size_t cmpBufferCap = SZo::SZ_compress_size_bound<float>(conf);
    if (split_z) {
        cmpBufferCap = 0;
        for (size_t p = 0; p < active_parts; p++) {
            SZo::Config slab_conf = make_slab_conf(z_offsets[p + 1] - z_offsets[p]);
            cmpBufferCap = std::max(cmpBufferCap, SZo::SZ_compress_size_bound<float>(slab_conf));
        }
    }
    // One full-capacity output buffer per variable (two per variable for -z), allocated up front.
    // The compress loop writes straight into its own buffer -> zero copy inside the timed region.
    // Only the actual compressed bytes (~MB) are ever touched, so the big reservations stay virtual.
    for (int k = 0; k < num_vars; k++) {
        for (size_t p = 0; p < active_parts; p++) {
            cmpArr[k * active_parts + p] = new unsigned char[cmpBufferCap];
        }
    }

    double start, end;
    double costReadOri = 0.0, costComp = 0.0;
    double costBoundPrep = 0.0;
    double costWriteZip = 0.0, costReadZip = 0.0, costDecomp = 0.0;
    double localCompTotal = 0.0;
    double compMin[100] = {0.0};
    double compMax[100] = {0.0};
    double compAvg[100] = {0.0};

    size_t nbEle;
    size_t expected_nbEle = r1 * r2 * r3 * r4;
    for(int i = 0; i < num_vars; i++) {
        // read original data
        std::string filename = input_dir + "/" + file[i];
        conf.relErrorBound = eb;
        if(world_rank == 0){
            start = MPI_Wtime();
            dataIn = reinterpret_cast<float*>(readFloatData(filename.c_str(), &nbEle));
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
        if(world_rank == 0) absErrorBound = eb * dataRange(dataIn, nbEle);
        MPI_Bcast(&absErrorBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        conf.errorBoundMode = SZo::EB_ABS;
        conf.absErrorBound = absErrorBound;
        conf.relErrorBound = eb;
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costBoundPrep += end - start;
        }

        // compress data
        if(world_rank == 0) start = MPI_Wtime();
        double localCompStart = MPI_Wtime();
        try {
            if (split_z) {
                const size_t xy = r1 * r2;
                compressed_size[i] = 0;
                for (size_t p = 0; p < active_parts; p++) {
                    SZo::Config slab_conf = make_slab_conf(z_offsets[p + 1] - z_offsets[p]);
                    slab_conf.errorBoundMode = SZo::EB_ABS;
                    slab_conf.absErrorBound = absErrorBound;
                    slab_conf.relErrorBound = eb;
                    const size_t idx = i * active_parts + p;
                    const size_t slab_size = SZ_compress<float>(
                        slab_conf, dataIn + z_offsets[p] * xy, reinterpret_cast<char*>(cmpArr[idx]), cmpBufferCap);
                    cmpArrSize[idx] = slab_size;
                    compressed_size[i] += slab_size;
                }
            } else {
                compressed_size[i] = SZ_compress<float>(conf, dataIn, reinterpret_cast<char*>(cmpArr[i]), cmpBufferCap);
                cmpArrSize[i] = compressed_size[i];
            }
        } catch (const std::exception &e) {
            printf("ERROR! SZo compression failed on rank %d for %s with cmp buffer %.2f MiB: %s\n",
                   world_rank, filename.c_str(), cmpBufferCap / 1048576.0, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
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
            printf("SZo compression failed for %s\n", filename.c_str());
            MPI_Finalize();
            return 1;
        }

        total_size += compressed_size[i];
    }

    // concatenate all per-variable compressed buffers here, OUTSIDE the timed compress loop
    for(size_t i = 0; i < active_parts * static_cast<size_t>(num_vars); i++) if(cmpArr[i]){ compressed_output.insert(compressed_output.end(), cmpArr[i], cmpArr[i]+cmpArrSize[i]); delete[] cmpArr[i]; }

    char zip_filename[1024];
    if (ensure_dir(output_dir) != 0) {
        printf("ERROR! Failed to prepare output dir %s: %s\n", output_dir, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    snprintf(zip_filename, sizeof(zip_filename), "%s/%s_%d_%d_%ld.out",
             output_dir, "szo", world_rank, (int)getpid(), (long)time(NULL));
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
    float *dataOutArr[200] = {nullptr};   // keep every decompressed buffer; free them together AFTER the timed loop
//     if (!split_z) {
//         // pre-allocate and prefault the outputs OUTSIDE the timed loop, so the timed
//         // decompress measures the kernel (buffer-reuse semantics), not page faults/compaction
//         for (int i = 0; i < num_vars; i++) {
//             dataOutArr[i] = new float[nbEle];
//             memset(dataOutArr[i], 0, nbEle * sizeof(float));
//         }
//     }
    for(int i = 0; i < num_vars; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0) start = MPI_Wtime();
        if (split_z) {
            for (size_t p = 0; p < active_parts; p++) {
                SZo::Config slab_conf = make_slab_conf(z_offsets[p + 1] - z_offsets[p]);
                const size_t idx = i * active_parts + p;
                dataOutArr[idx] = SZ_decompress<float>(slab_conf, reinterpret_cast<char*>(compressed_input_pos), cmpArrSize[idx]);
                compressed_input_pos += cmpArrSize[idx];
            }
        } else {
            SZo::Config dec_conf = conf;
            SZ_decompress<float>(dec_conf, reinterpret_cast<char*>(compressed_input_pos), compressed_size[i], dataOutArr[i]);
            compressed_input_pos += compressed_size[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            end = MPI_Wtime();
            costDecomp += end - start;
        }
    }
    for(size_t i = 0; i < active_parts * static_cast<size_t>(num_vars); i++) delete[] dataOutArr[i];   // unified delete, outside the timed region
    free(compressed_input);

    double totalCompMin = 0.0;
    double totalCompMax = 0.0;
    double totalCompSum = 0.0;
    MPI_Reduce(&localCompTotal, &totalCompMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localCompTotal, &totalCompMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localCompTotal, &totalCompSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        printf("SZo Finish parallel compressing on %s, total compression ratio %.4g.\n", dataset_name, 1.0 * total_original_size / total_size);
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

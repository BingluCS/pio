#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>

#include "mpi.h"
#include "zfp.h"

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

unsigned char* readData(const char *srcFilePath, size_t *nbEle, int *status)
{
    size_t inSize;
    std::ifstream inFile(srcFilePath, std::ios::binary);
    inFile.seekg(0, std::ios::end);
    inSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    unsigned char *daBuf = (unsigned char *)malloc(inSize);
    if (!inFile.read(reinterpret_cast<char*>(daBuf), inSize)) {
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

static size_t zfp_compress_bound_3D(double tolerance, size_t dimx, size_t dimy, size_t dimz)
{
    zfp_field* field = zfp_field_3d(NULL, zfp_type_float, dimx, dimy, dimz);
    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tolerance);
    const size_t buffer_size = zfp_stream_maximum_size(zfp, field);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    return buffer_size;
}

static size_t zfp_compress_3D(float* array, double tolerance,
                              size_t dimx, size_t dimy, size_t dimz,
                              unsigned char* buffer, size_t buffer_size)
{
    zfp_field* field = zfp_field_3d(array, zfp_type_float, dimx, dimy, dimz);
    zfp_stream* zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tolerance);
    bitstream* stream = stream_open(buffer, buffer_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    const size_t zfp_size = zfp_compress(zfp, field);

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return zfp_size;
}

static float* zfp_decompress_3D(unsigned char* comp_data, double tolerance, size_t buffer_size, size_t dimx, size_t dimy, size_t dimz)
{
    zfp_field* field;
    zfp_stream* zfp;
    bitstream* stream;

    float* array = (float*)malloc(dimx * dimy * dimz * sizeof(float));
    field = zfp_field_3d(array, zfp_type_float, dimx, dimy, dimz);
    zfp = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(zfp, tolerance);

    stream = stream_open((void*)comp_data, buffer_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (!zfp_decompress(zfp, field)) {
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
        free(array);
        return NULL;
    }

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return array;
}

static void usage()
{
    printf("Test case: pio_zfp -e error_bound -d dataset_name -i list_file -o output_dir -n nums [-z [parts]] [-1 r1 | -2 r1 r2 | -3 r1 r2 r3 | -4 r1 r2 r3 r4]\n");
    printf("Example: pio_zfp -e 1e-3 -d nyx -i /path/to/list.txt -o /path/to/out -n 4 -z 4 -3 r1 r2 r3\n");
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    size_t r1 = 1, r2 = 1, r3 = 1, r4 = 1;

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 6) {
        if (world_rank == 0) usage();
        MPI_Finalize();
        return 0;
    }

    double eb = 1e-3;
    const char *dataset_name = NULL;
    const char *folder = NULL;
    const char *output_dir = "./out";
    int num_vars = 0;
    bool split_z = false;
    size_t z_parts = 1;

    for (int i = 1; i < argc; i++) {
        switch (argv[i][1]) {
            case '1':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1) return 1;
                break;
            case '2':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r2) != 1) return 1;
                break;
            case '3':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1) return 1;
                break;
            case '4':
                if (++i == argc || sscanf(argv[i], "%zu", &r1) != 1 || ++i == argc ||
                    sscanf(argv[i], "%zu", &r2) != 1 || ++i == argc || sscanf(argv[i], "%zu", &r3) != 1 ||
                    ++i == argc || sscanf(argv[i], "%zu", &r4) != 1) return 1;
                break;
            case 'e':
                if (++i == argc) return 1;
                eb = atof(argv[i]);
                break;
            case 'i':
                if (++i == argc) return 1;
                folder = argv[i];
                break;
            case 'o':
                if (++i == argc) return 1;
                output_dir = argv[i];
                break;
            case 'n':
                if (++i == argc || sscanf(argv[i], "%d", &num_vars) != 1) return 1;
                break;
            case 'd':
                if (++i == argc) return 1;
                dataset_name = argv[i];
                break;
            case 'z':
                split_z = true;
                z_parts = 2;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    if (sscanf(argv[++i], "%zu", &z_parts) != 1) return 1;
                }
                break;
            default:
                if (world_rank == 0) usage();
                MPI_Finalize();
                return 1;
        }
    }

    if (dataset_name == NULL || folder == NULL || num_vars <= 0) {
        if (world_rank == 0) printf("ERROR: Missing required arguments: -d (dataset_name), -i (list file), or -n (num_vars).\n");
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
        printf("ERROR! Input information folder %s does not exist or is not accessible.\n", folder);
        MPI_Finalize();
        return 1;
    }

    std::string line;
    std::string input_dir;
    char file[100][50];
    bool found_dataset = false;
    int vi = 0;
    while (std::getline(list_file, line)) {
        if (!found_dataset) {
            if (strncasecmp(line.c_str(), dataset_name, strlen(dataset_name)) == 0) {
                found_dataset = true;
                std::istringstream iss(line);
                std::string dummy;
                iss >> dummy >> input_dir;
            }
        } else {
            sscanf(line.c_str(), "%s", file[vi]);
            vi++;
            if (vi >= num_vars) break;
        }
    }
    list_file.close();

    if (world_rank == 0) printf("Start parallel compressing ... \n");
    if (world_rank == 0) printf("size: %d\n", world_size);
    double start, end;
    double costReadOri = 0.0, costComp = 0.0;
    double costBoundPrep = 0.0;
    double costReadZip = 0.0, costWriteZip = 0.0, costDecomp = 0.0;
    double localCompTotal = 0.0;
    double compMin[100] = {0.0};
    double compMax[100] = {0.0};
    double compAvg[100] = {0.0};

    MPI_Barrier(MPI_COMM_WORLD);

    size_t compressed_size[100] = {0};
    size_t original_size[100] = {0};
    double absbound[100] = {0.0};
    size_t nbEle;
    size_t expected_nbEle = r1 * r2 * r3 * r4;
    const size_t effective_z = r3 * r4;
    size_t total_original_size = 0;
    size_t total_size = 0;
    int status;
    float* dataIn;
    std::vector<unsigned char> compressed_output;
    const size_t active_parts = split_z ? z_parts : 1;
    if (num_vars * active_parts > 200) {
        if (world_rank == 0) printf("ERROR: num_vars * z parts must be <= 200.\n");
        MPI_Finalize();
        return 1;
    }
    std::vector<size_t> part_compressed_size(num_vars * active_parts, 0);
    std::vector<size_t> part_capacity(num_vars * active_parts, 0);
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

    for (int i = 0; i < num_vars; i++) {
        char filename[4096];
        snprintf(filename, sizeof(filename), "%s/%s", input_dir.c_str(), file[i]);

        if (world_rank == 0) {
            start = MPI_Wtime();
            dataIn = reinterpret_cast<float*>(readData(filename, &nbEle, &status));
            if (dataIn == NULL || nbEle == 0) {
                printf("ERROR! Failed to read input file %s\n", filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            if (nbEle != expected_nbEle) {
                printf("ERROR! Dimension mismatch for %s: file has %zu elements, but dataset %s expects %zu (%zu x %zu x %zu x %zu)\n",
                       filename, nbEle, dataset_name, expected_nbEle, r1, r2, r3, r4);
                free(dataIn);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&nbEle, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            dataIn = (float*)malloc(nbEle * sizeof(float));
            if (dataIn == NULL) {
                printf("ERROR! Failed to allocate input buffer on rank %d\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            MPI_Bcast(dataIn, nbEle, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            costReadOri += end - start;
        }

        original_size[i] = nbEle * sizeof(float);
        total_original_size += original_size[i];

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) start = MPI_Wtime();
        if (world_rank == 0) {
            float max_val = dataIn[0];
            float min_val = dataIn[0];
            for (size_t j = 1; j < nbEle; j++) {
                const float val = dataIn[j];
                if (val > max_val) max_val = val;
                if (val < min_val) min_val = val;
            }
            absbound[i] = eb * static_cast<double>(max_val - min_val);
            if (absbound[i] <= 0.0) absbound[i] = eb;
        }
        MPI_Bcast(&absbound[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            costBoundPrep += end - start;
        }

        const size_t xy = r1 * r2;
        if (split_z) {
            for (size_t p = 0; p < active_parts; p++) {
                const size_t idx = i * active_parts + p;
                const size_t slab_z = z_offsets[p + 1] - z_offsets[p];
                part_capacity[idx] = zfp_compress_bound_3D(absbound[i], slab_z, r2, r1);
                part_output[idx] = static_cast<unsigned char *>(malloc(part_capacity[idx]));
                if (part_output[idx] == NULL) {
                    printf("ERROR! Failed to allocate ZFP output buffer on rank %d\n", world_rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            }
        } else {
            part_capacity[i] = zfp_compress_bound_3D(absbound[i], effective_z, r2, r1);
            part_output[i] = static_cast<unsigned char *>(malloc(part_capacity[i]));
            if (part_output[i] == NULL) {
                printf("ERROR! Failed to allocate ZFP output buffer on rank %d\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        if (world_rank == 0) start = MPI_Wtime();
        double localCompStart = MPI_Wtime();
        compressed_size[i] = 0;
        if (split_z) {
            for (size_t p = 0; p < active_parts; p++) {
                const size_t idx = i * active_parts + p;
                const size_t slab_z = z_offsets[p + 1] - z_offsets[p];
                part_compressed_size[idx] = zfp_compress_3D(
                    dataIn + z_offsets[p] * xy, absbound[i], slab_z, r2, r1,
                    part_output[idx], part_capacity[idx]);
                compressed_size[i] += part_compressed_size[idx];
            }
        } else {
            const size_t idx = i;
            part_compressed_size[idx] = zfp_compress_3D(
                dataIn, absbound[i], effective_z, r2, r1, part_output[idx], part_capacity[idx]);
            compressed_size[i] = part_compressed_size[idx];
        }
        double localCompTime = MPI_Wtime() - localCompStart;
        localCompTotal += localCompTime;
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            costComp += end - start;
        }

        double compSum = 0.0;
        MPI_Reduce(&localCompTime, &compMin[i], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localCompTime, &compMax[i], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localCompTime, &compSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (world_rank == 0) compAvg[i] = compSum / world_size;
        free(dataIn);

        if (compressed_size[i] == 0) {
            printf("ZFP compression failed for %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        total_size += compressed_size[i];
    }

    for (size_t idx = 0; idx < part_output.size(); idx++) {
        if (part_output[idx] != NULL) {
            compressed_output.insert(compressed_output.end(), part_output[idx],
                                     part_output[idx] + part_compressed_size[idx]);
            free(part_output[idx]);
        }
    }

    char zip_filename[1024];
    if (ensure_dir(output_dir) != 0) {
        printf("ERROR! Failed to prepare output dir %s: %s\n", output_dir, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    snprintf(zip_filename, sizeof(zip_filename), "%s/%s_%d_%d_%ld.out",
             output_dir, "zfp", world_rank, (int)getpid(), (long)time(NULL));
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
    if (writeData(compressed_output.data(), compressed_output.size(), zip_filename) != 0) {
        printf("ERROR! Failed to write compressed file %s: %s\n", zip_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end = MPI_Wtime();
        costWriteZip += end - start;
    }
    compressed_output.clear();
    compressed_output.shrink_to_fit();

    size_t inSize = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
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
    if (world_rank == 0) {
        end = MPI_Wtime();
        costReadZip += end - start;
    }
    unsigned char *compressed_input_pos = compressed_input;
    float *dataOutArr[200] = {nullptr};

    for (int i = 0; i < num_vars; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) start = MPI_Wtime();
        bool decomp_ok = true;
        if (split_z) {
            for (size_t p = 0; p < active_parts; p++) {
                const size_t idx = i * active_parts + p;
                const size_t slab_z = z_offsets[p + 1] - z_offsets[p];
                dataOutArr[idx] = zfp_decompress_3D(compressed_input_pos, absbound[i], part_compressed_size[idx],
                                                    slab_z, r2, r1);
                compressed_input_pos += part_compressed_size[idx];
                if (dataOutArr[idx] == NULL) decomp_ok = false;
            }
        } else {
            dataOutArr[i] = zfp_decompress_3D(compressed_input_pos, absbound[i], part_compressed_size[i],
                                              effective_z, r2, r1);
            compressed_input_pos += part_compressed_size[i];
            if (dataOutArr[i] == NULL) decomp_ok = false;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            costDecomp += end - start;
        }
        if (!decomp_ok) {
            printf("ZFP decompression failed for field %d\n", i);
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

    if (world_rank == 0) {
        printf("ZFP Finish parallel compressing on %s, total compression ratio %.4g.\n", dataset_name, 1.0 * total_original_size / total_size);
        printf("Separate ratios: ");
        for (int i = 0; i < num_vars; i++) {
            printf("%.4g ", 1.0 * original_size[i] / compressed_size[i]);
        }
        printf("\n");
        printf("Timecost of reading original files = %.4f seconds\n", costReadOri);
        printf("Timecost of preparing relative error bounds = %.4f seconds\n", costBoundPrep);
        printf("Timecost of compressing using %d processes = %.4f seconds\n", world_size, costComp);
        printf("Local compression time total min/max/avg = %.4f %.4f %.4f seconds\n",
               totalCompMin, totalCompMax, totalCompSum / world_size);
        printf("Local compression time per variable min/max/avg:\n");
        for (int i = 0; i < num_vars; i++) {
            printf("  %s: %.4f %.4f %.4f seconds\n", file[i], compMin[i], compMax[i], compAvg[i]);
        }
        printf("Timecost of writing compressed files = %.4f seconds\n", costWriteZip);
        printf("Timecost of reading compressed files = %.4f seconds\n", costReadZip);
        printf("Timecost of decompressing using %d processes = %.4f seconds\n", world_size, costDecomp);
    }

    MPI_Finalize();
    return 0;
}

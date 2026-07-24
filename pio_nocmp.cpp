#include <stdio.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mpi.h"

static int ensure_dir(const char *dir)
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

static int write_blocks(const std::vector<unsigned char *> &blocks,
                        const std::vector<size_t> &block_sizes,
                        const char *filename)
{
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) return -1;

    for (size_t i = 0; i < blocks.size(); i++) {
        fout.write(reinterpret_cast<const char *>(blocks[i]),
                   static_cast<std::streamsize>(block_sizes[i]));
        if (!fout.good()) return -1;
    }
    return 0;
}

static unsigned char *read_bytes(const char *src_file_path, size_t *byte_size)
{
    std::ifstream input(src_file_path, std::ios::binary);
    if (!input.good()) return NULL;
    input.seekg(0, std::ios::end);
    const std::streamoff input_size = input.tellg();
    if (input_size <= 0) return NULL;
    input.seekg(0, std::ios::beg);

    unsigned char *buffer = static_cast<unsigned char *>(malloc(static_cast<size_t>(input_size)));
    if (buffer == NULL) {
        std::cerr << "Failed to allocate input buffer" << std::endl;
        return NULL;
    }
    if (!input.read(reinterpret_cast<char *>(buffer), input_size)) {
        std::cerr << "Failed to read file" << std::endl;
        free(buffer);
        return NULL;
    }
    *byte_size = static_cast<size_t>(input_size);
    return buffer;
}

static unsigned char *read_float_data(const char *src_file_path, size_t *num_elements)
{
    size_t byte_size = 0;
    unsigned char *data = read_bytes(src_file_path, &byte_size);
    if (data == NULL) return NULL;
    if (byte_size % sizeof(float) != 0) {
        free(data);
        return NULL;
    }
    *num_elements = byte_size / sizeof(float);
    return data;
}

static inline void keep_decoded_memory(const void *ptr)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(ptr) : "memory");
#else
    (void)*static_cast<const volatile unsigned char *>(ptr);
#endif
}

static void usage()
{
    printf("Test case: pio_nocomp -e ignored -d dataset_name -i list_file -o output_dir -n nums [-z [parts]] [-1 r1 | -2 r1 r2 | -3 r1 r2 r3 | -4 r1 r2 r3 r4]\n");
    printf("Example: pio_nocomp -e 1e-3 -d nyx -i /path/to/list.txt -o /path/to/out -n 4 -z 4 -3 r1 r2 r3\n");
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 6) {
        if (world_rank == 0) usage();
        MPI_Finalize();
        return 0;
    }

    size_t r4 = 1;
    size_t r3 = 1;
    size_t r2 = 1;
    size_t r1 = 1;
    double ignored_error_bound = 1e-3;
    char *dataset_name = NULL;
    char *list_path = NULL;
    const char *output_dir = "./out";
    int num_vars = 0;
    bool split_z = false;
    size_t z_parts = 1;
    bool arguments_ok = true;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' || argv[i][1] == '\0') {
            arguments_ok = false;
            break;
        }
        switch (argv[i][1]) {
            case '1':
                arguments_ok = ++i < argc && sscanf(argv[i], "%zu", &r1) == 1;
                break;
            case '2':
                arguments_ok = ++i < argc && sscanf(argv[i], "%zu", &r1) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r2) == 1;
                break;
            case '3':
                arguments_ok = ++i < argc && sscanf(argv[i], "%zu", &r1) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r2) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r3) == 1;
                break;
            case '4':
                arguments_ok = ++i < argc && sscanf(argv[i], "%zu", &r1) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r2) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r3) == 1 &&
                               ++i < argc && sscanf(argv[i], "%zu", &r4) == 1;
                break;
            case 'e':
                arguments_ok = ++i < argc;
                if (arguments_ok) ignored_error_bound = atof(argv[i]);
                break;
            case 'i':
                arguments_ok = ++i < argc;
                if (arguments_ok) list_path = argv[i];
                break;
            case 'o':
                arguments_ok = ++i < argc;
                if (arguments_ok) output_dir = argv[i];
                break;
            case 'n':
                arguments_ok = ++i < argc && sscanf(argv[i], "%d", &num_vars) == 1;
                break;
            case 'd':
                arguments_ok = ++i < argc;
                if (arguments_ok) dataset_name = argv[i];
                break;
            case 'z':
                split_z = true;
                z_parts = 2;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    arguments_ok = sscanf(argv[++i], "%zu", &z_parts) == 1;
                }
                break;
            default:
                arguments_ok = false;
                break;
        }
        if (!arguments_ok) break;
    }
    (void)ignored_error_bound;

    if (!arguments_ok || dataset_name == NULL || list_path == NULL || num_vars <= 0 || num_vars > 100) {
        if (world_rank == 0) {
            printf("ERROR: Invalid or missing command-line arguments.\n");
            usage();
        }
        MPI_Finalize();
        return 1;
    }

    std::ifstream list_file(list_path);
    if (!list_file.good()) {
        if (world_rank == 0) {
            printf("ERROR! Input list %s does not exist or is not accessible.\n", list_path);
        }
        MPI_Finalize();
        return 1;
    }

    std::string line;
    std::string input_dir;
    std::vector<std::string> files;
    bool found_dataset = false;
    while (std::getline(list_file, line)) {
        if (!found_dataset) {
            if (strncasecmp(line.c_str(), dataset_name, strlen(dataset_name)) == 0) {
                found_dataset = true;
                std::istringstream stream(line);
                std::string dummy;
                stream >> dummy >> input_dir;
            }
            continue;
        }

        std::istringstream stream(line);
        std::string filename;
        if (stream >> filename) files.push_back(filename);
        if (files.size() >= static_cast<size_t>(num_vars)) break;
    }
    list_file.close();

    if (!found_dataset || input_dir.empty() || files.size() != static_cast<size_t>(num_vars)) {
        if (world_rank == 0) {
            printf("ERROR! Dataset %s or its %d variable names were not found in %s.\n",
                   dataset_name, num_vars, list_path);
        }
        MPI_Finalize();
        return 1;
    }

    if (split_z && (r4 != 1 || r3 <= 1 || z_parts < 2 || z_parts > r3)) {
        if (world_rank == 0) {
            printf("ERROR: -z supports only 3D data with z dimension > 1 and 2 <= parts <= z dimension.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t active_parts = split_z ? z_parts : 1;
    if (static_cast<size_t>(num_vars) * active_parts > 200) {
        if (world_rank == 0) printf("ERROR: num_vars * z parts must be <= 200.\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<size_t> z_offsets(active_parts + 1, 0);
    for (size_t p = 0; p <= active_parts; p++) {
        z_offsets[p] = split_z ? r3 * p / active_parts : (p == 0 ? 0 : r3);
    }

    if (world_rank == 0) {
        printf("Start parallel compressing ... \n");
        printf("size: %d\n", world_size);
        if (split_z) {
            printf("z-axis split into %zu slabs:", active_parts);
            for (size_t p = 0; p < active_parts; p++) {
                printf(" %zu", z_offsets[p + 1] - z_offsets[p]);
            }
            printf("\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const size_t expected_num_elements = r1 * r2 * r3 * r4;
    const size_t variable_bytes = expected_num_elements * sizeof(float);
    const size_t xy = r1 * r2;

    std::vector<size_t> original_size(num_vars, 0);
    std::vector<size_t> encoded_size(num_vars, 0);
    std::vector<unsigned char *> raw_blocks(static_cast<size_t>(num_vars) * active_parts, NULL);
    std::vector<size_t> raw_block_sizes(raw_blocks.size(), 0);

    for (int variable = 0; variable < num_vars; variable++) {
        for (size_t p = 0; p < active_parts; p++) {
            const size_t idx = static_cast<size_t>(variable) * active_parts + p;
            raw_block_sizes[idx] = split_z
                ? (z_offsets[p + 1] - z_offsets[p]) * xy * sizeof(float)
                : variable_bytes;
            raw_blocks[idx] = static_cast<unsigned char *>(malloc(raw_block_sizes[idx]));
            if (raw_blocks[idx] == NULL) {
                printf("ERROR! Failed to allocate %.2f MiB raw output block on rank %d.\n",
                       raw_block_sizes[idx] / 1048576.0, world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    double start = 0.0;
    double end = 0.0;
    double cost_read_original = 0.0;
    double cost_encode = 0.0;
    double cost_write = 0.0;
    double cost_read_encoded = 0.0;
    double cost_decode = 0.0;
    double local_encode_total = 0.0;
    std::vector<double> encode_min(num_vars, 0.0);
    std::vector<double> encode_max(num_vars, 0.0);
    std::vector<double> encode_avg(num_vars, 0.0);

    size_t total_original_size = 0;
    size_t total_encoded_size = 0;

    for (int variable = 0; variable < num_vars; variable++) {
        const std::string filename = input_dir + "/" + files[variable];
        size_t num_elements = 0;
        float *data_in = NULL;

        if (world_rank == 0) {
            start = MPI_Wtime();
            data_in = reinterpret_cast<float *>(read_float_data(filename.c_str(), &num_elements));
            if (data_in == NULL || num_elements == 0) {
                printf("ERROR! Failed to read input file %s\n", filename.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            if (num_elements != expected_num_elements) {
                printf("ERROR! Dimension mismatch for %s: file has %zu elements, but dataset %s expects %zu (%zu x %zu x %zu x %zu)\n",
                       filename.c_str(), num_elements, dataset_name, expected_num_elements,
                       r1, r2, r3, r4);
                free(data_in);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            MPI_Bcast(&num_elements, sizeof(num_elements), MPI_BYTE, 0, MPI_COMM_WORLD);
            MPI_Bcast(data_in, static_cast<int>(num_elements), MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&num_elements, sizeof(num_elements), MPI_BYTE, 0, MPI_COMM_WORLD);
            data_in = static_cast<float *>(malloc(num_elements * sizeof(float)));
            if (data_in == NULL) {
                printf("ERROR! Failed to allocate input buffer on rank %d.\n", world_rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            MPI_Bcast(data_in, static_cast<int>(num_elements), MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            cost_read_original += end - start;
        }

        original_size[variable] = num_elements * sizeof(float);
        total_original_size += original_size[variable];

        if (world_rank == 0) start = MPI_Wtime();
        const double local_encode_start = MPI_Wtime();
        encoded_size[variable] = 0;
        for (size_t p = 0; p < active_parts; p++) {
            const size_t idx = static_cast<size_t>(variable) * active_parts + p;
            const size_t source_offset = split_z ? z_offsets[p] * xy * sizeof(float) : 0;
            memcpy(raw_blocks[idx], reinterpret_cast<unsigned char *>(data_in) + source_offset,
                   raw_block_sizes[idx]);
            encoded_size[variable] += raw_block_sizes[idx];
        }
        const double local_encode_time = MPI_Wtime() - local_encode_start;
        local_encode_total += local_encode_time;

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            cost_encode += end - start;
        }

        double encode_sum = 0.0;
        MPI_Reduce(&local_encode_time, &encode_min[variable], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_encode_time, &encode_max[variable], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_encode_time, &encode_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (world_rank == 0) encode_avg[variable] = encode_sum / world_size;

        total_encoded_size += encoded_size[variable];
        free(data_in);
    }

    if (total_encoded_size != total_original_size) {
        printf("ERROR! Identity encoder size mismatch on rank %d.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (ensure_dir(output_dir) != 0) {
        printf("ERROR! Failed to prepare output dir %s: %s\n", output_dir, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    char raw_filename[1024];
    snprintf(raw_filename, sizeof(raw_filename), "%s/%s_%d_%d_%ld.out",
             output_dir, "nocomp", world_rank, static_cast<int>(getpid()), static_cast<long>(time(NULL)));

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
    if (write_blocks(raw_blocks, raw_block_sizes, raw_filename) != 0) {
        printf("ERROR! Failed to write raw file %s: %s\n", raw_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end = MPI_Wtime();
        cost_write += end - start;
    }

    for (unsigned char *block : raw_blocks) free(block);
    raw_blocks.clear();

    size_t input_size = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) start = MPI_Wtime();
    unsigned char *raw_input = read_bytes(raw_filename, &input_size);
    if (raw_input == NULL) {
        printf("ERROR! Failed to read raw file %s: %s\n", raw_filename, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (input_size != total_encoded_size) {
        printf("ERROR! Broken file %s: expected %zu bytes, got %zu bytes.\n",
               raw_filename, total_encoded_size, input_size);
        free(raw_input);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    remove(raw_filename);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        end = MPI_Wtime();
        cost_read_encoded += end - start;
    }

    unsigned char *decoded = static_cast<unsigned char *>(malloc(variable_bytes));
    if (decoded == NULL) {
        printf("ERROR! Failed to allocate raw decode buffer on rank %d.\n", world_rank);
        free(raw_input);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const unsigned char *raw_input_pos = raw_input;
    for (int variable = 0; variable < num_vars; variable++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) start = MPI_Wtime();
        memcpy(decoded, raw_input_pos, encoded_size[variable]);
        keep_decoded_memory(decoded);
        raw_input_pos += encoded_size[variable];
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            end = MPI_Wtime();
            cost_decode += end - start;
        }
    }

    if (raw_input_pos != raw_input + input_size) {
        printf("ERROR! Identity decoder consumed the wrong number of bytes on rank %d.\n", world_rank);
        free(decoded);
        free(raw_input);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    free(decoded);
    free(raw_input);

    double total_encode_min = 0.0;
    double total_encode_max = 0.0;
    double total_encode_sum = 0.0;
    MPI_Reduce(&local_encode_total, &total_encode_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_encode_total, &total_encode_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_encode_total, &total_encode_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("NoComp Finish parallel compressing on %s, total compression ratio %.4g.\n",
               dataset_name, static_cast<double>(total_original_size) / total_encoded_size);
        printf("Separate ratios: ");
        for (int variable = 0; variable < num_vars; variable++) {
            printf("%.4g ", static_cast<double>(original_size[variable]) / encoded_size[variable]);
        }
        printf("\n");
        printf("Timecost of reading original files = %.4f seconds\n", cost_read_original);
        printf("Timecost of compressing using %d processes = %.4f seconds\n", world_size, cost_encode);
        printf("Local compression time total min/max/avg = %.4f %.4f %.4f seconds\n",
               total_encode_min, total_encode_max, total_encode_sum / world_size);
        printf("Local compression time per variable min/max/avg:\n");
        for (int variable = 0; variable < num_vars; variable++) {
            printf("  %s: %.4f %.4f %.4f seconds\n", files[variable].c_str(),
                   encode_min[variable], encode_max[variable], encode_avg[variable]);
        }
        printf("Timecost of writing compressed files = %.4f seconds\n", cost_write);
        printf("Timecost of reading compressed files = %.4f seconds\n", cost_read_encoded);
        printf("Timecost of decompressing using %d processes = %.4f seconds\n",
               world_size, cost_decode);
    }

    MPI_Finalize();
    return 0;
}

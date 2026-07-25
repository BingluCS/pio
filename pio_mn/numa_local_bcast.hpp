#ifndef PIO_NUMA_LOCAL_BCAST_HPP
#define PIO_NUMA_LOCAL_BCAST_HPP

#include <mpi.h>

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#endif

struct NumaBroadcastContext {
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm numa_comm = MPI_COMM_NULL;
    int world_rank = -1;
    int world_size = 0;
    int node_rank = -1;
    int node_size = 0;
    int numa_rank = -1;
    int numa_size = 0;
    int cpu_id = -1;
    int numa_id = -1;
};

static inline void pio_numa_abort(const char *message, int world_rank)
{
    std::fprintf(stderr, "ERROR: world_rank=%d: %s\n", world_rank, message);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static inline int pio_numa_node_for_cpu(int cpu_id)
{
#if defined(__linux__)
    char cpu_path[128];
    std::snprintf(cpu_path, sizeof(cpu_path), "/sys/devices/system/cpu/cpu%d", cpu_id);

    DIR *directory = opendir(cpu_path);
    if (directory == NULL) return -1;

    int numa_id = -1;
    while (struct dirent *entry = readdir(directory)) {
        if (std::strncmp(entry->d_name, "node", 4) != 0) continue;

        char *end = NULL;
        const long parsed = std::strtol(entry->d_name + 4, &end, 10);
        if (end != entry->d_name + 4 && *end == '\0' && parsed >= 0 && parsed <= INT_MAX) {
            numa_id = static_cast<int>(parsed);
            break;
        }
    }
    closedir(directory);
    return numa_id;
#else
    (void)cpu_id;
    return -1;
#endif
}

static inline int pio_bound_numa_node(int *current_cpu)
{
#if defined(__linux__)
    const int running_cpu = sched_getcpu();
    if (running_cpu < 0) return -1;
    if (current_cpu != NULL) *current_cpu = running_cpu;

    const int running_numa = pio_numa_node_for_cpu(running_cpu);
    if (running_numa < 0) return -1;

    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) return -1;

    bool found_cpu = false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &affinity)) continue;
        found_cpu = true;
        const int numa_id = pio_numa_node_for_cpu(cpu);
        if (numa_id < 0 || numa_id != running_numa) {
            // The rank may migrate between NUMA domains. Refuse to create an
            // ambiguous NUMA communicator instead of silently mis-grouping it.
            return -2;
        }
    }
    return found_cpu ? running_numa : -1;
#else
    (void)current_cpu;
    return -1;
#endif
}

static inline NumaBroadcastContext pio_create_numa_broadcast_context()
{
    NumaBroadcastContext context;
    MPI_Comm_rank(MPI_COMM_WORLD, &context.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.world_size);

#if !defined(__linux__)
    pio_numa_abort("NUMA-local broadcast requires Linux.", context.world_rank);
#endif

    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                            &context.node_comm) != MPI_SUCCESS) {
        pio_numa_abort("MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) failed.",
                       context.world_rank);
    }

    MPI_Comm_rank(context.node_comm, &context.node_rank);
    MPI_Comm_size(context.node_comm, &context.node_size);

    context.numa_id = pio_bound_numa_node(&context.cpu_id);
    if (context.numa_id == -2) {
        pio_numa_abort(
            "CPU affinity spans multiple NUMA domains; bind every rank inside one NUMA domain.",
            context.world_rank);
    }
    if (context.numa_id < 0) {
        pio_numa_abort("Unable to determine the NUMA node from CPU affinity.",
                       context.world_rank);
    }

    // Split inside the shared-memory (single-node) communicator. NUMA IDs are
    // reused on different hosts and therefore must not be used as colors on
    // MPI_COMM_WORLD.
    if (MPI_Comm_split(context.node_comm, context.numa_id, context.node_rank,
                       &context.numa_comm) != MPI_SUCCESS) {
        pio_numa_abort("MPI_Comm_split for the NUMA communicator failed.",
                       context.world_rank);
    }

    MPI_Comm_rank(context.numa_comm, &context.numa_rank);
    MPI_Comm_size(context.numa_comm, &context.numa_size);

    if (context.numa_rank == 0) {
        char hostname[MPI_MAX_PROCESSOR_NAME] = {0};
        int hostname_length = 0;
        MPI_Get_processor_name(hostname, &hostname_length);
        std::printf(
            "PIO_NUMA_GROUP host=%s numa=%d ranks=%d leader_world_rank=%d leader_cpu=%d\n",
            hostname, context.numa_id, context.numa_size, context.world_rank,
            context.cpu_id);
        std::fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return context;
}

static inline void pio_numa_bcast_size(std::size_t *value,
                                        const NumaBroadcastContext &context)
{
    unsigned long long wire_value =
        context.numa_rank == 0 ? static_cast<unsigned long long>(*value) : 0ULL;

    MPI_Bcast(&wire_value, 1, MPI_UNSIGNED_LONG_LONG, 0, context.numa_comm);
    *value = static_cast<std::size_t>(wire_value);
}

static inline void pio_numa_bcast_floats(float *buffer, std::size_t count,
                                          const NumaBroadcastContext &context)
{
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t remaining = count - offset;
        const int chunk =
            remaining > static_cast<std::size_t>(INT_MAX)
                ? INT_MAX
                : static_cast<int>(remaining);
        MPI_Bcast(buffer + offset, chunk, MPI_FLOAT, 0, context.numa_comm);
        offset += static_cast<std::size_t>(chunk);
    }
}

static inline void pio_numa_bcast_double(double *value,
                                          const NumaBroadcastContext &context)
{
    MPI_Bcast(value, 1, MPI_DOUBLE, 0, context.numa_comm);
}

#endif

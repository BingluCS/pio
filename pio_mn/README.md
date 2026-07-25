# Parallel I/O Compression Benchmarks - NUMA-local Broadcast

本目录是原始 `pio` 程序的多节点、多 NUMA 版本。原目录源码保持不变。

## NUMA-local Input Replication

每个程序保持“每个 MPI rank 都压缩和解压一份完整数据”的实验语义，但不再由
全局 rank 0 向 `MPI_COMM_WORLD` 广播大数组。新的输入路径为：

1. 使用 `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` 创建单节点通信器；
2. 从 rank 的 CPU affinity 和 Linux sysfs 自动确定其 NUMA domain；
3. 在节点通信器内部按 NUMA ID 创建 `numa_comm`；
4. 每个 `numa_comm` 的本地 rank 0 独立读取一份完整输入；
5. 只在该 `numa_comm` 内广播数据大小和完整输入；
6. 每个 rank 继续独立执行压缩、写入、读取和解压缩。

NUMA 内的 rank 数量完全由实际 CPU 绑定决定，不要求每个 NUMA 固定为 16 个
rank。启动时每个 NUMA leader 会输出一行映射信息：

```text
PIO_NUMA_GROUP host=cn00001 numa=3 ranks=16 leader_world_rank=48 leader_cpu=114
```

如果一个 rank 的 CPU affinity 跨越多个 NUMA domain，程序会立即终止，避免
静默创建错误的 NUMA 通信器。

输入浮点数组使用分块 `MPI_Bcast`，因此也支持元素数量超过单次 MPI `int`
count 上限的数据。

## Required launch form

CPU 绑定仍由 HMPI rankfile 负责。`numactl --localalloc` 必须放在 `mpirun`
的应用程序位置，使它分别作用于每个 MPI rank：

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false

mpirun \
  -np "${CCS_TASK_REPLICA}" \
  --rankfile "$RANKFILE" \
  --mca rmaps_rank_file_physical true \
  --mca plm_rsh_agent /usr/bin/ssh \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x OMP_NUM_THREADS \
  -x OMP_PROC_BIND \
  -x OMP_PLACES \
  -x OMP_DYNAMIC \
  numactl --localalloc \
  ./pio_szo \
  -e 1e-3 -d NYX -i list.txt -o /path/to/output \
  -n 6 -3 512 512 512
```

不要写成：

```bash
numactl --localalloc mpirun ...
```

后一种写法只对 MPI 启动器设置策略，不能清晰保证每个远端 rank 都分别执行
`numactl --localalloc`。

编译时 `numa_local_bcast.hpp` 必须与 `.cpp` 文件位于同一目录。该实现通过
`/sys/devices/system/cpu/` 获取 NUMA ID，不需要链接 `libnuma`。在 Linux
上建议给原编译命令增加：

```text
-D_GNU_SOURCE
```

下面保留了原项目的编译说明。在服务器上使用这些命令时，请进入新的
`pio_mn` 目录（而不是原来的 `pio` 目录），并把 `-D_GNU_SOURCE` 加到每条
`mpicxx` 命令中；其余压缩器参数和链接参数保持不变。

该目录包含一组 MPI 并发压缩测试程序。它们使用相同的数据列表、命令行参数和计时流程，便于比较不同压缩器在单进程和多进程并发下的压缩率与性能。

## Programs

| 程序 | 压缩器 | 说明 |
| --- | --- | --- |
| `pio_szo` | SZo | 有损，相对误差转换为全局绝对误差界 |
| `pio_sz3` | SZ3 | 有损，相对误差转换为全局绝对误差界 |
| `pio_sperr` | SPERR | 有损，使用 point-wise error 模式 |
| `pio_pfpl` | PFPL | 有损，float32 NOA 路径 |
| `pio_zfp` | ZFP | 有损，accuracy 模式 |
| `pio_nocomp` | NoComp | 不压缩，只执行内存复制，用作基线 |

## Build

进入测试目录：

```bash
cd /home/lb/compressor/pio
```

### SZo

```bash
mpicxx -O3 -fvisibility=hidden pio_szo.cpp \
  /home/lb/compressor/SZo/include/SZo/encoder/zfse/*.c \
  -I/home/lb/compressor/SZo/build/include \
  -std=c++17 -lzstd -mavx2 -mfma -o pio_szo
```

SZo 自带 FSE 源码，位于 `include/SZo/encoder/zfse`。上面的单条 `mpicxx` 命令会同时编译 PIO 和内建 FSE；这样 `FSE_*` 使用 SZo 内建版本，而不是意外解析到系统 `libzstd` 导出的内部 FSE 符号。`-fvisibility=hidden` 将内建 FSE 符号限制在当前可执行文件内。

`-lzstd` 仍然必须保留，因为 SZo 后端还使用 `ZSTD_*` 完成最终的 Zstd 包装。它不再负责提供 FSE。链接参数应放在源码和 FSE 目标文件之后。

#### SZo on ARM with SVE2

当前 SZo 的 ARM SIMD 路径由编译器宏 `__ARM_FEATURE_SVE2` 控制，实际要求 **SVE2**，不是只有第一代 SVE。先在 ARM 机器上启用 SVE2 构建 SZo：

```bash
cd /home/lb/compressor/SZo
cmake -S . -B build -DENABLE_SVE2=ON
cmake --build build -j
```

然后编译 PIO 程序：

```bash
cd /home/lb/compressor/pio
mpicxx -O3 -march=armv8.6-a+sve2 -fvisibility=hidden pio_szo.cpp \
  /home/lb/compressor/SZo/include/SZo/encoder/zfse/*.c \
  -I/home/lb/compressor/SZo/build/include \
  -std=c++17 -lzstd -o pio_szo
```

ARM 构建不要使用 x86 专用的 `-mavx2 -mfma`。如果程序只在编译它的同一台机器上运行，也可以让编译器按本机 CPU 自动选择指令集：

```bash
mpicxx -O3 -march=native -fvisibility=hidden pio_szo.cpp \
  /home/lb/compressor/SZo/include/SZo/encoder/zfse/*.c \
  -I/home/lb/compressor/SZo/build/include \
  -std=c++17 -lzstd -o pio_szo
```

只有当本机 CPU 支持 SVE2 时，`-march=native` 才会定义 `__ARM_FEATURE_SVE2` 并启用 SZo 的 SVE2 代码。可以检查编译器是否启用了该宏：

```bash
echo | mpicxx -march=armv8.6-a+sve2 -dM -E -x c++ - | grep __ARM_FEATURE_SVE
```

预期至少包含：

```text
#define __ARM_FEATURE_SVE 1
#define __ARM_FEATURE_SVE2 1
```

如果 ARM CPU 只支持 SVE、不支持 SVE2（例如部分 A64FX 环境），当前 SZo 不会启用这条 SIMD 路径，需要使用 scalar 构建或另行实现 SVE-only 路径。用 SVE2 参数生成的可执行文件也只能在支持 SVE2 的 CPU 上运行，否则会触发 illegal instruction。

### SZ3

```bash
mpicxx -O3 pio_sz3.cpp -o pio_sz3 \
  -I/home/lb/compressor/SZ3/build/include \
  -L/home/lb/compressor/SZ3/build/lib \
  -std=c++17 -lzstd -mavx2 -mfma
```

### SPERR

先构建 SPERR。关闭单元测试可以避免下载 GoogleTest：

```bash
cd /home/lb/compressor/SPERR
cmake -S . -B build -DBUILD_UNIT_TESTS=OFF
cmake --build build -j
```

然后编译 PIO：

```bash
cd /home/lb/compressor/pio
mpicxx -O3 pio_sperr.cpp -o pio_sperr \
  -I/home/lb/compressor/SPERR/include \
  -I/home/lb/compressor/SPERR/build \
  -L/home/lb/compressor/SPERR/build/src \
  -Wl,-rpath,/home/lb/compressor/SPERR/build/src \
  -std=c++17 -lSPERR
```

`-I/home/lb/compressor/SPERR/build` 不能省略：该目录包含 CMake 生成的 `SperrConfig.h`，`pio_sperr.cpp` 会使用它与当前 `libSPERR.so` 的构建配置保持一致。

`rpath` 让运行时能够找到 `libSPERR.so`。也可以不写 `-Wl,-rpath,...`，运行前改用：

```bash
export LD_LIBRARY_PATH=/home/lb/compressor/SPERR/build/src:$LD_LIBRARY_PATH
```

### PFPL

```bash
mpicxx -O3 pio_pfpl.cpp -o pio_pfpl \
  -I/home/lb/compressor/PFPL/src \
  -std=c++17 -mavx2 -mfma
```

PFPL 的实现位于头文件中，不需要额外链接 PFPL 库。这里不使用 `-fopenmp`，因此编译器会忽略 PFPL 头文件中的 OpenMP pragma，每个 MPI rank 串行压缩。

### ZFP

先构建不带 OpenMP 的串行 ZFP 库：

```bash
cd /home/lb/compressor/ZFP
cmake -S . -B build-serial \
  -DZFP_WITH_OPENMP=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_UTILITIES=OFF
cmake --build build-serial -j
```

然后编译 PIO：

```bash
cd /home/lb/compressor/pio
mpicxx -O3 pio_zfp.cpp -o pio_zfp \
  -I/home/lb/compressor/ZFP/include \
  -L/home/lb/compressor/ZFP/build-serial/lib \
  -Wl,-rpath,/home/lb/compressor/ZFP/build-serial/lib \
  -std=c++17 -lzfp
```

现有 `/home/lb/compressor/ZFP/lib/libzfp.a` 是启用 OpenMP 构建的静态库，即使 PIO 使用 ZFP 的 serial execution policy，链接该库仍会被迫添加 OpenMP runtime。使用上面的 `build-serial` 可以彻底去掉这项依赖。

### NoComp

```bash
mpicxx -O3 pio_nocomp.cpp -o pio_nocomp -std=c++17
```

以上命令已在当前目录源码和当前依赖路径下完成试编译。

## Input List

`-i` 指向数据集列表文件，而不是原始数据文件。格式如下：

```text
NYX /data0/lb/sdrbench/NYX/
velocity_x.f32
dark_matter_density.f32
baryon_density.f32
temperature.f32
velocity_y.f32
velocity_z.f32

Hurricane /data0/lb/sdrbench/Hurricane/
Uf48.bin.f32
Vf48.bin.f32
...
```

数据集行由“数据集名称 + 数据目录”组成，后面的若干行是变量文件名。`-d` 选择数据集，`-n` 指定从该数据集条目下读取多少个变量。所有输入文件均按原始 `float32` 数组读取，不含文件头。

## Common Options

通用命令格式为：

```text
mpirun -np PROCESSES ./PROGRAM \
  -e ERROR -d DATASET -i LIST -o OUTPUT_DIR -n NUM_VARS \
  [-z [PARTS]] [-1 R1 | -2 R1 R2 | -3 R1 R2 R3 | -4 R1 R2 R3 R4]
```

| 参数 | 含义 |
| --- | --- |
| `-e ERROR` | 相对误差参数；NoComp 忽略该值 |
| `-d DATASET` | `list.txt` 中的数据集名称，不区分大小写 |
| `-i LIST` | 数据集列表文件路径 |
| `-o OUTPUT_DIR` | 每个 rank 写入压缩文件的目录 |
| `-n NUM_VARS` | 从列表中读取并压缩的变量数量 |
| `-1` ... `-4` | 数据维数及每一维长度，`R1` 是内存中最快变化的维度 |
| `-z` | 仅对 3D 数据沿 z 轴切成 2 个 slab |
| `-z PARTS` | 仅对 3D 数据沿 z 轴切成指定数量的 slab |

`-z` 当前支持 `pio_szo`、`pio_sz3`、`pio_sperr`、`pio_pfpl`、`pio_zfp` 和 `pio_nocomp`。要求 `R4=1`、`R3>1`，且 `2 <= PARTS <= R3`。每个 slab 独立压缩，最终压缩率按所有 slab 的压缩大小之和计算。

## Run

NYX，6 个变量，尺寸为 `512 x 512 x 512`：

```bash
mpirun -np 1 ./pio_szo \
  -e 1e-3 -d NYX -i list.txt -o /data0/lb/pio-output \
  -n 6 -3 512 512 512
```

使用 32 个 MPI 进程：

```bash
mpirun -np 32 ./pio_szo \
  -e 1e-3 -d NYX -i list.txt -o /data0/lb/pio-output \
  -n 6 -3 512 512 512
```

沿 z 轴切成 4 个 slab：

```bash
mpirun -np 32 ./pio_szo \
  -e 1e-3 -d NYX -i list.txt -o /data0/lb/pio-output \
  -n 6 -z 4 -3 512 512 512
```

Hurricane，13 个变量：

```bash
mpirun -np 32 ./pio_szo \
  -e 1e-3 -d Hurricane -i list.txt -o /data0/lb/pio-output \
  -n 13 -3 500 500 100
```

SPERR 可以用 `-t` 指定每个 MPI 进程内部的 SPERR 线程数：

```bash
OMP_NUM_THREADS=1 mpirun -np 8 ./pio_sperr \
  -e 1e-3 -d NYX -i list.txt -o /data0/lb/pio-output \
  -n 6 -t 1 -3 512 512 512
```

NoComp 基线：

```bash
mpirun -np 32 ./pio_nocomp \
  -e 0 -d NYX -i list.txt -o /data0/lb/pio-output \
  -n 6 -3 512 512 512
```

## MPI Workload

这些程序用于测量多个独立进程同时处理相同数据时的性能变化。MPI rank 之间不分摊变量：

- `-n 6 -np 1`：1 个进程依次压缩 6 个变量。
- `-n 6 -np 32`：32 个进程各自依次压缩同样的 6 个变量。
- `-n 6 -np 32 -z 4`：每个进程压缩 6 个变量，每个变量又分成 4 个独立 slab，即每个进程执行 24 次压缩。

Rank 0 从原始文件读取数据，再通过 `MPI_Bcast` 发送给所有 rank。每个 rank 独立压缩、写入自己的临时压缩文件、重新读回并解压；读回后临时文件会被删除。因此 `-np` 增大时，输出反映的是并发内存访问、压缩计算和文件系统压力，不是把一份压缩任务并行加速。

## Error Bounds

对 SZo、SZ3、SPERR、PFPL 和 ZFP，程序先在 rank 0 计算整个变量的：

```text
absolute_error_bound = ERROR * (max(data) - min(data))
```

然后将同一个绝对误差界广播给所有 rank。使用 `-z` 时，各 slab 仍使用整个变量的 range 所得到的误差界，保证切分前后的误差参数一致。

NoComp 只复制原始字节。

## Timing Output

主要输出字段包括：

- `Timecost of reading original files`：rank 0 读取原始文件，加上数据广播和同步时间。
- `Timecost of preparing relative error bounds`：计算 range、广播绝对误差界和同步的时间；仅部分有损程序输出。
- `Timecost of compressing using N processes`：rank 0 观察到的端到端压缩阶段时间，包含进程间 barrier 等待。
- `Local compression time total min/max/avg`：各 rank 本地压缩总时间的最小值、最大值和平均值。
- `Local compression time per variable min/max/avg`：每个变量在所有 rank 上的本地压缩时间统计。
- `Timecost of writing compressed files`：所有 rank 写压缩文件并同步所需时间。
- `Timecost of reading compressed files`：所有 rank 重新读入压缩文件并同步所需时间。
- `Timecost of decompressing using N processes`：解压所有变量并同步所需时间。

为了减少线程数混杂，比较 MPI 进程数时建议固定内部线程数：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

如果需要固定 CPU 绑定，可以使用：

```bash
mpirun --bind-to core --map-by core -np 32 ./pio_szo ...
```

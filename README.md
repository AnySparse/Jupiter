# 1. Compiling
## Thirdparty Requirement
+ [NVVIDIA NVSHMEM Library, version>=2.9.0](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html)

+ CUDA version 12.2, CMake version>=3.8, openmpi.

+ GPU-interconnect-topology:NVLink/PCIe or GPUDirect-RDMA for-all-GPU.

+ Modify CMakeLists to export these env config.


```
mkdir build && cd build && cmake ..
```

+ Datasets are available [here](https://drive.google.com/drive/folders/1ZV5oeyJfCV922bwwKoWfe6SLwIoiaNOG?usp=sharing). Please contact authors for more larger graph.

## Generate Jupiter-Delegation binary
+ Modify Config: open define "#define DELEGATION" in file (include/common.h)  

```
make jupiter_gpm
```

## Generate Jupiter-MassagePassing binary
+ Modify Config: open define "#define MESSAGE_PASSING" in file (include/common.h)  

```
make jupiter_gpm
```

# 2. Runing
### 2.1 Basic usage
>Usage: mpirun -n <num_GPUs> ./jupiter_gpm <graph_path> <pattern_name> <chunk_size>
>Support Patterns(pattern_name): Pattern-Enumeration(P1,P2,P3,P4...P16)

+ Note that NVSHMEM_SYMMETRIC_SIZE=20000000000 means memory used for NVSHMEM(Need larger when processing large graph)

+ Note that --bind-to numa means using openmp.

+  Note that the default chunk_size=1. When processing large graph, it may cause a crash. You need to adjust the `chunk_size`, which can be set to 100/1000/10000.
```
NVSHMEM_SYMMETRIC_SIZE=20000000000 mpirun -n 4  --bind-to numa ./jupiter_gpm ~/data/cit-Patents/graph  P4 1
```
### 2.2 Reproduce paper'results of Table 4.
```
./run_table4.sh
```


# 1. Compiling

```
mkdir build && cd build && cmake ..
```
## Generate Jupiter-Delegation binary
>Modify Config: open define "#define DELEGATION" in file (include/common.h)  

```
make make jupiter_gpm
```

## Generate Jupiter-MassagePassing binary
>Modify Config: open define "#define MESSAGE_PASSING" in file (include/common.h)  

```
make make jupiter_gpm
```

# 2. Runing
>Usage: mpirun -n <num_GPUs> ./jupiter_gpm <graph_path> <pattern_name> <chunk_size>
>Support Patterns(pattern_name): Pattern-Enumeration(P1,P2,P3,P4...P16)

> Note that NVSHMEM_SYMMETRIC_SIZE=20000000000 means memory used for NVSHMEM(Need larger when processing large graph)

> Note that --bind-to numa means using openmp. 
```
NVSHMEM_SYMMETRIC_SIZE=20000000000 mpirun -n 4  --bind-to numa ./jupiter_gpm ~/data/cit-Patents/graph  P4 1
```



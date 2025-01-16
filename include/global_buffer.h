#pragma once
#include "graph.h"
class GlobalBuffer
{
protected:
  vidType *buffer_list;
  vidType *buffer_size;
  vidType *buffer_offset;
  vidType *global_index;
  vidType buffer_capability;
  vidType *buffer_count;
  eidType index_capability;

  vidType *buffer_count_aprox;
public:
  void init(eidType ne)
  {
    // number = ne;
    
    //buffer_capability = std::max((eidType)2000000000, ne);
    //index_capability = std::max((eidType)200000000, ne);
    buffer_capability = 2000000000;
    //index_capability = 200000000;
    index_capability = 2000000000;

    buffer_list = (vidType *)nvshmem_malloc(static_cast<size_t>(buffer_capability) * sizeof(vidType));
    buffer_size = (vidType *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(vidType));
    buffer_offset = (vidType *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(vidType));

    CUDA_SAFE_CALL(cudaMemset(buffer_size, 0, static_cast<size_t>(index_capability) * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(buffer_offset, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&global_index, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(global_index, 0, sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_count, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(buffer_count, 0, sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_count_aprox, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(buffer_count_aprox, 0, sizeof(vidType)));
  }

  vidType test_index()
  {
    vidType h_global_index = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_global_index, global_index, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_global_index;
  }

  inline void clean_buffer()
  {
    CUDA_SAFE_CALL(cudaMemset(buffer_size, 0, static_cast<size_t>(index_capability) * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(buffer_offset, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMemset(global_index, 0, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(buffer_count, 0, sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMemset(buffer_count_aprox, 0, sizeof(vidType)));
  }

  inline __device__ int get_approximate_idx(int size)
  {
    int index = atomicAdd(&buffer_count_aprox[0], size);
    if (index + size >= index_capability)
    {
      printf("out of aprox memory! eroorrrrr!!!!!!!!!\n");
      return -1;
    }
    return index;
  }

  vidType get_buffer_total_num_approximate()
  {
    vidType h_buffer_count = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_buffer_count, buffer_count_aprox, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_buffer_count;
  }

  inline __device__ int get_buffer_idx()
  {
    int index = atomicAdd(&buffer_count[0], 1);
    if (index + 1 >= index_capability)
    {
      printf("out of index memory! eroorrrrr!!!!!!!!!\n");
      return -1;
    }
    return index;
  }

  vidType get_buffer_total_num()
  {
    vidType h_buffer_count = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_buffer_count, buffer_count, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_buffer_count;
  }

  inline __device__ vidType get_workload_size(eidType eid, int remote_gpu_id)
  {
    vidType remote_size[1] = {0};
    nvshmem_int_get((vidType *)&remote_size, &buffer_size[eid], 1, remote_gpu_id);
    return remote_size[0];
  }

  inline __device__ vidType get_workload_offset(eidType eid, int remote_gpu_id)
  {
    vidType remote_offset[1] = {0};
    nvshmem_int_get((vidType *)&remote_offset, &buffer_offset[eid], 1, remote_gpu_id);
    return remote_offset[0];
  }

  inline __device__ void fetch_remote_workload_warp(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  {
    // for(int gid=0; gid<num_gpu; gid++)
    // fetch from each gpu

    // TODO: all remote gpu should consider.
    // TODO: remote gpu's esize may be different.

    if (size == 0 || offset == -1)
      return;

    nvshmemx_int_get_warp(remote_list, &buffer_list[offset], size, remote_gpu_id);
    ;
  }

  inline __device__ int get_local_index(int size)
  {
    int index = atomicAdd(&global_index[0], size);
    if (index + size >= buffer_capability)
    {
      printf("out of buffer memory! eroorrrrr!!!!!!!!!\n");
      return -1;
    }
    return index;
  }
  inline __device__ void store_local(eidType eid, int offset, int size)
  {
    buffer_offset[eid] = offset;
    buffer_size[eid] = size;
  }

  inline __device__ void store_local_buffer(int lane, eidType eid, vidType *vlist)
  {
    int oft = buffer_offset[eid];
    if (oft >= buffer_capability || oft + buffer_size[eid] >= buffer_capability)
    {
      printf("out of buffer memory! erororoooo!!!!!!\n");
      return;
    }
    for (int ii = lane; ii < buffer_size[eid]; ii += 32)
    {
      buffer_list[oft + ii] = vlist[ii];
    }
  }

  inline __device__ int *load_local(eidType eid, int &size)
  {
    int offset = buffer_offset[eid];
    size = buffer_size[eid];
    if (size == 0)
      return NULL;
    return &buffer_list[offset];
  }
};
#pragma once
#include "graph.h"
#include "common.h"
#include "config.h"

typedef struct task_context
{                          // contex for a single task
  vidType task_total_size; // a task may contain 3 sub contexs, size of sub contex 3 can be compute by other sizes
  vidType task_size1;      // size of sub contex 1
  vidType task_size2;      // size of sub contex 2
  vidType task_offset;     // loc of the task
} task_context;

size_t check_memory(std::string output_info)
{
  size_t free_byte;
  size_t total_byte;

  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if (cudaSuccess != cuda_status)
  {
    std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }

  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  std::cout << output_info << "  GPU memory usage: used = " << used_db / 1024.0 / 1024.0 / 1024.0 << "GB, free = " << free_db / 1024.0 / 1024.0 / 1024.0 << " GB, total = " << total_db / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;

  free_byte = (size_t)(free_byte / 1024 / 1024 / 1024);
  free_byte = free_byte * 1024 * 1024 * 1024;
  return free_byte;
}

/*
  S is matched_list
  P is candidate_list
  [st,mi] store S.
  [mi,ed] store P.
  S,P | S,P | S,P | ...
*/

class MemoryBuffer
{
protected:
  vidType *buffer_list; // S,P chunk memory in global devices
  // vidType *buffer_size;   // S,P chunk size in global devices
  // vidType *buffer_mid;    // S,P chunk split for SP in global devices
  // vidType *buffer_offset; // S,P chunk offset in global devices
  task_context *task_content;
  vidType *mem_index;   // Current S,P chunk memory index in local device.
  vidType *count_index; // Current S,P chunk count index in local device.
  size_t buffer_capability;
  size_t index_capability;

  // COMM_TYPE *buffer_type; // S,P chunk type: pull neighbours(true) or PUSH

  vidType *count_index_aprox;
  Config *d_config;

  eidType *recv_ebuffer; // multi-node nvshmem need the recvieved buffer to be register.
  vidType *recv_vbuffer; // multi-node nvshmem need the recvieved buffer to be register.

  AccType *comm_volumn;
  int device_id;

public:
  void init(bool need_shared, size_t buffer_capability_, size_t index_capability_, int gpu_id)
  {
    buffer_capability = buffer_capability_;
    index_capability = index_capability_;
    device_id = gpu_id;
    // buffer_capability = 2000000000;
    // index_capability = 400000000;

    // buffer_capability = 300000000;
    // index_capability = 50000000;
    if (need_shared)
    {
      task_content = (task_context *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(task_context));
      buffer_list = (vidType *)nvshmem_malloc(static_cast<size_t>(buffer_capability) * sizeof(vidType));
      // buffer_size = (vidType *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(vidType));
      // buffer_mid = (vidType *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(vidType));
      // buffer_offset = (vidType *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(vidType));
      // buffer_type = (COMM_TYPE *)nvshmem_malloc(static_cast<size_t>(index_capability) * sizeof(COMM_TYPE));

      mem_index = (vidType *)nvshmem_malloc(sizeof(vidType));
      count_index = (vidType *)nvshmem_malloc(sizeof(vidType));
      count_index_aprox = (vidType *)nvshmem_malloc(sizeof(vidType));
    }
    else
    {
      CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_list, sizeof(vidType) * buffer_capability));
      // CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_size, sizeof(vidType) * index_capability));
      // CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_mid, sizeof(vidType) * index_capability));
      // CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_offset, sizeof(vidType) * index_capability));
      CUDA_SAFE_CALL(cudaMalloc((void **)&task_content, sizeof(task_context) * index_capability));
      // CUDA_SAFE_CALL(cudaMalloc((void **)&buffer_type, sizeof(COMM_TYPE) * index_capability));
      CUDA_SAFE_CALL(cudaMalloc((void **)&mem_index, sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&count_index, sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&count_index_aprox, sizeof(vidType)));

      // TODO: add additional ways to show potential error
      assert(nvshmemx_buffer_register(buffer_list, sizeof(vidType) * buffer_capability) == 0);
      // assert(nvshmemx_buffer_register(buffer_size, sizeof(vidType) * index_capability) == 0);
      // assert(nvshmemx_buffer_register(buffer_mid, sizeof(vidType) * index_capability) == 0);
      // assert(nvshmemx_buffer_register(buffer_offset, sizeof(vidType) * index_capability) == 0);
      assert(nvshmemx_buffer_register(task_content, sizeof(task_context) * index_capability) == 0);
      // assert(nvshmemx_buffer_register(buffer_type, sizeof(COMM_TYPE) * index_capability) == 0);
      assert(nvshmemx_buffer_register(mem_index, sizeof(vidType)) == 0);
      assert(nvshmemx_buffer_register(count_index, sizeof(vidType)) == 0);
      assert(nvshmemx_buffer_register(count_index_aprox, sizeof(vidType)) == 0);
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_config, sizeof(Config)));

    // CUDA_SAFE_CALL(cudaMemset(buffer_size, 0, static_cast<size_t>(index_capability) * sizeof(vidType)));
    // CUDA_SAFE_CALL(cudaMemset(buffer_mid, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));
    // CUDA_SAFE_CALL(cudaMemset(buffer_offset, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMemset(task_content, -1, static_cast<size_t>(index_capability) * sizeof(task_context)));

    CUDA_SAFE_CALL(cudaMemset(mem_index, 0, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(count_index, 0, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(count_index_aprox, 0, sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMalloc((void **)&recv_ebuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&recv_vbuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(vidType)));

    // TODO: add additional ways to show potential error
    assert(nvshmemx_buffer_register(recv_ebuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(eidType)) == 0);
    assert(nvshmemx_buffer_register(recv_vbuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(vidType)) == 0);

    CUDA_SAFE_CALL(cudaMalloc((void **)&comm_volumn, sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemset(comm_volumn, 0, sizeof(AccType)));
  }

  AccType calculate_comm_volumn()
  {
    AccType h_comm_volumn = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_comm_volumn, comm_volumn, sizeof(AccType), cudaMemcpyDeviceToHost));
    return h_comm_volumn;
  }

  vidType get_memory_size_host()
  {
    vidType h_mem_index = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_mem_index, mem_index, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_mem_index;
  }

  inline void clean_buffer()
  {
    // CUDA_SAFE_CALL(cudaMemset(buffer_size, 0, static_cast<size_t>(index_capability) * sizeof(vidType)));
    // CUDA_SAFE_CALL(cudaMemset(buffer_mid, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));
    // CUDA_SAFE_CALL(cudaMemset(buffer_offset, -1, static_cast<size_t>(index_capability) * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(task_content, -1, static_cast<size_t>(index_capability) * sizeof(task_context)));

    CUDA_SAFE_CALL(cudaMemset(mem_index, 0, sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(count_index, 0, sizeof(vidType)));

    CUDA_SAFE_CALL(cudaMemset(count_index_aprox, 0, sizeof(vidType)));
  }

  inline __device__ int get_count()
  {
    return *count_index;
  }
  inline __device__ int reset_count(int count)
  {
    *count_index = count;
  }
  inline __device__ int reset_mem(int mem)
  {
    *mem_index = mem;
  }

  __device__ void show_context(eidType eid)
  {
    printf("task loc:%d, task size1: %d, task size2: %d, task size all:%d\n", task_content[eid].task_offset, task_content[eid].task_size1, task_content[eid].task_size2, task_content[eid].task_total_size);
  }

  inline __device__ int *get_buffer_list_ptr() { return buffer_list; }
  // inline __device__ int *get_buffer_size_ptr() { return buffer_size; }
  // inline __device__ int *get_buffer_mid_ptr() { return buffer_mid; }
  // inline __device__ int *get_buffer_offset_ptr() { return buffer_offset; }
  inline __device__ task_context *get_context_ptr() { return task_content; }

  // todo:implement the same operations in communicator
  // only can handle single one

  inline __device__ void fetch_remote_context_warp(task_context *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  {
    nvshmemx_int_get_warp((int *)remote_list, (int *)&task_content[offset], size * (sizeof(task_context) / sizeof(int)), remote_gpu_id);
#ifdef PROFILING
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], size);
      }
#endif
  }
  inline __device__ void fetch_remote_context_block(task_context *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  {
    nvshmemx_int_get_block((int *)remote_list, (int *)&task_content[offset], size * (sizeof(task_context) / sizeof(int)), remote_gpu_id);
#ifdef PROFILING
    int thread_lane = threadIdx.x & (BLOCK_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], size);
      }
#endif
  }
  // inline __device__ void fetch_remote_workload_size(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_warp(remote_list, &buffer_size[offset], size, remote_gpu_id);
  //    #ifdef PROFILING
  //     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  //     int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }
  // inline __device__ void fetch_remote_workload_mid(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_warp(remote_list, &buffer_mid[offset], size, remote_gpu_id);
  //   #ifdef PROFILING
  //     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  //     int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }
  // inline __device__ void fetch_remote_workload_offset(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_warp(remote_list, &buffer_offset[offset], size, remote_gpu_id);
  //   #ifdef PROFILING
  //     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  //     int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }

  //  inline __device__ void fetch_remote_workload_size_block(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_block(remote_list, &buffer_size[offset], size, remote_gpu_id);
  //   #ifdef PROFILING
  //     int thread_lane = threadIdx.x & (BLOCK_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }
  // inline __device__ void fetch_remote_workload_mid_block(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_block(remote_list, &buffer_mid[offset], size, remote_gpu_id);
  //   #ifdef PROFILING
  //     int thread_lane = threadIdx.x & (BLOCK_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }
  // inline __device__ void fetch_remote_workload_offset_block(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  // {
  //   nvshmemx_int_get_block(remote_list, &buffer_offset[offset], size, remote_gpu_id);
  //   #ifdef PROFILING
  //     int thread_lane = threadIdx.x & (BLOCK_SIZE - 1);
  //     if(thread_lane==0)
  //     if(remote_gpu_id!=device_id){
  //         atomicAdd(&comm_volumn[0], size);
  //     }
  //   #endif
  // }

  inline __device__ vidType get_remote_ntasks_mem(int remote_gpu_id)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &mem_index[0], 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ int get_approximate_idx(int size)
  {
    int index = atomicAdd(&count_index_aprox[0], size);
    assert(index + size < index_capability);
    return index;
  }

  vidType get_buffer_total_num_approximate()
  {
    vidType h_count_index = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_count_index, count_index_aprox, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_count_index;
  }

  inline __device__ int get_buffer_idx()
  {
    int index = atomicAdd(&count_index[0], 1);
    assert(index + 1 < index_capability);
    return index;
  }

  vidType get_buffer_total_num()
  {
    vidType h_count_index = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_count_index, count_index, sizeof(vidType), cudaMemcpyDeviceToHost));
    return h_count_index;
  }

  inline __device__ vidType get_remote_ntasks(int remote_gpu_id)
  {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &count_index[0], 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ vidType get_workload_size(eidType eid, int remote_gpu_id)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &buffer_size[eid], 1, remote_gpu_id);
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &(task_content[eid].task_total_size), 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ vidType get_workload_offset(eidType eid, int remote_gpu_id)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &buffer_offset[eid], 1, remote_gpu_id);
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &(task_content[eid].task_offset), 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ vidType get_workload_mid(eidType eid, int remote_gpu_id)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &buffer_mid[eid], 1, remote_gpu_id);
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &(task_content[eid].task_size1), 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  // inline __device__ COMM_TYPE get_workload_type(eidType eid, int remote_gpu_id)
  // {
  //   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  //   nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], (vidType *)&buffer_type[eid], 1, remote_gpu_id);
  //   return static_cast<COMM_TYPE>(recv_vbuffer[thread_id * SLOT_SIZE]);
  // }

  inline __device__ void fetch_remote_workload_warp(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  {
    // for(int gid=0; gid<num_gpu; gid++)
    // fetch from each gpu

    // TODO: all remote gpu should consider.
    // TODO: remote gpu's esize may be different.

    if (size == 0 || offset == -1)
      return;

    nvshmemx_int_get_warp(remote_list, &buffer_list[offset], size, remote_gpu_id);

#ifdef PROFILING
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], size);
      }
#endif
  }

  inline __device__ void fetch_remote_workload_block(vidType *remote_list, int remote_gpu_id, eidType eid, vidType offset, vidType size)
  {
    // for(int gid=0; gid<num_gpu; gid++)
    // fetch from each gpu

    // TODO: all remote gpu should consider.
    // TODO: remote gpu's esize may be different.

    if (size == 0 || offset == -1)
      return;

    nvshmemx_int_get_block(remote_list, &buffer_list[offset], size, remote_gpu_id);
#ifdef PROFILING
    int thread_lane = threadIdx.x & (BLOCK_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], size);
      }
#endif
  }

  inline __device__ int get_local_index(int size)
  {

    int index = atomicAdd(&mem_index[0], size);
    assert(index + size < buffer_capability);
    return index;
  }

  inline __device__ void store_local_idx(eidType eid, int offset, int size1, int size2, int size)
  {
    assert(eid < index_capability && offset < buffer_capability);

    // buffer_offset[eid] = offset;
    // buffer_mid[eid] = mid;
    // buffer_size[eid] = size;
    task_content[eid].task_offset = offset;
    task_content[eid].task_size1 = size1;
    task_content[eid].task_size2 = size2;
    task_content[eid].task_total_size = size;
  }

  inline __device__ void store_local_idx(eidType eid, int offset, int size1, int size)
  {
    store_local_idx(eid, offset, size1, size - size1, size);
  }

  // inline __device__ void store_local_idx(eidType eid, int offset, int mid, int size, COMM_TYPE ctype)
  // {
  //   assert(eid < index_capability && offset < buffer_capability);
  //   buffer_offset[eid] = offset;
  //   buffer_mid[eid] = mid;
  //   buffer_size[eid] = size;
  //   buffer_type[eid] = ctype;
  // }

  inline __device__ void store_local_buffer(int lane, eidType eid, vidType *s_list, vidType *p_list, vidType *t_list)
  {
    int oft = task_content[eid].task_offset;
#ifdef EXTRA_CHECK
    assert(oft < buffer_capability && oft + task_content[eid].task_total_size < buffer_capability);
#endif
    // for (int ii = lane; ii < buffer_size[eid]; ii += 32)
    // {
    //   vidType value = -1;
    //   if (ii < buffer_mid[eid])
    //     value = s_list[ii];
    //   else
    //     value = p_list[ii - buffer_mid[eid]];
    //   buffer_list[oft + ii] = value;
    // }

    int size1 = task_content[eid].task_size1;
    int size2 = task_content[eid].task_size2;
    int size3 = task_content[eid].task_total_size - size1 - size2;
    for (int ii = lane; ii < size1; ii += 32)
    {
      buffer_list[oft + ii] = s_list[ii];
    }
    for (int ii = lane; ii < size2; ii += 32)
      buffer_list[oft + ii + size1] = p_list[ii];
#ifdef EXTRA_CHECK
    if (t_list == nullptr)
    {
      assert(size3 == 0);
      return;
    }
#endif
    for (int ii = lane; ii < size3; ii += 32)
      buffer_list[oft + ii + size1 + size2] = t_list[ii];
  }

  inline __device__ void store_local_buffer(int lane, eidType eid, vidType *s_list, vidType *p_list)
  {
    store_local_buffer(lane, eid, s_list, p_list, nullptr);
  }

  // inline __device__ int *load_local(eidType eid, int &size, int &mid)
  // {
  //   int offset = buffer_offset[eid];
  //   size = buffer_size[eid];
  //   mid = buffer_mid[eid];
  //   if (size == 0)
  //     return NULL;
  //   return &buffer_list[offset];
  // }

  // inline __device__ vidType *get_plist_ptr(eidType eid)
  // {
  //   int oft = buffer_offset[eid];
  //   int mid = buffer_mid[eid];
  //   return &buffer_list[oft + mid];
  // }

  // inline __device__ COMM_TYPE get_local_type(eidType eid)
  // {
  //   return buffer_type[eid];
  // }

  // Perform local computation...
  // TODO: int may be out of range.
  __device__ vidType *get_matched_list(eidType eid)
  {
    // int offset = buffer_offset[eid];
    int offset = task_content[eid].task_offset;
    return &buffer_list[offset];
  }

  __device__ int get_matched_size(eidType eid)
  {
    // int mi = buffer_mid[eid];
    int mi = task_content[eid].task_size1;
    return mi;
  }

  __device__ vidType *get_candidate_list(eidType eid)
  {
    // int st = buffer_offset[eid];
    // int mi = buffer_mid[eid];
    int st = task_content[eid].task_offset;
    int mi = task_content[eid].task_size1;
    int idx = st + mi;
    if (d_config->reused == true)
      idx = st;
    return &buffer_list[idx];
  }

  __device__ int get_candidate_size(eidType eid)
  {
    // int ed = buffer_size[eid];
    // int mi = buffer_mid[eid];
    // int mi = task_content[eid].task_size1;
    int size = task_content[eid].task_size2;
    if (d_config->reused == true)
    {
      // size = ed;
      size = task_content[eid].task_total_size;
    }
    return size;
  }

  __device__ vidType *get_new_list(eidType eid)
  {
    int st = task_content[eid].task_offset;
    int mi = task_content[eid].task_size1;
    int mi1 = task_content[eid].task_size2;
    int idx = st + mi + mi1;
    if (d_config->reused == true)
      idx = st;
    return &buffer_list[idx];
  }

  __device__ int get_new_size(eidType eid)
  {
    int size = task_content[eid].task_total_size - task_content[eid].task_size1 - task_content[eid].task_size2;
    if (d_config->reused == true)
    {
      // size = ed;
      size = task_content[eid].task_total_size;
    }
    return size;
  }

  __device__ vidType *get_workload_list(eidType eid)
  {
    // int offset = buffer_offset[eid];
    int offset = task_content[eid].task_offset;
    return &buffer_list[offset];
  }

  __device__ int get_workload_size(eidType eid)
  {
    // int ed = buffer_size[eid];
    int ed = task_content[eid].task_total_size;
    return ed;
  }

  __device__ vidType *get_left_operand(eidType eid, vidType *neighbour_list, vidType neighbour_size, vidType &operand_size)
  {
    auto P_list = get_candidate_list(eid);
    auto P_size = get_candidate_size(eid);
    if (d_config->left_operand == OPERAND_TYPE::P_TYPE)
    {
      operand_size = P_size;
      return P_list;
    }
    else
    {
      operand_size = neighbour_size;
      return neighbour_list;
    }
  }
  __device__ vidType *get_right_operand(eidType eid, vidType *neighbour_list, vidType neighbour_size, vidType &operand_size)
  {
    auto P_list = get_candidate_list(eid);
    auto P_size = get_candidate_size(eid);
    // auto S_list = get_matched_list(config);

    if (d_config->right_operand == OPERAND_TYPE::P_TYPE)
    {
      operand_size = P_size;
      return P_list;
    }
    else
    {
      operand_size = neighbour_size;
      return neighbour_list;
    }
  }

  void update_config(Config cfg)
  {
    CUDA_SAFE_CALL(cudaMemcpy(d_config, &cfg, sizeof(Config), cudaMemcpyHostToDevice));
  }

  __device__ OPERATOR_TYPE get_comp_type()
  {
    return d_config->op;
  }

  __device__ int get_level()
  {
    return d_config->level;
  }
  __device__ int get_reused()
  {
    return d_config->reused ? 1 : 0;
  }
};

class BufferBase
{
public:
  MemoryBuffer read_buffer;
  MemoryBuffer write_buffer;
  MemoryBuffer comp_buffer;
  vidType *tmp_buffer;
  PatternConfig patternConfig;
  int level;

  void init(PatternConfig pconfig, size_t tmp_size, int gpu_id)
  {
    // TODO: need remove it
    CUDA_SAFE_CALL(cudaMalloc((void **)&tmp_buffer, sizeof(vidType) * tmp_size));

    auto free_byte = check_memory("Estimate buffer memory.");
    // x=6y, xfor mem, y for count
    //(x+4y)*3 = free_byte

    auto y = size_t(free_byte / 1.3 / 4 / 30);
    auto x = 6 * y;

    printf("free byte:%ld,  buffer size:%ld,  index size:%ld\n", free_byte, x, y);

    level = 0;
    read_buffer.init(/*need_share=*/true, x, y, gpu_id);
    // check_memory("read buffer memory.");
    write_buffer.init(/*need_share=*/true, x, y, gpu_id);
    // check_memory("write buffer memory.");
    comp_buffer.init(/*need_share=*/false, x, y, gpu_id);
    // check_memory("comp buffer memory.");
    patternConfig = pconfig;
    patternConfig.init();
    auto config = patternConfig.first_config();
    write_buffer.update_config(config);
    read_buffer.update_config(config);
    comp_buffer.update_config(config);
  }

  void reload_pattern_config(PatternConfig pconfig, int k)
  {
    patternConfig = pconfig;
    patternConfig.init(k);
    auto config = patternConfig.first_config();
    write_buffer.update_config(config);
    read_buffer.update_config(config);
    comp_buffer.update_config(config);
  }

  void next_iteration()
  {
    if (level < 2)
    {
      level = 2;
    }
    else
      level++;
    auto config = patternConfig.next_config(level);
    std::swap(read_buffer, write_buffer);
    write_buffer.clean_buffer();
    comp_buffer.clean_buffer();

    write_buffer.update_config(config);
    read_buffer.update_config(config);
    comp_buffer.update_config(config);
  }

  bool is_done()
  {
    return level == patternConfig.nlevel;
  }

  void clean_buffer()
  {
    read_buffer.clean_buffer();
    write_buffer.clean_buffer();
    comp_buffer.clean_buffer();
    level = 0;
    auto config = patternConfig.first_config();
    write_buffer.update_config(config);
    read_buffer.update_config(config);
    comp_buffer.update_config(config);
  }
};

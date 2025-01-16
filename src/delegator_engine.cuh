// #include "context_manager_engine.cuh"
// typedef struct kernel_contex{
//   vidType IDs[6];
//   vidType *contex_loc[3];
//   vidType contex_size[3];
// }kernel_contex;
__device__ void comm_pull_edge_neighbours_by_warp(BufferBase buffer, NvsGraphGPU g, vidType max_deg,
                                                  vidType g_v0, vidType &v0_size, vidType **v0_ptr,
                                                  vidType g_v1, vidType &v1_size, vidType **v1_ptr)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  vidType v0_adj_size = 0;
  eidType v0_adj_start = 0;
  vidType v1_adj_size = 0;
  eidType v1_adj_start = 0;
  if (thread_lane == 0)
  {
    v0_adj_size = g.getOutDegree_remote(g_v0);
    v0_adj_start = g.get_remote_col_start(g_v0);
    v1_adj_size = g.getOutDegree_remote(g_v1);
    v1_adj_start = g.get_remote_col_start(g_v1);

    buffer.comp_buffer.store_local_idx(chunk_idx, mem_idx, /*mid=*/v0_adj_size, /*size=*/v0_adj_size + v1_adj_size);
  }

  v0_adj_size = __shfl_sync(0xffffffff, v0_adj_size, 0);
  v0_adj_start = __shfl_sync(0xffffffff, v0_adj_start, 0);
  v1_adj_size = __shfl_sync(0xffffffff, v1_adj_size, 0);
  v1_adj_start = __shfl_sync(0xffffffff, v1_adj_start, 0);

  __syncwarp();
  *v0_ptr = buffer.comp_buffer.get_matched_list(chunk_idx);
  *v1_ptr = buffer.comp_buffer.get_candidate_list(chunk_idx);
  if (g.is_local(g_v0))
    *v0_ptr = g.fetch_local_neigbours(g_v0);
  else
    g.fetch_remote_neigbours_warp(g_v0, *v0_ptr, v0_adj_size, v0_adj_start);
  v0_size = v0_adj_size;
  __syncwarp();

  if (g.is_local(g_v1))
    *v1_ptr = g.fetch_local_neigbours(g_v1);
  else
    g.fetch_remote_neigbours_warp(g_v1, *v1_ptr, v1_adj_size, v1_adj_start);
  v1_size = v1_adj_size;
  __syncwarp();
}

__device__ void comm_pull_neighbours_by_warp(BufferBase buffer, NvsGraphGPU g,
                                             vidType g_v0, vidType &v0_size, vidType *v0_ptr)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  vidType v0_adj_size = 0;
  eidType v0_adj_start = 0;
  if (thread_lane == 0)
  {
    v0_adj_size = g.getOutDegree_remote(g_v0);
    v0_adj_start = g.get_remote_col_start(g_v0);
  }

  v0_adj_size = __shfl_sync(0xffffffff, v0_adj_size, 0);
  v0_adj_start = __shfl_sync(0xffffffff, v0_adj_start, 0);
  __syncwarp();

  g.fetch_remote_neigbours_warp(g_v0, v0_ptr, v0_adj_size, v0_adj_start);
  v0_size = v0_adj_size;
  __syncwarp();

  // #ifdef MERGE_NEIGHBORS_COMM

  //   if (g.d_g_v1_frequency[g_v0] >= FREQ_THD)
  //   {
  //     int v0_start_ = g.d_v1_offset[g_v0];
  //     // v0_ptr = &buffer.read_buffer.get_buffer_list_ptr()[v0_start_];

  //     for(int idx=thread_lane; idx<v0_adj_size; idx+=32){
  //         v0_ptr[idx] = buffer.read_buffer.get_buffer_list_ptr()[v0_start_+idx];
  //     }
  //     __syncwarp();
  //     v0_size = v0_adj_size;
  //   }else{
  //     g.fetch_remote_neigbours_warp(g_v0, v0_ptr, v0_adj_size, v0_adj_start);
  //     v0_size = v0_adj_size;
  //   }
  //   __syncwarp();
  // #else
  //   g.fetch_remote_neigbours_warp(g_v0, v0_ptr, v0_adj_size, v0_adj_start);
  //   v0_size = v0_adj_size;
  //   __syncwarp();
  // #endif
}

__device__ bool handle_local(NvsGraphGPU g, vidType *tmp_list, vidType tmp_size, vidType total_size)
{
  __shared__ vidType local_sum[WARPS_PER_BLOCK * 2];
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int num = 0;
  // pull neoghbour total comm
  for (vidType idx = thread_lane; idx < tmp_size; idx += WARP_SIZE)
  {
    if (!g.is_local(tmp_list[idx]))
      num += g.get_degree_remote(tmp_list[idx]);
  }
  //
  // for(vidType idx = thread_lane; idx < tmp_size; idx += WARP_SIZE){
  //   if(!g.is_local(tmp_list[idx]))
  //     num += 1;
  // }
  auto count = warp_reduce(num);
  if (count > total_size * g.GPUNums() / 2) // need to check the batch's speed up
    return false;
  else
    return true;
}

__device__ COMM_TYPE analyze_buffer_type(NvsGraphGPU g, vidType *tmp_list, vidType tmp_size)
{
  // return PULL_NEIGHBOURS;
  return PULL_WORKLOADS;

  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int count = 0;
  COMM_TYPE ctype = PULL_WORKLOADS;
  int N_comm = 0;
  for (int idx = thread_lane; idx < tmp_size; idx += 32)
  {
    if (g.is_local(tmp_list[idx]))
      count++;
    else
    {
      N_comm += g.get_degree_remote(tmp_list[idx]);
    }
  }

  // Rule 1: |Local vertices|/|Total vertices| > TH, TH=0.8
  // auto local_count = warp_reduce(count);
  // if (local_count * 1.0 >= tmp_size * 0.8)
  //   ctype = PULL_NEIGHBOURS;

  // Rule 2: |Remote vertices comm| < |Remote workload comm|
  // auto local_count = warp_reduce(N_comm);
  // if(local_count < tmp_size){
  //      ctype = PULL_NEIGHBOURS;
  // }

  // ctype = PULL_WORKLOADS;
  return ctype;
}

__device__ void comm_fetch_SP_workload_number(BufferBase buffer, vidType &ntasks, int gpu_id)
{
#ifdef BATCH_LOAD
  ntasks = buffer.comp_buffer.get_count();

#else
  if (threadIdx.x == 0)
  {
    ntasks = buffer.read_buffer.get_remote_ntasks(gpu_id);
  }
  __syncthreads();
#endif
}

__device__ void comm_push_SP_workload_by_warp(BufferBase buffer, NvsGraphGPU g, vidType max_deg, vidType *tmp_list, vidType tmp_size, vidType *p_list, vidType p_size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  int shared_chunk_idx = 0;

  // COMM_TYPE ctype = analyze_buffer_type(g, tmp_list, tmp_size);

  if (thread_lane == 0)
  {
    int mem_idx = buffer.write_buffer.get_local_index(/*|S|+|P|*/ tmp_size + p_size);
    int chunk_idx = buffer.write_buffer.get_buffer_idx();
    shared_chunk_idx = chunk_idx;
    buffer.write_buffer.store_local_idx(chunk_idx, mem_idx, tmp_size, tmp_size + p_size);
  }
  __syncwarp();
  shared_chunk_idx = __shfl_sync(0xffffffff, shared_chunk_idx, 0);
  buffer.write_buffer.store_local_buffer(thread_lane, shared_chunk_idx, /*S_list=*/tmp_list, /*P_list=*/p_list);
  __syncwarp();
}

// __device__ void comm_push_SP_workload_by_warp(BufferBase buffer, NvsGraphGPU g, vidType max_deg, vidType *tmp_list, vidType tmp_size,  vidType *p_list, vidType p_size,
//                 vidType g_v0, vidType g_v1)
// {
//   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//   int warp_id = thread_id / WARP_SIZE;                  // global warp index
//   int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
//   int thread_lane = threadIdx.x & (WARP_SIZE - 1);
//   int warp_lane = threadIdx.x / WARP_SIZE;
//   int chunk_idx = warp_id;
//   int mem_idx = warp_id * (max_deg + max_deg);
//   int shared_chunk_idx = 0;

//   // COMM_TYPE ctype = analyze_buffer_type(g, tmp_list, tmp_size);

//   if (thread_lane == 0)
//   {
//     int mem_idx = buffer.write_buffer.get_local_index(/*|S|+|P|*/ tmp_size+p_size);
//     int chunk_idx = buffer.write_buffer.get_buffer_idx();
//     shared_chunk_idx = chunk_idx;
//     buffer.write_buffer.store_local_idx(chunk_idx, mem_idx, tmp_size, tmp_size+p_size);

//   }
//   __syncwarp();
//   shared_chunk_idx = __shfl_sync(0xffffffff, shared_chunk_idx, 0);
//   buffer.write_buffer.store_local_buffer(thread_lane, shared_chunk_idx, /*S_list=*/tmp_list, /*P_list=*/p_list);
//   __syncwarp();
//   if (thread_lane == 0)
//   {
//     buffer.write_buffer.get_buffer_list_ptr()[mem_idx + tmp_size - 2] = g_v0;
//     buffer.write_buffer.get_buffer_list_ptr()[mem_idx + tmp_size - 1] = g_v1;
//   }
// }

__device__ void comm_push_SP_workload_by_warp(BufferBase buffer, NvsGraphGPU g, vidType max_deg, vidType *tmp_list, vidType tmp_size, vidType *p_list, vidType p_size, vidType *new_list, vidType new_size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  int shared_chunk_idx = 0;

  // COMM_TYPE ctype = analyze_buffer_type(g, tmp_list, tmp_size);

  if (thread_lane == 0)
  {
    int mem_idx = buffer.write_buffer.get_local_index(/*|S|+|P|*/ tmp_size + p_size + new_size);
    int chunk_idx = buffer.write_buffer.get_buffer_idx();
    shared_chunk_idx = chunk_idx;
    buffer.write_buffer.store_local_idx(chunk_idx, mem_idx, tmp_size, p_size, tmp_size + p_size + new_size);
  }
  __syncwarp();
  shared_chunk_idx = __shfl_sync(0xffffffff, shared_chunk_idx, 0);
  buffer.write_buffer.store_local_buffer(thread_lane, shared_chunk_idx, /*S_list=*/tmp_list, /*P_list=*/p_list, new_list);
  __syncwarp();
}

__device__ void comm_push_workload_by_warp(BufferBase buffer, NvsGraphGPU g, vidType max_deg, vidType *tmp_list, vidType tmp_size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  int shared_chunk_idx = 0;

  // COMM_TYPE ctype = analyze_buffer_type(g, tmp_list, tmp_size);

  if (thread_lane == 0)
  {
    int mem_idx = buffer.write_buffer.get_local_index(/*|S|+0*/ tmp_size);
    int chunk_idx = buffer.write_buffer.get_buffer_idx();
    shared_chunk_idx = chunk_idx;
    buffer.write_buffer.store_local_idx(chunk_idx, mem_idx, tmp_size, tmp_size);
  }
  __syncwarp();
  shared_chunk_idx = __shfl_sync(0xffffffff, shared_chunk_idx, 0);
  buffer.write_buffer.store_local_buffer(thread_lane, shared_chunk_idx, /*S_list=*/tmp_list, /*useless, P_list=*/tmp_list);
  __syncwarp();
}

__device__ void comm_fetch_workload_number(BufferBase buffer, vidType &ntasks, int gpu_id)
{
  if (threadIdx.x == 0)
  {
    ntasks = buffer.read_buffer.get_remote_ntasks(gpu_id);
  }
  __syncthreads();
}

__device__ void comm_pull_SP_by_warp(BufferBase buffer, vidType **matched_list, vidType &matched_size, vidType **candidate_list, vidType &candidate_size, eidType eid, int gpu_id)
{
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  vidType adj_size = 0, adj_start = 0, adj_mid = 0;

#ifndef BATCH_LOAD
  if (thread_lane == 0)
  {
    adj_size = buffer.read_buffer.get_workload_size(eid, gpu_id);
    adj_start = buffer.read_buffer.get_workload_offset(eid, gpu_id);
    adj_mid = buffer.read_buffer.get_workload_mid(eid, gpu_id);
  }
  __syncwarp();
  adj_size = __shfl_sync(0xffffffff, adj_size, 0);
  adj_start = __shfl_sync(0xffffffff, adj_start, 0);
  adj_mid = __shfl_sync(0xffffffff, adj_mid, 0);
  buffer.comp_buffer.store_local_idx(eid, adj_start, adj_mid, adj_size);
  vidType *workload_list = buffer.comp_buffer.get_workload_list(eid);

  buffer.read_buffer.fetch_remote_workload_warp(workload_list, gpu_id, eid, adj_start, adj_size);
#endif

  *matched_list = buffer.comp_buffer.get_matched_list(eid);
  matched_size = buffer.comp_buffer.get_matched_size(eid);
  *candidate_list = buffer.comp_buffer.get_candidate_list(eid);
  candidate_size = buffer.comp_buffer.get_candidate_size(eid);
}

__device__ void comm_pull_SP_by_warp(BufferBase buffer, vidType **matched_list, vidType &matched_size, vidType **candidate_list, vidType &candidate_size, vidType **new_list, vidType &new_size, eidType eid, int gpu_id)
{
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  vidType adj_size = 0, adj_start = 0, adj_mid = 0;

#ifndef BATCH_LOAD
  if (thread_lane == 0)
  {
    adj_size = buffer.read_buffer.get_workload_size(eid, gpu_id);
    adj_start = buffer.read_buffer.get_workload_offset(eid, gpu_id);
    adj_mid = buffer.read_buffer.get_workload_mid(eid, gpu_id);
  }
  __syncwarp();
  adj_size = __shfl_sync(0xffffffff, adj_size, 0);
  adj_start = __shfl_sync(0xffffffff, adj_start, 0);
  adj_mid = __shfl_sync(0xffffffff, adj_mid, 0);
  buffer.comp_buffer.store_local_idx(eid, adj_start, adj_mid, adj_size);
  vidType *workload_list = buffer.comp_buffer.get_workload_list(eid);

  buffer.read_buffer.fetch_remote_workload_warp(workload_list, gpu_id, eid, adj_start, adj_size);
#endif

  *matched_list = buffer.comp_buffer.get_matched_list(eid);
  matched_size = buffer.comp_buffer.get_matched_size(eid);
  *candidate_list = buffer.comp_buffer.get_candidate_list(eid);
  candidate_size = buffer.comp_buffer.get_candidate_size(eid);
  *new_list = buffer.comp_buffer.get_new_list(eid);
  new_size = buffer.comp_buffer.get_new_size(eid);
}

__device__ void comm_pull_workload_by_warp(BufferBase buffer, vidType **matched_list, vidType &matched_size, eidType eid, int gpu_id)
{
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  vidType adj_size = 0, adj_start = 0, adj_mid = 0;
  if (thread_lane == 0)
  {
    adj_size = buffer.read_buffer.get_workload_size(eid, gpu_id);
    adj_start = buffer.read_buffer.get_workload_offset(eid, gpu_id);
    adj_mid = buffer.read_buffer.get_workload_mid(eid, gpu_id);
  }
  __syncwarp();
  adj_size = __shfl_sync(0xffffffff, adj_size, 0);
  adj_start = __shfl_sync(0xffffffff, adj_start, 0);
  adj_mid = __shfl_sync(0xffffffff, adj_mid, 0);
  buffer.comp_buffer.store_local_idx(eid, adj_start, adj_mid, adj_size);
  *matched_list = buffer.comp_buffer.get_matched_list(eid);
  matched_size = buffer.comp_buffer.get_matched_size(eid);
  buffer.read_buffer.fetch_remote_workload_warp(*matched_list, gpu_id, eid, adj_start, adj_size);
}

// __device__ COMM_TYPE comm_pull_workload_type(BufferBase buffer, eidType eid, int gpu_id)
// {
//   int ctype;
//   int thread_lane = threadIdx.x & (WARP_SIZE - 1);
//   if (thread_lane == 0)
//   {
//     ctype = static_cast<vidType>(buffer.read_buffer.get_workload_type(eid, gpu_id));
//   }
//   __syncwarp();
//   ctype = __shfl_sync(0xffffffff, ctype, 0);
//   return static_cast<COMM_TYPE>(ctype);
// }

__global__ void fetch_all_workload(BufferBase buffer, int gpu_id)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  __shared__ vidType ntasks;
  __shared__ vidType ntasks_mem;
  if (threadIdx.x == 0)
  {
    ntasks = buffer.read_buffer.get_remote_ntasks(gpu_id);
    ntasks_mem = buffer.read_buffer.get_remote_ntasks_mem(gpu_id);
    buffer.comp_buffer.reset_mem(ntasks_mem);
    buffer.comp_buffer.reset_count(ntasks);
  }
  __syncthreads();
  auto adj_size = ntasks_mem / num_warps;
  auto adj_start = warp_id * adj_size;
  if (warp_id == num_warps - 1)
    adj_size = ntasks_mem - adj_start;
  int *buffer_list = buffer.comp_buffer.get_buffer_list_ptr();
  // int *buffer_size = buffer.comp_buffer.get_buffer_size_ptr();
  // int *buffer_mid = buffer.comp_buffer.get_buffer_mid_ptr();
  // int *buffer_offset = buffer.comp_buffer.get_buffer_offset_ptr();
  task_context *context_ptr = buffer.comp_buffer.get_context_ptr();

  buffer.read_buffer.fetch_remote_workload_warp(&buffer_list[adj_start], gpu_id, 0, adj_start, adj_size);
  __syncwarp();

  adj_size = ntasks / num_warps;
  adj_start = warp_id * adj_size;
  // if(warp_id==num_warps-1) adj_size = adj_start+adj_size>=ntasks?ntasks-adj_start:adj_size;
  if (warp_id == num_warps - 1)
    adj_size = ntasks - adj_start;
  // buffer.read_buffer.fetch_remote_workload_size(&buffer_size[adj_start], gpu_id, 0, adj_start, adj_size);
  // buffer.read_buffer.fetch_remote_workload_mid(&buffer_mid[adj_start], gpu_id, 0, adj_start, adj_size);
  // buffer.read_buffer.fetch_remote_workload_offset(&buffer_offset[adj_start], gpu_id, 0, adj_start, adj_size);
  buffer.read_buffer.fetch_remote_context_warp(&context_ptr[adj_start], gpu_id, 0, adj_start, adj_size);
  __syncwarp();
  // if(gpu_id == 0 && thread_id == 0){
  //   buffer.comp_buffer.show_context(0);
  //   buffer.comp_buffer.show_context(1);
  //   buffer.comp_buffer.show_context(2);
  //   buffer.read_buffer.show_context(0);
  //   buffer.read_buffer.show_context(1);
  //   buffer.read_buffer.show_context(2);
  // }
}

__global__ void fetch_all_workload_by_block(BufferBase buffer, int gpu_id)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  int num_block = NUM_BLOCK;
  int block_id = thread_id / BLOCK_SIZE;

  __shared__ vidType ntasks;
  __shared__ vidType ntasks_mem;
  if (threadIdx.x == 0)
  {
    ntasks = buffer.read_buffer.get_remote_ntasks(gpu_id);
    ntasks_mem = buffer.read_buffer.get_remote_ntasks_mem(gpu_id);
    buffer.comp_buffer.reset_mem(ntasks_mem);
    buffer.comp_buffer.reset_count(ntasks);
  }
  __syncthreads();
  auto adj_size = ntasks_mem / num_block;
  auto adj_start = block_id * adj_size;
  if (block_id == num_block - 1)
    adj_size = ntasks_mem - adj_start;
  int *buffer_list = buffer.comp_buffer.get_buffer_list_ptr();
  // int *buffer_size = buffer.comp_buffer.get_buffer_size_ptr();
  // int *buffer_mid = buffer.comp_buffer.get_buffer_mid_ptr();
  // int *buffer_offset = buffer.comp_buffer.get_buffer_offset_ptr();
  task_context *context_ptr = buffer.comp_buffer.get_context_ptr();

  buffer.read_buffer.fetch_remote_workload_block(&buffer_list[adj_start], gpu_id, 0, adj_start, adj_size);
  __syncwarp();

  adj_size = ntasks / num_block;
  adj_start = block_id * adj_size;
  if (block_id == num_block - 1)
    adj_size = ntasks - adj_start;
  // buffer.read_buffer.fetch_remote_workload_size_block(&buffer_size[adj_start], gpu_id, 0, adj_start, adj_size);
  // buffer.read_buffer.fetch_remote_workload_mid_block(&buffer_mid[adj_start], gpu_id, 0, adj_start, adj_size);
  // buffer.read_buffer.fetch_remote_workload_offset_block(&buffer_offset[adj_start], gpu_id, 0, adj_start, adj_size);
  buffer.read_buffer.fetch_remote_context_block(&context_ptr[adj_start], gpu_id, 0, adj_start, adj_size);
  __syncwarp();
}

__device__ void comm_pull_edge_neighbours_by_warp_opt(BufferBase buffer, NvsGraphGPU g, vidType max_deg,
                                                      vidType g_v0, vidType &v0_size, vidType **v0_ptr,
                                                      vidType g_v1, vidType &v1_size, vidType **v1_ptr)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  vidType v0_adj_size = 0;
  eidType v0_adj_start = 0;
  vidType v1_adj_size = 0;
  eidType v1_adj_start = 0;
  if (thread_lane == 0)
  {
    v0_adj_size = g.getOutDegree_remote(g_v0);
    v0_adj_start = g.get_remote_col_start(g_v0);
    v1_adj_size = g.getOutDegree_remote(g_v1);
    v1_adj_start = g.get_remote_col_start(g_v1);

    buffer.comp_buffer.store_local_idx(chunk_idx, mem_idx, /*mid=*/v0_adj_size, /*size=*/v0_adj_size + v1_adj_size);
  }

  v0_adj_size = __shfl_sync(0xffffffff, v0_adj_size, 0);
  v0_adj_start = __shfl_sync(0xffffffff, v0_adj_start, 0);
  v1_adj_size = __shfl_sync(0xffffffff, v1_adj_size, 0);
  v1_adj_start = __shfl_sync(0xffffffff, v1_adj_start, 0);

  __syncwarp();
  *v0_ptr = buffer.comp_buffer.get_matched_list(chunk_idx);
  g.fetch_remote_neigbours_warp(g_v0, *v0_ptr, v0_adj_size, v0_adj_start);
  v0_size = v0_adj_size;
  __syncwarp();

  if (g.d_g_v1_frequency[g_v1] >= FREQ_THD)
  {
    int v1_start_ = g.d_v1_offset[g_v1];
    *v1_ptr = &buffer.read_buffer.get_buffer_list_ptr()[v1_start_];
    v1_size = v1_adj_size;
  }
  else
  {
    *v1_ptr = buffer.comp_buffer.get_candidate_list(chunk_idx);
    g.fetch_remote_neigbours_warp(g_v1, *v1_ptr, v1_adj_size, v1_adj_start);
    v1_size = v1_adj_size;
  }

  __syncwarp();
}

__device__ void comm_pull_edge_neighbours_by_warp_merge(BufferBase buffer, NvsGraphGPU g, vidType max_deg,
                                                        vidType g_v0, vidType &v0_size, vidType **v0_ptr,
                                                        vidType g_v1, vidType &v1_size, vidType **v1_ptr)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = thread_id / WARP_SIZE;                  // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  int chunk_idx = warp_id;
  int mem_idx = warp_id * (max_deg + max_deg);
  vidType v0_adj_size = 0;
  eidType v0_adj_start = 0;
  vidType v1_adj_size = 0;
  eidType v1_adj_start = 0;
  if (thread_lane == 0)
  {
    v0_adj_size = g.getOutDegree_remote(g_v0);
    v0_adj_start = g.get_remote_col_start(g_v0);
    v1_adj_size = g.getOutDegree_remote(g_v1);
    v1_adj_start = g.get_remote_col_start(g_v1);

    buffer.comp_buffer.store_local_idx(chunk_idx, mem_idx, /*mid=*/v0_adj_size, /*size=*/v0_adj_size + v1_adj_size);
  }

  v0_adj_size = __shfl_sync(0xffffffff, v0_adj_size, 0);
  v0_adj_start = __shfl_sync(0xffffffff, v0_adj_start, 0);
  v1_adj_size = __shfl_sync(0xffffffff, v1_adj_size, 0);
  v1_adj_start = __shfl_sync(0xffffffff, v1_adj_start, 0);
  __syncwarp();
#ifdef MERGE_NEIGHBORS_COMM

  if (g.d_g_v1_frequency[g_v0] >= FREQ_THD)
  {
    int v0_start_ = g.d_v1_offset[g_v0];
    *v0_ptr = &buffer.read_buffer.get_buffer_list_ptr()[v0_start_];
    v0_size = v0_adj_size;
  }
  else
  {
    *v0_ptr = buffer.comp_buffer.get_matched_list(chunk_idx);
    g.fetch_remote_neigbours_warp(g_v0, *v0_ptr, v0_adj_size, v0_adj_start);
    v0_size = v0_adj_size;
  }
  __syncwarp();

  if (g.d_g_v1_frequency[g_v1] >= FREQ_THD)
  {
    int v1_start_ = g.d_v1_offset[g_v1];
    *v1_ptr = &buffer.read_buffer.get_buffer_list_ptr()[v1_start_];
    v1_size = v1_adj_size;
  }
  else
  {
    *v1_ptr = buffer.comp_buffer.get_candidate_list(chunk_idx);
    g.fetch_remote_neigbours_warp(g_v1, *v1_ptr, v1_adj_size, v1_adj_start);
    v1_size = v1_adj_size;
  }

  __syncwarp();
#else
  *v0_ptr = buffer.comp_buffer.get_matched_list(chunk_idx);
  *v1_ptr = buffer.comp_buffer.get_candidate_list(chunk_idx);

  g.fetch_remote_neigbours_warp(g_v0, *v0_ptr, v0_adj_size, v0_adj_start);
  v0_size = v0_adj_size;
  __syncwarp();

  g.fetch_remote_neigbours_warp(g_v1, *v1_ptr, v1_adj_size, v1_adj_start);
  v1_size = v1_adj_size;
  __syncwarp();
#endif
}
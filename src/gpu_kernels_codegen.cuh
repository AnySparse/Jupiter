#include "delegator_engine.cuh"

//Codegen for k-cliques
//TODO: support any patterns coge-gen.

__global__ void pattern_task_generate(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
#ifdef PROFILING
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;
#endif
  // TODO: remove vlist, max_deg*NWARPS
  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    auto v0_size = 0;
    auto v1_size = 0;
    // if (g_v1 > g_v0)
    // {
    //   // gb.store_local_idx(eid, -1, 0);
    //   continue;
    // }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    // Step1. Communication(pull base): fetch v0 and v1' neighbours by remote.

    vidType *v0_ptr, *v1_ptr;

#ifdef MERGE_NEIGHBORS_COMM
    comm_pull_edge_neighbours_by_warp_opt(buffer, g, max_deg, g_v0, v0_size,
                                          &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
#ifdef PROFILING
    t_start = clock();
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    t_end = clock();
    total_comm += (t_end - t_start);
#else
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif
#endif

    // Step2. Local computation.
    //  TODO: chunk size can be known after set op is done
#ifdef PROFILING
    t_start = clock();
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    t_end = clock();
    total_comp += (t_end - t_start);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
#endif

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();
    if (thread_lane == 0)
      atomicAdd(&total[1], count1);

      // Step3. Write buffer to distributed shared memory.
#ifdef PROFILING
    t_start = clock();
    comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
    t_end = clock();
    total_comm += (t_end - t_start);
#else
    comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
#endif
  }
#ifdef PROFILING
  if (thread_id == 0 && g.GPUId() == 0)
  {
    printf("Level:%d total_comm:%.3f  total_comp:%.3f\n", buffer.level, total_comm / PEAK_CLK, total_comp / PEAK_CLK);
  }
#endif
}

// __global__ void buffer_division(BufferBase buffer, NvsGraphGPU g, int gpu_id)
// {
//   __shared__ vidType ntasks;
//   comm_fetch_workload_number(buffer, ntasks, gpu_id);
//   int type_a = 0, type_b = 0;
//   int total_ = 0;
//   for (eidType eid = 0; eid < ntasks; eid += 1)
//   {
//     auto ctype = buffer.read_buffer.get_local_type(eid);
//     if (ctype == PULL_NEIGHBOURS)
//       type_a++;
//     else
//       type_b++;
//   }
//   printf("pull neighbours:%d  pull workloads:%d  ntask:%d neighbours percent:%.3f%\n", type_a, type_b, ntasks, type_a * 100.0 / ntasks);
// }

__global__ void pattern_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

  __shared__ vidType tmp_size[WARPS_PER_BLOCK];

  AccType count = 0;
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;

  __shared__ vidType ntasks;
  comm_fetch_workload_number(buffer, ntasks, gpu_id);

  // TODO: we can load entire remote read_buffer to local comp buffer in the future
  // TODO: load partial read_buffer not entire read_buffer
  // for (eidType eid = warp_id + estart; eid < estart + ntasks; eid += num_warps)
  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
// Step1. Communication(psuh base): fetch workload from remote.
#ifdef PROFILING
    t_start = clock();
    vidType *matched_list;
    int matched_size = 0;
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
    t_end = clock();
    total_comm += (t_end - t_start);
#else
    vidType *matched_list;
    int matched_size = 0;
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
#endif
    for (vidType i1 = 0; i1 < matched_size; i1++)
    {
      auto v2 = matched_list[i1];
      if (!g.is_local(v2))
        continue;
      auto v2_ptr = g.fetch_local_neigbours(v2);
      auto v2_size = g.fetch_local_degree(v2);
      int left_size = 0, right_size = 0;
      auto left_operand = buffer.comp_buffer.get_left_operand(eid, v2_ptr, v2_size, left_size);
      auto right_operand = buffer.comp_buffer.get_right_operand(eid, v2_ptr, v2_size, right_size);

      // TODO: verify the set operations
      // if(buffer.comp_buffer.get_comp_type()==OPERATOR_TYPE::INTERSECTION)

      if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION_COUNT)
      {
#ifdef PROFILING
        t_start = clock();
        count += intersect_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += intersect_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION)
      {
#ifdef PROFILING
        t_start = clock();
        // TODO: chunk size can be known after set op is done
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
#endif

        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();

        if (tmp_size[warp_lane] == 0)
          continue;

#ifdef PROFILING
        t_start = clock();
        // Step3. Write buffer to distributed shared memory.
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
        t_end = clock();
        total_comm += (t_end - t_start);
#else
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE_COUNT)
      {
#ifdef PROFILING
        t_start = clock();
        count += difference_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += difference_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE)
      {
#ifdef PROFILING
        t_start = clock();
        // TODO: chunk size can be known after set op is done
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
#endif

        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();

        if (tmp_size[warp_lane] == 0)
          continue;

#ifdef PROFILING
        t_start = clock();
        // Step3. Write buffer to distributed shared memory.
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
        t_end = clock();
        total_comm += (t_end - t_start);
#else
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
#endif
      }
    }

    __syncwarp();
  }
#ifdef PROFILING
  if (thread_id == 0 && g.GPUId() == 0)
  {
    printf("Level:%d  from GPU:%d total_comm:%.3f  total_comp:%.3f\n", buffer.level, gpu_id, total_comm / PEAK_CLK, total_comp / PEAK_CLK);
  }
#endif
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void pattern_extend_pull_workloads(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

  __shared__ vidType tmp_size[WARPS_PER_BLOCK];

  AccType count = 0;
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;

#ifdef BATCH_LOAD
  int ntasks = buffer.comp_buffer.get_count();
#else
  __shared__ int ntasks;
  comm_fetch_workload_number(buffer, ntasks, gpu_id);
#endif
  // TODO: we can load entire remote read_buffer to local comp buffer in the future
  // TODO: load partial read_buffer not entire read_buffer
  // for (eidType eid = warp_id + estart; eid < estart + ntasks; eid += num_warps)
  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {

#ifdef BATCH_LOAD
    vidType *matched_list = buffer.comp_buffer.get_matched_list(eid);
    int matched_size = buffer.comp_buffer.get_matched_size(eid);
#else
    auto ctype = comm_pull_workload_type(buffer, eid, gpu_id);
    if (ctype == PULL_NEIGHBOURS)
      continue;
// Step1. Communication(psuh base): fetch workload from remote.
#ifdef PROFILING
    t_start = clock();
    vidType *matched_list;
    int matched_size = 0;
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
    t_end = clock();
    total_comm += (t_end - t_start);
#else
    vidType *matched_list;
    int matched_size = 0;
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
#endif

#endif

    for (vidType i1 = 0; i1 < matched_size; i1++)
    {
      auto v2 = matched_list[i1];
      if (!g.is_local(v2))
        continue;
      auto v2_ptr = g.fetch_local_neigbours(v2);
      auto v2_size = g.fetch_local_degree(v2);
      int left_size = 0, right_size = 0;
      auto left_operand = buffer.comp_buffer.get_left_operand(eid, v2_ptr, v2_size, left_size);
      auto right_operand = buffer.comp_buffer.get_right_operand(eid, v2_ptr, v2_size, right_size);

      bool need_write = false;
      if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION_COUNT)
      {
#ifdef PROFILING
        t_start = clock();
        count += intersect_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += intersect_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION)
      {
#ifdef PROFILING
        t_start = clock();
        // TODO: chunk size can be known after set op is done
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
#endif

        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();
        need_write = true;
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE_COUNT)
      {
#ifdef PROFILING
        t_start = clock();
        count += difference_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += difference_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE)
      {
#ifdef PROFILING
        t_start = clock();
        // TODO: chunk size can be known after set op is done
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
#endif

        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();
        need_write = true;
      }
      if (need_write)
      {

        if (tmp_size[warp_lane] == 0)
          continue;

#ifdef PROFILING
        t_start = clock();
        // Step3. Write buffer to distributed shared memory.
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
        t_end = clock();
        total_comm += (t_end - t_start);
#else
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
#endif
      }
    }

    __syncwarp();
  }
#ifdef PROFILING
  if (thread_id == 0 && g.GPUId() == 0)
  {
    printf("Level:%d pull workloads from GPU:%d total_comm:%.3f  total_comp:%.3f\n", buffer.level, gpu_id, total_comm / PEAK_CLK, total_comp / PEAK_CLK);
  }
#endif
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void pattern_extend_pull_neighbours(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

  __shared__ vidType tmp_size[WARPS_PER_BLOCK];

  AccType count = 0;
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;

  __shared__ vidType ntasks;
  comm_fetch_workload_number(buffer, ntasks, gpu_id);

  // TODO: we can load entire remote read_buffer to local comp buffer in the future
  // TODO: load partial read_buffer not entire read_buffer
  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    // auto ctype = comm_pull_workload_type(buffer, eid, gpu_id);
    // if (ctype == PULL_WORKLOADS)
    //   continue;

    // Step1. Communication(psuh base): fetch workload from remote.
    vidType *matched_list;
    int matched_size = 0;

#ifdef PROFILING
    t_start = clock();
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
    t_end = clock();
    total_comm += (t_end - t_start);
#else
    comm_pull_workload_by_warp(buffer, &matched_list, matched_size, eid, gpu_id);
#endif

    for (vidType i1 = 0; i1 < matched_size; i1++)
    {
      auto v2 = matched_list[i1];
      // if (!g.is_local(v2))
      //   continue;
      // auto v2_ptr = g.fetch_local_neigbours(v2);
      // auto v2_size = g.fetch_local_degree(v2);

      vidType v2_size = 0;
      vidType *v2_ptr = tmp_list;

#ifdef PROFILING
      t_start = clock();
      comm_pull_neighbours_by_warp(buffer, g, v2, v2_size, v2_ptr);
      t_end = clock();
      total_comm += (t_end - t_start);
#else
      comm_pull_neighbours_by_warp(buffer, g, v2, v2_size, v2_ptr);
#endif

      int left_size = 0, right_size = 0;
      auto left_operand = buffer.comp_buffer.get_left_operand(eid, v2_ptr, v2_size, left_size);
      auto right_operand = buffer.comp_buffer.get_right_operand(eid, v2_ptr, v2_size, right_size);
      bool need_write = false;
      if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION_COUNT)
      {

#ifdef PROFILING
        t_start = clock();
        count += intersect_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += intersect_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::INTERSECTION)
      {
#ifdef PROFILING
        t_start = clock();
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = intersect(left_operand, left_size, right_operand, right_size, tmp_list);
#endif

        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();
        need_write = true;
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE_COUNT)
      {

#ifdef PROFILING
        t_start = clock();
        count += difference_num(left_operand, left_size, right_operand, right_size);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        count += difference_num(left_operand, left_size, right_operand, right_size);
#endif
      }
      else if (buffer.comp_buffer.get_comp_type() == OPERATOR_TYPE::DIFFERENCE)
      {
#ifdef PROFILING
        t_start = clock();
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
        t_end = clock();
        total_comp += (t_end - t_start);
#else
        auto count1 = difference_set(left_operand, left_size, right_operand, right_size, tmp_list);
#endif
        if (thread_lane == 0)
          tmp_size[warp_lane] = count1;
        __syncwarp();
        need_write = true;
      }

      if (need_write)
      {
        if (tmp_size[warp_lane] == 0)
          continue;

#ifdef PROFILING
        t_start = clock();
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
        t_end = clock();
        total_comm += (t_end - t_start);
#else
        comm_push_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane]);
#endif
      }
    }

    __syncwarp();
  }
#ifdef PROFILING
  if (thread_id == 0 && g.GPUId() == 0)
  {
    printf("Level:%d  pull neighbours from GPU:%d total_comm:%.3f  total_comp:%.3f\n", buffer.level, gpu_id, total_comm / PEAK_CLK, total_comp / PEAK_CLK);
  }
#endif
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void count_freq(BufferBase buffer, eidType estart,
                           eidType ne, eidType true_ne,
                           vidType max_deg, NvsGraphGPU g,
                           AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int num_threads = (BLOCK_SIZE)*gridDim.x;
  // TODO: remove vlist, max_deg*NWARPS
  for (eidType eid = thread_id + estart; eid < estart + ne; eid += num_threads)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    auto v0_size = 0;
    auto v1_size = 0;
    if (g_v1 > g_v0)
    {
      // gb.store_local_idx(eid, -1, 0);
      continue;
    }
    // int v1_adj_size = g.getOutDegree_remote(g_v1);
    atomicAdd(&g.d_g_v1_frequency[g_v1], 1);
  }
}

__global__ void count_offset(BufferBase buffer, eidType estart,
                             eidType ne, eidType true_ne,
                             vidType max_deg, NvsGraphGPU g,
                             AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int num_threads = (BLOCK_SIZE)*gridDim.x;
  // TODO: remove vlist, max_deg*NWARPS
  for (vidType vid = thread_id; vid < g.total_nv; vid += num_threads)
  {
    if (g.d_g_v1_frequency[vid] >= FREQ_THD)
    {
      int v_size = g.getOutDegree_remote(vid);
      g.d_v1_offset[vid] = atomicAdd(g.d_global_index, v_size);
    }
  }
}

__global__ void write_result(BufferBase buffer, eidType estart,
                             eidType ne, eidType true_ne,
                             vidType max_deg, NvsGraphGPU g,
                             AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int num_threads = (BLOCK_SIZE)*gridDim.x;
  int warp_id = thread_id / WARP_SIZE; // global warp index
  int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (vidType vid = warp_id; vid < g.total_nv; vid += num_warps)
  {
    if (g.d_g_v1_frequency[vid] >= FREQ_THD)
    {
      int v_size = 0;
      int v_start = g.d_v1_offset[vid];
      int *v_ptr = &buffer.read_buffer.get_buffer_list_ptr()[v_start];
      comm_pull_neighbours_by_warp(buffer, g, vid, v_size, v_ptr);
    }
  }
}
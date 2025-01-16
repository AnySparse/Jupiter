__global__ void P1_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;
  // max_deg += 2;
  // __syncwarp();
  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  vidType *tmp_list1 = &buffer.read_buffer.get_buffer_list_ptr()[int64_t(warp_id) * int64_t(max_deg) * 2];

  AccType count = 0;

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    // if(thread_lane==0) printf("eid:%d\n",eid);
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    // todo:add check the size of v0 and v1
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    if (g_v1 >= g_v0)
    { // swap context
      auto tmp = g_v0;
      g_v0 = g_v1;
      g_v1 = tmp;
      tmp = v0_size;
      v0_size = v1_size;
      v1_size = tmp;
      auto tmp_ptr = v0_ptr;
      v0_ptr = v1_ptr;
      v1_ptr = tmp_ptr;
    }
#else
    if (g_v1 >= g_v0)
    {
      continue;
    }
    auto v0_size = 0;
    auto v1_size = 0;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif

    // auto v3_size = 0;

    // int chunk_idx = warp_id;
    // int mem_idx = warp_id * (max_deg + max_deg);

#ifdef USE_FUSE
    // todo add short cut to enable direct pull compute if necessary
    if (handle_local(g, v0_ptr, v0_size, v0_size + v1_size))
    {
      vidType *v2_ptr = tmp_list;
      int v2_size = 0;
      for (int v2_idx = 0; v2_idx < v0_size; v2_idx++)
      {
        auto g_v2 = v0_ptr[v2_idx];
        if (g_v2 >= g_v1)
          break;

        comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);

        int *left_operand = v1_ptr;
        int *right_operand = v2_ptr;
        int left_size = v1_size;
        int right_size = v2_size;
        count += intersect_num(left_operand, left_size, right_operand, right_size, g_v0);
      }
    }
    else
    {
#ifndef USE_FILTER
      auto count1 = intersect(v0_ptr, v0_size, v0_ptr, v0_size, tmp_list);
      auto count2 = v1_size;
      auto tmp_list1 = v1_ptr;
#else
      auto count1 = list_smaller(g_v1, v0_ptr, v0_size, tmp_list); // filt unfit v2
      auto count2 = list_smaller(g_v0, v1_ptr, v1_size, tmp_list1);
#endif
      if (thread_lane == 0)
      {
        tmp_size[warp_lane] = count1 + 2;
        tmp_list[count1 + 0] = g_v0;
        tmp_list[count1 + 1] = g_v1;
      }
      __syncwarp();
      comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], tmp_list1, count2);
    }
#else
#ifndef USE_FILTER
    auto count1 = intersect(v0_ptr, v0_size, v0_ptr, v0_size, tmp_list);
    auto count2 = v1_size;
    auto tmp_list1 = v1_ptr;
#else
    auto count1 = list_smaller(g_v1, v0_ptr, v0_size, tmp_list); // filt unfit v2
    auto count2 = list_smaller(g_v0, v1_ptr, v1_size, tmp_list1);
#endif
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], tmp_list1, count2);
// comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], nullptr, 0, tmp_list1, count2);
#endif
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P1_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  // vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

  __shared__ vidType tmp_size[WARPS_PER_BLOCK];

  AccType count = 0;
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;

  __shared__ vidType ntasks;
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list, *new_list;
    int matched_size = 0, candidate_size = 0, new_size;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    // comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &new_list, new_size,&candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = matched_list;
    auto v0_size = matched_size;

    vidType *v1_ptr = candidate_list;
    auto v1_size = candidate_size;

    for (int idx2 = 0; idx2 < v0_size; idx2++)
    {
      auto g_v2 = v0_ptr[idx2];
#ifndef USE_FILTER
      if (g_v2 >= g_v1)
        break;
#endif
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);

      count += intersect_num(v2_ptr, v2_size, v1_ptr, v1_size, g_v0);
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P2_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    // if(thread_lane==0) printf("eid:%d\n",eid);
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    // todo:add check the size of v0 and v1
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    if (g_v1 >= g_v0)
    {
      continue;
    }
    auto v0_size = 0;
    auto v1_size = 0;
    vidType *v0_ptr, *v1_ptr;
#endif
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();
    for (vidType i = 0; i < tmp_size[warp_lane]; i++)
    {
      auto g_v2 = tmp_list[i];
      AccType count3 = count_smaller(g_v2, tmp_list, tmp_size[warp_lane]);
      if (thread_lane == 0)
        count += count3;
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P3_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    // if(thread_lane==0) printf("eid:%d\n",eid);
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    auto v0_size = 0;
    auto v1_size = 0;
    auto v3_size = 0;

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#ifdef USE_FILTER
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
#endif
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size);
  }
}

__global__ void P3_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    for (int idx3 = 0; idx3 < candidate_size; idx3++)
    {
      auto g_v3 = candidate_list[idx3];
      if (g_v3 == g_v1)
        continue;
      if (!g.is_local(g_v3))
        continue;
      auto v3_size = g.fetch_local_degree(g_v3);
      vidType *v3_ptr = g.fetch_local_neigbours(g_v3);

      for (vidType i = 0; i < matched_size; i++)
      {
        auto g_v2 = matched_list[i];
#ifndef USE_FILTER
        if (g_v2 <= g_v1 || g_v2 == g_v3 || g_v2 == g_v0)
          continue;
#else
        if (g_v2 == g_v3 || g_v2 == g_v0)
          continue;
#endif
        for (int idx4 = thread_lane; idx4 < v3_size; idx4 += 32)
        {
          auto g_v4 = v3_ptr[idx4];
          if (g_v4 == g_v0 || g_v4 == g_v1 || g_v4 == g_v3 || g_v4 == g_v2)
            continue;
          else
            count++;
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P4_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    // if(thread_lane==0) printf("eid:%d\n",eid);
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    auto v0_size = 0;
    auto v1_size = 0;
    // auto v3_size = 0;

    // int chunk_idx = warp_id;
    // int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    if (g_v1 >= g_v0)
      continue;
#endif

#ifdef USE_LAZY
    for (vidType id = thread_lane; id < v0_size; id += WARP_SIZE)
    {
      tmp_list[id] = v0_ptr[id];
    }
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = v0_size + 2;
      tmp_list[v0_size + 0] = g_v0;
      tmp_list[v0_size + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v1_ptr, v1_size);
// comm_push_SP_workload_by_warp(buffer, g, max_deg, v0_ptr, v0_size + 2, v1_ptr, v1_size, g_v0, g_v1);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size, v1_ptr, v1_size);
#endif
  }
}

__global__ void P4_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list, *new_list;
    int matched_size = 0, candidate_size = 0, new_size = 0;
#ifdef USE_LAZY
    comm_pull_SP_by_warp(buffer, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = candidate_list[candidate_size - 2];
    auto g_v1 = candidate_list[candidate_size - 1];
    candidate_size = candidate_size - 2;
    matched_list = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    matched_size = intersect(candidate_list, candidate_size, new_list, new_size, matched_list);
#else
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;
#endif
    for (vidType v3_id = 0; v3_id < new_size; v3_id++)
    {
      vidType g_v3 = new_list[v3_id];
      if (!g.is_local(g_v3) || g_v3 == g_v0)
        continue;

      vidType *v3_ptr = g.fetch_local_neigbours(g_v3);
      vidType v3_degree = g.fetch_local_degree(g_v3);
      int v4_size = intersect(v3_ptr, v3_degree, candidate_list, candidate_size, tmp_list);

      for (vidType v2_id = 0; v2_id < matched_size; v2_id++)
      {
        vidType g_v2 = matched_list[v2_id];
        if (g_v2 == g_v3)
          continue;

        for (vidType v4_id = thread_lane; v4_id < v4_size; v4_id += warpSize)
        {
          vidType g_v4 = tmp_list[v4_id];
          if (g_v4 == g_v1 || g_v4 == g_v2)
            continue;
          else
            count++;
        }
      }
    }
    // int revise = binary_search(matched_list,g_v3,matched_size);

    // if(thread_lane == 0)
    //   tmp_size[warp_lane] = 0;
    // __syncwarp();

    // int thread_size = intersect_num(tmp_list, v4_size, candidate_list, candidate_size);
    // atomicAdd(&tmp_size[warp_lane],thread_size);
    // __syncwarp();

    // int recovered_size = tmp_size[warp_lane];
    // if(thread_lane == 0)
    //   count += (matched_size - revise)*(v4_size - recovered_size) + (matched_size - revise - 1)*(recovered_size);
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P5_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

#ifdef USE_COMP
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    if (g_v1 <= g_v0)
    {
      continue;
    }
    auto v0_size = 0;
    auto v1_size = 0;
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif

    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], /*useless=*/v0_ptr, 0);
  }
}

__global__ void P5_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    for (int idx3 = 0; idx3 < matched_size; idx3++)
    {
      auto g_v2 = matched_list[idx3];
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(matched_list, matched_size, v2_ptr, v2_size, g_v2, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < matched_size; i++)
      {
        auto g_v4 = matched_list[i];
        if (g_v2 == g_v4)
          continue;
        for (int idx_ = thread_lane; idx_ < tmp_size[warp_lane]; idx_ += 32)
        {
          auto v_ = tmp_list[idx_];
          if (v_ == g_v4)
            continue;
          else
            count += 1;
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P6_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    auto v0_size = 0;
    auto v1_size = 0;
    auto v3_size = 0;
    if (g_v1 >= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], /*useless=*/v0_ptr, 0);
  }
}

__global__ void P6_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType ancestors[2];
    for (vidType idx2 = 0; idx2 < matched_size; idx2++)
    {
      vidType g_v2 = matched_list[idx2];
      // OPT1:
      //  if(!g.is_local(g_v2)) continue;
      //  auto v2_size = g.fetch_local_degree(g_v2);
      //  vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      //  auto v3_size = 0;
      //  vidType* v3_ptr = tmp_list;
      //  for (vidType idx3 = 0; idx3 < idx2; idx3++)
      //  {
      //    vidType g_v3 = matched_list[idx3];
      //    comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);
      //    ancestors[0] = g_v0;
      //    ancestors[1] = g_v1;
      //    count += intersect_num(v2_ptr, v2_size, v3_ptr, v3_size, ancestors, 2); // v4 != v1 && v4 != v2
      //  }

      // OPT2:
      auto v2_size = 0;
      vidType *v2_ptr = tmp_list;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      for (vidType idx3 = 0; idx3 < idx2; idx3++)
      {
        vidType g_v3 = matched_list[idx3];
        if (!g.is_local(g_v3))
          continue;
        auto v3_size = g.fetch_local_degree(g_v3);
        vidType *v3_ptr = g.fetch_local_neigbours(g_v3);
        ancestors[0] = g_v0;
        ancestors[1] = g_v1;
        count += intersect_num(v2_ptr, v2_size, v3_ptr, v3_size, ancestors, 2); // v4 != v1 && v4 != v2
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

// __global__ void P6_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
// {
//   __shared__ typename BlockReduce::TempStorage temp_storage;
//   int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
//   int warp_id = thread_id / WARP_SIZE;                   // global warp index
//   int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
//   int thread_lane = threadIdx.x & (WARP_SIZE - 1);
//   int warp_lane = threadIdx.x / WARP_SIZE;

//   vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
//   __shared__ vidType tmp_size[WARPS_PER_BLOCK];

//   AccType count = 0;

//   for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
//   {
//     if (eid >= true_ne)
//     {
//       continue;
//     }
//     auto g_v0 = g.get_src(eid);
//     auto g_v1 = g.get_dst(eid);
//     auto v0_size = 0;
//     auto v1_size = 0;
//     auto v3_size = 0;
//     if (g_v1 >= g_v0)
//     {
//       continue;
//     }
//     __syncwarp();

//     int chunk_idx = warp_id;
//     int mem_idx = warp_id * (max_deg + max_deg);

//     vidType *v0_ptr, *v1_ptr;

//     vidType *v2_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
//     comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
//                                       g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
//     auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

//     for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
//     {
//       vidType g_v2 = tmp_list[idx2];
//       int v2_size = 0;
//       comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);

//       if (thread_lane == 0){
//         v2_ptr[v2_size+0] = g_v0;
//         v2_ptr[v2_size+1] = g_v1;
//       }
//       __syncwarp();
//       comm_push_SP_workload_by_warp(buffer, g, max_deg, v2_ptr, v2_size+2, tmp_list, idx2);
//     }
//   }

// }

// __global__ void P6_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
// {
//   __shared__ typename BlockReduce::TempStorage temp_storage;
//   int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
//   int warp_id = thread_id / WARP_SIZE;                   // global warp index
//   int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
//   int thread_lane = threadIdx.x & (WARP_SIZE - 1);
//   int warp_lane = threadIdx.x / WARP_SIZE;

//   vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

//   __shared__ vidType tmp_size[WARPS_PER_BLOCK];

//   AccType count = 0;
//   unsigned long long t_start, t_end;
//   unsigned long long total_comm = 0, total_comp = 0;

//   __shared__ vidType ntasks;
//   comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

//   for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
//   {
//     vidType *matched_list, *candidate_list;
//     int matched_size = 0, candidate_size= 0;
//     comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
//     auto g_v0 = matched_list[matched_size-2];
//     auto g_v1 = matched_list[matched_size-1];
//     matched_size = matched_size - 3;

//     vidType* v2_ptr = matched_list;
//     auto v2_size = matched_size;
//     vidType ancestors[2];

//     for (vidType idx3 = 0; idx3 < candidate_size; idx3++)
//     {
//       vidType g_v3 = candidate_list[idx3];
//       if(!g.is_local(g_v3)) continue;
//       auto v3_size = g.fetch_local_degree(g_v3);
//       vidType *v3_ptr = g.fetch_local_neigbours(g_v3);
//       ancestors[0] = g_v0;
//       ancestors[1] = g_v1;
//       count += intersect_num(v2_ptr, v2_size, v3_ptr, v3_size, ancestors, 2); // v4 != v1 && v4 != v2
//     }
//   AccType block_num = BlockReduce(temp_storage).Sum(count);
//   if (threadIdx.x == 0)
//     atomicAdd(&total[0], block_num);
// }

__global__ void P7_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    // if(thread_lane==0) printf("eid:%d\n",eid);
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    auto v0_size = 0;
    auto v1_size = 0;

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size);
  }
}

__global__ void P7_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = candidate_list;
    auto v0_size = candidate_size;

    for (int idx2 = 0; idx2 < matched_size; idx2++)
    {
      auto g_v2 = matched_list[idx2];
      if (g_v2 >= g_v1)
        break;
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(v2_ptr, v2_size, v0_ptr, v0_size, tmp_list);
      // if(thread_lane == 0)
      //   tmp_size[warp_lane] = count2;
      // int revision1 = binary_search(tmp_list,g_v1,count2) ? -1 : 0;
      // for (vidType i = 0; i < matched_size; i++)
      // {
      //   auto g_v3 = matched_list[i];
      //   // assert(g_v3 != g_v1);
      //   if (g_v3 == g_v2)
      //     continue;
      //   int revision2 = binary_search(tmp_list,g_v3,count2) ? -1 : 0;
      //   if(thread_lane == 0){
      //     assert((count2 + revision1 + revision2) >= 0);
      //     count += count2 + revision1 + revision2;
      //   }
      // }

      for (vidType i = 0; i < count2; i++)
      {
        auto g_v4 = tmp_list[i];
        if (g_v1 == g_v4)
          continue;
        // v3_size = 0;
        for (int idx_ = thread_lane; idx_ < matched_size; idx_ += 32)
        {
          auto v_ = matched_list[idx_];
          if (v_ == g_v4 || v_ == g_v2)
            continue;
          else
            count += 1;
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P8_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  vidType *tmp_list1 = &buffer.read_buffer.get_buffer_list_ptr()[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];

  AccType count = 0;

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
    auto v3_size = 0;

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#ifdef USE_FILTER
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count2 = list_smaller(g_v1, v0_ptr, v0_size, tmp_list1);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    auto count2 = v0_size;
    tmp_list1 = v0_ptr;
#endif
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], tmp_list1, count2);
  }
}

__global__ void P8_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = candidate_list;
    auto v0_size = candidate_size;

    vidType *v2_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg];
    for (vidType idx2 = 0; idx2 < matched_size; idx2++)
    {
      vidType g_v2 = matched_list[idx2];
#ifndef USE_FILTER
      if (g_v2 >= g_v1)
        continue;
#endif
      int v2_size = 0;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      {
        auto g_v3 = tmp_list[i];
#ifndef USE_FILTER
        if (g_v3 >= g_v1)
          continue;
#endif
        if (!g.is_local(g_v3))
          continue;
        auto v3_size = g.fetch_local_degree(g_v3);
        vidType *v3_ptr = g.fetch_local_neigbours(g_v3);

        count += intersect_num(matched_list, matched_size, v3_ptr, v3_size, g_v2);
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P9_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    // todo:add check the size of v0 and v1
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    if (g_v1 <= g_v0)
    { // swap context
      auto tmp = g_v0;
      g_v0 = g_v1;
      g_v1 = tmp;
      tmp = v0_size;
      v0_size = v1_size;
      v1_size = tmp;
      auto tmp_ptr = v0_ptr;
      v0_ptr = v1_ptr;
      v1_ptr = tmp_ptr;
    }
#else
    if (g_v1 <= g_v0)
    {
      continue;
    }
    auto v0_size = 0;
    auto v1_size = 0;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], /*useless=*/v0_ptr, 0);
  }
}

__global__ void P9_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    for (int idx2 = 0; idx2 < matched_size; idx2++)
    {
      auto g_v2 = matched_list[idx2];
      if (g_v2 <= g_v1)
        continue;
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(matched_list, matched_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      {
        auto g_v3 = tmp_list[i];
        AccType count3 = count_smaller(g_v3, tmp_list, tmp_size[warp_lane]);
        if (thread_lane == 0)
          count += count3;
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P10_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size);
  }
}

__global__ void P10_extend_step1(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id); // intersect and v0
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = candidate_list;
    auto v0_size = candidate_size;

    // vidType *v2_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg];
    for (vidType idx2 = 0; idx2 < matched_size; idx2++)
    {
      vidType g_v2 = matched_list[idx2];
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      if (thread_lane == 0)
      {
        tmp_size[warp_lane] = count2 + 3;
        tmp_list[count2 + 0] = g_v0;
        tmp_list[count2 + 1] = g_v1;
        tmp_list[count2 + 2] = g_v2;
      }
      __syncwarp();
      comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], matched_list, matched_size);
    }
  }

  // AccType block_num = BlockReduce(temp_storage).Sum(count);
  // if (threadIdx.x == 0)
  //   atomicAdd(&total[0], block_num);
}

__global__ void P10_extend_step2(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);

    auto g_v0 = matched_list[matched_size - 3];
    auto g_v1 = matched_list[matched_size - 2];
    auto g_v2 = matched_list[matched_size - 1];
    matched_size = matched_size - 3;

    // vidType *v5_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType v0_size = 0;
    vidType *v0_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    comm_pull_neighbours_by_warp(buffer, g, g_v0, v0_size, v0_ptr);

    for (vidType v3_id = 0; v3_id < matched_size; v3_id++)
    {
      vidType g_v3 = matched_list[v3_id];
      if (g_v3 >= g_v1)
        break;
      if (!g.is_local(g_v3))
        continue;
      auto v3_ptr = g.fetch_local_neigbours(g_v3);
      auto v3_size = g.fetch_local_degree(g_v3);
      auto v5_size = intersect(v3_ptr, v3_size, v0_ptr, v0_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = v5_size;
      __syncwarp();
      for (vidType v4_id = 0; v4_id < candidate_size; v4_id++)
      {
        vidType g_v4 = candidate_list[v4_id];
        if (g_v4 == g_v3 || g_v4 == g_v2)
          continue;
        for (vidType v5_id = thread_lane; v5_id < tmp_size[warp_lane]; v5_id += WARP_SIZE)
        {
          vidType g_v5 = tmp_list[v5_id];
          if (g_v5 == g_v1 || g_v5 == g_v2 || g_v5 == g_v4)
            continue;
          count++;
        }
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P11_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    // todo:add check the size of v0 and v1
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    if (g_v1 < g_v0)
    { // swap context
      auto tmp = g_v0;
      g_v0 = g_v1;
      g_v1 = tmp;
      tmp = v0_size;
      v0_size = v1_size;
      v1_size = tmp;
      auto tmp_ptr = v0_ptr;
      v0_ptr = v1_ptr;
      v1_ptr = tmp_ptr;
    }
#else
    auto v0_size = 0;
    auto v1_size = 0;
    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif

#ifdef USE_LAZY
    for (vidType id = thread_lane; id < v0_size; id += WARP_SIZE)
    {
      tmp_list[id] = v0_ptr[id];
    }
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = v0_size + 2;
      tmp_list[v0_size + 0] = g_v0;
      tmp_list[v0_size + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v1_ptr, v1_size);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size, v1_ptr, v1_size);
#endif
  }
}

__global__ void P11_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];

  __shared__ vidType tmp_size[WARPS_PER_BLOCK * 2];

  AccType count = 0;
  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;

  __shared__ vidType ntasks;
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list, *new_list;
    int matched_size = 0, candidate_size = 0, new_size = 0;
#ifdef USE_LAZY
    comm_pull_SP_by_warp(buffer, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = candidate_list[candidate_size - 2];
    auto g_v1 = candidate_list[candidate_size - 1];
    candidate_size = candidate_size - 2;
    matched_list = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    matched_size = intersect(candidate_list, candidate_size, new_list, new_size, matched_list);
    vidType *v5_list = &matched_list[max_deg];
#else
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;
    vidType *v5_list = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
#endif
    vidType v1_size = new_size;
    vidType *v1_ptr = new_list; // not work need new api

    // comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);

    for (int idx2 = 0; idx2 < matched_size; idx2++)
    {
      auto g_v2 = matched_list[idx2];
      if (g_v2 <= g_v1)
        continue;
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);

      auto count1 = intersect(v1_ptr, v1_size, v2_ptr, v2_size, v5_list);
      if (thread_lane == 0)
      {
        tmp_size[warp_lane * 2] = count1;
      }
      __syncwarp();
      auto count2 = intersect(candidate_list, candidate_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
      {
        tmp_size[warp_lane * 2 + 1] = count2;
      }
      __syncwarp();
      for (vidType v3_id = 0; v3_id < matched_size; v3_id++)
      {
        vidType g_v3 = matched_list[v3_id];
        if (g_v3 == g_v2)
          continue;
        for (vidType v4_id = 0; v4_id < tmp_size[warp_lane * 2 + 1]; v4_id++)
        {
          vidType g_v4 = tmp_list[v4_id];
          if (g_v4 == g_v1 || g_v4 == g_v3)
            continue;
          for (vidType v5_id = thread_lane; v5_id < tmp_size[warp_lane * 2]; v5_id += WARP_SIZE)
          {
            vidType g_v5 = v5_list[v5_id];
            if (g_v5 == g_v0 || g_v5 == g_v3 || g_v5 == g_v4)
              continue;
            count++;
          }
        }
      }
      __syncwarp();
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P12_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    auto v0_size = 0;
    auto v1_size = 0;
    auto v3_size = 0;
    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], /*useless=*/v0_ptr, 0);
  }
}

__global__ void P12_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    for (int idx2 = 0; idx2 < matched_size; idx2++)
    {
      auto g_v2 = matched_list[idx2];
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(matched_list, matched_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size[warp_lane]; idx3++)
      {
        auto g_v3 = tmp_list[idx3];
        for (vidType idx4 = idx3 + 1; idx4 < tmp_size[warp_lane]; idx4++)
        {
          auto g_v4 = tmp_list[idx4];
          for (int idx_ = thread_lane; idx_ < matched_size; idx_ += 32)
          {
            auto v_ = matched_list[idx_];
            if (v_ == g_v4 || v_ == g_v1 || v_ == g_v2 || v_ == g_v3 || v_ == g_v0)
              continue;
            else
              count += 1;
          }
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void vsgm_last_pattern_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
#ifdef USE_COMP
    auto v0_size = g.fetch_local_degree(g_v0);
    auto v1_size = g.get_degree_remote(g_v1);
    if (v0_size < v1_size || v0_size == v1_size && g_v1 >= g_v0)
      continue;
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#else
    auto v0_size = 0;
    auto v1_size = 0;
    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
#endif

#ifdef USE_LAZY
    for (vidType id = thread_lane; id < v0_size; id += WARP_SIZE)
    {
      tmp_list[id] = v0_ptr[id];
    }
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = v0_size + 2;
      tmp_list[v0_size + 0] = g_v0;
      tmp_list[v0_size + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v1_ptr, v1_size);
#else
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, v0_size, v1_ptr, v1_size);
#endif
  }
}

__global__ void vsgm_last_pattern_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list, *new_list;
    int matched_size = 0, candidate_size = 0, new_size = 0;
#ifdef USE_LAZY
    comm_pull_SP_by_warp(buffer, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = candidate_list[candidate_size - 2];
    auto g_v1 = candidate_list[candidate_size - 1];
    candidate_size = candidate_size - 2;
    matched_list = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    matched_size = intersect(candidate_list, candidate_size, new_list, new_size, matched_list);
    vidType *v5_list = &matched_list[max_deg];
#else
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, &new_list, new_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;
    vidType *v5_list = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
#endif
    vidType v1_size = 0;
    vidType *v1_ptr = new_list; // not work need new api

    v1_size = new_size;

    for (int idx2 = 0; idx2 < candidate_size; idx2++)
    {
      auto g_v2 = candidate_list[idx2];
      if (!g.is_local(g_v2) || g_v2 == g_v1)
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(v1_ptr, v1_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();

      for (vidType idx5 = 0; idx5 < tmp_size[warp_lane]; idx5++)
      {
        auto g_v5 = tmp_list[idx5];
        if (g_v5 == g_v0)
          continue;
        for (vidType idx3 = 0; idx3 < matched_size; idx3++)
        {
          auto g_v3 = matched_list[idx3];
          if (g_v3 == g_v2 || g_v3 == g_v1 || g_v3 == g_v0 || g_v3 == g_v5)
            continue;
          for (int idx_ = thread_lane; idx_ < matched_size; idx_ += 32)
          {
            auto v_ = matched_list[idx_];
            if (v_ == g_v0 || v_ == g_v1 || v_ == g_v2 || v_ <= g_v3 || v_ == g_v5)
              continue;
            else
              count += 1;
          }
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P14_kernel_producer(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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
    auto v3_size = 0;
    // if (g_v1 <= g_v0)
    // {
    //   continue;
    // }
    // __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp(buffer, g, max_deg,
                                      g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
    {
      tmp_size[warp_lane] = count1 + 2;
      tmp_list[count1 + 0] = g_v0;
      tmp_list[count1 + 1] = g_v1;
    }
    __syncwarp();
    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], /*useless=*/v0_ptr, 0);
  }
}

__global__ void P14_extend(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    for (int idx2 = 0; idx2 < matched_size; idx2++)
    {
      auto g_v2 = matched_list[idx2];
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      count += intersect_num(matched_list, matched_size, v2_ptr, v2_size);
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

//===========================================
__global__ void P8_extend_step1(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v0 = matched_list[matched_size - 2];
    auto g_v1 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = candidate_list;
    auto v0_size = candidate_size;

    // vidType *v2_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg];
    for (vidType idx2 = 0; idx2 < matched_size; idx2++)
    {
      vidType g_v2 = matched_list[idx2];
      if (g_v2 >= g_v1)
        continue;
      // comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      if (!g.is_local(g_v2))
        continue;
      auto v2_size = g.fetch_local_degree(g_v2);
      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_list);
      if (thread_lane == 0)
        tmp_size[warp_lane] = count2;
      __syncwarp();
      if (thread_lane == 0)
      {
        tmp_size[warp_lane] = count2 + 2;
        tmp_list[count2 + 0] = g_v1;
        tmp_list[count2 + 1] = g_v2;
      }
      __syncwarp();
      comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], matched_list, matched_size);

      // for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      // {
      //   auto g_v3 = tmp_list[i];
      //   if (g_v3 >= g_v1)
      //     continue;

      //   if (!g.is_local(g_v3))
      //     continue;
      //   auto v3_size = g.fetch_local_degree(g_v3);
      //   vidType *v3_ptr = g.fetch_local_neigbours(g_v3);

      //   count += intersect_num(matched_list, matched_size, v3_ptr, v3_size, g_v2);
      // }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P8_extend_step2(BufferBase buffer, int gpu_id, AccType *total, vidType max_deg, NvsGraphGPU g)
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
  comm_fetch_SP_workload_number(buffer, ntasks, gpu_id);

  for (eidType eid = warp_id; eid < ntasks; eid += num_warps)
  {
    vidType *matched_list, *candidate_list;
    int matched_size = 0, candidate_size = 0;
    comm_pull_SP_by_warp(buffer, &matched_list, matched_size, &candidate_list, candidate_size, eid, gpu_id);
    auto g_v1 = matched_list[matched_size - 2];
    auto g_v2 = matched_list[matched_size - 1];
    matched_size = matched_size - 2;

    vidType *v0_ptr = candidate_list;
    auto v0_size = candidate_size;

    for (vidType i = 0; i < matched_size; i++)
    {
      auto g_v3 = matched_list[i];
      if (g_v3 >= g_v1)
        continue;

      if (!g.is_local(g_v3))
        continue;
      auto v3_size = g.fetch_local_degree(g_v3);
      vidType *v3_ptr = g.fetch_local_neigbours(g_v3);

      count += intersect_num(candidate_list, candidate_size, v3_ptr, v3_size, g_v2);
    }

    // vidType *v2_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg];
    //  for (vidType idx2 = 0; idx2 < matched_size; idx2++)
    //  {
    //    vidType g_v2 = matched_list[idx2];
    //    if (g_v2 >= g_v1)
    //      continue;
    //    int v2_size = 0;
    //    // comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
    //    if (!g.is_local(g_v2))
    //        continue;
    //      auto v2_size = g.fetch_local_degree(g_v2);
    //      vidType *v2_ptr = g.fetch_local_neigbours(g_v2);
    //    auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_list);
    //    if (thread_lane == 0)
    //      tmp_size[warp_lane] = count2;
    //    __syncwarp();
    //    if (thread_lane == 0)
    //    {
    //      tmp_size[warp_lane] = count1 + 2;
    //      tmp_list[count1 + 0] = g_v1;
    //      tmp_list[count1 + 1] = g_v2;
    //    }
    //    __syncwarp();
    //    comm_push_SP_workload_by_warp(buffer, g, max_deg, tmp_list, tmp_size[warp_lane], v0_ptr, 0);

    // }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}
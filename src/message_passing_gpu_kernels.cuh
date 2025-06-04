__global__ void count_freq_p1(BufferBase buffer, eidType estart,
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
    if (g_v1 >= g_v0)
    {
      continue;
    }

    atomicAdd(&g.d_g_v1_frequency[g_v1], 1);

    // // OPT1:
    // for (int v2_idx = 0; v2_idx < v0_size; v2_idx++)
    // {
    //   auto g_v2 = v0_ptr[v2_idx];
    //   if (g_v2 >= g_v1)
    //     break;
    //   atomicAdd(&g.d_g_v1_frequency[g_v2], 1);
    // }
  }
}

__global__ void count_freq_p3(BufferBase buffer, eidType estart,
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
    v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    atomicAdd(&g.d_g_v1_frequency[g_v1], 1);
  }
}

__global__ void P0_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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
    comm_pull_edge_neighbours_by_warp_merge(buffer, g, max_deg,
                                            g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    count += intersect_num(v0_ptr, v0_size, v1_ptr, v1_size);
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P1_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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
  unsigned long long ts, te;
  ts = clock();

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    if (g_v1 >= g_v0)
    {
      continue;
    }

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    int v1_size = 0;
    int v2_size = 0;

    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v2_ptr = &v1_ptr[max_deg];

    t_start = clock();
    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    t_end = clock();
    total_comm += (t_end - t_start);

    for (int v2_idx = 0; v2_idx < v0_size; v2_idx++)
    {
      auto g_v2 = v0_ptr[v2_idx];
      if (g_v2 >= g_v1)
        break;

      t_start = clock();
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      t_end = clock();
      total_comm += (t_end - t_start);

      int *left_operand = v1_ptr;
      int *right_operand = v2_ptr;
      int left_size = v1_size;
      int right_size = v2_size;
      t_start = clock();
      count += intersect_num(left_operand, left_size, right_operand, right_size, g_v0);
      t_end = clock();
      total_comp += (t_end - t_start);
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);

  te = clock();
  // auto total_time = (te - ts);
  // if (thread_id == 0 && g.GPUId() == 0)
  // {
  //   printf("Total time:%.3f  total_comm:%.3f  total_comp:%.3f\n", total_time / PEAK_CLK, total_comm / PEAK_CLK, total_comp / PEAK_CLK);
  // }
}

__global__ void P2_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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
  unsigned long long ts, te;
  ts = clock();
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
    if (g_v1 >= g_v0)
    {
      continue;
    }

    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * (max_deg + max_deg);

    t_start = clock();
    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp_merge(buffer, g, max_deg,
                                            g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    t_end = clock();
    total_comm += (t_end - t_start);

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
  te = clock();
  auto total_time = (te - ts);
}

__global__ void P3_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
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

  // if(thread_lane==0) printf("warp_id:%d\n",warp_id);
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

    vidType *v3_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v0_ptr, *v1_ptr;
    comm_pull_edge_neighbours_by_warp_merge(buffer, g, max_deg,
                                            g_v0, v0_size, &v0_ptr, g_v1, v1_size, &v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    // for (int idx3 = thread_lane; idx3 < v0_size; idx3 += 32)
    // {
    //   auto g_v3 = v0_ptr[idx3];
    //   if (g_v3 == g_v1)
    //     continue;
    //   if (!g.is_local(g_v3))
    //   {
    //     int v3_size = g.getOutDegree_remote(g_v3);
    //     atomicAdd(&g.comm_volumn[0], v3_size);
    //   }
    // }

    for (int idx3 = 0; idx3 < v0_size; idx3++)
    {
      auto g_v3 = v0_ptr[idx3];
      if (g_v3 == g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);

      for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      {
        auto g_v2 = tmp_list[i];
        if (g_v2 <= g_v1 || g_v2 == g_v3 || g_v2 == g_v0)
          continue;
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

__global__ void P4_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    if (g_v1 >= g_v0)
    {
      continue;
    }

    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *new_v1_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *new_v0_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx3 = 0; idx3 < v1_size; idx3++)
    {
      vidType g_v3 = v1_ptr[idx3];
      if (g_v3 == g_v0)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);
      auto count2 = intersect(v0_ptr, v0_size, v3_ptr, v3_size, new_v1_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();
      for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      {
        auto g_v2 = tmp_list[i];
        if (g_v3 == g_v2)
          continue;
        v3_size = 0;

        for (int idx_ = thread_lane; idx_ < tmp_size1[warp_lane]; idx_ += 32)
        {
          auto v_ = new_v1_ptr[idx_];
          if (v_ == g_v1 || v_ == g_v2)
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

__global__ void P5_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    if (g_v1 >= g_v0)
    {
      continue;
    }

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx3 = 0; idx3 < tmp_size[warp_lane]; idx3++)
    {
      vidType g_v2 = tmp_list[idx3];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, g_v2, tmp_v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();
      for (vidType i = 0; i < tmp_size[warp_lane]; i++)
      {
        auto g_v4 = tmp_list[i];
        if (g_v2 == g_v4)
          continue;
        v3_size = 0;
        for (int idx_ = thread_lane; idx_ < tmp_size1[warp_lane]; idx_ += 32)
        {
          auto v_ = tmp_v3_ptr[idx_];
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

__global__ void P6_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    if (g_v1 >= g_v0)
    {
      continue;
    }

    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);

      for (vidType idx3 = 0; idx3 < idx2; idx3++)
      {
        vidType g_v3 = tmp_list[idx3];
        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);
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

__global__ void P7_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v4_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      if (g_v2 <= g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_v4_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size1[warp_lane]; i++)
      {
        auto g_v4 = tmp_v4_ptr[i];
        if (g_v1 == g_v4)
          continue;
        v3_size = 0;
        for (int idx_ = thread_lane; idx_ < tmp_size[warp_lane]; idx_ += 32)
        {
          auto v_ = tmp_list[idx_];
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

__global__ void P8_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      if (g_v2 >= g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size1[warp_lane]; i++)
      {
        auto g_v3 = tmp_v3_ptr[i];
        if (g_v3 >= g_v1)
          continue;

        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);
        count += intersect_num(tmp_list, tmp_size[warp_lane], v3_ptr, v3_size, g_v2);
        //   if (thread_lane == 0)
        //     tmp_size1[warp_lane] = count2;
        //  __syncwarp();

        //   for(int idx_ = thread_lane; idx_<tmp_size[warp_lane]; idx_ +=32){
        //     auto v_ = tmp_list[idx_];
        //     if(v_ == g_v4 || v_ == g_v2) continue;
        //     else count+=1;
        //   }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P9_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      if (g_v2 <= g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, tmp_v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size1[warp_lane]; i++)
      {
        auto g_v3 = tmp_v3_ptr[i];
        AccType count3 = count_smaller(g_v3, tmp_v3_ptr, tmp_size1[warp_lane]);
        if (thread_lane == 0)
          count += count3;
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P10_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    vidType *v5_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, tmp_v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType i = 0; i < tmp_size1[warp_lane]; i++)
      {
        auto g_v3 = tmp_v3_ptr[i];
        if (g_v3 <= g_v1)
          continue;
        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v3_ptr);
        auto count2 = intersect(v0_ptr, v0_size, v3_ptr, v3_size, v5_ptr);
        if (thread_lane == 0)
          tmp_size2[warp_lane] = count2;
        __syncwarp();

        for (int idx4 = 0; idx4 < tmp_size[warp_lane]; idx4++)
        {
          vidType g_v4 = tmp_list[idx4];
          if (g_v4 == g_v2 || g_v4 == g_v1 || g_v4 == g_v3)
            continue;
          for (int idx_ = thread_lane; idx_ < tmp_size2[warp_lane]; idx_ += 32)
          {
            auto v_ = v5_ptr[idx_];
            if (v_ == g_v4 || v_ == g_v1 || v_ == g_v2 || v_ == g_v3)
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

__global__ void P11_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    vidType *v5_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      if (g_v2 <= g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);

      auto count2 = intersect(v0_ptr, v0_size, v2_ptr, v2_size, v4_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      auto count3 = intersect(v1_ptr, v1_size, v2_ptr, v2_size, v5_ptr);
      if (thread_lane == 0)
        tmp_size2[warp_lane] = count3;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size[warp_lane]; idx3++)
      {
        vidType g_v3 = tmp_list[idx3];
        if (g_v3 == g_v2 || g_v3 == g_v1 || g_v3 == g_v0)
          continue;

        for (vidType idx4 = 0; idx4 < tmp_size1[warp_lane]; idx4++)
        {
          vidType g_v4 = v4_ptr[idx4];
          if (g_v4 == g_v3 || g_v4 == g_v2 || g_v4 == g_v1)
            continue;
          for (int idx_ = thread_lane; idx_ < tmp_size2[warp_lane]; idx_ += 32)
          {
            auto v_ = v5_ptr[idx_];
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

__global__ void P12_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *tmp_v3_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, tmp_v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size1[warp_lane]; idx3++)
      {
        auto g_v3 = tmp_v3_ptr[idx3];
        for (vidType idx4 = idx3 + 1; idx4 < tmp_size1[warp_lane]; idx4++)
        {
          auto g_v4 = tmp_v3_ptr[idx4];
          for (int idx_ = thread_lane; idx_ < tmp_size[warp_lane]; idx_ += 32)
          {
            auto v_ = tmp_list[idx_];
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

__global__ void vsgm_last_pattern_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);

    if (g_v1 <= g_v0)
    {
      continue;
    }
    __syncwarp();

    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    vidType *v5_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < v0_size; idx2++)
    {
      vidType g_v2 = v0_ptr[idx2];
      if (g_v2 == g_v1)
        continue;
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);

      auto count3 = intersect(v1_ptr, v1_size, v2_ptr, v2_size, v5_ptr);
      if (thread_lane == 0)
        tmp_size2[warp_lane] = count3;
      __syncwarp();

      for (int idx5 = 0; idx5 < tmp_size2[warp_lane]; idx5++)
      {
        auto g_v5 = v5_ptr[idx5];
        if (g_v5 == g_v0)
          continue;
        for (vidType idx3 = 0; idx3 < tmp_size[warp_lane]; idx3++)
        {
          vidType g_v3 = tmp_list[idx3];
          if (g_v3 == g_v2 || g_v3 == g_v1 || g_v3 == g_v0 || g_v3 == g_v5)
            continue;
          for (int idx_ = thread_lane; idx_ < tmp_size[warp_lane]; idx_ += 32)
          {
            auto v_ = tmp_list[idx_];
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

__global__ void P13_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  unsigned long long t_start, t_end;
  unsigned long long total_comm = 0, total_comp = 0;
  unsigned long long ts, te;
  ts = clock();
  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    t_start = clock();
    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    t_end = clock();
    total_comm += (t_end - t_start);

    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);

    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      t_start = clock();
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      t_end = clock();
      total_comm += (t_end - t_start);
      count += intersect_num(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size);
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
  te = clock();
  auto total_time = (te - ts);
  // if (thread_id == 0 && g.GPUId() == 0)
  // {
  //   auto t_ = total_time / PEAK_CLK;
  //   auto t1_ = total_comm / PEAK_CLK;
  //   printf("Total time:%.3f  total_comm:%.3f  total_comp:%.3f    percent_: %.2f  \n", total_time / PEAK_CLK, total_comm / PEAK_CLK, (total_time - total_comm) / PEAK_CLK, t1_ / t_);
  // }
}

__global__ void P14_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size1[warp_lane]; idx3++)
      {
        vidType g_v3 = v3_ptr[idx3];
        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v4_ptr);
        count += intersect_num(v3_ptr, tmp_size1[warp_lane], v4_ptr, v3_size);
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P15_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType v4_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    vidType *v5_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v6_ptr = &v5_ptr[max_deg];

    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size1[warp_lane]; idx3++)
      {
        vidType g_v3 = v3_ptr[idx3];
        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v4_ptr);
        auto count3 = intersect(v3_ptr, tmp_size1[warp_lane], v4_ptr, v3_size, v5_ptr);
        if (thread_lane == 0)
          tmp_size2[warp_lane] = count3;
        __syncwarp();

        for (vidType idx4 = 0; idx4 < tmp_size2[warp_lane]; idx4++)
        {
          vidType g_v4 = v5_ptr[idx4];
          comm_pull_neighbours_by_warp(buffer, g, g_v4, v4_size, v6_ptr);
          count += intersect_num(v5_ptr, tmp_size2[warp_lane], v6_ptr, v4_size);
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

__global__ void P16_kernel(BufferBase buffer, eidType estart, eidType ne, eidType true_ne, vidType max_deg, NvsGraphGPU g, AccType *total)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_lane = threadIdx.x / WARP_SIZE;

  vidType *tmp_list = &buffer.tmp_buffer[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ vidType tmp_size[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size1[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size2[WARPS_PER_BLOCK];
  __shared__ vidType tmp_size3[WARPS_PER_BLOCK];

  AccType count = 0;
  vidType ancestors[2];

  for (eidType eid = warp_id + estart; eid < estart + ne; eid += num_warps)
  {
    if (eid >= true_ne)
    {
      continue;
    }
    auto g_v0 = g.get_src(eid);
    auto g_v1 = g.get_dst(eid);
    int chunk_idx = warp_id;
    int mem_idx = warp_id * max_deg;

    auto v0_size = g.fetch_local_degree(g_v0);
    vidType *v0_ptr = g.fetch_local_neigbours(g_v0);

    vidType v1_size = 0;
    vidType v2_size = 0;
    vidType v3_size = 0;
    vidType v4_size = 0;
    vidType *v1_ptr = &buffer.comp_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];

    vidType *v2_ptr = &v1_ptr[max_deg];
    vidType *v3_ptr = &buffer.read_buffer.get_buffer_list_ptr()[warp_id * max_deg * 2];
    vidType *v4_ptr = &v3_ptr[max_deg];

    vidType *v5_ptr = &buffer.write_buffer.get_buffer_list_ptr()[warp_id * max_deg * 4];
    vidType *v6_ptr = &v5_ptr[max_deg];
    vidType *v7_ptr = &v5_ptr[max_deg * 2];
    vidType *v8_ptr = &v5_ptr[max_deg * 3];
    comm_pull_neighbours_by_warp(buffer, g, g_v1, v1_size, v1_ptr);
    // auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, g_v1, tmp_list);
    auto count1 = intersect(v0_ptr, v0_size, v1_ptr, v1_size, tmp_list);
    if (thread_lane == 0)
      tmp_size[warp_lane] = count1;
    __syncwarp();

    for (vidType idx2 = 0; idx2 < tmp_size[warp_lane]; idx2++)
    {
      vidType g_v2 = tmp_list[idx2];
      comm_pull_neighbours_by_warp(buffer, g, g_v2, v2_size, v2_ptr);
      auto count2 = intersect(tmp_list, tmp_size[warp_lane], v2_ptr, v2_size, v3_ptr);
      if (thread_lane == 0)
        tmp_size1[warp_lane] = count2;
      __syncwarp();

      for (vidType idx3 = 0; idx3 < tmp_size1[warp_lane]; idx3++)
      {
        vidType g_v3 = v3_ptr[idx3];
        comm_pull_neighbours_by_warp(buffer, g, g_v3, v3_size, v4_ptr);
        auto count3 = intersect(v3_ptr, tmp_size1[warp_lane], v4_ptr, v3_size, v5_ptr);
        if (thread_lane == 0)
          tmp_size2[warp_lane] = count3;
        __syncwarp();

        for (vidType idx4 = 0; idx4 < tmp_size2[warp_lane]; idx4++)
        {
          vidType g_v4 = v5_ptr[idx4];
          comm_pull_neighbours_by_warp(buffer, g, g_v4, v4_size, v6_ptr);
          auto count4 = intersect(v5_ptr, tmp_size2[warp_lane], v6_ptr, v4_size, v7_ptr);
          if (thread_lane == 0)
            tmp_size3[warp_lane] = count4;
          __syncwarp();

          for (vidType idx5 = 0; idx5 < tmp_size3[warp_lane]; idx5++)
          {
            vidType g_v5 = v7_ptr[idx5];
            int v5_size = 0;
            comm_pull_neighbours_by_warp(buffer, g, g_v5, v5_size, v8_ptr);
            count += intersect_num(v7_ptr, tmp_size3[warp_lane], v8_ptr, v5_size);
          }
        }
      }
    }
  }

  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&total[0], block_num);
}

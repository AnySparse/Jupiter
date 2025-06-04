// Copyright (c) 2023 ICT
// Author: Zhiheng Lin
#include <cub/cub.cuh>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "graph_gpu_nvs.h"
#include "graph_gpu.h"
#include "graph_partition.h"
#include "scheduler.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "scan.h"
#include "utils.h"
#include "VertexSet.h"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#include "context_manager_engine.cuh"
#include "gpu_kernels_codegen.cuh"

#include "message_passing_gpu_kernels.cuh"
#include "delegation_gpu_kernels.cuh"

void SubgraphSolver(Graph &g, uint64_t &total, int argc, char *argv[], MPI_Comm &mpi_comm, int rank, int nranks)
{
  Timer t;
  // Step1: Initialize mpi comm
#if 1
  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));

  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  // create NVSHMEM common world.
  cudaStream_t stream;
  nvshmemx_init_attr_t attr;

  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  // <- end initialization.

  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("gpu_id:%d  n_gpus:%d ndevices:%d  mype_node:%d\n", rank, nranks, ndevices, mype_node);

  int n_gpus = nranks;
  int gpu_id = rank;
  int local_gpu_id = mype_node;

  cudaSetDevice(local_gpu_id);
  cudaStreamCreate(&stream);

  int mype, npes;
  mype = nvshmem_my_pe();
  npes = nvshmem_n_pes();
#endif
  vidType *part_id_list, *local_id_list;
  // Step2: Graph 1D partition
// #define SUBGRAPH_LOADING
#ifdef SUBGRAPH_LOADING

  int subgraph_vsize = nv;
  int v_size = subgraph_vsize;

  Graph *subgraph = &g;
  eidType e_size = ne;
  eidType subgraph_esize = 0;
  subgraph_vsize += 1;
  MPI_Allreduce(&e_size, &subgraph_esize, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  vidType *subgraph_src_list = (vidType *)malloc(sizeof(vidType) * e_size);
#pragma omp parallel for
  for (int v = 0; v < nv; v++)
  {
    auto begin = subgraph->edge_begin(v);
    // TODO: error here
    int gv = (v * ndevices + gpu_id);
    int vdeg = subgraph->get_degree(v);
    for (int xx = 0; xx < vdeg; xx++)
    {
      subgraph_src_list[begin + xx] = gv;
    }
  }
  t.Stop();
  subgraph->set_max_degree(md);
  std::cout << "Construct Subgraph Time: " << t.Seconds() << " sec \n";
#else

#ifdef HASH_1D_PARTITION
  t.Start();
  // g.sort_graph();
  int subgraph_vsize = nv / n_gpus;
  int left_size = nv % n_gpus;
  auto v_size = subgraph_vsize;
  if (gpu_id < left_size)
  {
    v_size++;
  }
  // Calculate new subgraph meta infomation(NE, NV...)
  std::vector<vidType> degrees(v_size, 0);
  vidType *degree_list = custom_alloc_global<vidType>(nv);
#pragma omp parallel for
  for (vidType v = gpu_id; v < nv; v += n_gpus)
  {
    vidType vid = v / n_gpus;
    degrees[vid] = g.get_degree(v);
  }
#pragma omp parallel for
  for (vidType v = 0; v < nv; v += 1)
  {
    degree_list[v] = g.get_degree(v);
  }

  eidType *offsets = custom_alloc_global<eidType>(v_size + 1);
  parallel_prefix_sum<vidType, eidType>(degrees, offsets);
  eidType e_size = offsets[v_size];
  std::cout << " |V| = " << v_size << " |E| = " << e_size << "\n";

  Graph *subgraph = new Graph();
  subgraph->allocateFrom(v_size, e_size);

  eidType subgraph_esize = 0;
  subgraph_vsize += 1;
  MPI_Allreduce(&e_size, &subgraph_esize, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  printf("Process %d: Max edge value is %ld\n", rank, subgraph_esize);

  vidType *subgraph_src_list = (vidType *)malloc(sizeof(vidType) * e_size);

// Construct new subgraph.
#pragma omp parallel for
  for (vidType v = gpu_id; v < nv; v += n_gpus)
  {
    vidType vid = v / n_gpus;
    auto begin = offsets[vid];
    auto end = offsets[vid + 1];
    subgraph->fixEndEdge(vid, end);
    vidType j = 0;
    for (auto u : g.N(v))
    {
      subgraph->constructEdge(begin + j, u);
      subgraph_src_list[begin + j] = v;
      j++;
    }
  }
  t.Stop();
  subgraph->set_max_degree(md);
  std::cout << "Construct Subgraph Time: " << t.Seconds() << " sec \n";
#endif // end of HASH_1D_PARTITION
#ifdef LOCALITY_1D_PARTITION
  t.Start();
  int subgraph_vsize = (nv + n_gpus - 1) / n_gpus;
  auto v_size = subgraph_vsize;
  if (gpu_id == n_gpus - 1)
  {
    v_size = nv - gpu_id * subgraph_vsize;
  }
  std::vector<vidType> degrees(v_size, 0);

  auto v_start = gpu_id * subgraph_vsize;
  auto v_end = v_start + v_size;
#pragma omp parallel for
  for (vidType v = v_start; v < v_end; v += 1)
  {
    vidType vid = v - v_start;
    degrees[vid] = g.get_degree(v);
  }
  vidType *degree_list = custom_alloc_global<vidType>(nv);
#pragma omp parallel for
  for (vidType v = 0; v < nv; v += 1)
  {
    degree_list[v] = g.get_degree(v);
  }

  eidType *offsets = custom_alloc_global<eidType>(v_size + 1);
  parallel_prefix_sum<vidType, eidType>(degrees, offsets);
  eidType e_size = offsets[v_size];
  std::cout << " |V| = " << v_size << " |E| = " << e_size << "\n";

  Graph *subgraph = new Graph();
  subgraph->allocateFrom(v_size, e_size);

  eidType subgraph_esize = 0;
  printf("=========subgraph vsize:%d  v_size:%d\n", subgraph_vsize, v_size);
  // subgraph_vsize += 1;
  MPI_Allreduce(&e_size, &subgraph_esize, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  printf("Process %d: Max edge value is %ld\n", rank, subgraph_esize);

  // new add.........
  vidType *subgraph_src_list = (vidType *)malloc(sizeof(vidType) * e_size);

#pragma omp parallel for
  for (vidType v = v_start; v < v_end; v += 1)
  {
    vidType vid = v - v_start;
    auto begin = offsets[vid];
    auto end = offsets[vid + 1];
    subgraph->fixEndEdge(vid, end);
    vidType j = 0;
    for (auto u : g.N(v))
    {
      subgraph->constructEdge(begin + j, u);
      subgraph_src_list[begin + j] = v;
      j++;
    }
  }
  t.Stop();
  subgraph->set_max_degree(md);
  std::cout << "Construct Subgraph Time: " << t.Seconds() << " sec \n";
#endif
#ifdef METIS_1D_PARTITION
  std::string prefix = argv[1];
  std::ifstream f_parts((prefix + ".parts").c_str());
  assert(f_parts);
  int idx = 0;
  // std::vector<vidType> part_id_list(nv, 0);
  // std::vector<vidType> local_id_list(nv, 0);
  part_id_list = custom_alloc_global<vidType>(nv);
  local_id_list = custom_alloc_global<vidType>(nv);
  std::vector<vidType> part_size(n_gpus, 0);
  while (idx < nv)
  {
    f_parts >> part_id_list[idx];
    part_size[part_id_list[idx]]++;
    idx++;
  }

  t.Start();
  int v_size = part_size[gpu_id];
  std::vector<vidType> degrees(v_size, 0);
  vidType *degree_list = custom_alloc_global<vidType>(nv);
  int vid = 0;

  std::vector<vidType> cnt_index(n_gpus, 0);
  for (vidType v = 0; v < nv; v += 1)
  {
    int dst_gpu = part_id_list[v];

    local_id_list[v] = cnt_index[dst_gpu];
    cnt_index[dst_gpu]++;

    if (dst_gpu != gpu_id)
      continue;
    degrees[vid] = g.get_degree(v);
    // local_id_list[v] = vid;
    vid++;
  }
#pragma omp parallel for
  for (vidType v = 0; v < nv; v += 1)
  {
    degree_list[v] = g.get_degree(v);
  }
  eidType *offsets = custom_alloc_global<eidType>(v_size + 1);
  parallel_prefix_sum<vidType, eidType>(degrees, offsets);
  eidType e_size = offsets[v_size];
  std::cout << " |V| = " << v_size << " |E| = " << e_size << "\n";
  Graph *subgraph = new Graph();
  subgraph->allocateFrom(v_size, e_size);

  eidType subgraph_esize = 0;
  vidType subgraph_vsize = 0;
  MPI_Allreduce(&e_size, &subgraph_esize, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&v_size, &subgraph_vsize, 1, MPI_INT32_T, MPI_MAX, MPI_COMM_WORLD);

  printf("Process %d: Max edge value is %ld, vertex value:%d\n", rank, subgraph_esize, subgraph_vsize);

  vidType *subgraph_src_list = (vidType *)malloc(sizeof(vidType) * e_size);
  // Construct new subgraph.
#pragma omp parallel for
  for (vidType v = 0; v < nv; v += 1)
  {
    // vidType vid = v / n_gpus;
    int dst_gpu = part_id_list[v];
    int vid_ = local_id_list[v];
    if (dst_gpu != gpu_id)
      continue;

    auto begin = offsets[vid_];
    auto end = offsets[vid_ + 1];
    subgraph->fixEndEdge(vid_, end);
    vidType j = 0;
    for (auto u : g.N(v))
    {
      subgraph->constructEdge(begin + j, u);
      subgraph_src_list[begin + j] = v;
      j++;
    }
    // vid++;
  }
  t.Stop();
  subgraph->set_max_degree(md);
  std::cout << "Construct Subgraph Time: " << t.Seconds() << " sec \n";

#endif // end of METIS_1D_PARTITION
#endif

  // Dump sorted graph .......
  // std::string filename(argv[1]);
  // subgraph->dump_binary(filename + std::to_string(rank));

  // MPI_Barrier(MPI_COMM_WORLD);
  // nvshmem_finalize();
  // MPI_Finalize();
  // exit(0);

  // Step3: GPU graph initializing
#if 1
  t.Start();
  CUDA_SAFE_CALL(cudaSetDevice(local_gpu_id));
  NvsGraphGPU d_graph;
#ifdef USE_COMP || USE_FUSE
  d_graph.init(*subgraph, gpu_id, n_gpus, subgraph_vsize, subgraph_esize, degree_list, part_id_list, local_id_list, nv);
#else
  d_graph.init(*subgraph, gpu_id, n_gpus, subgraph_vsize, subgraph_esize);
#endif

  d_graph.copy_edgelist_to_device(subgraph_src_list);

  t.Stop();
  std::cout << "Total GPU copy time (graph+edgelist) = " << t.Seconds() << " sec\n";
#endif

  // Step4: Kernel configurations
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = NUM_BLOCK;
  int result_num = 6;
  int k = 1;
  AccType *d_count;
  CUDA_SAFE_CALL(cudaMalloc(&d_count, sizeof(AccType) * result_num));
  AccType *h_total = (AccType *)malloc(sizeof(AccType) * result_num);

  size_t nwarps = WARPS_PER_BLOCK;
  md += 10;
  size_t per_block_vlist_size = nwarps * size_t(k) * size_t(md) * sizeof(vidType);
  size_t list_size = nblocks * per_block_vlist_size;

  // Step5: Memory buffer initializing
  PatternConfig pattern_config; // 5-clique for test

  if (gpu_id == 0)
    check_memory("After graph construcion ");

#ifdef MERGE_NEIGHBORS_COMM
  d_graph.init_merge_buffer(nv);
#endif

  BufferBase buffer;
  buffer.init(pattern_config, list_size, gpu_id);

  if (gpu_id == 0)
    check_memory("After buffer construction ");
  MPI_Barrier(MPI_COMM_WORLD);

  // Step6: Launch kernel
  int total_round = 1;
  std::string pattern_name(argv[2]);
  if (argc == 4)
    total_round = atoi(argv[3]);

  int round = 0;
  eidType enumber = subgraph_esize / total_round;
  eidType estart = 0;

  vidType max_index = 0;
  vidType current_index = 0;
  vidType buffer_num = 0;
  vidType max_buffer_num = 0;

  uint64_t total_count = 0;
  double total_time = 0;

  for (eidType estart = 0; estart < subgraph_esize; estart += enumber)
  {
    eidType ecurent = (estart + enumber) >= subgraph_esize ? (subgraph_esize - estart) : enumber;
    double producer_time = 0;
    double consumer_time = 0;
    double comm_time = 0;
    size_t comm_volume = 0;

#ifdef MERGE_NEIGHBORS_COMM
    d_graph.clear_merge_buffer();
    count_freq_p1<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    // count_freq_p3<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);

    count_offset<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    write_result<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif

#ifdef MESSAGE_PASSING
    t.Start();
    if (pattern_name == "P1")
      P1_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P2")
      P2_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P3")
      P3_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P4")
      P4_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P5")
      P5_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P6")
      P6_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P7")
      P7_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P8")
      P8_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P9")
      P9_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P10")
      P10_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P11")
      P11_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P12")
      P12_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "vsgm_last_pattern")
      vsgm_last_pattern_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P13")
      P13_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P14")
      P14_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P15")
      P15_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else if (pattern_name == "P16")
      P16_kernel<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
    else
      assert(0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    t.Stop();
    producer_time = t.Seconds();
#endif
#ifdef DELEGATION
    if (pattern_name == "P1")
    {
      t.Start();
      P1_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        // printf("completed single transfer\n");
        P1_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P2")
    {
      t.Start();
      P2_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      // buffer.next_iteration();
      // MPI_Barrier(MPI_COMM_WORLD);
      // consumer_time = t.Seconds();
    }
    else if (pattern_name == "P3")
    {
      t.Start();
      P3_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P3_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P4")
    {
      t.Start();
      P4_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P4_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P5")
    {
      t.Start();
      P5_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P5_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P6")
    {
      t.Start();
      P6_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P6_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P7")
    {
      t.Start();
      P7_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P7_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P8")
    {
      t.Start();
      P8_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);

      Timer t1;
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD

        t1.Start();
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        t1.Stop();

        comm_time += t1.Seconds();
#endif
        P8_extend_step1<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);

      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD

        t1.Start();
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        t1.Stop();

        comm_time += t1.Seconds();
#endif
        P8_extend_step2<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);

      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P9")
    {
      t.Start();
      P9_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P9_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P10")
    {
      t.Start();
      P10_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P10_extend_step1<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();

      // update level:2->3
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P10_extend_step2<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time += t.Seconds();
    }
    else if (pattern_name == "P11")
    {
      t.Start();
      P11_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P11_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P12")
    {
      t.Start();
      P12_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        P12_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "vsgm_last_pattern")
    {
      t.Start();
      vsgm_last_pattern_kernel_producer<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();
      MPI_Barrier(MPI_COMM_WORLD);
      t.Start();
      for (int gid = 0; gid < n_gpus; gid++)
      {
#ifdef BATCH_LOAD
        fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
#endif
        vsgm_last_pattern_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t.Stop();
      consumer_time = t.Seconds();
    }
    else if (pattern_name == "P13" || pattern_name == "P14" || pattern_name == "P15" || pattern_name == "P16")
    {
      int k_ = 4;
      if (pattern_name == "P13")
        k_ = 4;
      else if (pattern_name == "P14")
        k_ = 5;
      else if (pattern_name == "P15")
        k_ = 6;
      else if (pattern_name == "P16")
        k_ = 7;
      PatternConfig config_;
      buffer.reload_pattern_config(config_, k_);
      // Level 0: edge-centric
      t.Start();
      pattern_task_generate<<<nblocks, nthreads>>>(buffer, estart, ecurent, e_size, md, d_graph, d_count);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      t.Stop();

      producer_time = t.Seconds();
      MPI_Barrier(MPI_COMM_WORLD);
      // update level:0->2
      buffer.next_iteration();

      //=============Sync mode for each level=========
      //=============Fetch tasks(S,P) to local========
      // read_buffer->comp_buffer->write_buffer

      while (!buffer.is_done())
      {
        double level_time = 0;
        double level_comm = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        t.Start();
#ifdef BASE_COMM
        for (int gid = 0; gid < n_gpus; gid++)
        {
          pattern_extend<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
          CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }
        MPI_Barrier(MPI_COMM_WORLD);

#else
        Timer t1;
        for (int gid_ = 0; gid_ < n_gpus; gid_++)
        {
          int gid = (gid_ + gpu_id) % n_gpus;

#ifdef BATCH_LOAD

          t1.Start();
          fetch_all_workload<<<nblocks, nthreads>>>(buffer, gid);
          CUDA_SAFE_CALL(cudaDeviceSynchronize());
          t1.Stop();
          level_comm += t1.Seconds();

          uint64_t task_mem = buffer.comp_buffer.get_memory_size_host();
          uint64_t task_num = buffer.comp_buffer.get_buffer_total_num();
          comm_volume += task_mem;
          comm_volume += task_num;

#endif
          pattern_extend_pull_workloads<<<nblocks, nthreads>>>(buffer, gid, d_count, md, d_graph);
          CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        buffer.next_iteration();
        t.Stop();

        level_time = t.Seconds();
        comm_time += level_comm;
        consumer_time += level_time;
      }
    }
    else
      assert(0);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    buffer.clean_buffer();
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = producer_time + consumer_time;
    total_time += tt;
    CUDA_SAFE_CALL(cudaMemcpy(h_total, d_count, sizeof(AccType) * result_num, cudaMemcpyDeviceToHost));
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&h_total[0], &total_count, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    uint64_t total_count4 = 0;
    MPI_Reduce(&h_total[1], &total_count4, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (gpu_id == 0)
    // {
    //   printf("GPUID:%d Round:%d Now process range[%ld,%ld], size=%ld, time cost:%.3fsec(first_level:%.3fsec, other_levels:%.3fsec, comm_time:%.3fsec) count:%ld \n", gpu_id, round, estart, estart + ecurent, ecurent, tt, producer_time, consumer_time, comm_time, total_count);
    // }

    round++;
  }
  // Step7: Collect results.
  CUDA_SAFE_CALL(cudaMemcpy(h_total, d_count, sizeof(AccType) * result_num, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  std::cout << "GPU:" << gpu_id << " Kernel runtime = " << total_time << " sec << total_num_triangles = " << h_total[0] << " \n";

  MPI_Barrier(MPI_COMM_WORLD);
  float local_time = total_time;
  float global_time = 0;
  total_count = 0;
  MPI_Reduce(&h_total[0], &total_count, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &global_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    printf("=====================gpu count:%ld  runtime:%.6fs\n", total_count, global_time);
#ifdef PROFILING
    AccType comm1 = buffer.read_buffer.calculate_comm_volumn();
    AccType comm2 = buffer.write_buffer.calculate_comm_volumn();
    AccType comm3 = buffer.comp_buffer.calculate_comm_volumn();
    AccType comm4 = d_graph.calculate_comm_volumn();
    printf("=======================total communication volumns:%.3f GB (%.3fGB,  %3fGB,  %.3fGB,  %.3fGB)!!\n",
           (comm1 + comm2 + comm3 + comm4) * 4.0 / 1e9,
           comm1 * 4.0 / 1e9,
           comm2 * 4.0 / 1e9,
           comm3 * 4.0 / 1e9,
           comm4 * 4.0 / 1e9);
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // nvshmem_finalize();
  MPI_Finalize();
}

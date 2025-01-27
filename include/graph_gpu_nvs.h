#pragma once
#include "graph.h"
#include "operations.cuh"
#include "cutil_subset.h"
#include "common.h"

class NvsGraphGPU
{
protected:
  vidType num_vertices;             // number of vertices
  eidType num_edges;                // number of edges
  vidType subgraph_vsize;           // number of max vertices
  eidType subgraph_esize;           // number of max edges
  int device_id, n_gpu;             // no. of GPUs
  eidType *d_rowptr;                // row pointers of CSR format
  vidType *d_colidx;                // column induces of CSR format
  vidType *d_src_list, *d_dst_list; // for COO format

  eidType *recv_ebuffer; // multi-node nvshmem need the recvieved buffer to be register.
  vidType *recv_vbuffer; // multi-node nvshmem need the recvieved buffer to be register.
#ifdef USE_COMP || USE_FUSE
  vidType *graph_degrees;
  vidType graph_nv;

  vidType *d_part_ids;
  vidType *d_local_ids;
#endif
public:
  int *d_g_v1_frequency, *d_v1_offset, *d_global_index; // merge neibours comm.
  int total_nv;
  AccType *comm_volumn;

  vidType *freq_list;

  NvsGraphGPU() : device_id(0), n_gpu(1) {}
  NvsGraphGPU(Graph &g) : device_id(0), n_gpu(1) { init(g); }
  NvsGraphGPU(Graph &g, int n, int m) : device_id(n), n_gpu(m) { init(g); }
  inline __device__ __host__ vidType V() { return num_vertices; }
  inline __device__ __host__ vidType size() { return num_vertices; }
  inline __device__ __host__ eidType E() { return num_edges; }
  inline __device__ __host__ eidType sizeEdges() { return num_edges; }
  inline __device__ __host__ bool valid_vertex(vidType vertex) { return (vertex < num_vertices); }
  inline __device__ __host__ bool valid_edge(eidType edge) { return (edge < num_edges); }
  inline __device__ __host__ vidType get_src(eidType eid) const { return d_src_list[eid]; }
  inline __device__ __host__ vidType get_dst(eidType eid) const { return d_dst_list[eid]; }
  inline __device__ __host__ vidType *get_src_ptr(eidType eid) const { return d_src_list; }
  inline __device__ __host__ vidType *get_dst_ptr(eidType eid) const { return d_dst_list; }
  inline __device__ __host__ vidType *N(vidType vid) { return d_colidx + d_rowptr[vid]; }
  inline __device__ __host__ eidType *out_rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType *out_colidx() { return d_colidx; }
  inline __device__ __host__ eidType getOutDegree(vidType src) { return d_rowptr[src + 1] - d_rowptr[src]; }
  inline __device__ __host__ vidType get_degree(vidType src) { return vidType(d_rowptr[src + 1] - d_rowptr[src]); }
  inline __device__ __host__ vidType getDestination(vidType src, eidType edge) { return d_colidx[d_rowptr[src] + edge]; }
  inline __device__ __host__ vidType getAbsDestination(eidType abs_edge) { return d_colidx[abs_edge]; }
  inline __device__ __host__ vidType getEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ eidType edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType edge_end(vidType src) { return d_rowptr[src + 1]; }

  inline __device__ __host__ vidType GPUId() { return device_id; }
  inline __device__ __host__ vidType GPUNums() { return n_gpu; }
  inline __device__ __host__ vidType VPart() { return subgraph_vsize; }
  inline __device__ __host__ vidType EPart() { return subgraph_esize; }

// #define SORT_GRAPH
#ifdef HASH_1D_PARTITION
  inline __device__ __host__ vidType convert_to_local(vidType gv) { return gv / n_gpu; }
  inline __device__ __host__ vidType convert_to_global(vidType lv) { return lv * n_gpu + device_id; }
  inline __device__ __host__ vidType dst_GPUId(vidType gv) { return gv % n_gpu; }
  inline __device__ __host__ vidType is_local(vidType gv) { return dst_GPUId(gv) == device_id; }
#endif
#ifdef LOCALITY_1D_PARTITION
  inline __device__ __host__ vidType convert_to_local(vidType gv) { return gv % subgraph_vsize; }
  inline __device__ __host__ vidType convert_to_global(vidType lv) { return lv + device_id * subgraph_vsize; }
  inline __device__ __host__ vidType dst_GPUId(vidType gv) { return gv / subgraph_vsize; }
  inline __device__ __host__ vidType is_local(vidType gv) { return dst_GPUId(gv) == device_id; }
#endif

#ifdef METIS_1D_PARTITION
  inline __device__ __host__ vidType convert_to_local(vidType gv) { return d_local_ids[gv];}
  inline __device__ __host__ vidType convert_to_global(vidType lv) { return -1;} //TODO:
  inline __device__ __host__ vidType dst_GPUId(vidType gv) { return d_part_ids[gv];}
  inline __device__ __host__ vidType is_local(vidType gv) { return dst_GPUId(gv) == device_id; }
#endif

  inline __device__ __host__ vidType *fetch_local_neigbours(vidType vid)
  {
    #ifdef EXTRA_CHECK
    assert(is_local(vid) == 1);
    #endif
    int local_vid = convert_to_local(vid);
    eidType tmp_eid = d_rowptr[local_vid];
    return &d_colidx[tmp_eid];
  }
  inline __device__ __host__ vidType fetch_local_degree(vidType vid)
  {
    #ifdef EXTRA_CHECK
    assert(is_local(vid) == 1);
    #endif
    int local_vid = convert_to_local(vid);
    return d_rowptr[local_vid + 1] - d_rowptr[local_vid];
  }

  inline __device__ __host__ void fetch_remote_neigbours(vidType vid, vidType *adj_list, vidType adj_size)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int64_get(&recv_ebuffer[thread_id * SLOT_SIZE], &d_rowptr[remote_local_vid], 1, remote_gpu_id);
    nvshmem_int_get(adj_list, &d_colidx[recv_ebuffer[thread_id * SLOT_SIZE]], adj_size, remote_gpu_id);
  }

  inline __device__ vidType get_degree_remote(vidType src) {
    return vidType(getOutDegree_remote(src));
  }
  inline __device__ eidType getOutDegree_remote(vidType src)
  {

    int remote_gpu_id = dst_GPUId(src);
    int remote_local_vid = convert_to_local(src);

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int64_get(&recv_ebuffer[thread_id * SLOT_SIZE], &d_rowptr[remote_local_vid], 2, remote_gpu_id);

#ifdef PROFILING
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], 2);
      }
#endif
    auto test_size = recv_ebuffer[thread_id * SLOT_SIZE + 1] - recv_ebuffer[thread_id * SLOT_SIZE];
    // if(test_size!=graph_degrees[src]) printf("hahahahaah src:%d  true:%d false:%d\n",src, graph_degrees[src], test_size);
    return recv_ebuffer[thread_id * SLOT_SIZE + 1] - recv_ebuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ void fetch_remote_neigbours_warp(vidType vid, vidType *adj_list, vidType adj_size, eidType col_start)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);
    // nvshmem_int_get((vidType* )&adj_list, &d_colidx[col_start], adj_size, remote_gpu_id);
    nvshmemx_int_get_warp(adj_list, &d_colidx[col_start], adj_size, remote_gpu_id);

#ifdef PROFILING
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], adj_size);
      }
#endif
  }

  inline __device__ vidType fetch_remote_one_neigbour_thread(vidType vid, eidType col_start, int oft)
  {

    int remote_gpu_id = dst_GPUId(vid);

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &d_colidx[col_start + oft], 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ void fetch_remote_neigbours_thread(vidType vid, vidType *adj_list, vidType adj_size, eidType col_start)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);
    nvshmem_int_get(adj_list, &d_colidx[col_start], adj_size, remote_gpu_id);
  }
  inline __device__ void fetch_remote_neigbours_thread_async(vidType vid, vidType *adj_list, vidType adj_size, eidType col_start)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);
    nvshmem_int_get_nbi(adj_list, &d_colidx[col_start], adj_size, remote_gpu_id);
  }

  inline __device__ void fetch_remote_neigbours_warp_async(vidType vid, vidType *adj_list, vidType adj_size, eidType col_start)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);

    nvshmemx_int_get_nbi_warp(adj_list, &d_colidx[col_start], adj_size, remote_gpu_id);
  }

  inline __device__ eidType get_remote_col_start(vidType src)
  {
    int remote_gpu_id = dst_GPUId(src);
    int remote_local_vid = convert_to_local(src);

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int64_get(&recv_ebuffer[thread_id * SLOT_SIZE], &d_rowptr[remote_local_vid], 1, remote_gpu_id);

#ifdef PROFILING
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    if (thread_lane == 0)
      if (remote_gpu_id != device_id)
      {
        atomicAdd(&comm_volumn[0], 2);
      }
#endif

    return recv_ebuffer[thread_id * SLOT_SIZE];
  }
  //===============
  inline __device__ vidType fetch_remote_neigbour(vidType vid, vidType offset, eidType col_start)
  {

    int remote_gpu_id = dst_GPUId(vid);
    int remote_local_vid = convert_to_local(vid);

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &d_colidx[col_start + offset], 1, remote_gpu_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }

  inline __device__ vidType get_dst_remote(eidType eid)
  {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    nvshmem_int_get(&recv_vbuffer[thread_id * SLOT_SIZE], &d_colidx[eid], 1, device_id);
    return recv_vbuffer[thread_id * SLOT_SIZE];
  }
  //===============

  void clean()
  {
    CUDA_SAFE_CALL(cudaFree(d_rowptr));
    CUDA_SAFE_CALL(cudaFree(d_colidx));
  }
  void clean_edgelist()
  {
    CUDA_SAFE_CALL(cudaFree(d_src_list));
    CUDA_SAFE_CALL(cudaFree(d_dst_list));
  }
  #ifdef USE_COMP || USE_FUSE
  void init(Graph &g, int n, int m, vidType j, eidType k, vidType *degrees, vidType *part_ids, vidType *local_ids, int total_nv){
    graph_nv = total_nv;
    CUDA_SAFE_CALL(cudaMalloc((void **)&graph_degrees, total_nv * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(graph_degrees, degrees, total_nv * sizeof(vidType), cudaMemcpyHostToDevice));

  #ifdef METIS_1D_PARTITION
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_part_ids, total_nv * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_part_ids, part_ids, total_nv * sizeof(vidType), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_local_ids, total_nv * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_local_ids, local_ids, total_nv * sizeof(vidType), cudaMemcpyHostToDevice));
   #endif
    init(g,n,m,j,k);
  }
  #endif
  void init(Graph &g, int n, int m, vidType j, eidType k)
  {
    device_id = n;
    n_gpu = m;
    subgraph_vsize = j;
    subgraph_esize = k;
    init(g);
  }
  void init(Graph &hg)
  {
    auto m = hg.num_vertices();
    auto nnz = hg.num_edges();
    num_vertices = m;
    num_edges = nnz;
    auto h_rowptr = hg.out_rowptr();
    auto h_colidx = hg.out_colidx();
    // size_t mem_vert = size_t(m + 1) * sizeof(eidType);
    // size_t mem_edge = size_t(nnz) * sizeof(vidType);
    // size_t mem_graph = mem_vert + mem_edge;
    // size_t mem_el = mem_edge; // memory for the edgelist
    // size_t mem_all = mem_graph + mem_el;
    auto mem_gpu = get_gpu_mem_size();
    Timer t;

    std::cout << "GPU ID:   " << device_id << "  " << subgraph_vsize << "  " << subgraph_esize << " m:" << m << "   nnz:" << nnz << std::endl;
    // if(device_id==0)
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "GPU ID:   " << device_id << "  Vertex mem cost" << static_cast<size_t>(subgraph_vsize + 1) * sizeof(eidType) * 1.0 / 1e9 << " GB , Edge memory cost" << static_cast<size_t>(subgraph_esize + 1) * sizeof(vidType) * 1.0 / 1e9 << "GB" << std::endl;
    d_rowptr = (eidType *)nvshmem_malloc(static_cast<size_t>(subgraph_vsize + 1) * sizeof(eidType));
    MPI_Barrier(MPI_COMM_WORLD);

    d_colidx = (vidType *)nvshmem_malloc(static_cast<size_t>(subgraph_esize + 1) * sizeof(vidType));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, h_rowptr, (m + 1) * sizeof(eidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx, h_colidx, (nnz) * sizeof(vidType), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&freq_list, (n_gpu + 1) * m * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemset(freq_list, 0, sizeof(vidType) * (n_gpu + 1) * m));

    CUDA_SAFE_CALL(cudaMalloc((void **)&recv_ebuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&recv_vbuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(vidType)));
    nvshmemx_buffer_register(recv_ebuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(eidType));
    nvshmemx_buffer_register(recv_vbuffer, NUM_BLOCK * BLOCK_SIZE * SLOT_SIZE * sizeof(vidType));

    CUDA_SAFE_CALL(cudaMalloc((void **)&comm_volumn, sizeof(AccType)));
    CUDA_SAFE_CALL(cudaMemset(comm_volumn, 0, sizeof(AccType)));
  }

  // copy from Gminer
  void copy_edgelist_to_device(size_t nnz, Graph &hg, bool sym_break = false)
  {
    copy_edgelist_to_device(0, nnz, hg, sym_break);
  }
  void copy_edgelist_to_device(size_t begin, size_t end, Graph &hg, bool sym_break = false)
  {
    copy_edgelist_to_device(begin, end, hg.get_src_ptr(), hg.get_dst_ptr(), sym_break);
  }
  void copy_edgelist_to_device(size_t nnz, vidType *h_src_list, vidType *h_dst_list, bool sym_break)
  {
    copy_edgelist_to_device(0, nnz, h_src_list, h_dst_list, sym_break);
  }
  void copy_edgelist_to_device(size_t begin, size_t end, vidType *h_src_list, vidType *h_dst_list, bool sym_break)
  {
    auto n = end - begin;
    eidType n_tasks_per_gpu = eidType(n - 1) / eidType(n_gpu) + 1;
    eidType start = begin + device_id * n_tasks_per_gpu;
    if (!sym_break)
      d_dst_list = d_colidx + start;
    eidType num = n_tasks_per_gpu;
    if (start + num > end)
      num = end - start;
    // std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num
    //           << " [" << start << ", " << start+num << ")\n";
    // Timer t;
    // t.Start();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, h_src_list + start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    if (sym_break)
    {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, h_dst_list + start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // t.Stop();
    // std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void copy_edgelist_to_device(std::vector<eidType> lens, std::vector<vidType *> &srcs, std::vector<vidType *> &dsts)
  {
    Timer t;
    t.Start();
    vidType *src_ptr = srcs[device_id];
    vidType *dst_ptr = dsts[device_id];
    auto num = lens[device_id];
    // std::cout << "src_ptr = " << src_ptr << " dst_ptr = " << dst_ptr << "\n";
    // std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num << "\n";

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, src_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, dst_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();
    std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void copy_edgelist_to_device(vidType *src_ptr)
  {
    Timer t;
    t.Start();
    d_dst_list = d_colidx;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num_edges * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, src_ptr, num_edges * sizeof(vidType), cudaMemcpyHostToDevice));

    t.Stop();
    std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }

#ifdef MERGE_NEIGHBORS_COMM
  void init_merge_buffer(int total_nv_)
  {
    total_nv = total_nv_;
    cudaMalloc(&d_g_v1_frequency, total_nv * sizeof(int));
    cudaMalloc(&d_v1_offset, (total_nv + 1) * sizeof(int));
    cudaMalloc(&d_global_index, sizeof(int));
  }
  void clear_merge_buffer()
  {
    cudaMemset(d_g_v1_frequency, 0, total_nv * sizeof(int));
    cudaMemset(d_v1_offset, 0, (total_nv + 1) * sizeof(int));
    cudaMemset(d_global_index, 0, sizeof(int));
  }
#endif

  AccType calculate_comm_volumn()
  {
    AccType h_comm_volumn = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_comm_volumn, comm_volumn, sizeof(AccType), cudaMemcpyDeviceToHost));
    return h_comm_volumn;
  }
};

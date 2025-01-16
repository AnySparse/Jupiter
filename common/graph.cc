#include "graph.h"
#include "scan.h"
#include <thread>
#include <mutex>
#include <pthread.h>
#include <omp.h>

Graph::Graph(std::string prefix, bool use_dag, bool directed,
             bool use_vlabel, bool use_elabel) : is_directed_(directed), vlabels(NULL), elabels(NULL), nnz(0)
{

  Timer t;
  t.Start();
  std::cout <<prefix<<std::endl;
  // parse file name
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos)
    inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos)
    name = inputfile_path.substr(i + 1);
  std::cout << "input file path: " << inputfile_path << ", graph name: " << name << "\n";

  // read meta information
  VertexSet::release_buffers();
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0;
  // f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
  f_meta >> n_vertices >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  assert(sizeof(elabel_t) == elabel_size);
  assert(max_degree > 0 && max_degree < n_vertices);
  f_meta.close();
  // read row pointers
  if (map_vertices)
    map_file(prefix + ".vertex.bin", vertices, n_vertices + 1);
  else
    read_file(prefix + ".vertex.bin", vertices, n_vertices + 1);
  // read column indices
  if (map_edges)
    map_file(prefix + ".edge.bin", edges, n_edges);
  else
    read_file(prefix + ".edge.bin", edges, n_edges);
  // read vertex labels
  if (use_vlabel)
  {
    assert(num_vertex_classes > 0);
    assert(num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good())
    {
      if (map_vlabels)
        map_file(vlabel_filename, vlabels, n_vertices);
      else
        read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    }
    else
    {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++)
      {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels + n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  if (use_elabel)
  {
    assert(num_edge_classes > 0);
    assert(num_edge_classes < 65535); // we use 16-bit edge label dtype
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good())
    {
      if (map_elabels)
        map_file(elabel_filename, elabels, n_edges);
      else
        read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    }
    else
    {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
      elabels = new elabel_t[n_edges];
      for (eidType e = 0; e < n_edges; e++)
      {
        elabels[e] = rand() % num_edge_classes + 1;
      }
    }
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels + n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // orientation: convert the undirected graph into directed. Only for k-cliques. This may change max_degree.
  t.Stop();

  std::cout << "========================Load mtx time cost:" << t.Seconds() << "sec." << std::endl;
  t.Start();
  if (use_dag)
  {
    assert(!directed); // must be undirected before orientation
    orientation();
  }
  // compute maximum degree
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);

  labels_frequency_.clear();
  t.Stop();
  std::cout << "========================Orientation cost:" << t.Seconds() << "sec." << std::endl;
}

Graph::~Graph()
{
  if (map_edges)
    munmap(edges, n_edges * sizeof(vidType));
  else
    custom_free(edges, n_edges);
  if (map_vertices)
  {
    munmap(vertices, (n_vertices + 1) * sizeof(eidType));
  }
  else
    custom_free(vertices, n_vertices + 1);
  if (vlabels != NULL)
    delete[] vlabels;
}

VertexSet Graph::N(vidType vid) const
{
  assert(vid >= 0);
  assert(vid < n_vertices);
  eidType begin = vertices[vid], end = vertices[vid + 1];
  if (begin > end)
  {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin, end - begin, vid);
}

void Graph::allocateFrom(vidType nv, eidType ne)
{
  n_vertices = nv;
  n_edges = ne;
  vertices = new eidType[nv + 1];
  edges = new vidType[ne];
  vertices[0] = 0;
}

vidType Graph::compute_max_degree()
{
  std::cout << "computing the maximum degree\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
#pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++)
  {
    degrees[v] = vertices[v + 1] - vertices[v];
  }
  vidType max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  t.Start();
  return max_degree;
}

// struct Params {
//   int threadId;
//   int numElements;
//   vidType *dataPtr;

//   Params(int tid, int numElements, vidType *ptr)
//       : threadId(tid), numElements(numElements), dataPtr(ptr) {}
// };

// void *test_func(void *args) {
//  Params *param = static_cast<Params *>(args);

//   int tid = param->threadId;
//   const int numElements = param->numElements;
//   vidType *data = param->dataPtr;
//   int result = 0;
//       int start = tid;
//       for (int src = start; src < n_vertices; src += NUM_THREADS) {
//         for (auto dst : N(src)) {
//           if (degrees[dst] > degrees[src] ||
//               (degrees[dst] == degrees[src] && dst > src)) {
//             data[src]++;
//           }
//         }
//       }
//   pthread_exit(NULL);

// }

void Graph::orientation()
{
  std::cout << "Orientation enabled, using DAG\n";
  Timer t;
  t.Start();
  Timer t1;
  t1.Start();
  std::vector<vidType> degrees(n_vertices, 0);
#pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++)
  {
    degrees[v] = get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
#pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src++)
  {
    for (auto dst : N(src))
    {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src))
      {
        new_degrees[src]++;
      }
    }
  }
  t1.Stop();
  std::cout << "Orientation Step1 cost" << t1.Seconds() << std::endl;

  // std::vector<vidType> new_degrees1(n_vertices, 0);
  // t1.Start();
  // std::vector<std::thread> threads;

  // for (int jj = 0; jj < 10; jj++)
  // {
  //   threads.emplace_back([&](int index)
  //                        {
  //     int start = index;
  //     int step = 10;
  //     printf("Thread %d \n", index);
  //     for (int src = start; src < n_vertices; src += step) {
  //       //printf("i:%d src:%d\n",jj,src);
  //       // int count = 0;
  //       for (auto dst : N(src)) {
  //         //std::lock_guard<std::mutex> lock(mutex);
  //         //if(src==2) printf("===%d    \n",dst);
  //         if (degrees[dst] > degrees[src] ||
  //             (degrees[dst] == degrees[src] && dst > src)) {
  //               // count++;
  //           new_degrees1[src]++;
  //         }
  //         // new_degrees1[src]+=count;
  //       }
  //     } }, jj);
  // }

  // for (auto &thread : threads)
  // {
  //   thread.join();
  // }

  // #define NUM_THREADS 10
  // std::vector<pthread_t> threads(NUM_THREADS);
  //   for (int t = 1; t <= NUM_THREADS; ++t) {
  //   Params params(t, n_vertices, new_degrees1.data());
  //   int rc = pthread_create(&threads[t - 1], NULL, test_func, (void *)(&params));
  //   if (rc) {
  //     std::cout << "ERROR; return code from pthread_create() is " << rc
  //               << std::endl;
  //     exit(-1);
  //   }
  // }
  //   // wait for all threads to complete
  // for (int t = 0; t < NUM_THREADS; ++t) {
  //   pthread_join(threads[t], NULL);
  // }

  // #pragma omp parallel num_threads(2)
  //  {
  //          printf("%d : %d\n", __LINE__, omp_get_thread_num());
  //  }

  /*
  int num_threads = 16; // 指定线程数量
  //omp_set_num_threads(4);
  #pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();  // 获取当前线程ID
   //  printf("Thread %d\n",thread_id);

    // int chunk_size = n_vertices / num_threads;
    // int start = thread_id * chunk_size;
    // int end = (thread_id + 1) * chunk_size;

    // // 最后一个线程额外处理剩余的数据
    // if (thread_id == num_threads - 1) {
    //   end = n_vertices;
    // }
    //  printf("Thread %d processed from %d to %d\n", thread_id, start, end-1);
    size_t count = 0;
    // for (vidType src = start; src < end; src++)

    int start = thread_id;
    int stride = num_threads;
    for (vidType src = start; src < n_vertices; src+=num_threads)
    {
      //for (auto dst : N(src))

      eidType vbegin = vertices[src], vend = vertices[src + 1];
      auto deg_src =  degrees[src];
      vidType sum = 0;
      for(eidType vidx = vbegin; vidx<vend; vidx++)
      {
        count++;
        auto dst = edges[vidx];

        if (degrees[dst] > deg_src ||
           (degrees[dst] == deg_src && dst > src))
        {
          sum++;
        }
      }
      new_degrees1[src] = sum;
    }
    printf("Thread %d count = %zu\n", thread_id, count);
  }
  */

  // #pragma omp parallel for
  //   for (vidType src = 0; src < n_vertices; src++)
  //   {
  //     for (auto dst : N(src))
  //     {
  //       if (degrees[dst] > degrees[src] ||
  //          (degrees[dst] == degrees[src] && dst > src))
  //       {
  //         new_degrees1[src]++;
  //       }
  //     }

  //   }

  //   t1.Stop();
  //   std::cout << "Step1 opt cost" << t1.Seconds() << std::endl;

  //   for (int i = 0; i < n_vertices; i++)
  //   {
  //     if (new_degrees1[i] != new_degrees[i])
  //     {
  //       printf("vid:%d   true:%d  error:%d !!!\n", i, new_degrees[i], new_degrees1[i]);
  //       break;
  //     }
  //   }
  // printf("success!!!\n");

  t1.Start();
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices + 1);
  parallel_prefix_sum<vidType, eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  t1.Stop();
  std::cout << "Step2 cost" << t1.Seconds() << std::endl;
  t1.Start();
#pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src++)
  {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src))
    {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src))
      {
        new_edges[begin + offset] = dst;
        offset++;
      }
    }
  }
  t1.Stop();
  std::cout << "Orientation Step3 cost" << t1.Seconds() << std::endl;
  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
  n_edges = num_edges;
  t.Stop();
  std::cout << "Orientation Time on generating the DAG: " << t.Seconds() << " sec\n";
  // exit(0);
}

void Graph::print_graph() const
{
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++)
  {
    std::cout << "vertex " << n << ": degree = "
              << get_degree(n) << " edgelist = [ ";
    for (auto e = edge_begin(n); e != edge_end(n); e++)
      std::cout << getEdgeDst(e) << " ";
    std::cout << "]\n";
  }
}

// eidType Graph::init_edgelist(bool sym_break, bool ascend)
// {
//   Timer t;
//   t.Start();
//   if (nnz != 0)
//     return nnz; // already initialized
//   nnz = E();
//   if (sym_break)
//     nnz = nnz / 2;
//   sizes.resize(V());
//   src_list = new vidType[nnz];
//   if (sym_break)
//     dst_list = new vidType[nnz];
//   else
//     dst_list = edges;
//   size_t i = 0;
//   for (vidType v = 0; v < V(); v++)
//   {
//     for (auto u : N(v))
//     {
//       if (u == v)
//         continue; // no selfloops
//       if (ascend)
//       {
//         if (sym_break && v > u)
//           continue;
//       }
//       else
//       {
//         if (sym_break && v < u)
//           break;
//       }
//       src_list[i] = v;
//       if (sym_break)
//         dst_list[i] = u;
//       sizes[v]++;
//       i++;
//     }
//   }
//   // assert(i == nnz);
//   t.Stop();
//   std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
//   return nnz;
// }


eidType Graph::init_edgelist(bool sym_break, bool ascend)
{
  Timer t;
  t.Start();
  if (nnz != 0)
    return nnz; // already initialized
  nnz = E();
  if (sym_break)
    nnz = nnz / 2;
  //sizes.resize(V());
  src_list = new vidType[nnz];
  dst_list = edges;
  size_t i = 0;
  // for (vidType v = 0; v < V(); v++)
  // {
  //   for (auto u : N(v))
  //   {
  //     if (u == v)
  //       continue; // no selfloops
  //     src_list[i] = v;
  //   }
  // }

    int num_threads = 10;
  #pragma omp parallel num_threads(num_threads)
  {
    int thread_id = omp_get_thread_num();  // 获取当前线程ID
    size_t count = 0;
    int start = thread_id;
    int stride = num_threads;
   
    for (vidType src = start; src < V(); src+=num_threads)
    {
      //auto deg = g.get_degree(src);
      auto end = vertices[src + 1];
      auto start = vertices[src];

      for(auto xx=start; xx<end; xx++){
        src_list[xx] = src;
      }
    }
  }
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

bool Graph::is_connected(vidType v, vidType u) const
{
  auto v_deg = get_degree(v);
  auto u_deg = get_degree(u);
  bool found;
  if (v_deg < u_deg)
  {
    found = binary_search(u, edge_begin(v), edge_end(v));
  }
  else
  {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

bool Graph::is_connected(std::vector<vidType> sg) const
{
  return false;
}

bool Graph::binary_search(vidType key, eidType begin, eidType end) const
{
  auto l = begin;
  auto r = end - 1;
  while (r >= l)
  {
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key)
      return true;
    if (value < key)
      l = mid + 1;
    else
      r = mid - 1;
  }
  return false;
}

vidType Graph::intersect_num(vidType v, vidType u, vlabel_t label)
{
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = get_degree(v);
  vidType u_size = get_degree(u);
  vidType *v_ptr = &edges[vertices[v]];
  vidType *u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size)
  {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b)
      idx_l++;
    if (b <= a)
      idx_r++;
    if (a == b && vlabels[a] == label)
      num++;
  }
  return num;
}

void Graph::print_meta_data() const
{
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0)
  {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty())
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
    std::cout << "\n";
  }
  else
  {
    std::cout << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0)
  {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  }
  else
  {
    std::cout << "This graph does not have edge labels\n";
  }
  if (feat_len > 0)
  {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  }
  else
  {
    std::cout << "This graph has no input vertex features\n";
  }
}

void Graph::sort_graph()
{

  Timer t1;
  // t1.Start();
  // //std::vector<int> index(n_vertices);
  // std::vector<int> r_index(n_vertices);
  // // for (size_t i = 0; i < index.size(); i++)
  // //   index[i] = i;
  // // std::stable_sort(index.begin(), index.end(), [&](int a, int b)
  // //                  { return get_degree(a) > get_degree(b); });

  // t1.Stop();
  // std::cout << "Step1 cost" << t1.Seconds() << std::endl;

  t1.Start();
  std::vector<int> r_index(n_vertices);
  // std::vector<std::pair<int, int>> index1;
  // index1.resize(index.size());
  // #pragma omp parallel for
  // for (size_t i = 0; i < index.size(); i++) {
  //     index1[i] = std::make_pair(i, get_degree(i));
  // }

  // std::stable_sort(index1.begin(), index1.end(), [](const auto& a, const auto& b) {
  //     return a.second > b.second;
  // });

  std::vector<int> count(max_degree+1, 0);
  std::vector<int> index(n_vertices);

  for (size_t i = 0; i < index.size(); i++)
  {
    count[get_degree(i)]++;
  }

  for (size_t i = 1; i < count.size(); i++)
  {
    count[i] += count[i - 1];
  }

  for (int i = index.size() - 1; i >= 0; i--)
  {
    int degree = get_degree(i);
    index[count[degree] - 1] = i;
    count[degree]--;
  }

  // index_ = std::move(sortedIndex); // 将排序后的索引赋值给 index_

  // for(int i=0; i<index.size()-1; i++){
  //   if(get_degree(index[i])>get_degree(index[i+1])) {
  //     printf("error!!! i:%d vi:%d  vi+1:%d\n",i,get_degree(index[i]),get_degree(index[i+1]));
  //     break;
  //   }
  // }

  t1.Stop();
  std::cout << "Sort Step1 opt cost" << t1.Seconds() << std::endl;

  // for(int i=0; i<index.size(); i++){
  //   if(index[i] != index1[i].first) {printf("error!!!!!\n"); break;}
  // }

  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices + 1);
  vidType *new_edges = custom_alloc_global<vidType>(n_edges);
  std::vector<vidType> new_degrees(n_vertices, 0);

  t1.Start();
#pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src++)
  {
    vidType v = index[src];
    r_index[v] = src;
  }

#pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src++)
  {
    vidType v = index[src];
    new_degrees[src] = get_degree(v);
  }
  parallel_prefix_sum<vidType, eidType>(new_degrees, new_vertices);
  t1.Stop();
  std::cout << "Sort Step2 cost" << t1.Seconds() << std::endl;
  t1.Start();
#pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src++)
  {
    auto begin = new_vertices[src];
    eidType offset = 0;
    vidType v = index[src];
    for (auto dst : N(v))
    {
      new_edges[begin + offset] = r_index[dst];
      offset++;
    }
    std::sort(&new_edges[begin], &new_edges[begin + offset]);
  }
  t1.Stop();
  std::cout << "Sort Step3 cost" << t1.Seconds() << std::endl;

  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  // eidType edge_end = vertices[n_vertices];
  // std::cout<<n_vertices<<"  "<<n_edges<<"   "<<vertices[n_vertices]<<std::endl;

  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);

  // printf("0:adj\n");
  // for (auto dst : N(0)) {
  //   printf("%d  ",dst);
  // }
  // printf("\n");

  //   printf("1000:adj\n");
  // for (auto dst : N(1000)) {
  //   printf("%d  ",dst);
  // }
  // printf("\n");
}

void Graph::dump_binary(std::string filename){
        std::cout << "Writing graph to file\n";
        //std::string outfilename =  "/data/linzhiheng/sort_g30/graph";
        //std::string outfilename =  "/data/linzhiheng/sort_uk2014/graph";
        //std::string outfilename =  "/data/linzhiheng/"+filename+"/graph";
        std::string outfilename = filename;
        std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(edges), n_edges*sizeof(vidType));
    outfile1.close();
    std::ofstream outfile2((outfilename+".meta.txt").c_str());
  outfile2<<n_vertices<<"\n";
  outfile2<<n_edges<<"\n";
  outfile2<<"4 8 1 2\n";
  outfile2<<max_degree<<"\n";
  outfile2<<"0\n0\n0\n";
  }
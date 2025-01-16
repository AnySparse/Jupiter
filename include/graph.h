#pragma once
#include "VertexSet.h"

using namespace std;

// for out-of-core solution, enable mmap flags below
constexpr bool map_edges = false;
constexpr bool map_vertices = false;
constexpr bool map_vlabels = false;
constexpr bool map_elabels = false;
constexpr bool map_features = false;

class Graph
{
private:
  std::string name;                            // name of the graph
  std::string inputfile_path;                  // file path of the graph
  bool is_directed_;                           // is it a directed graph?
  vidType max_degree;                          // maximun degree
  vidType n_vertices;                          // number of vertices
  eidType n_edges;                             // number of edges
  vidType *edges;                              // column indices of CSR format
  eidType *vertices;                           // row pointers of CSR format
  vlabel_t *vlabels;                           // vertex labels
  elabel_t *elabels;                           // edge labels
  feat_t *features;                            // vertex features; one feature vector per vertex
  int feat_len;                                // vertex feature vector length: '0' means no features
  int num_vertex_classes;                      // number of distinct vertex labels: '0' means no vertex labels
  int num_edge_classes;                        // number of distinct edge labels: '0' means no edge labels
  int max_label;                               // maximum label
  eidType nnz;                                 // number of edges in COO format (may be halved due to orientation)
  vidType *src_list, *dst_list;                // source and destination vertices of COO format
  vidType max_label_frequency_;                // maximum label frequency
  std::vector<vidType> labels_frequency_;      // vertex count of each label
  std::vector<nlf_map> nlf_;                   // neighborhood label frequency
  std::vector<vidType> sizes;                  // neighbor count of each source vertex in the edgelist
  VertexList reverse_index_;                   // indices to vertices grouped by vertex label
  std::vector<eidType> reverse_index_offsets_; // pointers to each vertex group

public:
  Graph(std::string prefix, bool use_dag = false, bool directed = false,
        bool use_vlabel = false, bool use_elabel = false);
  Graph() : n_vertices(0), n_edges(0) {}
  Graph(vidType nv, eidType ne) { allocateFrom(nv, ne); }
  ~Graph();
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  // get methods for graph meta information
  vidType V() const { return n_vertices; }
  eidType E() const { return n_edges; }
  eidType get_num_tasks() const { return nnz; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  std::string get_name() const { return name; }
  std::string get_inputfile_path() const { return inputfile_path; }
  bool is_directed() const { return is_directed_; }
  void set_max_degree(vidType md) { max_degree = md; }
  vidType get_max_degree() const { return max_degree; }

  // get methods for graph topology information
  vidType get_degree(vidType v) const { return vertices[v + 1] - vertices[v]; }
  vidType out_degree(vidType v) const { return vertices[v + 1] - vertices[v]; }
  eidType edge_begin(vidType v) const { return vertices[v]; }
  eidType edge_end(vidType v) const { return vertices[v + 1]; }
  vidType *adj_ptr(vidType v) const { return &edges[vertices[v]]; }
  vidType N(vidType v, vidType n) const { return edges[vertices[v] + n]; } // get the n-th neighbor of v
  VertexSet N(vidType v) const;                                            // get the neighbor list of vertex v
  eidType *out_rowptr() { return vertices; }                               // get row pointers array
  vidType *out_colidx() { return edges; }                                  // get column indices array
  bool is_connected(vidType v, vidType u) const;                           // is vertex v and u connected by an edge
  bool is_connected(std::vector<vidType> sg) const;                        // is the subgraph sg a connected one

  // get methods for local graph topology information
  // vidType get_local_degree(vidType v) const { return local_vertices[v + 1] - local_vertices[v]; }
  // vidType local_out_degree(vidType v) const { return local_vertices[v + 1] - local_vertices[v]; }
  // eidType local_edge_begin(vidType v) const { return local_vertices[v]; }
  // eidType local_edge_end(vidType v) const { return local_vertices[v + 1]; }
  // vidType *local_adj_ptr(vidType v) const { return &local_edges[local_vertices[v]]; }
  // vidType local_N(vidType v, vidType n) const { return local_edges[local_vertices[v] + n]; } // get the n-th neighbor of v
  // VertexSet local_N(vidType v) const;                                            // get the neighbor list of vertex v
  // eidType *local_out_rowptr() { return local_vertices; }                               // get row pointers array
  // vidType *local_out_colidx() { return local_edges; }                                  // get column indices array
  // bool local_is_connected(vidType v, vidType u) const;                           // is vertex v and u connected by an edge
  // bool local_is_connected(std::vector<vidType> sg) const;                        // is the subgraph sg a connected one

  // // get methods for remote graph topology information
  // vidType get_remote_degree(vidType v) const { return remote_vertices[v + 1] - remote_vertices[v]; }
  // vidType remote_out_degree(vidType v) const { return remote_vertices[v + 1] - remote_vertices[v]; }
  // eidType remote_edge_begin(vidType v) const { return remote_vertices[v]; }
  // eidType remote_edge_end(vidType v) const { return remote_vertices[v + 1]; }
  // vidType *remote_adj_ptr(vidType v) const { return &remote_edges[remote_vertices[v]]; }
  // vidType remote_N(vidType v, vidType n) const { return remote_edges[remote_vertices[v] + n]; } // get the n-th neighbor of v
  // VertexSet remote_N(vidType v) const;                                            // get the neighbor list of vertex v
  // eidType *remote_out_rowptr() { return remote_vertices; }                               // get row pointers array
  // vidType *remote_out_colidx() { return remote_edges; }                                  // get column indices array
  // bool remote_is_connected(vidType v, vidType u) const;                           // is vertex v and u connected by an edge
  // bool remote_is_connected(std::vector<vidType> sg) const;                        // is the subgraph sg a connected one

  // Galois compatible APIs
  vidType size() const { return n_vertices; }
  eidType sizeEdges() const { return n_edges; }
  vidType getEdgeDst(eidType e) const { return edges[e]; } // get target vertex of the edge e
  vlabel_t getData(vidType v) const { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) const { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) const { return elabels[e]; }
  void fixEndEdge(vidType vid, eidType row_end) { vertices[vid + 1] = row_end; }
  void allocateFrom(vidType nv, eidType ne);
  void constructEdge(eidType eid, vidType dst) { edges[eid] = dst; }

  // get methods for labels
  vlabel_t get_vlabel(vidType v) const { return vlabels[v]; }
  elabel_t get_elabel(eidType e) const { return elabels[e]; }
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return num_edge_classes; }     // number of distinct edge labels
  int get_frequent_labels(int threshold);
  int get_max_label() { return max_label; }
  vlabel_t *getVlabelPtr() { return vlabels; }
  elabel_t *getElabelPtr() { return elabels; }
  vlabel_t *get_vlabel_ptr() { return vlabels; }
  elabel_t *get_elabel_ptr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }

  // edgelist or COO
  vidType *get_src_ptr() { return &src_list[0]; }
  vidType *get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  std::vector<vidType> get_sizes() const { return sizes; }
  eidType init_edgelist(bool sym_break = false, bool ascend = false);

  // edge orientation: convert the graph from undirected to directed
  void orientation();
  void sort_graph();

  void dump_binary(std::string filename);

  // print graph information
  void print_meta_data() const;
  void print_graph() const;

  
private:
  vidType compute_max_degree();
  bool binary_search(vidType key, eidType begin, eidType end) const;
  vidType intersect_num(vidType v, vidType u, vlabel_t label);
};

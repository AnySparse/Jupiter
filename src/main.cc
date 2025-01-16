
#include "graph.h"
#include <mpi.h>

void SubgraphSolver(Graph &g, uint64_t &total, int argc, char *argv[], MPI_Comm &mpi_comm, int rank, int nranks);

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "Usage: " << argv[0] << " <graph> [pattern(P1)] [round(1)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph P1\n";
    exit(1);
  }

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  //====for dump subgraph
  // step 1: close SUBGRAPH_LOADING
  // step 2: open dump_binary
  //====for load subgraph
  // step 1: open SUBGRAPH_LOADING
  // step 2: close dump_binary

bool use_dag_ = false;
std::string pattern_name(argv[2]);
//These patterns are k-clique with graph orientatation
if(pattern_name == "P13" || pattern_name == "P14" || pattern_name == "P15" || pattern_name == "P16")
 use_dag_ = true;
  

// #define SUBGRAPH_LOADING
#ifdef SUBGRAPH_LOADING
  printf("loading subgraph\n");
  std::string filename(argv[1]);
  Graph g(filename + std::to_string(rank));
#else
  Graph g(argv[1],use_dag_); 
#endif
  g.print_meta_data();
  uint64_t total = 0;
  SubgraphSolver(g, total, argc, argv, mpi_comm, rank, nranks);
  return 0;
}

#pragma once
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

typedef float   feat_t;    // vertex feature type
typedef uint8_t patid_t;   // pattern id type
typedef uint8_t mask_t;    // mask type
typedef uint8_t label_t;   // label type
typedef uint8_t vlabel_t;  // vertex label type
typedef uint16_t elabel_t; // edge label type
typedef uint8_t cmap_vt;   // cmap value type
typedef int32_t vidType;   // vertex ID type
typedef int64_t eidType;   // edge ID type
typedef int32_t IndexT;
typedef uint64_t emb_index_t; // embedding index type
typedef unsigned long long AccType;

typedef std::vector<patid_t> PidList;    // pattern ID list
typedef std::vector<vidType> VertexList; // vertex ID list
typedef std::vector<std::vector<vidType>> VertexLists;
typedef std::unordered_map<vlabel_t, int> nlf_map;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256
#define NUM_BLOCK    2592
#define SLOT_SIZE 10
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))


#define ROUND_UP(x) x%32==0 ?x/32:x/32+1
#define DIVIDE32(x) x>>5
#define MULTIPLY32(x) x<<5
#define MOD32(x) x-((x>>5)<<5)
static constexpr uint32_t ONE_BIT[32] = {
        0x80000000, 0x40000000, 0x20000000, 0x10000000,
        0x08000000, 0x04000000, 0x02000000, 0x01000000,
        0x00800000, 0x00400000, 0x00200000, 0x00100000,
        0x00080000, 0x00040000, 0x00020000, 0x00010000,
        0x00008000, 0x00004000, 0x00002000, 0x00001000,
        0x00000800, 0x00000400, 0x00000200, 0x00000100,
        0x00000080, 0x00000040, 0x00000020, 0x00000010,
        0x00000008, 0x00000004, 0x00000002, 0x00000001,
};

enum Status {
  Idle,
  Extending,
  IteratingEdge,
  Working,
  ReWorking
};

#define OP_INTERSECT 'i'
#define OP_DIFFERENCE 'd'
extern std::map<char,double> time_ops;

const std::string long_separator = "--------------------------------------------------------------------\n";
const std::string short_separator = "-----------------------\n";


enum OPERAND_TYPE {
  P_TYPE,
  S_TYPE
};

enum OPERATOR_TYPE {
  INTERSECTION,
  INTERSECTION_COUNT,
  DIFFERENCE,
  DIFFERENCE_COUNT
};

enum COMM_TYPE {
  PULL_NEIGHBOURS,
  PULL_WORKLOADS
};

//#define BITMAP
// #define EXTRA_CHECK
#define HASH_1D_PARTITION
// #define LOCALITY_1D_PARTITION
// #define PROFILING
#define BATCH_LOAD
// #define MERGE_NEIGHBORS_COMM
#define FREQ_THD 2
#define PEAK_CLK (float)1410000

// #define MESSAGE_PASSING
#define DELEGATION

#define USE_FILTER
#define USE_FUSE
#define USE_COMP
// #define MERGE_TASK
// #define USE_LAZY
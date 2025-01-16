#include "common.h"
// For sync module.
class Config
{
public:
  int level;
  bool reused;                // S=P can be used
  OPERAND_TYPE left_operand;  // P or N(vi) in S
  OPERAND_TYPE right_operand; // N(vi) in S, P, P_inion
  OPERATOR_TYPE op;           // intersection or difference
};

class PatternConfig
{
public:
  Config configs[8];
  int nlevel;
  void init()
  {
    nlevel = 4;
    for (int i = 0; i < nlevel; i++)
      configs[i].level = i;
    configs[0].reused = false; // edge parallel
    configs[1].reused = false;
    configs[2].reused = false;
    configs[2].left_operand = OPERAND_TYPE::P_TYPE;
    configs[2].right_operand = OPERAND_TYPE::S_TYPE;
    configs[2].op = OPERATOR_TYPE::INTERSECTION;
    configs[3].reused = false;
    configs[3].left_operand = OPERAND_TYPE::P_TYPE;
    configs[3].right_operand = OPERAND_TYPE::S_TYPE;
    configs[3].op = OPERATOR_TYPE::INTERSECTION;
    configs[4].reused = false;
    configs[4].left_operand = OPERAND_TYPE::P_TYPE;
    configs[4].right_operand = OPERAND_TYPE::S_TYPE;
    configs[4].op = OPERATOR_TYPE::INTERSECTION_COUNT;

  }
  void init(int k)
  {
    // nlevel = 4;
    assert(k>=4);
    nlevel = k - 1;
    for (int i = 0; i < nlevel; i++)
      configs[i].level = i;
    configs[0].reused = false;
    configs[1].reused = false;
    for (int i = 2; i < nlevel; i++)
    {
      configs[i].reused = true;
      configs[i].left_operand = OPERAND_TYPE::P_TYPE;
      configs[i].right_operand = OPERAND_TYPE::S_TYPE;
      if (i != nlevel - 1)
        configs[i].op = OPERATOR_TYPE::INTERSECTION;
      else
        configs[i].op = OPERATOR_TYPE::INTERSECTION_COUNT;
    }
  }
  Config next_config(int level)
  {
    return configs[level];
  }
  Config first_config()
  {
    return configs[0];
  }
};

// TODO: genral for any patterns
// Special for 5-clique now
// class PatternConfig
// {
// public:
//   Config configs[4];
//   int nlevel;
//   void init()
//   {
//     nlevel = 4;
//     for (int i = 0; i < nlevel; i++)
//       configs[i].level = i;
//     configs[0].reused = false; // edge parallel
//     configs[1].reused = false;
//     configs[2].reused = true;
//     configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[2].op = OPERATOR_TYPE::INTERSECTION;
//     configs[3].reused = true;
//     configs[3].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[3].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[3].op = OPERATOR_TYPE::INTERSECTION_COUNT;
//   }
//   Config next_config(int level)
//   {
//     return configs[level];
//   }
//   Config first_config()
//   {
//     return configs[0];
//   }
// };

// Special for 4-clique now
// class PatternConfig
// {
// public:
//   Config configs[3];
//   int nlevel;
//   void init()
//   {
//     nlevel = 3;
//     for (int i = 0; i < nlevel; i++)
//       configs[i].level = i;
//     configs[0].reused = false; // edge parallel
//     configs[1].reused = false;
//     configs[2].reused = true;
//     configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[2].op = OPERATOR_TYPE::INTERSECTION_COUNT;
//   }
//   Config next_config(int level)
//   {
//     return configs[level];
//   }
//   Config first_config()
//   {
//     return configs[0];
//   }
// };

// 6 clique
// class PatternConfig
// {
// public:
//   Config configs[5];
//   int nlevel;
//   void init()
//   {
//     nlevel = 5;
//     for (int i = 0; i < nlevel; i++)
//       configs[i].level = i;
//     configs[0].reused = false; // edge parallel
//     configs[1].reused = false;
//     configs[2].reused = true;
//     configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[2].op = OPERATOR_TYPE::INTERSECTION;
//     configs[3].reused = true;
//     configs[3].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[3].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[3].op = OPERATOR_TYPE::INTERSECTION;
//     configs[4].reused = true;
//     configs[4].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[4].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[4].op = OPERATOR_TYPE::INTERSECTION_COUNT;
//   }
//   Config next_config(int level)
//   {
//     return configs[level];
//   }
//   Config first_config()
//   {
//     return configs[0];
//   }
// };

// 7 clique
//  class PatternConfig
//  {
//  public:
//    Config configs[6];
//    int nlevel;
//    void init()
//    {
//      nlevel = 6;
//      for (int i = 0; i < nlevel; i++)
//        configs[i].level = i;
//      configs[0].reused = false; // edge parallel
//      configs[1].reused = false;
//      configs[2].reused = true;
//      configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[2].op = OPERATOR_TYPE::INTERSECTION;
//      configs[3].reused = true;
//      configs[3].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[3].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[3].op = OPERATOR_TYPE::INTERSECTION;
//      configs[4].reused = true;
//      configs[4].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[4].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[4].op = OPERATOR_TYPE::INTERSECTION;
//      configs[5].reused = true;
//      configs[5].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[5].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[5].op = OPERATOR_TYPE::INTERSECTION_COUNT;
//    }
//    Config next_config(int level)
//    {
//      return configs[level];
//    }
//    Config first_config()
//    {
//      return configs[0];
//    }
//  };

// // Special for diamond now
// class PatternConfig
// {
// public:
//   Config configs[3];
//   int nlevel;
//   void init()
//   {
//     nlevel = 3;
//     for (int i = 0; i < nlevel; i++)
//       configs[i].level = i;
//     configs[0].reused = false; // edge parallel
//     configs[1].reused = false;
//     configs[2].reused = true;
//     configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//     configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//     configs[2].op = OPERATOR_TYPE::DIFFERENCE_COUNT;
//   }
//   Config next_config(int level)
//   {
//     return configs[level];
//   }
//   Config first_config()
//   {
//     return configs[0];
//   }
// };

// Test config
//  class PatternConfig
//  {
//  public:
//    Config configs[5];
//    int nlevel;
//    void init()
//    {
//      nlevel = 4;
//      for (int i = 0; i < nlevel; i++)
//        configs[i].level = i;
//      configs[0].reused = false; // edge parallel
//      configs[1].reused = false;
//      configs[2].reused = false;
//      configs[2].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[2].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[2].op = OPERATOR_TYPE::INTERSECTION;
//      configs[3].reused = false;
//      configs[3].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[3].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[3].op = OPERATOR_TYPE::INTERSECTION;
//      configs[4].reused = false;
//      configs[4].left_operand = OPERAND_TYPE::P_TYPE;
//      configs[4].right_operand = OPERAND_TYPE::S_TYPE;
//      configs[4].op = OPERATOR_TYPE::INTERSECTION_COUNT;
//    }
//    Config next_config(int level)
//    {
//      return configs[level];
//    }
//    Config first_config()
//    {
//      return configs[0];
//    }
//  };
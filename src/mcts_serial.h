#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include <gomoku.h>
#include <thread_pool.h>
#include <libtorch.h>

class TreeNode_serial {
 public:
  // friend class can access private variables
  friend class MCTS_serial;

  TreeNode_serial();
  TreeNode_serial(const TreeNode_serial &node);
  TreeNode_serial(TreeNode_serial *parent, double p_sa, unsigned action_size);

  TreeNode_serial &operator=(const TreeNode_serial &p);

  unsigned int select(double c_puct);
  void expand(const std::vector<double> &action_priors);
  void backup(double leaf_value);

  double get_value(double c_puct,
                   unsigned int sum_n_visited) const;
  inline bool get_is_leaf() const { return this->is_leaf; }

 private:
  // store tree
  TreeNode_serial *parent;
  std::vector<TreeNode_serial *> children;
  bool is_leaf;

  unsigned int n_visited;
  double p_sa;
  double q_sa;
};

class MCTS_serial {
 public:
  MCTS_serial(NeuralNetwork *neural_network, double c_puct,
       unsigned int num_mcts_sims,
       unsigned int action_size);
  std::vector<double> get_action_probs(Gomoku *gomoku, double temp = 1e-3);
  void update_with_move(int last_move);

 private:
  void simulate(std::shared_ptr<Gomoku> game);
  static void tree_deleter(TreeNode_serial *t);

  // variables
  std::unique_ptr<TreeNode_serial, decltype(MCTS_serial::tree_deleter) *> root;
  NeuralNetwork *neural_network;

  unsigned int action_size;
  unsigned int num_mcts_sims;
  double c_puct;
  int selection_time;
  int expansion_time;
  int backprop_time;
};

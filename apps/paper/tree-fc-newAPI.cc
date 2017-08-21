#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch_size, 20,       "batch");
DEFINE_int32 (hidden,     100,      "hidden size");
DEFINE_int32 (tree_size,  128,      "epochs");
DEFINE_int32 (iters,      100,       "iterations");
DEFINE_double(init_scale, 0.1f,     "init random scale of variables");
DEFINE_double(lr,         0.00001f, "learning rate");


class TreeFCModel : public GraphSupport {
 public:
  TreeFCModel(const Sym& graph_ph, const Sym& vertex_ph) : 
    GraphSupport(graph_ph, vertex_ph) {

    W = Sym::Variable(DT_FLOAT, {2 * FLAGS_hidden, FLAGS_hidden}, Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//v2
    //B = Sym::Variable(DT_FLOAT, {FLAGS_hidden}, Sym::Zeros());
  }

  void Node() override {
    Sym left = Gather(0, {FLAGS_hidden});//g3
    Sym right = Gather(1, {FLAGS_hidden});//g4
    Sym fc = Sym::MatMul(Sym::Concat({left, right}).Reshape({1, 2*FLAGS_hidden}), W.Mirror()/*mirror_5*/);// + B.Reshape({1, FLAGS_hidden}).Mirror();
    Sym res = fc.Relu();
    Scatter(res.Mirror());
    Push(res.Mirror());
  }

 private:
  Sym W, B;
};

//class TreeGenerator {
 //public:
  //int genTree(int leaf_size) {
    //if (leaf_size == 1)
      //return;
    //int left_size = random() % (leaf_size-1) + 1;
    //int right_size = leaf_size - left_size;
    
    //int left_depth = genTree(left_size);
    //int right_depth = genTree(right_size);
    //return std::max(left_depth, right_depth);
  //}

  //void toInt() {}

 //private:
  //struct Node {
    //int depth = -1;
    //Node* father;
  //};
  //Node* node;
//};

void binaryTree(vector<int>* graph) {
  vector<int> one_tree;
  int count = 0;
  for (int width = FLAGS_tree_size; width > 1; width >>= 1) {
    count += width;
    for (int i = 0; i < width; i++) {
      one_tree.push_back(i/2+count);
    }
  }
  one_tree.push_back(-1);
  CHECK(one_tree.size() == 2*FLAGS_tree_size-1);

  graph->clear();
  for (int i = 0; i < FLAGS_batch_size; i++) {
    std::copy(one_tree.begin(), one_tree.end(), graph->begin() + i*(2*FLAGS_tree_size-1));
  }
}

//void RandomTree(vector<int>* graph) {
   
//}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, 2*FLAGS_tree_size-1}, "CPU");//p0
  Sym vertex   = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, 2*FLAGS_tree_size-1});//p1

  TreeFCModel model(graph, vertex);
  Sym graph_output = model.Output();
  Sym loss = graph_output.Reduce_sum();

  Sym train      = loss.Optimizer({}, FLAGS_lr);
  Sym perplexity = loss.Reduce_mean();

  //Session sess;
  Session sess(OPT_BATCHING);
  vector<int>   graph_data(FLAGS_batch_size*(2*FLAGS_tree_size-1), -1);

  binaryTree(&graph_data);
  for (int j = 0; j < FLAGS_iters; j++) {
    sess.Run({train}, {{graph, graph_data.data()}});
    //sess.Run({loss}, {{graph, graph_data.data()}});
    //if (j % 10 == 0)
    LOG(INFO) << "\tIteration:\t" << j;
  }
  

  return 0;
}

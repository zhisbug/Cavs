#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch,      20,       "batch"                         );
DEFINE_int32 (hidden,     100,      "hidden size"                   );
DEFINE_int32 (tree,       128,      "the leaf number of the tree"   );
DEFINE_int32 (iters,      100,      "iterations"                    );
DEFINE_int32 (input_size,  21701,    "input size");
DEFINE_double(init_scale, 0.1f,     "init random scale of variables");
DEFINE_double(lr,         0.00001f, "learning rate"                 );


class TreeLSTMModel : public GraphSupport {
 public:
  TreeLSTMModel(const Sym& graph_ph, const Sym& vertex_ph) : 
    GraphSupport(graph_ph, vertex_ph) {
    embedding = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    W = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden* FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    U = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden}, Sym::Zeros());

    // prepare parameter symbols
    b_i = B.Slice(0, FLAGS_hidden);
    b_f = B.Slice(FLAGS_hidden, FLAGS_hidden);
    b_u = B.Slice(2 * FLAGS_hidden, FLAGS_hidden);
    b_o = B.Slice(3 * FLAGS_hidden, FLAGS_hidden);

    U_iou = U.Slice(0, 3 * FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, 3 * FLAGS_hidden});
    U_f   = U.Slice(3 * FLAGS_hidden * FLAGS_hidden, FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, FLAGS_hidden});
  }

  void Node() override {
    Sym left = Gather(0, {2 * FLAGS_hidden});
    Sym right = Gather(1, {2 * FLAGS_hidden});
    Sym h_l, c_l, h_r, c_r;
    tie(h_l, c_l) = left.Split2();
    tie(h_r, c_r) = right.Split2();
    Sym h_lr = h_l + h_r;
    
    // Pull the input word
    Sym x = Pull(0, {1});
    x = x.EmbeddingLookup(embedding.Mirror());

    // layout: i, o, u, f
    // start computation
    // xW is 1 x 4*FLAGS_hidden
    Sym xW = Sym::MatMul(x, W.Reshape({FLAGS_hidden, 4 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 4});
    Sym xW_i, xW_o, xW_u, xW_f;
    tie(xW_i, xW_o, xW_u, xW_f) = xW.Split4();
    
    // hU_iou is 1 x 3*FLAGS_hidden
    Sym hU_iou = Sym::MatMul(h_lr.Reshape({1, FLAGS_hidden}), U_iou.Mirror()).Reshape({FLAGS_hidden * 3});
    Sym hU_i, hU_o, hU_u;
    tie(hU_i, hU_o, hU_u) = hU_iou.Split3();

    // forget gate for every child
    Sym hU_fl = Sym::MatMul(h_l.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});
    Sym hU_fr = Sym::MatMul(h_r.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});

    // Derive i, f_l, f_r, o, u
    Sym i = (xW_i + hU_i + b_i.Mirror()).Sigmoid();
    Sym o = (xW_o + hU_o + b_o.Mirror()).Sigmoid();
    Sym u = (xW_u + hU_u + b_u.Mirror()).Tanh();

    Sym f_l = (xW_f + hU_fl + b_f.Mirror()).Sigmoid();
    Sym f_r = (xW_f + hU_fr + b_f.Mirror()).Sigmoid();

    Sym c = i * u + f_l * c_l + f_r * c_r;
    Sym h = o * Sym::Tanh(c.Mirror());

    Scatter(Sym::Concat({h.Mirror(), c.Mirror()}));
    Push(h.Mirror());
  }

 private:
  Sym W, U, B;
  Sym embedding;
  Sym b_i;
  Sym b_f;
  Sym b_u;
  Sym b_o;
            
  Sym U_iou;
  Sym U_f;
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
  for (int width = FLAGS_tree; width > 1; width >>= 1) {
    count += width;
    for (int i = 0; i < width; i++) {
      one_tree.push_back(i/2+count);
    }
  }
  one_tree.push_back(-1);
  CHECK(one_tree.size() == 2*FLAGS_tree-1);

  graph->clear();
  for (int i = 0; i < FLAGS_batch; i++) {
    std::copy(one_tree.begin(), one_tree.end(), graph->begin() + i*(2*FLAGS_tree-1));
  }
}

//void RandomTree(vector<int>* graph) {
   
//}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, 2*FLAGS_tree-1}, "CPU");//p0
  Sym vertex   = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, 2*FLAGS_tree-1});//p1

  TreeLSTMModel model(graph, vertex);
  Sym graph_output = model.Output();
  Sym loss = graph_output.Reduce_sum();

  Sym train      = loss.Optimizer({}, FLAGS_lr);
  Sym perplexity = loss.Reduce_mean();

  //Session sess;
  Session sess(OPT_BATCHING);
  vector<int>   graph_data(FLAGS_batch*(2*FLAGS_tree-1), -1);

  binaryTree(&graph_data);
  for (int j = 0; j < FLAGS_iters; j++) {
    sess.Run({train}, {{graph, graph_data.data()}});
    //sess.Run({loss}, {{graph, graph_data.data()}});
    //if (j % 10 == 0)
    LOG(INFO) << "\tIteration:\t" << j;
  }
  

  return 0;
}

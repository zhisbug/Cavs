#ifndef CAVS_MIDEND_BATCHING_WEIGHT_UPDATER_H_
#define CAVS_MIDEND_BATCHING_WEIGHT_UPDATER_H_

#include "cavs/midend/node.h"

#include <list>

namespace midend {

class BatchingWeightUpdater {
 public:
  BatchingWeightUpdater(std::list<Node*>* nodes, std::list<Node*>* finalize_node) {
    CHECK(!nodes->empty());
    CHECK(finalize_node->empty());
    
    Scope* s = nodes->front()->scope();
    std::list<Node*> updater;
    for (auto iter = nodes->begin(); iter != nodes->end(); ) {
      CHECK((*iter)->scope() == s);
      bool cond0 = ((*iter)->output_size() == 1);
      bool cond1 = (*iter)->output(0)->isGradient();
      if (cond0 && cond1) {
        Edge* backward_e = (*iter)->output(0);
        Edge* forward_e = s->FindEdge(GetOriginName(backward_e->name()));  
        CHECK(forward_e);
        bool cond2 = !forward_e->IsBatchEnabled();
        if (cond2) {
          updater.push_back(*iter);
          finalize_node->push_back(*iter);
          nodes->erase(iter++);
          continue;
        }
      }
      iter++;
    }

    for (Node* n : *finalize_node) {
      bool cond0 = (n->output(0)->scope() == n->scope());
      if (cond0) {
        for (Node* pn : n->output(0)->dst(true)) {
          CHECK(std::find(finalize_node->begin(), finalize_node->end(), pn) !=
                          finalize_node->end());
        }
      }
    }
  }
};

} //namespace midend 

#endif


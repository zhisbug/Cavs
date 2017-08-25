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
    std::set<Node*> blacklist;
    for (auto iter = nodes->begin(); iter != nodes->end(); ) {
      CHECK((*iter)->scope() == s);
      bool isSingleOutput = ((*iter)->output_size() == 1);
      bool isBackwardAndNotInGatherPullPath = false;
      for (Edge* e : (*iter)->output()) {
        if (e->isGradient()) {
          Edge* forward_e     = s->FindEdge(GetOriginName(e->name()));  
          CHECK(forward_e);
          if (!forward_e->IsBatchEnabled()) {
            isBackwardAndNotInGatherPullPath = true; 
            break;
          }
        } 
      }

      VLOG(V_DEBUG) << (*iter)->debug_info();
      VLOG(V_DEBUG) << isSingleOutput << "\t" << isBackwardAndNotInGatherPullPath;

      if (blacklist.find(*iter) != blacklist.end()) {
        for (Edge* edge : (*iter)->output()) {
          if (edge->scope() == (*iter)->scope()) {
            for (Node* pnode : edge->dst(true)) {
              blacklist.insert(pnode);
            }
          }
        }
      }else {
        if (isBackwardAndNotInGatherPullPath) {
          if (isSingleOutput) {
            finalize_node->push_back(*iter);
            nodes->erase(iter++);
            continue;
          }else {
            blacklist.insert(*iter);
            for (Edge* edge : (*iter)->output()) {
              if (edge->scope() == (*iter)->scope()) {
                for (Node* pnode : edge->dst(true)) {
                  blacklist.insert(pnode);
                }
              }
            }
          }
        }
      }
      iter++;
    }

    for (Node* n : *finalize_node) {
      CHECK(n->output_size() == 1);
      bool cond0 = (n->output(0)->scope() == n->scope());
      if (cond0) {
        for (Node* pn : n->output(0)->dst(true)) {
          CHECK(std::find(finalize_node->begin(), finalize_node->end(), pn) !=
                          finalize_node->end());
        }
      }
    }

    for (auto *n : *finalize_node) {
      VLOG(V_DEBUG) << n->debug_info();
    }
    CHECK(!blacklist.empty());
    for (auto *n : blacklist) {
      VLOG(V_DEBUG) << n->debug_info();
    }
    //LOG(FATAL) << "here";
  }
};

} //namespace midend 

#endif


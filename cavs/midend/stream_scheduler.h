#ifndef CAVS_MIDEND_STREAM_SCHEDULER_H_
#define CAVS_MIDEND_STREAM_SCHEDULER_H_

#include "cavs/midend/node.h"
#include "cavs/midend/edge.h"
#include "cavs/util/stream_event_handle_pool.h"

#include <unordered_map>
#include <list>
#include <vector>

namespace midend {

class StreamScheduler {
 public:
  StreamScheduler(std::vector<Statement*>* stmts, const std::vector<std::vector<int>>& dependency) {
    CHECK(dependency.size() == stmts->size());
    std::vector<int> stream_ids(stmts->size(), -1);
    std::vector<int> event_ids(stmts->size(), -1);
    std::vector<std::vector<int>> input_event_ids(stmts->size());
    std::vector<bool> sync_me(stmts->size(), false);
    //0) set the stream of me //initialization
    //1) set the event of me 
    //2) set the stream of father
    //3) set the input event of father
    for (int id = 0; id < dependency.size(); id++) {
      if (dependency[id].size() == 1) {
        if (stream_ids[id] == -1) {
          stream_ids[id] = StreamEventHandlePool::GenNewStreamID();
        }
        int pid = dependency[id][0];
        CHECK(pid > id);
        stream_ids[pid] = stream_ids[id];
      }else if (dependency[id].size() > 1) {
        if (stream_ids[id] == -1) {
          stream_ids[id] = StreamEventHandlePool::GenNewStreamID();
        }
        CHECK(event_ids[id] == -1);
        event_ids[id] = StreamEventHandlePool::GenNewEventID();
        
        bool reuse_stream = false;
        for (int pid : dependency[id]) {
          CHECK(pid > id);
          if (stream_ids[pid] == -1 && !reuse_stream) {
            reuse_stream = true; 
            stream_ids[pid] = stream_ids[id];
          }else {
            CHECK(event_ids[id] != -1);
            input_event_ids[pid].push_back(event_ids[id]); 
          }
        }
      }else {
        CHECK(stream_ids[id] != -1);
        sync_me[id] = true;
      }
    }
    VLOG(V_DEBUG) << "Streamming info " << stream_ids[0];

    for (int id = 0; id < stmts->size(); id++) {
      CHECK(stream_ids[id] != -1);
      if (stmts->at(id)->type() == Statement::EXPR) {
        ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
        es->GetContext()->SetStreamId(stream_ids[id]);
        for (int input_eid : input_event_ids[id]) {
          es->GetContext()->AddInputEventId(input_eid);
        }
        if (sync_me[id])
          es->GetContext()->SetSyncMe();
      }
    }

    for (int id = 0; id < stmts->size(); id++) {
      VLOG(V_DEBUG) << "Streamming info " << stream_ids[id];
    }
  }

  static void DependencyExtractor(std::vector<std::vector<int>>* dependency, const std::list<Node*>& nodes) {
    dependency->clear();
    std::unordered_map<Node*, int> node2idx;
    auto iter = nodes.begin();
    for (int i = 0; i < nodes.size(); i++, iter++) {
      CHECK(node2idx.find(*iter) == node2idx.end());
      node2idx[*iter] = i; 
    }

    iter = nodes.begin();
    for (int i = 0; i < nodes.size(); i++, iter++) {
      for (Edge* edge : (*iter)->output()) {
        if (edge->scope() == (*iter)->scope()) {
          for (Node* pnode : edge->dst(true)) {
            CHECK(node2idx.find(pnode) == node2idx.end());
            dependency->at(i).push_back(node2idx.at(pnode));
          }
        }
      }
    }
  }
};

} //namespace midend 

#endif

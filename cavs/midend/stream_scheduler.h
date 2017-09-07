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
  //1: not on gather-to-scatter but on to-scatter
  //2: on gather-to-scatter
  //3: not on to-scatter(on to-push, all the other nodes)
  StreamScheduler(std::vector<Statement*>* stmts, const std::list<Node*>& nodes) {
    std::vector<int> label(stmts->size(), 0);
    int anchor[2] = {-1};
    {
      //to-scatter
      //it is DFS actually, but considering the fusion optimization which breaks the chain
      //we use the reverse of the typological sorted order
      std::set<Edge*> on_scatter_path_edge;
      auto riter = nodes.rbegin();
      for (int i = nodes.size()-1; i >= 0; i--, riter++) {
        if ((*riter)->name() == "Scatter") {
          label[i] = 1;
          for (Edge* edge : (*riter)->input()) {
            CHECK(edge->scope() == (*riter)->scope());
            CHECK(on_scatter_path_edge.find(edge) == on_scatter_path_edge.end());
            on_scatter_path_edge.insert(edge);
          }
        }else {
          for (Edge* edge : (*riter)->output()) {
            if (on_scatter_path_edge.find(edge) != on_scatter_path_edge.end()) {
              for (Edge* cedge : (*riter)->input()) {
                CHECK(on_scatter_path_edge.find(cedge) == on_scatter_path_edge.end());
                on_scatter_path_edge.insert(cedge);
              }
              label[i] = 1; 
              break;
            }
            label[i] = 3;
          }
        }
      }

      //the first label which equals 3 is the parting line of stream 2 and stream 3
      for (int i = 0; i < nodes.size(); i++) {
        if (label[i] == 3) {
          anchor[1] = i;
          break;
        }
      }
    }

    {
      //gather-to-
      //it is a reverse of DFS actually
      //but again we have to consider the funsion optimization
      std::set<Edge*> on_gather_to_path_edge;
      auto iter = nodes.begin();
      for (int i = 0; i < nodes.size(); i++, iter++) {
        if ((*iter)->name() == "Gather") {
          CHECK(label[i] == 1);
          label[i] = 2;
          for (Edge* edge : (*iter)->output()) {
            CHECK(edge->scope() == (*iter)->scope());
            CHECK(on_gather_to_path_edge.find(edge) == on_gather_to_path_edge.end());
            on_gather_to_path_edge.insert(edge);
          }
        }else {
          bool on_path = false;
          for (Edge* edge : (*iter)->input()) {
            if (on_gather_to_path_edge.find(edge) != on_gather_to_path_edge.end()) {
              for (Edge* pedge : (*iter)->output()) {
                CHECK(on_gather_to_path_edge.find(pedge) == on_gather_to_path_edge.end());
                on_gather_to_path_edge.insert(pedge); 
              }
              CHECK(label[i] == 1 || label[i] == 3);
              if (label[i] == 1)  label[i] = 2;
              on_path = true;
              break;
            }
          }

          //the first node which depends on a node in stream 1 is the anchor of stream 1 and 2
          if (on_path && anchor[0] == -1) {
            for (Edge* edge : (*iter)->input()) {
              if (on_gather_to_path_edge.find(edge) == on_gather_to_path_edge.end()) {
                anchor[0] = i; 
                break;
              }
            }
          }
        }
      }
    }

    std::vector<int> stream_ids(3, -1);
    for (int id = 0; id < 3; id++) {
      stream_ids[id] = StreamEventHandlePool::GenNewStreamID();
    }
    std::vector<int> event_ids(2, -1);
    for (int id = 0; id < 2; id++) {
      event_ids[id] = StreamEventHandlePool::GenNewEventID();
    }

    std::vector<Statement*> reordered_stmts;
    for (int label_id = 1; label_id <= 3; label_id++) {
      for (int id = 0; id < stmts->size(); id++) {
        CHECK(label[id] >= 0 && label[id] <= 3);
        if (label[id] == label_id) {
          reordered_stmts.push_back(stmts->at(id));
          ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
          es->GetContext()->SetStreamId(stream_ids[label_id]);
        }
      }
      if (label_id == 1 || label_id == 2) {
        ExprStatement* es = dynamic_cast<ExprStatement*>(reordered_stmts.back());
        es->GetContext()->SetEventRecord(event_ids[label_id-1]);
      }
    }

    for (int i = 0; i < 2; i++) {
      Statement* anchor_stmt = stmts->at(anchor[i]);
      dynamic_cast<ExprStatement*>(anchor_stmt)->GetContext()->SetWaitForEventId(event_ids[i]);
    }
    CHECK(stmts->size() == reordered_stmts.size());
    *stmts = std::move(reordered_stmts);
  }

  
  //StreamScheduler(std::vector<Statement*>* stmts, const std::list<Node*>& nodes) {
    //CHECK(dependency.size() == stmts->size());
    //std::vector<int> stream_ids(stmts->size(), -1);
    //std::vector<int> event_ids(stmts->size(), -1);
    //std::vector<std::vector<int>> input_event_ids(stmts->size());
    //std::vector<bool> sync_me(stmts->size(), false);
    ////0) set the stream of me //initialization
    ////1) reuse the stream for father
    ////2) set the event of me(if needed) and make it the input event of father
    //for (int id = 0; id < dependency.size(); id++) {
      ////step 0
      //if (stream_ids[id] == -1) {
        //stream_ids[id] = StreamEventHandlePool::GenNewStreamID();
      //}
      //if (dependency[id].size() == 1) {
        //int pid = dependency[id][0];
        //CHECK(pid > id);
        //stream_ids[pid] = stream_ids[id];
      //}else if (dependency[id].size() > 1) {
        //CHECK(event_ids[id] == -1);
        //event_ids[id] = StreamEventHandlePool::GenNewEventID();
        //bool reuse_stream = false;
        //for (int pid : dependency[id]) {
          //CHECK(pid > id);
          //if (stream_ids[pid] == -1 && !reuse_stream) {
            //reuse_stream = true; 
            //stream_ids[pid] = stream_ids[id];
          //}else {
            //CHECK(event_ids[id] != -1);
            //input_event_ids[pid].push_back(event_ids[id]); 
          //}
        //}
      //}else {
        //CHECK(stream_ids[id] != -1) << id;
        //sync_me[id] = true;
      //}
    //}
    //VLOG(V_DEBUG) << "Streamming info " << stream_ids[0];

    //for (int id = 0; id < stmts->size(); id++) {
      //CHECK(stream_ids[id] != -1);
      //if (stmts->at(id)->type() == Statement::EXPR) {
        //ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
        //es->GetContext()->SetStreamId(stream_ids[id]);
        //for (int input_eid : input_event_ids[id]) {
          //es->GetContext()->AddInputEventId(input_eid);
        //}
        //if (sync_me[id])
          //es->GetContext()->SetSyncMe();
      //}
    //}

    //for (int id = 0; id < stmts->size(); id++) {
      //VLOG(V_DEBUG) << "Streamming info " << stream_ids[id];
    //}
  //}

  //static void DependencyExtractor(std::vector<std::vector<int>>* dependency, const std::list<Node*>& nodes) {
    //dependency->clear();
    //std::unordered_map<Node*, int> node2idx;
    //auto iter = nodes.begin();
    //for (int i = 0; i < nodes.size(); i++, iter++) {
      //CHECK(node2idx.find(*iter) == node2idx.end());
      //node2idx[*iter] = i; 
    //}

    //iter = nodes.begin();
    //for (int i = 0; i < nodes.size(); i++, iter++) {
      //for (Edge* edge : (*iter)->output()) {
        //if (edge->scope() == (*iter)->scope()) {
          //for (Node* pnode : edge->dst(true)) {
            ////CHECK(node2idx.find(pnode) != node2idx.end());
            ////we loose this constraint because batchweightupdater may remove some nodes in the scope
            ////but, these node must be the last nodes and will not break the dependency chain.
            //if (node2idx.find(pnode) == node2idx.end()) continue;
            //dependency->at(i).push_back(node2idx.at(pnode));
          //}
        //}
      //}
    //}
  //}
};

} //namespace midend 

#endif

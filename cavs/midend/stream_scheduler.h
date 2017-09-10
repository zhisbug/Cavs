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
    CHECK(stmts->size() == nodes.size());
    //record stream in edges
    //record event  in nodes
    std::set<int> using_stream_ids;
    std::set<int> used_stream_ids;
    std::set<Edge*> inherited_stream;
    //null stream will not be recorded
    std::unordered_map<Edge*, int> edge_stream_record;
    std::unordered_map<Edge*, int> edge_to_last_src_nodeid;
    std::unordered_map<Edge*, int> edge_dealed_times;
    std::unordered_map<int, int> nodeid_to_eventid;

    auto iter = nodes.begin();
    for (int id = 0; id < nodes.size(); id++, iter++) {
      CHECK(stmts->at(id)->type() == Statement::EXPR);
      int stream_id = -1;
      if ((*iter)->name() != "Gather" &&
          (*iter)->name() != "Scatter") {
        //bool assigned = false;
        CHECK(stmts->at(id)->type() == Statement::EXPR);
        ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
        for (Edge* edge : (*iter)->input()) {
          if (edge->scope() == (*iter)->scope()) {
            if (edge_dealed_times.find(edge) == edge_dealed_times.end())
              edge_dealed_times[edge] = 0;

            if (stream_id == -1) {
              //inherit logic
              if (inherited_stream.find(edge) == inherited_stream.end() &&
                  edge_stream_record.find(edge) != edge_stream_record.end()) {
                stream_id = edge_stream_record[edge];
                inherited_stream.insert(edge);
                VLOG(V_DEBUG) << (*iter)->name() << ": Inheriting stream[ " << stream_id << "]";
              }
              
              if (stream_id == -1){//fork logic
                //get an id first
                if (!used_stream_ids.empty()) {
                  stream_id = *used_stream_ids.begin();
                  CHECK(using_stream_ids.find(stream_id) == using_stream_ids.end())
                    << stream_id;
                  used_stream_ids.erase(stream_id);
                  VLOG(V_DEBUG) << (*iter)->name() << ": Swapping stream[ "
                                << stream_id << "] in used";
                }else {
                  stream_id = StreamEventHandlePool::GenNewStreamID();
                  CHECK(used_stream_ids.find(stream_id) == used_stream_ids.end());
                }
              }
              CHECK(stream_id != -1);
              using_stream_ids.insert(stream_id);
              VLOG(V_DEBUG) << (*iter)->name() << ": Inserting stream[ "
                            << stream_id << "] in using";
              es->GetContext()->SetStreamId(stream_id);
            }

            CHECK(stream_id != -1); 
            if (edge_stream_record.find(edge) != edge_stream_record.end() &&
                stream_id != edge_stream_record[edge]) {
              //sync then
              CHECK(edge_to_last_src_nodeid.find(edge) != edge_to_last_src_nodeid.end());
              int last_node_id = edge_to_last_src_nodeid[edge];
              ExprStatement* prev_stmt = dynamic_cast<ExprStatement*>(stmts->at(last_node_id));
              int event_id;
              if (nodeid_to_eventid.find(last_node_id) == nodeid_to_eventid.end()) {
                event_id = StreamEventHandlePool::GenNewEventID();
                nodeid_to_eventid[id] = event_id;
              }else {
                event_id = nodeid_to_eventid[last_node_id];
              }
              prev_stmt->GetContext()->SetEventRecord(event_id);
              es->GetContext()->SetWaitForEventId(event_id);
            }

            if (edge_dealed_times.find(edge) != edge_dealed_times.end() &&
                ++edge_dealed_times[edge] == edge->dst_size(true) &&
                edge_stream_record.find(edge) != edge_stream_record.end() &&
                stream_id != edge_stream_record[edge]) {
              VLOG(V_DEBUG) << (*iter)->name() << ": Erasing stream["
                            << edge_stream_record[edge]
                            << "] in using and swapping it to used";
              using_stream_ids.erase(edge_stream_record[edge]);
              used_stream_ids.insert(edge_stream_record[edge]);
            }
          }
        }

        if (stream_id == -1) {
          if (!used_stream_ids.empty()) {
            stream_id = *used_stream_ids.begin();
            CHECK(using_stream_ids.find(stream_id) == using_stream_ids.end()) 
              << stream_id;
            used_stream_ids.erase(stream_id);
          }else {
            stream_id = StreamEventHandlePool::GenNewStreamID();
          }
          CHECK(stream_id != -1);
          using_stream_ids.insert(stream_id);
          es->GetContext()->SetStreamId(stream_id);
        }

        for (Edge* edge : (*iter)->output()) {
          if (edge->scope() == (*iter)->scope()) {
            if (edge_stream_record.find(edge) == edge_stream_record.end()) {
              edge_stream_record[edge] = stream_id; 
            }else {
              CHECK(edge_stream_record[edge] == stream_id);
              //otherwise it is a write-write collision
            }
            edge_to_last_src_nodeid[edge] = id;
          }
        }
      }

      LOG(INFO) << "Streaming policy: " << (*iter)->name()
                << " ==> Stream[" << stream_id << "]";
    }
    //LOG(FATAL) << "here";

    //CHECK(stmts->at(node_id_prev)->type() == Statement::EXPR);
    //ExprStatement* record_es = dynamic_cast<ExprStatement*>(stmts->at(node_id_prev));
    //int new_event_id = StreamEventHandlePool::GenNewEventID();
    //record_es->GetContext()->SetEventRecord(new_event_id);
    //CHECK(stmts->at(node_id_next)->type() == Statement::EXPR);
    //ExprStatement* wait_for_stmt = dynamic_cast<ExprStatement*>(stmts->at(node_id_next));
    //wait_for_stmt->GetContext()->SetWaitForEventId(new_event_id);
  }
};

} //namespace midend 

#endif

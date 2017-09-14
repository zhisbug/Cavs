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
  //StreamScheduler(std::vector<Statement*>* stmts, const std::list<Node*>& nodes) {
    //CHECK(stmts->size() == nodes.size());
    ////record stream in edges
    ////record event  in nodes
    //std::set<int> using_stream_ids;
    //std::set<int> used_stream_ids;
    //std::set<Edge*> inherited_stream;
    ////null stream will not be recorded
    //std::unordered_map<Edge*, int> edge_stream_record;
    //std::unordered_map<Edge*, int> edge_to_last_src_nodeid;
    //std::unordered_map<Edge*, int> edge_dealed_times;
    //std::unordered_map<int, int> nodeid_to_eventid;

    //auto iter = nodes.begin();
    //for (int id = 0; id < nodes.size(); id++, iter++) {
      //CHECK(stmts->at(id)->type() == Statement::EXPR);
      //int stream_id = -1;
      //if ((*iter)->name() != "Gather" &&
          //(*iter)->name() != "Scatter") {
        ////bool assigned = false;
        //CHECK(stmts->at(id)->type() == Statement::EXPR);
        //ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
        //for (Edge* edge : (*iter)->input()) {
          //if (edge->scope() == (*iter)->scope()) {
            //if (edge_dealed_times.find(edge) == edge_dealed_times.end())
              //edge_dealed_times[edge] = 0;

            //if (stream_id == -1) {
              ////inherit logic
              //if (inherited_stream.find(edge) == inherited_stream.end() &&
                  //edge_stream_record.find(edge) != edge_stream_record.end()) {
                //stream_id = edge_stream_record[edge];
                //inherited_stream.insert(edge);
                //VLOG(V_DEBUG) << (*iter)->name() << ": Inheriting stream[ " << stream_id << "]";
              //}
              
              //if (stream_id == -1){//fork logic
                ////get an id first
                //if (!used_stream_ids.empty()) {
                  //stream_id = *used_stream_ids.begin();
                  //CHECK(using_stream_ids.find(stream_id) == using_stream_ids.end())
                    //<< stream_id;
                  //used_stream_ids.erase(stream_id);
                  //VLOG(V_DEBUG) << (*iter)->name() << ": Swapping stream[ "
                                //<< stream_id << "] in used";
                //}else {
                  //stream_id = StreamEventHandlePool::GenNewStreamID();
                  //CHECK(used_stream_ids.find(stream_id) == used_stream_ids.end());
                //}
              //}
              //CHECK(stream_id != -1);
              //using_stream_ids.insert(stream_id);
              //VLOG(V_DEBUG) << (*iter)->name() << ": Inserting stream[ "
                            //<< stream_id << "] in using";
              //es->GetContext()->SetStreamId(stream_id);
            //}

            //CHECK(stream_id != -1); 
            //if (edge_stream_record.find(edge) != edge_stream_record.end() &&
                //stream_id != edge_stream_record[edge]) {
              ////sync then
              //CHECK(edge_to_last_src_nodeid.find(edge) != edge_to_last_src_nodeid.end());
              //int last_node_id = edge_to_last_src_nodeid[edge];
              //ExprStatement* prev_stmt = dynamic_cast<ExprStatement*>(stmts->at(last_node_id));
              //int event_id;
              //if (nodeid_to_eventid.find(last_node_id) == nodeid_to_eventid.end()) {
                //event_id = StreamEventHandlePool::GenNewEventID();
                //nodeid_to_eventid[id] = event_id;
              //}else {
                //event_id = nodeid_to_eventid[last_node_id];
              //}
              //prev_stmt->GetContext()->SetEventRecord(event_id);
              //es->GetContext()->SetWaitForEventId(event_id);
            //}

            //if (edge_dealed_times.find(edge) != edge_dealed_times.end() &&
                //++edge_dealed_times[edge] == edge->dst_size(true) &&
                //edge_stream_record.find(edge) != edge_stream_record.end() &&
                //stream_id != edge_stream_record[edge]) {
              //VLOG(V_DEBUG) << (*iter)->name() << ": Erasing stream["
                            //<< edge_stream_record[edge]
                            //<< "] in using and swapping it to used";
              //using_stream_ids.erase(edge_stream_record[edge]);
              //used_stream_ids.insert(edge_stream_record[edge]);
            //}
          //}
        //}

        //if (stream_id == -1) {
          //if (!used_stream_ids.empty()) {
            //stream_id = *used_stream_ids.begin();
            //CHECK(using_stream_ids.find(stream_id) == using_stream_ids.end()) 
              //<< stream_id;
            //used_stream_ids.erase(stream_id);
          //}else {
            //stream_id = StreamEventHandlePool::GenNewStreamID();
          //}
          //CHECK(stream_id != -1);
          //using_stream_ids.insert(stream_id);
          //es->GetContext()->SetStreamId(stream_id);
        //}

        //for (Edge* edge : (*iter)->output()) {
          //if (edge->scope() == (*iter)->scope()) {
            //if (edge_stream_record.find(edge) == edge_stream_record.end()) {
              //edge_stream_record[edge] = stream_id; 
            //}else {
              //CHECK(edge_stream_record[edge] == stream_id);
              ////otherwise it is a write-write collision
            //}
            //edge_to_last_src_nodeid[edge] = id;
          //}
        //}
      //}

      //LOG(INFO) << "Streaming policy: " << (*iter)->name()
                //<< " ==> Stream[" << stream_id << "]";
    //}
  //}
  
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
            VLOG(V_DEBUG) << "Inserting " << edge->name();
          }
        }else {
          for (Edge* edge : (*riter)->output()) {
            if (on_scatter_path_edge.find(edge) != on_scatter_path_edge.end()) {
              for (Edge* cedge : (*riter)->input()) {
                VLOG(V_DEBUG) << "Want to insert " << cedge->name();
                //CHECK(on_scatter_path_edge.find(cedge) == on_scatter_path_edge.end());
                on_scatter_path_edge.insert(cedge);
                VLOG(V_DEBUG) << "Inserting " << cedge->name();
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
      
      //first we conclude the specifical edge: from global variable and then mirror/reshape
      //these edges are not dependent on any nodes
      std::set<Edge*> independent_edges;
      auto iter = nodes.begin();
      for (int i = 0; i < nodes.size(); i++, iter++) {
        for (Edge* edge : (*iter)->input()) {
          if (edge->scope() != (*iter)->scope()) {
            independent_edges.insert(edge); 
          }
        }

        std::vector<std::string> non_dependent_ops = {"Mirror", "Reshape"};
        if (std::find(non_dependent_ops.begin(), non_dependent_ops.end(), (*iter)->name())
            != non_dependent_ops.end()) {
          CHECK((*iter)->output_size() == 1); 
          independent_edges.insert((*iter)->output(0));
        }
      }

      if (VLOG_IS_ON(V_DEBUG)) {
        for (auto* edge : independent_edges)
          VLOG(V_DEBUG) << "independent egde: " << edge->name();
      }

      //auto iter = nodes.begin();
      //for (int id = 0; id < nodes.size(); id++, iter++) {
        //CHECK(stmts->at(id)->type() == Statement::EXPR);
        
      std::set<Edge*> on_gather_to_path_edge;
      iter = nodes.begin();
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
                VLOG(V_DEBUG) << pedge->name() << edge->name();
                VLOG(V_DEBUG) << (*iter)->debug_info();
                //we loose this constraint because of the backward of split function
                //several operators output to the same tensor
                //CHECK(on_gather_to_path_edge.find(pedge) == on_gather_to_path_edge.end());
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
              if (on_gather_to_path_edge.find(edge) == on_gather_to_path_edge.end() &&
                  independent_edges.find(edge) == independent_edges.end()) {
                VLOG(V_DEBUG) << edge->name() << (*iter)->debug_info();
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

    {
      //debug stream
      auto iter = nodes.begin();
      for (int i = 0; i < nodes.size(); i++, iter++) {
        VLOG(V_DEBUG) << "Stream[" << label[i] << "]: " << (*iter)->name();
      }
      //LOG(FATAL) << "here";
    }

    std::vector<Statement*> reordered_stmts;
    for (int label_id = 1; label_id <= 3; label_id++) {
      for (int id = 0; id < stmts->size(); id++) {
        CHECK(label[id] >= 0 && label[id] <= 3);
        if (label[id] == label_id) {
          reordered_stmts.push_back(stmts->at(id));
          ExprStatement* es = dynamic_cast<ExprStatement*>(stmts->at(id));
          es->GetContext()->SetStreamId(stream_ids[label_id-1]);
        }
      }
      if (label_id == 1 || label_id == 2) {
        ExprStatement* es = dynamic_cast<ExprStatement*>(reordered_stmts.back());
        es->GetContext()->SetEventRecord(event_ids[label_id-1]);
        VLOG(V_DEBUG) << "event record: [" << label_id-1 << "]: " << es->debug_info();
      }
    }

    {
      //debug statement order
      for (int i = 0; i < nodes.size(); i++) {
        CHECK(reordered_stmts[i]->type() == Statement::EXPR);
        ExprStatement* es = dynamic_cast<ExprStatement*>(reordered_stmts[i]);
        VLOG(V_DEBUG) << "execute[" << i << "]: " << es->debug_info();
      }
    }

    for (int i = 0; i < 2; i++) {
      ExprStatement* anchor_stmt = dynamic_cast<ExprStatement*>(stmts->at(anchor[i]));
      anchor_stmt->GetContext()->SetWaitForEventId(event_ids[i]);
        VLOG(V_DEBUG) << "stream( " << anchor_stmt->debug_info()
                      << ") wait for event: [" << event_ids[i] << "]";
    }
    //LOG(FATAL) << "here";
    CHECK(stmts->size() == reordered_stmts.size());
    *stmts = std::move(reordered_stmts);
  }
};

} //namespace midend 

#endif

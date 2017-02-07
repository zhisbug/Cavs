#include "cavs/midend/edge.h"

namespace midend {

void Edge::AddDst(Node* node) {
  dsts_.push_back(node); 
  if (node->isSink()) {
    CHECK(sink_ == NULL); 
    sink_ = node;
  }else {
    if (sink_) {
      sink_->RemoveInput(this);
      RemoveDst(node);
      sink_ = NULL;
    }
  }
}

} //namespace midend

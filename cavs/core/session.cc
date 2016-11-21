#include "cavs/core/session.h"

namespace cavs {

const Tensor* Session::GetTensor(const string& name) const {
    if (tensor_map.count(name) == 0)
        return NULL;
    else
        return tensor_map.at(name);
}


bool Session::InsertTensor(Tensor* t){
    if (tensor_map.count(t->name()) > 0)
        return false;
    else
        tensor_map[t->name()] = t;
    return true;
}


Session* simple_session() {
    static Session sess;
    return &sess;
}

} //namespace cavs

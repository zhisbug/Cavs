#include "session.h"

bool Session::InsertTensor(string name, Tensor* t){
    if (tensor_map.count(t) > 0)
        return false;
    else
        tensor_map[name] = t;
    return true;
}

Tensor* Session::GetTensor(string name){
    if (tensor_map.count(name) == 0)
        return NULL;
    else
        return tensor_map[name];
}

Session* simple_session() {
    static Session sess;
    return &sess;
}

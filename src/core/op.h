#ifndef OP_H_
#define OP_H_

using std::string;

class Op {
 public:
  explicit Op(const OpDef& def); 
  explicit Op(const vector<OpDef>& def); //for JIT
  virtual void Compute(OpContext*) = 0;
};

class OpContext {

};

#define REGISTER_OP_BUILDER(key, constructor)           \
    static OpRegister register__body__##name##(       \
        op_factory::key.LowerToString(),                \
            [](OpDef& def) -> Op* {                   \
                    return new constructor(def);            \
                });

namespace op_factory {


class OpRegister {
 public:
  typedef Op* (*Factory)(OpDef& def);

  OpRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

}
        
#endif

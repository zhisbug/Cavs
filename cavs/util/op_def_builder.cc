#include "cavs/util/op_def_builder.h"
#include "cavs/util/logging.h"

using std::string;
using std::vector;
using std::initializer_list;

OpDefBuilder& OpDefBuilder::Input(const string& input) {
  op_def_.add_input(input);
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(const vector<string>& inputs) {
  for (auto& s : inputs)
    op_def_.add_input(s);
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(const OpDef& def) {
  for (auto& inp : def.input())
    op_def_.add_input(inp);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(const string& output) {
  op_def_.add_output(output);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(const vector<string>& outputs) {
  for (auto& s : outputs)
    op_def_.add_output(s);
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(const OpDef& def) {
  for (auto& out : def.output())
    op_def_.add_output(out);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(const string& dev) {
  if (dev == "GPU")
    op_def_.set_device(GPU);
  else 
    op_def_.set_device(CPU);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(const DeviceType type) {
  op_def_.set_device(type);
  return *this;
}

OpDefBuilder& OpDefBuilder::Device(const OpDef& def) {
  op_def_.set_device(def.device());
  return *this;
}

bool OpDefBuilder::CheckValid() const {
  CHECK(op_def_.output_size() == op_def_.shape_size() ||
        op_def_.shape_size() == 0)
        << op_def_.DebugString();
  return true;
}

OpDefBuilder& OpDefBuilder::Shape(const vector<int>& shape) {
  op_def_.clear_shape();
  TensorShapeDef* shape_def = op_def_.add_shape();
  for (int dim : shape)
    shape_def->add_dim(dim);
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const TensorShapeDef& shape) {
  op_def_.clear_shape();
  *(op_def_.add_shape()) = shape;
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const vector<TensorShapeDef>& shapes) {
  op_def_.clear_shape();
  for (auto& s : shapes)
    *(op_def_.add_shape()) = s;
  return *this;
}

OpDefBuilder& OpDefBuilder::Shape(const OpDef& def) {
  *(op_def_.mutable_shape()) = def.shape();
  return *this;
}

OpDefBuilder& OpDefBuilder::Dtype(const DataType type) {
  op_def_.set_dtype(type);
}

OpDefBuilder& OpDefBuilder::Label(const string& label) {
  op_def_.set_label(label);
  return *this;
}

OpDefBuilder& OpDefBuilder::Attr(const OpDef& def) {
  *(op_def_.mutable_attr()) = def.attr();
  return *this;
}

OpDefBuilder& OpDefBuilder::Attr(const OpDef::AttrDef& def) {
  *(op_def_.add_attr()) = def;
  return *this;
}

OpDefBuilder& OpDefBuilder::Attr(const vector<OpDef::AttrDef>& def) {
  for (auto& d : def) {
    *(op_def_.add_attr()) = d;
  }
  return *this;
}

#define INSTANTIATE_SETATTR(T, fieldname)           \
  template <>                                       \
  OpDefBuilder& OpDefBuilder::AttrSingle<T>(        \
      const string& key, T value) {                 \
    for (auto& attr : *(op_def_.mutable_attr())) {  \
      if (attr.name() == key) {                     \
        LOG(WARNING) << "Rewriting " << key;        \
        attr.mutable_value()->                      \
          set_##fieldname(value);                   \
          return *this;                             \
      }                                             \
    }                                               \
    OpDef::AttrDef *attr = op_def_.add_attr();      \
    attr->set_name(key);                            \
    attr->mutable_value()->                         \
      set_##fieldname(value);                       \
    return *this;                                   \
  }                                                 \
  template <>                                       \
  OpDefBuilder& OpDefBuilder::AttrList<T>(          \
      const string& key, T value) {                 \
    for (auto& attr : *(op_def_.mutable_attr())) {  \
      if (attr.name() == key) {                     \
        attr.mutable_value()->                      \
          mutable_list()->add_##fieldname(value);   \
          return *this;                             \
      }                                             \
    }                                               \
    OpDef::AttrDef *attr = op_def_.add_attr();      \
    attr->set_name(key);                            \
    attr->mutable_value()->                         \
      mutable_list()->add_##fieldname(value);       \
    return *this;                                   \
  }                                                 \
  template <>                                       \
  OpDefBuilder& OpDefBuilder::AttrList<T>(          \
      const string& key, const vector<T> value) {   \
    for (auto& attr : *(op_def_.mutable_attr())) {  \
      if (attr.name() == key) {                     \
        auto* l = attr.mutable_value()->            \
          mutable_list();                           \
        for (auto t : value)                        \
          l->add_##fieldname(t);                    \
        return *this;                               \
      }                                             \
    }                                               \
    OpDef::AttrDef *attr = op_def_.add_attr();      \
    attr->set_name(key);                            \
    auto* l = attr->mutable_value()->               \
      mutable_list();                               \
    for (auto t : value)                            \
      l->add_##fieldname(t);                        \
    return *this;                                   \
  }

INSTANTIATE_SETATTR(float,  f)
INSTANTIATE_SETATTR(int,    i)
INSTANTIATE_SETATTR(bool,   b)
INSTANTIATE_SETATTR(string, s)

//void BuildConstantOpDef(OpDef* op_def, 
    //const string& output,
    //const TensorShapeDef& shape,
    //float val) {
  //OpDefBuilder("ConstOp")
    //.Output(output)
    //.Shape(shape)
    //.AttrSingle("init", val)
    //.Device("GPU")
    //.Finalize(op_def);
//}

//float GetConstFromConstantOp(const OpDef& def) {
  //for (auto& attr : def.attr()) {
    //if (attr.name() == "init") 
      //return attr.value().f();
  //}
  //LOG(FATAL) << "init value not found";
//}

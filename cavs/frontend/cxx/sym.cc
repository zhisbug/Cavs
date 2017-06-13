#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/c_api.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/util/logging.h"

#include <algorithm>
#include <iomanip>
#include <tuple>

using std::vector;
using std::unordered_map;

//void Sym::node::Finalize(OpDef* op_def) const {
  //op_def->set_name(op_name_);
  //for (const string& str: input_)
    //op_def->add_input(str);
  //for (const string& str: output_)
    //op_def->add_output(str);
  //op_def->set_dtype(DataType((int)type_));
  //op_def->set_label(label_);
  ////device
  //if (device_ == "GPU")
    //op_def->set_device(GPU);
  //else
    //op_def->set_device(CPU);
  ////shape
  //op_def->clear_shape();
  //TensorShapeDef* shape_def = op_def->add_shape();
  //for (auto dim : shape_)
    //shape_def->add_dim(dim);
//}

Sym::MODE Sym::mode_ = Sym::STATIC_SYM;
string Sym::func_name_;
unordered_map<string, FunctionDef> Sym::func_def_;

//Sym::Sym(const string& op_name,
         //const vector<string>& inputs, 
         //const C_Dtype type,
         //const string& label,
         //const string& device,
         //const vector<int>& shape,
         //const vector<OpDef::AttrDef>& attrs) {
  //static int id = 0;  
  //node_.reset(new node());
  //node_->op_name_ = op_name;
  //node_->output_.push_back(op_name + std::to_string(id++));
  //node_->input_ = inputs;
  //node_->type_ = type;
  //node_->label_ = label;
  //node_->shape_ = shape; 
  //node_->device_ = device; 

  //OpDef op_def;
  //node_->Finalize(&op_def);
  //for (auto& attr : attrs)
    //*(op_def.add_attr()) = attr;
  //string serial_def;
  //op_def.SerializeToString(&serial_def);
  //int *dim = NULL;
  //size_t dim_length;
  //C_AddNode(C_GetDefaultDG(),
      //serial_def.c_str(), serial_def.length(),
      //&dim, &dim_length);
  //node_->shape_.clear();
  //for (int i = 0; i < dim_length; i++)
    //node_->shape_.push_back(dim[i]);
  //free(dim);
  ////LOG(INFO) << op_def.DebugString();
//}

//Sym::Sym(const string& op_name,
    //const string& loss,
    //const vector<Sym>& variables,
    //const float lr,
    //const float clip,
    //const int iters,
    //const string& projection) {
  //CHECK(op_name == "Optimizer");
  //static int id = 0;
  //node_.reset(new node());
  //node_->op_name_ = op_name;
  //node_->output_.push_back(op_name + std::to_string(id++));
  //node_->input_ = {loss};

  //OpDef op_def;
  //node_->Finalize(&op_def);

  //if (variables.size()) {
    //OpDef::AttrDef* var_attr = op_def.add_attr();
    //var_attr->set_name("Vars");
    //OpDef::AttrType::ListValue* str_list
      //= var_attr->mutable_value()->mutable_list();
    //for (auto& sym : variables)
      //str_list->add_s(sym.output(0));
  //}
  //OpDef::AttrDef* solver_attr = op_def.add_attr();
  //solver_attr->set_name("Solver");
  //solver_attr->mutable_value()->set_s("SGD");
  //if (projection.length() > 0) {
    //OpDef::AttrDef* proj_attr = op_def.add_attr();
    //proj_attr->set_name("Projection");
    //proj_attr->mutable_value()->set_s(projection);
  //}
  //OpDef::AttrDef* lr_attr = op_def.add_attr();
  //lr_attr->set_name("learning_rate");
  //lr_attr->mutable_value()->set_f(lr);

  //if (clip != 0) {
    //OpDef::AttrDef* clip_attr = op_def.add_attr();
    //clip_attr->set_name("clip");
    //clip_attr->mutable_value()->set_f(clip);
  //}

  //OpDef::AttrDef* iters_attr = op_def.add_attr();
  //iters_attr->set_name("Iters");
  //iters_attr->mutable_value()->set_i(iters);

  //string serial_def;
  //op_def.SerializeToString(&serial_def);
  //C_OptimizeWithLoss(C_GetDefaultDG(),
    //serial_def.c_str(), serial_def.length());
//}

Sym::Sym(const OpDef& op_def) {
  node_.reset(new node_t());
  node_->op_def = op_def;

  static int id_ = 0;
  mutable_def()->add_output(op_name() + "_" + std::to_string(id_++));

  string serialization;
  def().SerializeToString(&serialization);
  if (op_name() != "Optimizer") {
    if (mode_ == STATIC_SYM) {
      int *dim = NULL;
      size_t dim_length;
      //C_AddNode(C_GetDefaultDG(),
          //serialization.c_str(), serialization.length(),
          //&dim, &dim_length);
      C_AddOp(serialization.c_str(), serialization.length(),
              &dim, &dim_length);
      mutable_def()->clear_shape();
      TensorShapeDef* shape = mutable_def()->add_shape();
      for (int i = 0; i < dim_length; i++)
        shape->add_dim(dim[i]);
      free(dim);
    }else {
      if (func_def_.find(func_name_) == func_def_.end()) {
        FunctionDef fdef; 
        fdef.set_name(func_name_);
        fdef.add_ops()->CopyFrom(def());
        func_def_.emplace(func_name_, fdef);
      }else {
        func_def_[func_name_].add_ops()->CopyFrom(def());
      }
    }
  }else {
    CHECK(mode_ != DYNAMIC_SYM);
    //C_OptimizeWithLoss(C_GetDefaultDG(),
      //serialization.c_str(), serialization.length());
    C_AddOptimizerOp(
      serialization.c_str(), serialization.length());
  }
}

template <>
Sym::Sym<float> (float c) {
  //OpDef::AttrDef attr;
  //attr.set_name("init");
  //attr.mutable_value()->set_f(c);
  //new (this)Sym("ConstOp", {}, C_FLOAT, "", "GPU", {1}, {attr});
  OpDef def = OpDefBuilder("ConstOp")
                .Dtype(DT_FLOAT)
                .Device("GPU")
                .Shape(vector<int>({1}))
                .AttrSingle("init", c)
                .Finalize();
  new (this)Sym(def);
}

Sym Sym::Variable(DataType type, const vector<int>& shape,
    const ATTRIBUTE& filler, string device) {
  CHECK(shape.size() > 0);
  OpDef def = OpDefBuilder("Variable")
                .Dtype(type)
                .Label(filler.first)
                .Device(device)
                .Shape(shape)
                .Attr(filler.second)
                .Finalize();
  LOG(INFO) << def.DebugString();
  //return Sym("Variable", {}, type, filler.first, device, shape, {filler.second});
  return Sym(def);
}

Sym Sym::Placeholder(DataType type, const vector<int>& shape,
    string device) {
  CHECK(shape.size() > 0);
  OpDef def = OpDefBuilder("Placeholder")
                .Dtype(type)
                .Device(device)
                .Shape(shape)
                .Finalize();
  //return Sym("Placeholder", {}, type, "", device, shape);
  return Sym(def);
}

Sym Sym::Constant(DataType type, float value, const vector<int>& shape,
      string device) {
  CHECK(shape.size() > 0);
  OpDef def = OpDefBuilder("Constant")
                .Dtype(type)
                .Device(device)
                .Shape(shape)
                .AttrSingle("init", value)
                .Finalize();
  //return Sym("Placeholder", {}, type, "", device, shape);
  return Sym(def);
}

Sym Sym::MnistInput(int batch, string source, string file, string device) {
  //OpDef::AttrDef batch_attr;
  //batch_attr.set_name("Batch");
  //batch_attr.mutable_value()->set_i(batch);
  //OpDef::AttrDef source_attr;
  //source_attr.set_name("Source");
  //source_attr.mutable_value()->set_s(source);
  //OpDef::AttrDef file_attr;
  //file_attr.set_name("ImageDir");
  //file_attr.mutable_value()->set_s(file);
  //return Sym("MnistInput", {}, C_FLOAT, "", device, {},
      //{batch_attr, source_attr, file_attr}); 
  OpDef def = OpDefBuilder("MnistInput")
                .Dtype(DT_FLOAT)
                .Device(device)
                .AttrSingle("Batch", batch)
                .AttrSingle("Source", source)
                .AttrSingle("ImageDir", file)
                .Finalize();
  return Sym(def);
}

Sym Sym::Data(DataType type, const vector<int>& shape,
    int batch, const ATTRIBUTE& reader, string device) {
  //OpDef::AttrDef batch_attr;
  //batch_attr.set_name("Batch");
  //batch_attr.mutable_value()->set_i(batch);
  //OpDef::AttrDef shape_attr;
  //shape_attr.set_name("Shape");
  //OpDef::AttrType::ListValue* lv = shape_attr.mutable_value()->mutable_list();
  //for (int s : shape) {
    //lv->add_i(s);
  //}
  //vector<OpDef::AttrDef> attrs = {batch_attr, shape_attr};
  //for (auto& attr : reader.second)
    //attrs.push_back(attr);
  //return Sym("Data", {}, C_FLOAT, reader.first, device, {}, attrs);
  OpDef def = OpDefBuilder("Data")
                .Dtype(DT_FLOAT)
                .Label(reader.first)
                .Device(device)
                .AttrSingle("Batch", batch)
                .AttrList("Shape", shape)
                .Attr(reader.second)
                .Finalize();
  return Sym(def);
}

Sym Sym::DDV(DataType type, const vector<int>& shape, int batch,
    const ATTRIBUTE& filler, string device) {
  //OpDef::AttrDef batch_attr;
  //batch_attr.set_name("Batch");
  //batch_attr.mutable_value()->set_i(batch);
  //OpDef::AttrDef shape_attr;
  //shape_attr.set_name("Shape");
  //OpDef::AttrType::ListValue* lv = shape_attr.mutable_value()->mutable_list();
  //for (int s : shape) {
    //lv->add_i(s);
  //}
  //vector<OpDef::AttrDef> attrs = {batch_attr, shape_attr};
  //for (auto& attr : filler.second)
    //attrs.push_back(attr);
  //return Sym("DDV", {}, type, filler.first, device, {}, attrs); 
  OpDef def = OpDefBuilder("DDV")
                .Dtype(type)
                .Label(filler.first)
                .Device(device)
                .AttrSingle("Batch", batch)
                .AttrList("Shape", shape)
                .Attr(filler.second)
                .Finalize();
  return Sym(def);
}

Sym Sym::Abs(const Sym& a, string device) {
  CHECK(a.output_size() == 1);
  //Sym s("Abs", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Abs")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Argmax(const Sym& a, int axis, string device) {
  CHECK(a.output_size() == 1);
  //OpDef::AttrDef attr;
  //attr.set_name("axis");
  //attr.mutable_value()->set_i(axis);
  //Sym s("Argmax", {a.node_->output_[0]},
        //a.node_->type_, "", device, {}, {attr});
  //return s;
  OpDef def = OpDefBuilder("Argmax")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Square(const Sym& a, string device) {
  CHECK(a.output_size() == 1);
  //Sym s("Square", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Square")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Reduce_mean(const Sym& a, string device) {
  CHECK(a.output_size() == 1);
  //Sym s("Reduce_mean", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Reduce_mean")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Reduce_sum(const Sym& a, string device) {
  CHECK(a.output_size() == 1);
  //Sym s("Reduce_sum", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Reduce_sum")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Maxpooling(const Sym& a,
    int HightWindow, int WidthWindow, string device) {
  //vector<OpDef::AttrDef> attrs;
  //{
    //OpDef::AttrDef attr;
    //attr.set_name("HightWindow");
    //attr.mutable_value()->set_i(HightWindow);
    //attrs.push_back(std::move(attr));
  //}
  //{
    //OpDef::AttrDef attr;
    //attr.set_name("WidthWindow");
    //attr.mutable_value()->set_i(HightWindow);
    //attrs.push_back(std::move(attr));
  //}
  //{
    //OpDef::AttrDef attr;
    //attr.set_name("PoolingMode");
    //attr.mutable_value()->set_s("Max");
    //attrs.push_back(std::move(attr));
  //}
  //return Sym("Pooling", {a.node_->output_[0]}, a.node_->type_, "",
         //device, {}, attrs);
  OpDef def = OpDefBuilder("Maxpooling")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .AttrSingle("HightWindow", HightWindow)
                .AttrSingle("WidthWindow", WidthWindow)
                .AttrSingle("PoolingMode", string("Max"))
                .Finalize();
  return Sym(def);
}

Sym Sym::Relu(const Sym& a, string device) {
  //return Sym("Relu", {a.node_->output_[0]}, a.node_->type_, "", device);
  OpDef def = OpDefBuilder("Relu")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Sigmoid(const Sym& a, string device) {
  OpDef def = OpDefBuilder("Sigmoid")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Tanh(const Sym& a, string device) {
  OpDef def = OpDefBuilder("Tanh")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Slice(const Sym& a, int offset, int stride) {
  OpDef def = OpDefBuilder("Slice")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(a.device())
                .AttrSingle("Offset", offset)
                .AttrSingle("Stride", stride)
                .Finalize();
  return Sym(def);
}

std::tuple<Sym, Sym, Sym> Sym::Split3(const Sym& a) {
  OpDef def[3];
  for (int i = 0; i < 3; i++) {
    def[i] = OpDefBuilder("Slice")
               .Input(a.output(0))
               .Dtype(a.type())
               .Device(a.device())
               .AttrSingle("Split", 3)
               .AttrSingle("Index", i)
               .Finalize();
  }
  return std::make_tuple(Sym(def[0]), Sym(def[1]), Sym(def[2]));
}

std::tuple<Sym, Sym, Sym, Sym> Sym::Split4(const Sym& a) {
  OpDef def[4];
  for (int i = 0; i < 4; i++) {
    def[i] = OpDefBuilder("Slice")
               .Input(a.output(0))
               .Dtype(a.type())
               .Device(a.device())
               .AttrSingle("Split", 3)
               .AttrSingle("Index", i)
               .Finalize();
  }
  return std::make_tuple(Sym(def[0]), Sym(def[1]), Sym(def[2]), Sym(def[3]));
}

Sym Sym::Flatten(const Sym& a) {
  //OpDef::AttrDef attr;
  //attr.set_name("ShareMemory");
  //attr.mutable_value()->set_b(true);
  //return Sym("Flatten", { a.node_->output_[0] },
      //a.node_->type_, "", a.node_->device_, {}, {attr});
  OpDef def = OpDefBuilder("Flatten")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(a.device())
                .AttrSingle("ShareMemory", true)
                .Finalize();
  return Sym(def);
}

Sym Sym::Reshape(const Sym& a, const std::vector<int>& shape) {
  //OpDef::AttrDef attr;
  //attr.set_name("ShareMemory");
  //attr.mutable_value()->set_b(true);
  //return Sym("Reshape", { a.node_->output_[0] },
      //a.node_->type_, "", a.node_->device_, shape, {attr});
  OpDef def = OpDefBuilder("Reshape")
                .Input(a.output(0))
                .Dtype(a.type())
                .Device(a.device())
                .Shape(shape)
                .AttrSingle("ShareMemory", true)
                .Finalize();
  return Sym(def);
}

Sym Sym::SoftmaxEntropyLogits(const Sym& a, const Sym& b, string device) {
  //return Sym("SoftmaxEntropyLogits",
      //{ a.node_->output_[0], b.node_->output_[0] },
        //a.node_->type_, "", device);
  OpDef def = OpDefBuilder("SoftmaxEntropyLogits")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::SoftmaxEntropyLoss(const Sym&a, const Sym& b, string device) {
  //return Sym("SoftmaxEntropyLoss",
      //{ a.node_->output_[0], b.node_->output_[0] },
        //a.node_->type_, "", device);
  OpDef def = OpDefBuilder("SoftmaxEntropyLoss")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Equal(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == 1 &&
        b.output_size() == 1);
  //Sym s("Equal", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Equal")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Add(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == 1 &&
        b.output_size() == 1);
  //Sym s("Add", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Add")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Sub(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == 1 &&
        b.output_size() == 1);
  //Sym s("Sub", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Sub")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Mul(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == b.output_size());
  CHECK(a.output_size() == 1);
  //Sym s("Mul", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("Mul")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::MatMul(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == 1 &&
        b.output_size() == 1);
  //Sym s("MatMul", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("MatMul")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::EmbeddingLookup(const Sym& a, const Sym& b, string device) {
  CHECK(a.type() == b.type());
  CHECK(a.output_size() == 1 &&
        b.output_size() == 1);
  //Sym s("EmbeddingLookup", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  OpDef def = OpDefBuilder("EmbeddingLookup")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Conv(const Sym& a, const Sym& b, const Sym& c, string device) {
  CHECK(b.op_name() == "Variable");
  CHECK(c.op_name() == "Variable");
  CHECK(a.type() == b.type() &&
        b.type() == c.type());
  //return Sym("Conv",
      //{a.node_->output_[0], b.node_->output_[0], c.node_->output_[0]},
      //a.node_->type_, "", device);
  OpDef def = OpDefBuilder("Conv")
                .Input(a.output(0))
                .Input(b.output(0))
                .Input(c.output(0))
                .Dtype(a.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::FullyConnected(const Sym& x, const Sym& w, const Sym& b, string device) {
  CHECK(w.op_name() == "Variable");
  CHECK(b.op_name() == "Variable");
  CHECK(x.type() == w.type() &&
        x.type() == b.type());
  CHECK(x.output_size() == 1 &&
        w.output_size() == 1 &&
        b.output_size() == 1);
  //return Sym("FullyConnected", {x.node_->output_[0], w.node_->output_[0], b.node_->output_[0]},
             //x.node_->type_, "", device, {}, {});
  OpDef def = OpDefBuilder("FullyConnected")
                .Input(x.output(0))
                .Input(w.output(0))
                .Input(b.output(0))
                .Dtype(x.type())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::LSTM(const Sym& a, const Sym& b, int layer, int hidden, string device) {
  CHECK(b.op_name() == "Variable");
  CHECK(a.type() == b.type());
  //OpDef::AttrDef layer_attr;
  //layer_attr.set_name("num_layers");
  //layer_attr.mutable_value()->set_i(layer);
  //OpDef::AttrDef hidden_attr;
  //hidden_attr.set_name("hidden_size");
  //hidden_attr.mutable_value()->set_i(hidden);
  //return Sym("LSTM",
      //{a.node_->output_[0], b.node_->output_[0]},
      //a.node_->type_, "", device, {}, {layer_attr, hidden_attr});
  OpDef def = OpDefBuilder("LSTM")
                .Input(a.output(0))
                .Input(b.output(0))
                .Dtype(a.type())
                .Device(device)
                .AttrSingle("num_layers", layer)
                .AttrSingle("hidden_size", hidden)
                .Finalize();
  return Sym(def);
}

Sym Sym::Concat(const vector<Sym>& syms, string device) {
  vector<string> inputs;
  for (auto& s : syms) {
    CHECK(s.output_size() == 0);
    CHECK(s.type() == syms[0].type());
    CHECK(s.device() == syms[0].device());
    inputs.push_back(s.output(0));
  }
  OpDef def = OpDefBuilder("Concat")
                .Input(inputs)
                .Dtype(syms[0].type())
                .Device(syms[0].device())
                .Finalize();
  return Sym(def);
}
  //Sym(const string& op_name, const string& input,
      //const vector<Sym>& variables = {},
      //const float lr = 1,
      //const float clip = 0,
      //const int iters = 1,
      //const string& projections = "");

Sym Sym::Optimizer(const Sym& a) {
  CHECK(a.output_size() == 1);
  //Sym s("Optimizer", a.node_->output_[0]);
  //return s;
  OpDef def = OpDefBuilder("Optimizer")
                .Input(a.output(0))
                .Dtype(a.type())
                .AttrSingle("Learning_rate", 1)
                .AttrSingle("Iters", 1)
                .Finalize();
  return Sym(def);
}

//filler operation
Sym::ATTRIBUTE Sym::Ones() {
  vector<OpDef::AttrDef> vec(1);
  vec[0].set_name("const_value");
  vec[0].mutable_value()->set_f(1.f);
  return std::make_pair("ConstantFiller", vec);
}

Sym::ATTRIBUTE Sym::Zeros() {
  vector<OpDef::AttrDef> vec(1);
  vec[0].set_name("const_value");
  vec[0].mutable_value()->set_f(0.f);
  return std::make_pair("ConstantFiller", vec);
}

Sym::ATTRIBUTE Sym::Const(float c) {
  vector<OpDef::AttrDef> vec(1);
  vec[0].set_name("const_value");
  vec[0].mutable_value()->set_f(c);
  return std::make_pair("ConstantFiller", vec);
}

Sym::ATTRIBUTE Sym::UniformNormalizer(int stride) {
  vector<OpDef::AttrDef> vec(1);
  vec[0].set_name("stride");
  vec[0].mutable_value()->set_i(stride);
  return std::make_pair("UniformNormalizer", vec);
}

Sym::ATTRIBUTE Sym::Uniform(float minval, float maxval) {
  vector<OpDef::AttrDef> vec(2);
  vec[0].set_name("minval");
  vec[0].mutable_value()->set_f(minval);
  vec[1].set_name("maxval");
  vec[1].mutable_value()->set_f(maxval);
  return std::make_pair("Uniform", vec);
}

Sym::ATTRIBUTE Sym::Xavier() {
  vector<OpDef::AttrDef> vec;
  return std::make_pair("Xavier", vec);
}

Sym::ATTRIBUTE Sym::NormalRandom() {
  vector<OpDef::AttrDef> vec;
  return std::make_pair("Normal", vec);
}

Sym::ATTRIBUTE Sym::BinaryReader(const string& filename) {
  vector<OpDef::AttrDef> vec(1);
  vec[0].set_name("filename");
  vec[0].mutable_value()->set_s(filename);
  return std::make_pair("BinaryReader", vec);
}

Sym Sym::Optimizer(const Sym& a, vector<Sym> variables,
    float lr, float clip, int iters, const string& projection) {
  //CHECK(variables.size() > 0);
  CHECK(iters > 0);
  CHECK(a.output_size() == 1);
  //Sym s("Optimizer", a.node_->output_[0],
      //variables, lr, clip, iters, projection);
  //return s;
  vector<string> vars;
  for (auto& v : variables)
    vars.push_back(v.output(0));
  OpDef def = OpDefBuilder("Optimizer")
                .Input(a.output(0))
                .Dtype(a.type())
                .AttrList("Vars", vars)
                .AttrSingle("Learning_rate", lr)
                .AttrSingle("Clip", clip)
                .AttrSingle("Iters", iters)
                .AttrSingle("Projection", projection)
                .AttrSingle("Solver", string("SGD"))
                .Finalize();
  return Sym(def);
}

Sym& Sym::operator= (const Sym& sym) {
  this->node_ = sym.node_; 
  return *this;
}

void Sym::print() {
  //hack here
  int length = 1;
  CHECK_NOTNULL(node_.get());
  CHECK_NOTNULL(data());
  for (int dim : shape(0))
    length *= dim;
  if (type() == DT_FLOAT) {
    for (int i = 0; i < std::min(length, 10); i++) {
      LOG(INFO) << "[" << i << "]:\t"
                << std::fixed << std::setprecision(15)
                << ((float*)data())[i];
    }
  }
}

const void* Sym::eval() const {
  //hack here
  //currently, eval only support single element
  int length = 1;
  CHECK_NOTNULL(node_.get());
  CHECK_NOTNULL(data());
  for (int dim : shape(0))
    length *= dim;
  CHECK(length == 1);
  return data();
}

void Sym::DumpGraph() {
  //C_DumpGraph(C_GetDefaultDG());
}


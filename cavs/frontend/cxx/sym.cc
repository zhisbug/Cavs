#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/c_api.h"
#include "cavs/proto/devices.pb.h"
#include "cavs/util/op_def_builder.h"
#include "cavs/util/logging.h"

#include <algorithm>
#include <iomanip>

using std::vector;

#define _genOutputName(ret, op_name)                    \
    do {                                                \
      static int id = 0;                                \
      ret = op_name + std::to_string(id_++) + "_"       \
                    + std::to_string(id++);             \
    } while(0)

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

int Sym::id_ = 0;

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

template <>
Sym::Sym<float> (float c) {
  //OpDef::AttrDef attr;
  //attr.set_name("init");
  //attr.mutable_value()->set_f(c);
  //new (this)Sym("ConstOp", {}, C_FLOAT, "", "GPU", {1}, {attr});
  string out;
  _genOutputName(out, "Variable");
  vector<int> shape = {1};
  OpDef def = OpDefBuilder("ConstOp")
                .Output(out)
                .Dtype(DT_FLOAT)
                .Device("GPU")
                .Shape(shape)
                .AttrSingle("init", c)
                .Finalize();
  new (this)Sym(def);
}

Sym Sym::Variable(DataType type, const vector<int>& shape,
    const ATTRIBUTE& filler, string device) {
  CHECK(shape.size() > 0);
  string out;
  _genOutputName(out, "Variable");
  OpDef def = OpDefBuilder("Variable")
                .Output(out)
                .Dtype(type)
                .Label(filler.first)
                .Device(device)
                .Shape(shape)
                .Attr(filler.second)
                .Finalize();
  //return Sym("Variable", {}, type, filler.first, device, shape, {filler.second});
  return Sym(def);
}

Sym Sym::Placeholder(DataType type, const vector<int>& shape,
    string device) {
  CHECK(shape.size() > 0);
  string out;
  _genOutputName(out, "Placeholder");
  OpDef def = OpDefBuilder("Placeholder")
                .Output(out)
                .Dtype(type)
                .Device(device)
                .Shape(shape)
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
  string out;
  _genOutputName(out, "MnistInput");
  OpDef def = OpDefBuilder("MnistInput")
                .Output(out)
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
  string out;
  _genOutputName(out, "Data");
  OpDef def = OpDefBuilder("Data")
                .Output(out)
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
  string out;
  _genOutputName(out, "DDV");
  OpDef def = OpDefBuilder("DDV")
                .Output(out)
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
  CHECK(a.def().output_size() == 1);
  //Sym s("Abs", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Abs");
  OpDef def = OpDefBuilder("Abs")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Argmax(const Sym& a, int axis, string device) {
  CHECK(a.def().output_size() == 1);
  //OpDef::AttrDef attr;
  //attr.set_name("axis");
  //attr.mutable_value()->set_i(axis);
  //Sym s("Argmax", {a.node_->output_[0]},
        //a.node_->type_, "", device, {}, {attr});
  //return s;
  string out;
  _genOutputName(out, "Argmax");
  OpDef def = OpDefBuilder("Argmax")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Square(const Sym& a, string device) {
  CHECK(a.def().output_size() == 1);
  //Sym s("Square", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Square");
  OpDef def = OpDefBuilder("Square")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Reduce_mean(const Sym& a, string device) {
  CHECK(a.def().output_size() == 1);
  //Sym s("Reduce_mean", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Reduce_mean");
  OpDef def = OpDefBuilder("Reduce_mean")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Reduce_sum(const Sym& a, string device) {
  CHECK(a.def().output_size() == 1);
  //Sym s("Reduce_sum", {a.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Reduce_sum");
  OpDef def = OpDefBuilder("Reduce_sum")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
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
  string out;
  _genOutputName(out, "Maxpooling");
  OpDef def = OpDefBuilder("Maxpooling")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .AttrSingle("HightWindow", HightWindow)
                .AttrSingle("WidthWindow", WidthWindow)
                .AttrSingle("PoolingMode", string("Max"))
                .Finalize();
  return Sym(def);
}

Sym Sym::Relu(const Sym& a, string device) {
  //return Sym("Relu", {a.node_->output_[0]}, a.node_->type_, "", device);
  string out;
  _genOutputName(out, "Relu");
  OpDef def = OpDefBuilder("Relu")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}


Sym Sym::Flatten(const Sym& a) {
  //OpDef::AttrDef attr;
  //attr.set_name("ShareMemory");
  //attr.mutable_value()->set_b(true);
  //return Sym("Flatten", { a.node_->output_[0] },
      //a.node_->type_, "", a.node_->device_, {}, {attr});
  string out;
  _genOutputName(out, "Flatten");
  OpDef def = OpDefBuilder("Flatten")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(a.def().device())
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
  string out;
  _genOutputName(out, "Reshape");
  OpDef def = OpDefBuilder("Reshape")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(a.def().device())
                .Shape(shape)
                .AttrSingle("ShareMemory", true)
                .Finalize();
  return Sym(def);
}

Sym Sym::SoftmaxEntropyLogits(const Sym& a, const Sym& b, string device) {
  //return Sym("SoftmaxEntropyLogits",
      //{ a.node_->output_[0], b.node_->output_[0] },
        //a.node_->type_, "", device);
  string out;
  _genOutputName(out, "SoftmaxEntropyLogits");
  OpDef def = OpDefBuilder("SoftmaxEntropyLogits")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::SoftmaxEntropyLoss(const Sym&a, const Sym& b, string device) {
  //return Sym("SoftmaxEntropyLoss",
      //{ a.node_->output_[0], b.node_->output_[0] },
        //a.node_->type_, "", device);
  string out;
  _genOutputName(out, "SoftmaxEntropyLoss");
  OpDef def = OpDefBuilder("SoftmaxEntropyLoss")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Equal(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //Sym s("Equal", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Equal");
  OpDef def = OpDefBuilder("Equal")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Add(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //Sym s("Add", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Add");
  OpDef def = OpDefBuilder("Add")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Sub(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //Sym s("Sub", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Sub");
  OpDef def = OpDefBuilder("Sub")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Mul(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == b.def().output_size());
  CHECK(a.def().output_size() == 1);
  //Sym s("Mul", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "Mul");
  OpDef def = OpDefBuilder("Mul")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::MatMul(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //Sym s("MatMul", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "MatMul");
  OpDef def = OpDefBuilder("MatMul")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::EmbeddingLookup(const Sym& a, const Sym& b, string device) {
  CHECK(a.def().dtype() == b.def().dtype());
  CHECK(a.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //Sym s("EmbeddingLookup", {a.node_->output_[0], b.node_->output_[0]},
        //a.node_->type_, "", device);
  //return s;
  string out;
  _genOutputName(out, "EmbeddingLookup");
  OpDef def = OpDefBuilder("EmbeddingLookup")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::Conv(const Sym& a, const Sym& b, const Sym& c, string device) {
  CHECK(b.op_name() == "Variable");
  CHECK(c.op_name() == "Variable");
  CHECK(a.def().dtype() == b.def().dtype() &&
        b.def().dtype() == c.def().dtype());
  //return Sym("Conv",
      //{a.node_->output_[0], b.node_->output_[0], c.node_->output_[0]},
      //a.node_->type_, "", device);
  string out;
  _genOutputName(out, "Conv");
  OpDef def = OpDefBuilder("Conv")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Input(c.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::FullyConnected(const Sym& x, const Sym& w, const Sym& b, string device) {
  CHECK(w.op_name() == "Variable");
  CHECK(b.op_name() == "Variable");
  CHECK(x.def().dtype() == w.def().dtype() &&
        x.def().dtype() == b.def().dtype());
  CHECK(x.def().output_size() == 1 &&
        w.def().output_size() == 1 &&
        b.def().output_size() == 1);
  //return Sym("FullyConnected", {x.node_->output_[0], w.node_->output_[0], b.node_->output_[0]},
             //x.node_->type_, "", device, {}, {});
  string out;
  _genOutputName(out, "FullyConnected");
  OpDef def = OpDefBuilder("FullyConnected")
                .Input(x.def().output(0))
                .Input(w.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(x.def().dtype())
                .Device(device)
                .Finalize();
  return Sym(def);
}

Sym Sym::LSTM(const Sym& a, const Sym& b, int layer, int hidden, string device) {
  CHECK(b.op_name() == "Variable");
  CHECK(a.def().dtype() == b.def().dtype());
  //OpDef::AttrDef layer_attr;
  //layer_attr.set_name("num_layers");
  //layer_attr.mutable_value()->set_i(layer);
  //OpDef::AttrDef hidden_attr;
  //hidden_attr.set_name("hidden_size");
  //hidden_attr.mutable_value()->set_i(hidden);
  //return Sym("LSTM",
      //{a.node_->output_[0], b.node_->output_[0]},
      //a.node_->type_, "", device, {}, {layer_attr, hidden_attr});
  string out;
  _genOutputName(out, "LSTM");
  OpDef def = OpDefBuilder("LSTM")
                .Input(a.def().output(0))
                .Input(b.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .Device(device)
                .AttrSingle("num_layers", layer)
                .AttrSingle("hidden_size", hidden)
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
  CHECK(a.def().output_size() == 1);
  //Sym s("Optimizer", a.node_->output_[0]);
  //return s;
  string out;
  _genOutputName(out, "Optimizer");
  OpDef def = OpDefBuilder("Optimizer")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .AttrSingle("learning_rate", 1)
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
  CHECK(a.def().output_size() == 1);
  //Sym s("Optimizer", a.node_->output_[0],
      //variables, lr, clip, iters, projection);
  //return s;
  vector<string> vars;
  for (auto& v : variables)
    vars.push_back(v.def().output(0));
  string out;
  _genOutputName(out, "Optimizer");
  OpDef def = OpDefBuilder("Optimizer")
                .Input(a.def().output(0))
                .Output(out)
                .Dtype(a.def().dtype())
                .AttrList("Vars", vars)
                .AttrSingle("learning_rate", lr)
                .AttrSingle("clip", clip)
                .AttrSingle("Iters", iters)
                .AttrSingle("Projection", projection)
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
  for (int dim : shape(0))
    length *= dim;
  if (def().dtype() == DT_FLOAT) {
    for (int i = 0; i < std::min(length, 10); i++) {
      LOG(INFO) << "[" << i << "]:\t"
                << std::fixed << std::setprecision(15)
                << ((float*)node_->raw_data)[i];
    }
  }
}

void* Sym::eval() {
  //hack here
  //currently, eval only support single element
  int length = 1;
  CHECK_NOTNULL(node_.get());
  for (int dim : shape(0))
    length *= dim;
  CHECK(length == 1);
  CHECK(node_->raw_data);
  return node_->raw_data;
}

void Sym::DumpGraph() {
  C_DumpGraph(C_GetDefaultDG());
}


#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
//#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_util.h"

#include <string>

using std::string;

namespace backend {

using ::midend::Tensor;

class IOOpBase: public OpImpl {
 public:
  explicit IOOpBase(const OpDef& def) : OpImpl(def) {}
};

#define TRAINING_LABEL_MAGIC 0x0801
#define TRAINING_IMAGE_MAGIC 0x0803
#define bswap(x) __builtin_bswap32(x)
struct LabelFileDescriptor {
  int magic;
  int N;
  void b2l() {
    magic = bswap(magic);
    N = bswap(N);
  }
  int bytes() {
    return N;
  }
  string DebugInfo() {
    return "magic: "    + std::to_string(magic)
          + "\titems: " + std::to_string(N);
  }
};

struct ImageFileDescriptor {
  int magic;
  int N;
  int H;
  int W;
  void b2l(){
    magic = bswap(magic);
    N = bswap(N);
    H = bswap(H);
    W = bswap(W);
  }
  int bytes() {
    return N*H*W; 
  }
  string DebugInfo() {
    return "magic: "       + std::to_string(magic)
          + "\timages: "   + std::to_string(N)
          + "\trows: "     + std::to_string(H)
          + "\tcolumns: "  + std::to_string(W);
  }
};

const static string train_image_name = "train-images.idx3-ubyte";
const static string train_label_name = "train-labels.idx1-ubyte";

class MnistInputOp : public IOOpBase {
 public:
  explicit MnistInputOp(const OpDef& def) :
    IOOpBase(def), image_buf_(NULL), label_buf_(NULL),
    image_curr_idx_(-1), label_curr_idx_(-1) {
    batch_ = GetSingleArg<int>(def, "Batch");
    dir_ = GetSingleArg<string>(def, "ImageDir");
    string source = GetSingleArg<string>(def, "Source");
    if (source == "Image")
      image_ = true;
    else if (source == "Label")
      image_ = false;
    else
      LOG(FATAL) << "Image or Label not specified";
    VLOG(V_DEBUG) << op_def_.DebugString();
  }
  void Compute(OpContext* context) override;

 private:
  int batch_;
  string dir_;
  bool image_;
  float* image_buf_;
  float* label_buf_;
  ImageFileDescriptor image_desc_;
  LabelFileDescriptor label_desc_;
  int image_curr_idx_;
  int label_curr_idx_;
};

void MnistInputOp::Compute(OpContext* context) {
  //static int steps = 0;
  if (image_) {
    Tensor* image = context->Output(0);
    if (!image_buf_) {
      FILE *image_fp;
      if (!(image_fp = fopen((dir_+"/"+train_image_name).c_str(), "rb"))) {
        LOG(FATAL) << "can not open image file:" << dir_ + "/" + train_image_name;
      }
      if (fread(&image_desc_, sizeof(image_desc_), 1, image_fp) != 1) {
        LOG(FATAL) << "Invalid image dataset header";  
      } 
      image_desc_.b2l();
      unsigned char* image_raw_buf = (unsigned char*)malloc(image_desc_.bytes());
      image_buf_ = (float*)malloc(sizeof(float)*image_desc_.bytes());
      if (fread(image_raw_buf, sizeof(unsigned char), image_desc_.bytes(), image_fp)
          != image_desc_.bytes()) {
        LOG(FATAL) << "Read image dataset error";
      }
      for (int i = 0; i < image_desc_.bytes(); i++) {
        image_buf_[i] = (float)(image_raw_buf[i]) / 255.f;
      }
      free(image_raw_buf);
      fclose(image_fp);
    }
    int next_idx = context->GetRound() % (image_desc_.N/batch_);
    if (image_curr_idx_ != next_idx) {
      checkCudaError(cudaMemcpy(image->mutable_data<float>(),
            image_buf_+next_idx*batch_*image_desc_.H*image_desc_.W, 
            batch_*image_desc_.H*image_desc_.W*sizeof(float),
            cudaMemcpyHostToDevice));
      image_curr_idx_ = next_idx;
    }
  }else {
    Tensor* label = context->Output(0);
    if (!label_buf_) {
      FILE *label_fp;
      if (!(label_fp = fopen((dir_ + "/" + train_label_name).c_str(), "rb"))) {
        LOG(FATAL) << "can not open label file" << dir_ + "/" + train_label_name;
      }
      if (fread(&label_desc_, sizeof(label_desc_), 1, label_fp) != 1) {
        LOG(FATAL) << "Invalid label dataset header";  
      } 
      label_desc_.b2l();
      unsigned char* label_raw_buf = (unsigned char*)malloc(label_desc_.bytes());
      label_buf_ = (float*)malloc(sizeof(float)*label_desc_.bytes());
      if (fread(label_raw_buf, sizeof(unsigned char), label_desc_.bytes(), label_fp)
          != label_desc_.bytes()) {
        LOG(FATAL) << "Read label dataset error";
      }
      for (int i = 0; i < label_desc_.bytes(); i++) {
        label_buf_[i] = (float)(label_raw_buf[i]);
      }
      free(label_raw_buf);
      fclose(label_fp);
    }
    int next_idx = context->GetRound() % (label_desc_.N/batch_);
    if (label_curr_idx_ != next_idx) {
      checkCudaError(cudaMemcpy(label->mutable_data<float>(),
            label_buf_+next_idx*batch_, 
            batch_*sizeof(float),
            cudaMemcpyHostToDevice));
      label_curr_idx_ = next_idx;
    }
    //label->DebugNumerical<float>();
  }
}

REGISTER_OP_IMPL_BUILDER(Key("MnistInput").Device("GPU"), MnistInputOp);

} //namespace backend;

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void MilLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_instance = bottom.size();
  for (int i = 0 ;i < num_instance; ++i) {
    CHECK_EQ(bottom[i]->count(), bottom[0]->count());
  }
  
}

template <typename Dtype>
void MilLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MilLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  if (p_x.size() < bottom[0]->count()) p_x.resize(count);
  
  vector<const Dtype*> p_x_i_j;
  p_x_i_j.clear(); 
  
  for (int j = 0; j < num_instance; ++j) {
    p_x_i_j.push_back(bottom[j]->cpu_data());
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int i = 0; i < count; ++i){
    p_x[i] = Dtype(1);
    for (int j = 0; j < num_instance; ++j) {
        p_x[i] *= 1 - p_x_i_j[j][i];
    }
    p_x[i] = 1 - p_x[i];
    top_data[i] = p_x[i];
  }
}

template <typename Dtype>
void MilLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MilLayer);
#endif

INSTANTIATE_CLASS(MilLayer);
REGISTER_LAYER_CLASS(Mil);

}  // namespace caffe

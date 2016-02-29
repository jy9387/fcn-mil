#include <algorithm>
#include <cfloat>
#include <vector>
#include <memory>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MilLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_top_vec_.clear();
  // the last bottom is label
  num_instance = bottom.size() - 1;
  p_x = new Dtype[bottom[0]->count()];
  vector<Blob<Dtype>* > temp;
  for (int i = 0; i < num_instance; ++i) {
    temp.clear();
    temp.push_back(bottom[i]);
    sigmoid_bottom_vec_.push_back(temp);
    temp.clear();
    if (i == 0) {
      temp.push_back(sigmoid_output_0.get());
    } else if (i == 1){
      temp.push_back(sigmoid_output_1.get());
    }
    sigmoid_top_vec_.push_back(temp);
  }
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_[0], sigmoid_top_vec_[0]);
}

template <typename Dtype>
void MilLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < num_instance; i++) {
    CHECK_EQ(bottom[0]->count(), bottom[i]->count())<<"MilLossLayer must have bottom with same shape";
  }
  LossLayer<Dtype>::Reshape(bottom, top);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_[0], sigmoid_top_vec_[0]);
}

template <typename Dtype>
void MilLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<<"flag0";
  for (int i = 0; i < num_instance; i++) {
    sigmoid_bottom_vec_[i][0] = bottom[i];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_[i], sigmoid_top_vec_[i]);
  }
  Dtype loss = Dtype(0);
  Dtype temp_loss_neg = Dtype(0);
  Dtype temp_loss_pos = Dtype(0);
  int count = bottom[0]->count();
  int count_pos = 0, count_neg = 0;

  const Dtype* target = bottom[num_instance]->cpu_data();
  for (int i = 0; i < count; i++)
    if (target[i])count_pos++;else count_neg++;
  for (int i = 0; i < count; ++i) {
    p_x[i] = Dtype(1);
    for (int j = 0; j < num_instance; ++j) {
      p_x[i] *= 1 - sigmoid_top_vec_[j][0]->cpu_data()[i];
    }
    p_x[i] = 1 - p_x[i];
    if (target[i])
      temp_loss_pos -= target[i] * log(p_x[i]) + (1 - target[i]) * (1 - p_x[i]);
    else
      temp_loss_neg -= target[i] * log(p_x[i]) + (1 - target[i]) * (1 - p_x[i]);
  }
  loss = temp_loss_neg * count_pos / count + temp_loss_pos * count_neg / count;
  top[0]->mutable_cpu_data()[0] = loss;
  LOG(INFO)<<"loss: "<<loss;
}

template <typename Dtype>
void MilLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(INFO)<<"INTO BACKWARD";
  if (propagate_down[num_instance]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    int count_pos = 0, count_neg = 0;
    const Dtype* target = bottom[num_instance]->cpu_data();
    for (int i = 0; i < count; i++)
      if (target[i])count_pos++;else count_neg++;

    const int num = bottom[0]->num();
    vector<Dtype*> bottom_diff; bottom_diff.clear();// = bottom[0]->mutable_cpu_diff();
    vector<const Dtype*> p_x_ij; p_x_ij.clear();
    for (int j = 0; j < num_instance; ++j) {
      p_x_ij.push_back(sigmoid_top_vec_[j][0]->cpu_data());
      bottom_diff.push_back(bottom[j]->mutable_cpu_diff());
    }
    
    for (int j = 0; j < num_instance; ++j) {
      for (int i = 0; i < count; ++i) {
        if (target[i]) {
          bottom_diff[j][i] = count_neg / (count_neg + count_pos) * (p_x_ij[j][i] - target[i] / (1 - p_x[i]));
        } else {
          bottom_diff[j][i] = count_pos / (count_neg + count_pos) * (p_x_ij[j][i] - target[i] / (1 - p_x[i]));
        }
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //caffe_scal(count, loss_weight / num, bottom_diff);
    LOG(INFO)<<"OUT BACKWARD";
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(MilLossLayer, Backward);
#endif

INSTANTIATE_CLASS(MilLossLayer);
REGISTER_LAYER_CLASS(MilLoss);

}  // namespace caffe

#include <vector>

#include "caffe/layers/conv_layer.hpp"

// MINSOO haha
#include <fstream>
#include <iostream>
#include "minsoo/Statis.hpp"
extern bool log_report;
bool statis_on;
bool mult_dump(false);
Statis<float>* statis;
Batch<float>* batch;
int batch_size = 100;
int num_clayer = 2;
int num_ilayer = 2;

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      if (batch->reportID() == 0 && batch->returnInfer(n)->reportID() == 1 && n == 1) {
        mult_dump = true; 
      }
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      mult_dump = false;
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }

  // MINSOO status gathering
  if (statis_on) {
    static int count_conv = 0;
    for (int h = 0; h < bottom.size(); h++) { //probably for colors
      if (h != 0) {
        perror("conv_layer.cpp:58: h is not 0");
        exit(EXIT_FAILURE);
      }
      Dtype* top_data = top[h]->mutable_cpu_data();
      for (int n = 0; n < this->num_; n++) {
        Infer<float>* curr_infer = batch->returnInfer(n);
        ConvLayer<float>* layer = new ConvLayer<float>(count_conv, \
                                          this->conv_out_channels_);
        for (int i = 0; i < this->conv_out_channels_; i++) {
          ConvChannel<float>* channel = new ConvChannel<float>(i, \
                                          this->conv_out_spatial_dim_);
          for (int j = 0; j < this->conv_out_spatial_dim_; j++) {
            channel->addVal(top_data[n*this->top_dim_ + \
                                  i*this->conv_out_spatial_dim_ + j]);  
          }
          layer->addChannel(channel);
        }
        curr_infer->addConvLayer(layer);
      }
    }

    count_conv++;
    if (count_conv >= num_clayer) {
        count_conv = 0;
    }
  }

  /*
  if (log_report) {
      // MINSOO report the values
      static int count_conv = 0;
      count_conv++;
      std::ofstream fout;
      fout.open("conv.log",ios::out | ios::app);
      fout << "\n\n";
      fout << "Beginning the output of Conv at Number " << count_conv << "\n";
      for (int i = 0; i < bottom.size(); i++) {
        int total = top[i]->count();
        Dtype* top_data = top[i]->mutable_cpu_data();
        fout << "\nBeginning Channel " << i << "\n";
        for (int n = 0; n < total; n++) {
          fout << top_data[n] << "\n";
        } 
        fout << "\n";
      }
      fout << "\n\n";
      
      fout << "Beginning the weight of Conv at Number " << count_conv << "\n";
      fout << "kernel_dim_ : " << this->kernel_dim_ << "\n";
      fout << "conv_out_spatial_dim_ : " << this->conv_out_spatial_dim_ << "\n";
      for (int i = 0; i < this->kernel_dim_ ; i++) {
        for (int j = 0; j < this->conv_out_spatial_dim_ ; j++) {
            fout << weight[j + i*this->conv_out_spatial_dim_] << "\t";
        }
        fout << "\n";
      }
      fout << "\n\n";
      fout.close();
    }
 */
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe

// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <chrono>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
//#include "tensorflow/contrib/lite/string_util.h"
//#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#include <opencv2/opencv.hpp>

using namespace cv;

#if 1
#define LOG(x) std::cerr
#define CHECK(x)                  \
  if (!(x)) {                     \
    LOG(ERROR) << #x << "failed"; \
    exit(1);                      \
  }

static std::vector<uint8_t> LoadImageFromFile(char * image_path,
                                              int desired_image_width,
                                              int desired_image_height,
                                              int desired_image_channels)
{
  // Load the image
  auto img = imread(image_path);

  LOG(INFO) << "input image size: " << img.cols << "x" << img.rows << "x" << img.channels() << std::endl;

  auto newSize = Size(desired_image_width, desired_image_width);
  Mat reimg;
  resize(img, reimg, newSize);

  LOG(INFO) << "output image size: " << reimg.cols << "x" << reimg.rows << "x" << reimg.channels() << std::endl;

  std::vector<uint8_t> imgData(img.cols * img.rows * img.channels());

  if (reimg.isContinuous())
  {
    imgData.assign(reimg.datastart, reimg.dataend);
  }
  else
  {
    for (int i = 0; i < reimg.rows; ++i)
    {
      imgData.insert(imgData.end(), reimg.ptr<uint8_t>(i), reimg.ptr<uint8_t>(i)+reimg.cols);
    }
  }

  return imgData;
}


// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

std::string RunInferenceOnImage(char * graph, char * labels, char * image_path) {
  const int num_threads = 1;
  std::string input_layer_type = "float";
  std::vector<int> sizes = {1, 224, 224, 3};

  std::unique_ptr<tflite::FlatBufferModel> model(
      tflite::FlatBufferModel::BuildFromFile(graph));
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph << std::endl;
  }
  LOG(INFO) << "Loaded model " << graph << std::endl;
  model->error_reporter();
  LOG(INFO) << "resolved reporter" << std::endl;

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  LOG(INFO) << "Getting the input tensor" << std::endl;

  int input = interpreter->inputs()[0];

  LOG(INFO) << "Resizing input tensor" << std::endl;

  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes);
  }



  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // Read the label list
  std::vector<std::string> label_strings;
  std::ifstream t;
  t.open(labels);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings.push_back(line);
  }
  t.close();

  // Read the Grace Hopper image.
  const int wanted_width = 224;
  const int wanted_height = 224;
  const int wanted_channels = 3;

  std::vector<uint8_t> image_data =
      LoadImageFromFile(image_path, wanted_width, wanted_height, wanted_channels);

  const float input_mean = 127.5f;
  const float input_std = 127.5f;
  //assert(image_channels >= wanted_channels);
  uint8_t* in = image_data.data();


  LOG(INFO) << "Requesting output node" << std::endl;
  float* out = interpreter->typed_tensor<float>(input);
  if (out == nullptr)
  {
    LOG(INFO) << "output node is null" << std::endl;
  }

  for (int y = 0; y < wanted_height; ++y)
  {
    for (int x = 0; x < wanted_width; ++x)
    {
      for (int c = 0; c < wanted_channels; ++c)
      {
        *out = (*in - input_mean) / input_std;
        out++;
        in++;
      }
    }
  }


  // for (int y = 0; y < wanted_height; ++y) {
  //   const int in_y = (y * wanted_height) / wanted_height;
  //   uint8_t* in_row = in + (in_y * wanted_width * wanted_channels);
  //   float* out_row = out + (y * wanted_width * wanted_channels);
  //   for (int x = 0; x < wanted_width; ++x) {
  //     const int in_x = (x * wanted_width) / wanted_width;
  //     uint8_t* in_pixel = in_row + (in_x * wanted_channels);
  //     float* out_pixel = out_row + (x * wanted_channels);
  //     for (int c = 0; c < wanted_channels; ++c) {
  //       LOG(INFO) << "y/x/c: " << y << "/" << x << "/" << c << std::endl;
  //       out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
  //     }
  //   }
  // }

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke!";
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "TFLite took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << "us.\n";


  float* output = interpreter->typed_output_tensor<float>(0);
  const int output_size = 1000;
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output, output_size, kNumResults, kThreshold, &top_results);

  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;

    ss << index << " " << confidence << "  ";

    // Write out the result as a string
    if (index < label_strings.size()) {
      // just for safety: theoretically, the output is under 1000 unless there
      // is some numerical issues leading to a wrong prediction.
      ss << label_strings[index];
    } else {
      ss << "Prediction: " << index;
    }

    ss << "\n";
  }

  LOG(INFO) << "Predictions: " << ss.str();

  std::string predictions = ss.str();
  std::string result;

  return result;
}
#endif

int main(int argc, char * argv[])
{
  if (argc != 4)
  {
    std::cout << "usage: test graph.lite labels.txt image.jpg" << std::endl;
    return -1;
  }
  std::cout << "Using graph " << argv[1];
  std::cout << " with labels " << argv[2];
  std::cout << " and image " << argv[3] << std::endl;
  RunInferenceOnImage(argv[1], argv[2], argv[3]);
}
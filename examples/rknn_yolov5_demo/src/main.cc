// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include "xsy_yolo.h"

#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

/*-------------------------------------------
  Function:  rknn_init
  Descrition: initial the following processing
    1. Load model file
    2. Query the API version and Driver version of the SDK
    3. Query the input/output infomation of the model, such as input/output channels' number
    4. Initialize the input/output tensors' attributes according to Step3
  Inputs:
    const char* model_path : model fullpath name
  Outputs:
    0: successful,  !=0: error
  Version: V1.0.0
  Update history:
-------------------------------------------*/
// init rga context
src = calloc(1,sizeof(rga_buffer_t));
dst = calloc(1,sizeof(rga_buffer_t));
src_rect = calloc(1, sizeof(im_rect));
dst_rect = calloc(1, sizeof(im_rect));
// rga_buffer_t src;
// rga_buffer_t dst;
// im_rect      src_rect;
// im_rect      dst_rect;
// memset(&src_rect, 0, sizeof(src_rect));
// memset(&dst_rect, 0, sizeof(dst_rect));
// memset(&src, 0, sizeof(src));
// memset(&dst, 0, sizeof(dst));

static rknn_input_output_num io_num;
static rknn_sdk_version version;
static rknn_tensor_attr* input_attrs = NULL;
static rknn_tensor_attr* output_attrs = NULL;
static rknn_input inputs[1];

static int model_channel = 3;
// static int model_width   = 0;  
// static int model_height  = 0;  // this is the img height of the model's input, not the original image!
static RECT model_res;  // this is the model's input resolution, not the original image!

int model_init(const char* model_path) {
    // 初始化模型加载操作
    /* Create the neural network */
    printf("Loading model...\n");
    int            model_data_size = 0;
    unsigned char* model_data = load_model(model_path, &model_data_size);
    ret                       = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    /* Create the neural network end */

    // 获取SDK信息
    /* Get the api version and driver version of the SDK */
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    /* Get the api version and driver version of the SDK */

    // 设置输入输出参数
    /* Get the IO_num of the model */
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    /* Get the IO_num of the model */

    /* Set the input/output tensor attributes according to the IO_num */
    //input_attrs = malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    input_attrs = calloc(io_num.n_input, sizeof(rknn_tensor_attr)); // request memory for input tensor attributes, and init to all 0
    if (NULL == input_attrs) {
        // 内存分配失败的错误处理
        // ...
    }
    for (int i = 0; i < io_num.n_input; i++) {
      (*(input_attrs + i)).index = i;
      ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, (input_attrs + i), sizeof(rknn_tensor_attr));
      if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
      }
      dump_tensor_attr(input_attrs + i);
    }

    output_attrs = calloc(io_num.n_output, sizeof(rknn_tensor_attr)); // request memory for output tensor attributes, and init to all 0
    if (NULL == output_attrs) {
        // 内存分配失败的错误处理
        // ...
    }
    for (int i = 0; i < io_num.n_output; i++) {
      (*(output_attrs + i)).index = i;
      ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, (output_attrs + i), sizeof(rknn_tensor_attr));
      dump_tensor_attr(output_attrs + i);
    }
  /* Set the input/output tensor attributes according to the IO_num */
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
      printf("model is NCHW input fmt\n");
      model_channel     = input_attrs[0].dims[1];
      model_res.height  = input_attrs[0].dims[2];
      model_res.width   = input_attrs[0].dims[3];
    } else {
      printf("model is NHWC input fmt\n");
      model_res.height  = input_attrs[0].dims[1];
      model_res.width   = input_attrs[0].dims[2];
      model_channel     = input_attrs[0].dims[3];
    }

    printf("model input model_height=%d, model_width=%d, model_channel=%d\n", model_height, model_width, model_channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = model_width * model_height * model_channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    return 0; // 或者其他错误码
}

/*-------------------------------------------
  Function:  run_inference
  Descrition: inference processing

  Inputs:

  Outputs:

  Version: V1.0.0
  Update history:
-------------------------------------------*/
int run_inference(const char* image_path) {
    // 执行推理操作
    // ...

    return 0; // 或者其他错误码
}

/*-------------------------------------------
  Function:  rga_resize
  Descrition: resize img with RGA
      You may not need resize when src resulotion equals to dst resulotion
      Resizing operation is depent on the difference 
        between the original img resolution(img_width/height) and the model input resolution(width/height)!!!!
      If they are different, resizing operation is required!
      In order to ensure the compatibility of the code under different models and input images, the following resize code should be retained.
      Resize operation
  Inputs:

  Outputs:

  Version: V1.0.0
  Update history:
-------------------------------------------*/
int rga_resize(){
  void* resize_buf = nullptr;
  if (img_width != model_res.width || img_height != model_res.height) {
    printf("resize with RGA!\n");
    resize_buf = malloc(model_res.height * model_res.width * model_channel);
    memset(resize_buf, 0x00, model_res.height * model_res.width * model_channel);

    src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void*)resize_buf, model_width, model_height, RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      return ret;
    }
    IM_STATUS STATUS = imresize(src, dst);

    // for debug
    cv::Mat resize_img(cv::Size(model_width, model_height), CV_8UC3, resize_buf);
    cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;
  } else {
    inputs[0].buf = (void*)img.data;
  }
  // Resize operation
  return 0
}


void rknn_cleanup() {
    // 清理资源
    // ...
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  int            status     = 0;
  char*          model_name = NULL;
  rknn_context   ctx;
  size_t         actual_size        = 0;
  int            img_width          = 0;
  int            img_height         = 0;
  int            img_channel        = 0;
  const float    nms_threshold      = NMS_THRESH;
  const float    box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;
  int            ret;

  // init rga context
  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect      src_rect;
  im_rect      dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  if (argc != 3) {
    printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    return -1;
  }

  printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  model_name       = (char*)argv[1];
  char* image_name = argv[2];

  printf("Read %s ...\n", image_name);
  cv::Mat orig_img = cv::imread(image_name, 1);
  if (!orig_img.data) {
    printf("cv::imread %s fail!\n", image_name);
    return -1;
  }
  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  img_width  = img.cols;
  img_height = img.rows;
  printf("img width = %d, img height = %d\n", img_width, img_height);

  ret = model_init(model_name);
  if (0 != ret){
    // 出错处理
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  /* Load model and initialization */
  /* Create the neural network */
  // printf("Loading model...\n");
  // int            model_data_size = 0;
  // unsigned char* model_data = load_model(model_name, &model_data_size);
  // ret                       = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  // if (ret < 0) {
  //   printf("rknn_init error ret=%d\n", ret);
  //   return -1;
  // }
  // /* Create the neural network end */

  // /* Get the api version and driver version of the SDK */
  // rknn_sdk_version version;
  // ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  // if (ret < 0) {
  //   printf("rknn_init error ret=%d\n", ret);
  //   return -1;
  // }
  // printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
  // /* Get the api version and driver version of the SDK */

  // /* Get the IO_num of the model */
  // rknn_input_output_num io_num;
  // ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  // if (ret < 0) {
  //   printf("rknn_init error ret=%d\n", ret);
  //   return -1;
  // }
  // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
  // /* Get the IO_num of the model */

  // /* Set the input/output tensor attributes according to the IO_num */
  // rknn_tensor_attr input_attrs[io_num.n_input];
  // memset(input_attrs, 0, sizeof(input_attrs));
  // for (int i = 0; i < io_num.n_input; i++) {
  //   input_attrs[i].index = i;
  //   ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
  //   if (ret < 0) {
  //     printf("rknn_init error ret=%d\n", ret);
  //     return -1;
  //   }
  //   dump_tensor_attr(&(input_attrs[i]));
  // }

  // rknn_tensor_attr output_attrs[io_num.n_output];
  // memset(output_attrs, 0, sizeof(output_attrs));
  // for (int i = 0; i < io_num.n_output; i++) {
  //   output_attrs[i].index = i;
  //   ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
  //   dump_tensor_attr(&(output_attrs[i]));
  // }
  /* Set the input/output tensor attributes according to the IO_num */
  
  int model_channel = 3;
  int model_width   = 0;  // this is the img width of the model's input, not the original image!
  int model_height  = 0;  // this is the img height of the model's input, not the original image!
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    model_channel = input_attrs[0].dims[1];
    model_height  = input_attrs[0].dims[2];
    model_width   = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    model_height  = input_attrs[0].dims[1];
    model_width   = input_attrs[0].dims[2];
    model_channel = input_attrs[0].dims[3];
  }

  printf("model input model_height=%d, model_width=%d, model_channel=%d\n", model_height, model_width, model_channel);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = model_width * model_height * model_channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  // You may not need resize when src resulotion equals to dst resulotion
  // Resizing operation is depent on the difference 
  //   between the original img resolution(img_width/height) and the model input resolution(width/height)!!!!
  // If they are different, resizing operation is required!
  // In order to ensure the compatibility of the code under different models and input images, the following resize code should be retained.
  // Resize operation
  void* resize_buf = nullptr;

  if (img_width != model_width || img_height != model_height) {
    printf("resize with RGA!\n");
    resize_buf = malloc(model_height * model_width * model_channel);
    memset(resize_buf, 0x00, model_height * model_width * model_channel);

    src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void*)resize_buf, model_width, model_height, RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);

    // for debug
    cv::Mat resize_img(cv::Size(model_width, model_height), CV_8UC3, resize_buf);
    cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;
  } else {
    inputs[0].buf = (void*)img.data;
  }
  // Resize operation

  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs)

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = 0;
  }

  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // post process
  float scale_w = (float)model_width / img_width;
  float scale_h = (float)model_height / img_height;

  detect_result_group_t detect_result_group;
  std::vector<float>    out_scales;
  std::vector<int32_t>  out_zps;
  for (int i = 0; i < io_num.n_output; ++i) {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }
  post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, model_height, model_width,
               box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

  // Draw Objects
  char text[256];
  for (int i = 0; i < detect_result_group.count; i++) {
    detect_result_t* det_result = &(detect_result_group.results[i]);
    sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
    printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
           det_result->box.right, det_result->box.bottom, det_result->prop);
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
    putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  imwrite("./out.jpg", orig_img);
  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  // loop test
  int test_count = 10;
  gettimeofday(&start_time, NULL);
  for (int i = 0; i < test_count; ++i) {
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
    post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, model_height, model_width,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
#endif
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }
  gettimeofday(&stop_time, NULL);
  printf("loop count = %d , average run  %f ms\n", test_count,
         (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

  deinitPostProcess();

  // release
  ret = rknn_destroy(ctx);

  if (model_data) {
    free(model_data);
  }

  if (resize_buf) {
    free(resize_buf);
  }

  return 0;
}

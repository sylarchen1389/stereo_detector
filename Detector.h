
#pragma once

#include<iostream>
#include<string>
#include<cmath>
#include<time.h>
#include<algorithm>
#include <numeric>

#include"NvInfer.h"
#include"NvOnnxParser.h"

#include"Utils.h"
#include"EntropyCalibrator.h"
#include<cudnn.h>

#include<opencv2/opencv.hpp>
#include<chrono>


static const int INPUT_CHANNEL=3;      // =1 detect with grey image
static const int CHECK_COUNT = 3;       // output achor box for each yolo layer
static const int BATCH_SIZE = 1;

struct YoloKernel
{
    int width;
    int height;
    float anchors[CHECK_COUNT*2];
};

struct Detection
{
    float bbox[4];
    int classId;
    float prob;
};

class Detector{

public:
    Detector(std::string onnxFile,std::string trtFile,std::string calibFileList,int input_w,int input_h,int num_classes,float yolo_thresh,float nms_thresh,bool use_int8);
    ~Detector();
    std::vector<std::vector<Detection>> doInference(std::vector<cv::Mat>& img);

private:
    void postProcessImg(cv::Mat& img,std::vector<Detection>& detections);
    void doNms(std::vector<Detection>& detections,float nmsThresh);
    std::vector<std::vector<Detection>>interpretOutputTensor(float *tensor,int batchSize);

    Logger logger_;

    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IRuntime* runtime_;

    std::unique_ptr<float[]> inputData_;
    std::unique_ptr<float[]> outputData_;
    cudaStream_t stream_;
    std::unique_ptr<void*[]> buffers_;

    std::vector<YoloKernel> yoloKernel_;

    // param
    int inputW_;
    int inputH_;
    int numClasses_;
    float yoloThresh_;
    float nmsThresh_;

    const YoloKernel yolo1_;
    const YoloKernel yolo2_;
    const YoloKernel yolo3_;

};
#pragma once

#include"NvInfer.h"
#include"NvOnnxParser.h"
#include<iostream>
//#include<string>
#include"NvInferRuntime.h"
#include<vector>
#include<map>
#include<fstream>
#include<iomanip>
#include<assert.h>
#include<numeric>


#include<opencv/highgui.h>
#include<opencv2/opencv.hpp>
//#include<opencv2/gpu/gpu.hpp>


#define CUDA_CHECK(callstr)                                                                    \
{                                                                                          \
    cudaError_t error_code = callstr;                                                      \
    if (error_code != cudaSuccess) {                                                       \
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
        assert(0);                                                                         \
    }                                                                                      \
}



class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cout << "INFO: "; break;
        default: std::cout << "UNKNOWN: "; break;
        }
        std::cout << msg << std::endl;
    }

    Severity reportableSeverity;
};

nvinfer1::ICudaEngine*  engineFromFiles(std::string onnxFile,std::string  trtFile,nvinfer1::IRuntime* runtime,int batchSize,Logger& logger,bool useInt8,bool markOutput,nvinfer1::IInt8EntropyCalibrator* calibrator);
void prepareImage(cv::Mat& img,float* data,int  w,int h,int c,bool cvtColor = true,bool padCenter = true,bool pad = true,bool normalize =true );
float* prepareImage2(cv::Mat& img,cv::Mat& img_pad,float* data,int w,int h,int c,bool cvtColor=true,bool normalize = true);
float* calculate_padding(int orig_width,int orig_height,int  new_width,int new_height);


inline int64_t volume(const nvinfer1::Dims& d )
{
    return std::accumulate(d.d,d.d+d.nbDims,1,std::multiplies<int64_t>());
}
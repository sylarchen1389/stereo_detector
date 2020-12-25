#pragma once 

#include<iostream>
#include<string>

#include"Detector.h"
#include"Utils.h"
#include"Types.h"

#include<opencv2/opencv.hpp>
#include<boost/thread.hpp>
#include<mutex>
#include<string>

using namespace std;

struct TensorRTOption
{
    std::string onnxFile;
    std::string trtFile;
    std::string calibFile;

    bool useInt8;
    int yoloClasses;
    double yoloThresh;
    double yoloNms;
    int maxBatch;
    double timeout;

    int yoloW;
    int yoloH;

    TensorRTOption(){}
};
 

class TensorRT{

public:

    TensorRT(TensorRTOption option);    
    ~TensorRT();

    void drawDetections(cv::Mat &img, std::vector<BoundingBox> &boxes);
    void detect(cv::Mat& img,std::vector<BoundingBox>& bboxes);
private:
    TensorRTOption option_;
    std::vector<BoundingBox> processDetections(std::vector<Detection> &detections,cv::Mat &img);
    void darwDetections(cv::Mat& img,std::vector<BoundingBox> &boxes);


    // detector对象
    std::unique_ptr<Detector> detector_;

    //线程操作
    std::shared_ptr<boost::thread> detectThread_;

    std::vector<cv::Mat> images_to_detect_;
    std::vector<BoundingBox> detect_results_;
     
    int yoloW_;
    int yoloH_;
    int boxMinSizeRatio_ = 0.12;
};

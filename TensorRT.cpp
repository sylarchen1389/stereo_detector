#include"TensorRT.h"
#include"Utils.h"
#include <chrono>

using namespace std;

TensorRT::TensorRT(TensorRTOption option){
    option_ = option;
    yoloH_ = option.yoloH;
    yoloW_ = option.yoloW;
    
    cout<<"tensorrt param ===================================="<<endl;
    cout<<"onnx: "<<option_.onnxFile<<endl;
    cout<<"trt: "<<option_.trtFile<<endl;
    cout<<"calib: "<<option_.calibFile<<endl;
    cout<<"yoloW: "<<yoloW_<<endl;
    cout<<"yoloH: "<<yoloH_<<endl;
    cout<<"classes: "<<option_.yoloClasses<<endl;
    cout<<"thresh: "<<option_.yoloThresh<<endl;
    cout<<"nms: "<<option_.yoloNms<<endl;
    cout<<"useInt8: "<<option_.useInt8<<endl;
    cout<<"tensorrt param ===================================="<<endl;
 
    detector_.reset(new Detector(option_.onnxFile,option_.trtFile,option_.calibFile,yoloW_,yoloH_,
        option_.yoloClasses,option_.yoloThresh,option_.yoloNms,option_.useInt8));
}


TensorRT::~TensorRT(){
    if(detectThread_){
        detectThread_->interrupt();
        detectThread_->join();
    }
}

void TensorRT::detect(cv::Mat& img,std::vector<BoundingBox>& bboxes){

    if(img.cols == 0 || img.rows ==0|| img.empty()){
        cout<<"empty image"<<endl;
        return;
    }

    images_to_detect_.clear();
    images_to_detect_.push_back(img.clone());
    vector<vector<Detection>> detect_results_ = detector_->doInference(images_to_detect_);
    vector<BoundingBox> boxes = processDetections(detect_results_[0],images_to_detect_[0]);

    bboxes.clear();
    for(BoundingBox box:boxes){
        bboxes.push_back(box);
    }
    
}


std::vector<BoundingBox> TensorRT::processDetections(std::vector<Detection> &detections,cv::Mat &img){
    vector<BoundingBox> boxes;
    vector<cv::Mat> rois;
    float* pad = calculate_padding(img.cols,img.rows,yoloW_,yoloH_);
    cout<<"[INFO]pad: "<<pad[0]<<","<<pad[1]<<endl;
    int img_width = img.cols+pad[0]*2;
    int img_height = img.rows + pad[1]*2;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        cout<<"bbox: "<<b[0]<<" "<<b[1]<<" "<<b[2]<<" "<<b[3]<<endl;
        int left  = max((b[0]-b[2]/2.)*img_width- pad[0], 0.0);
        int right = min((b[0]+b[2]/2.)*img_width - pad[0], double(img.cols));
        int top   = max((b[1]-b[3]/2.)*img_height - pad[1], 0.0);
        int bot   = min((b[1]+b[3]/2.)*img_height - pad[1], double(img.rows));
        cout<<"point: "<<left<< " "<<top<<" "<<right<<" "<<bot<<endl;
	    int h = bot - top;
	    int w = right - left;
	    double adder = 0.0;
	    left = max((int)(left - adder * w), 0);
        top = max((int)(top - adder * h), 0);
        right = min((int)(right + adder * w), img.cols);
        bot = min((int)(bot + adder * h), img.rows);

        if (right - left <= img.cols * boxMinSizeRatio_ || bot - top <= img.rows * boxMinSizeRatio_)
            continue;
	    double x = (right + left) / 2. / img.cols;
	    double y = bot / (double)img.rows;
	
        BoundingBox boundingBox;
        boundingBox.probability = item.prob;
        boundingBox.xmin = left;
        boundingBox.ymin = top;
        boundingBox.xmax = right;
        boundingBox.ymax = bot;
        boundingBox.Class = std::to_string(item.classId);

        cv::Rect box(cv::Point(left, top), cv::Point(right, bot));
        cv::Mat roi = img(box);
        
        boxes.push_back(boundingBox);
    }
    
    return boxes;
}

void TensorRT::drawDetections(cv::Mat &img, std::vector<BoundingBox> &boxes){
    for (auto &b : boxes)
    {
        cv::Rect box(cv::Point(b.xmin,b.ymin), cv::Point(b.xmax,b.ymax));
        cv::Scalar boxColor(255, 0, 0);
	    cv::rectangle(img, box, boxColor,2,8,0);
    }
}
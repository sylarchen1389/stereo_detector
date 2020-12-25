#include"Utils.h"
#include<string>
#include<opencv2/core/version.hpp>

using namespace nvinfer1;
using namespace std;

//#define USE_GPU

float* calculate_padding(int orig_width,int orig_height,int  new_width,int new_height)
{
    int new_img_width,new_img_height;
    float *res = new float[3];   //pad_w,pad_h,scale_factor
    if (orig_height>=orig_width)
    {
        res[2] = new_height/orig_height;
        new_img_height = orig_height;
        new_img_width = new_width*orig_height/new_height;
        res[1] = 0;
        res[0] = int((new_img_width-orig_width)/2);
        if(res[0]<0)
            res[0] = 0;
    }
    else
    {
        res[2] = new_width/orig_width;
        new_img_width = orig_width;
        new_img_height = new_height*orig_width/new_width;
        res[0] = 0;
        res[1] = int((new_img_height- orig_height)/2);
        if(res[1]<0)
            res[1] = 0;
    }
    return res;
}

void prepareImage(cv::Mat& img, float *data, int w, int h, int c, bool cvtColor, bool padCenter, bool pad, bool normalize)
{
    float scale = min(float(w) / img.cols, float(h) / img.rows);
    auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);
    if (scaleSize.height < 1 || scaleSize.width < 1)
        pad = false;


    cv::Mat rgb;
    if (pad)
        cv::resize(img, rgb, scaleSize, 0, 0, cv::INTER_NEAREST);
    else
        cv::resize(img, rgb, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    if (cvtColor)
    {
        cv::cvtColor(rgb, rgb, CV_BGR2RGB);
    }


    cv::Mat cropped(h, w, CV_8UC3, cv::Scalar(127, 127, 127));
    if (pad)
    {
        if (padCenter)
        {
            cv::Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
            rgb.copyTo(cropped(rect));
        }
        else
        {
            cv::Rect rect(0, 0, scaleSize.width, scaleSize.height);
            rgb.copyTo(cropped(rect));
        }

    }
    else
    {
        rgb.copyTo(cropped);
    }

    float factor = 1.0;
    if (normalize)
        factor = 1/255.0;
    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, factor);
    else
        cropped.convertTo(img_float, CV_32FC1, factor);

    cv::Mat input_channels[c];
    cv::split(img_float, input_channels);

    int channelLength = h * w;
    for (int i = 0; i < c; ++i) 
    {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
}

float* prepareImage2(cv::Mat& img,cv::Mat& img_pad,float* data,int w,int h,int c,bool cvtColor,bool normalize)
{
    //assert(img.rows%32==0&&img.cols%32==0);
    float* scale = calculate_padding(img.cols,img.rows,w,h);
    cv::Mat rgb;
    cout<<"[INFO] Origin image w,h: "<<img.cols<<" "<<img.rows<<endl;
    cout<<"[INFO] Image scale:"<<int(scale[0])<<" "<<int(scale[1])<<" "<<scale[2]<<endl;
    cv::copyMakeBorder(img,img_pad,scale[1],scale[1],scale[0],scale[0],cv::BORDER_CONSTANT,0);
    // cv::imshow("img_pad",img_pad);
    // while (true)
    // {
    //     if(cv::waitKey(1) == 'q')
    //         break;
    // }
    
    cv::resize(img_pad,rgb,cv::Size(w,h),0,0,cv::INTER_NEAREST);
    //cv::imshow("rgb",rgb);
    // while (true)
    // {
    //     if(cv::waitKey(1) == 'q')
    //         break;
    // }
    if (cvtColor)
    {
        cv::cvtColor(rgb, rgb, CV_BGR2RGB);
    }
     
    float factor;
    if (normalize)
        factor = 1/255.0;
    cv::Mat img_float;
    if (c == 3)
        rgb.convertTo(img_float, CV_32FC3, factor);
    else
        rgb.convertTo(img_float, CV_32FC1, factor);

    cv::Mat input_channels[c];
    cv::split(img_float, input_channels);

    int channelLength = h * w;
    for (int i = 0; i < c; ++i) 
    {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }

    return scale;
}

void onnxToTRTModel(const std::string& modelFile,
                    unsigned int maxBatchSize,
                    IHostMemory*& trtModelStream, Logger &logger, bool useInt8, bool markOutput, IInt8EntropyCalibrator* calibrator)
{
    IBuilder* builder = createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, logger);

    // ifstream
    std::ifstream onnx_file(modelFile.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
   // 使用char 数组存放 onnx文件
    std::vector<char> onnx_buf(file_size);
    
    // 读入
    if(!onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) 
    {
        // 这里的logger是继承的消息发布类，会把消息msg通过std输出到屏幕上
        // 可能有保存log文件的功能
        string msg("failed to open onnx file");
        logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    }

    //解析onnx文件
    // parser nv  封装的类，用来解析onnx文件
    /** 将序列化的ONNX模型解析到TensorRT网络中。
    *这种方法的诊断能力非常有限。如果由于任何原因(例如，不支持的IR版本、不支持的操作集等)解析序列化模型失败。
    *用户有责任拦截并报告错误。若要获得更好的诊断，请使用下面的parseFromFile方法。
    */
    if (!parser->parse(onnx_buf.data(), onnx_buf.size()))
    {
        string msg("failed to parse onnx file");
        logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    }

    //batch_size 
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    if (useInt8 && builder->platformHasFastInt8())
    {
      builder->setInt8Mode(true);
      builder->setInt8Calibrator(calibrator);
      
      //setLayerPrecision(network);
      //setDynamicRange(network);
    }
    else
    {
        builder->setFp16Mode(true);
    }
    builder->setStrictTypeConstraints(true);
    if (markOutput)
    {
        network->markOutput(*network->getLayer(network->getNbLayers()-1)->getOutput(0));
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // serialize the engine, then close everything down
    parser->destroy();
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}


ICudaEngine*  engineFromFiles(std::string onnxFile,string  trtFile,IRuntime* runtime,int batchSize ,Logger& logger,bool useInt8,bool markOutput,IInt8EntropyCalibrator* calibrator)
{
    ICudaEngine *engine;
    fstream file;
    
    file.open(trtFile,ios::binary | ios::in);

    if(!file.is_open())
    {
        std::cerr<<"open trt fail !"<<std::endl;
        IHostMemory* trtModelStream{nullptr};
        onnxToTRTModel(onnxFile, batchSize, trtModelStream, logger, useInt8, markOutput, calibrator);
        assert(trtModelStream != nullptr);

        engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
        assert(engine != nullptr);
        trtModelStream->destroy();

        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream save_file;
        save_file.open(trtFile, std::ios::binary | std::ios::out);

        save_file.write((const char*)data->data(), data->size());
        save_file.close();
    }else
    {
        // 对输入文件定位，param（偏移量，基地址），指针在文件结束位置
        file.seekg(0,ios::end);
        int length = file.tellg();
        file.seekg(0,ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(),length);

        file.close();

        engine = runtime->deserializeCudaEngine(data.get(),length,nullptr);
        assert(engine!=nullptr);
    }
    
    return engine;

}
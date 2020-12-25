#pragma once

// TODO!!!!!!!!!!!!!!!!!!!!!!


#include<cudnn.h>
#include<string>
#include<vector>
#include"NvInfer.h"
#include"Utils.h"
#include<string>

namespace nvinfer1{

class Int8EntropyCalibrator:public IInt8EntropyCalibrator{

public:
    Int8EntropyCalibrator(int  BatchSize,const std::vector<std::vector<float>>& data,const std::string& CalibDataName="",bool readChace = true);   

    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const override{ return mBatchSize;}


private:
    std::string mCalibDataName;
    std::vector<std::vector<float>> mDatas;
	int mBatchSize;

};






}





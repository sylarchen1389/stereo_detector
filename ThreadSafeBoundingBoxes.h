#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <initializer_list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include"Types.h"
#include<vector>

using namespace std;

class ThreadSafeBoundingBoxes
{
private:
    // data_queue访问信号量
    mutable std::mutex mut;
    mutable std::condition_variable data_cond;

    std::vector<BoundingBox> bboxes;
    int change = 0;
public:
    ThreadSafeBoundingBoxes()=default;
    ThreadSafeBoundingBoxes(const ThreadSafeBoundingBoxes&)=delete;
    ThreadSafeBoundingBoxes& operator=(const ThreadSafeBoundingBoxes&)=delete;
    ~ThreadSafeBoundingBoxes()=default;


    bool wirteBBoxws(std::vector<BoundingBox>& detections){
        std::unique_lock<std::mutex>lk(mut);
        bboxes.clear();
        for(BoundingBox box : detections){
            bboxes.push_back(box);
        }
        data_cond.notify_one();
        change = 1;
        return true;
    }

    bool getBBoxes(std::vector<BoundingBox>& detections){
        cout<<"[INFO] get bboxes"<<endl;
        std::unique_lock<std::mutex>lk(mut);
        data_cond.wait(lk,[this]{return change == 1; });
        detections.clear();
        for(BoundingBox box : bboxes){
            detections.push_back(box);
        }
        change = 0;
        return true;
    }
};

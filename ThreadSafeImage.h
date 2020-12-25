#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <initializer_list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;

class ThreadSafeImage
{
private:
    // data_queue访问信号量
    mutable std::mutex mut;
    mutable std::condition_variable data_cond;
    int grayChange = 0;
    int rectifyChange = 0;
    int xyzChange = 0;
    
    cv::Mat grayImageL;
    cv::Mat grayImageR;
    cv::Mat rectifyImageL;
    cv::Mat xyz;
public:
    ThreadSafeImage()=default;
    ThreadSafeImage(const ThreadSafeImage&)=delete;
    ThreadSafeImage& operator=(const ThreadSafeImage&)=delete;
    ~ThreadSafeImage()=default;

    void writeGrayImage(cv::Mat& grayL,cv::Mat& grayR){
        std::unique_lock<std::mutex>lk(mut);
        grayL.copyTo(grayImageL);
        grayR.copyTo(grayImageR);
        grayChange = 1;
        data_cond.notify_one();
    }

    void getGrayImage(cv::Mat& grayL,cv::Mat& grayR){
        std::unique_lock<std::mutex>lk(mut);
        data_cond.wait(lk,[this]{return this->grayImageL.cols!=0&&grayChange; });
        grayImageL.copyTo(grayL);
        grayImageR.copyTo(grayR);
        grayChange = 0;
    }

    
    void writeXYZ(cv::Mat& XYZ){
        std::unique_lock<std::mutex>lk(mut);
        XYZ.copyTo(xyz);
        xyzChange = 1 ;
        data_cond.notify_one();
    }
    void getXYZ(cv::Mat& XYZ){
        std::unique_lock<std::mutex>lk(mut);
        data_cond.wait(lk,[this]{return this->xyz.cols!=0&&xyzChange; });
        xyz.copyTo(XYZ);

    }

    void wirteRectifyImageL(cv::Mat& left){
        std::unique_lock<std::mutex>lk(mut);
        left.copyTo(rectifyImageL);
        rectifyChange = 1;
        data_cond.notify_one();
    }

    void getRectifyImageL(cv::Mat& left){
        std::unique_lock<std::mutex>lk(mut);
        data_cond.wait(lk,[this]{return this->xyz.cols!=0&&rectifyChange; });
        rectifyImageL.copyTo(left);
        rectifyChange = 0;
    }
};


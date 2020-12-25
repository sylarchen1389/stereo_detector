#pragma once 
#include<iostream>
#include<ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include"stereo_detector/BoundingBox.h"
#include"stereo_detector/BoundingBoxes.h"

#include<Types.h>
#include<ThreadSafeImage.h>
#include<ThreadSafeBoundingBoxes.h>
#include<TensorRT.h>
#include<SGM.h>
#include<chrono>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include <mutex>
#include <condition_variable>


class StereoDetectorNode
{

public:
    StereoDetectorNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    StereoDetectorNode():StereoDetectorNode(ros::NodeHandle(),ros::NodeHandle("~")){}
    ~StereoDetectorNode();

    void computeXYZ();
    void computeBBox();
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg1, const sensor_msgs::ImageConstPtr& msg2);
    void rectify(cv::Mat& left,cv::Mat& right);
private:

    void drawDetections(cv::Mat &img, stereo_detector::BoundingBoxes &boxes);

    /*Publisher and Subscriber*/
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport imageTransport_;
    image_transport::SubscriberFilter imageSubscriber1_;
    image_transport::SubscriberFilter imageSubscriber2_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>> sync_;

    ros::Publisher boundingBoxesPublisher_;
    image_transport::Publisher detectionImagePublisher_;
    image_transport::Publisher disparityImagePublisher_;

    /*param*/
    CamParam camOption_;
    SGMOption sgmOption_;
    TensorRTOption trtOption_;

    float baseline_;
    float force_;
    int cxl_;
    int cyl_;
    int cxr_;
    int cyr_;

    cv::Mat Q_;

    int sgmPadding_;
    bool drawDetections_ = true;

    std::shared_ptr<boost::thread> tensorrtThread_;
    std::shared_ptr<boost::thread> sgmThread_;

    /*queue*/
    std::shared_ptr<ThreadSafeBoundingBoxes> bboxes_;
    std::shared_ptr<ThreadSafeImage> images_;
};

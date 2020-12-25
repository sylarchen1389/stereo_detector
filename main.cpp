#include"StereoDetectorNode.h"
#include<chrono>

int main(int argc, char** argv){
    ros::init(argc,argv,"stereo_detector");
    StereoDetectorNode node;
    ros::spin();
    return 0;
}
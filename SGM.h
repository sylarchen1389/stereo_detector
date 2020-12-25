#pragma once 
#include"Types.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <libsgm.h>
#include <vector>

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \


struct SGMOption
{
    int imageWidth;
    int imageHeight;

    int num_paths;
    sint32 disp_size;
	sint32 min_disparity;
    
	sint32 p1;
	sint32 p2_int;
    float32 uniqueness;
    bool subpixel = false;
    sint32 min_disp;
    sint32 LR_max_diff;
    
    float force;
    float baseline;

    int cxl;
    int cxr;
    int cyl;
    int cyr;

	SGMOption() :disp_size(128),num_paths(8), min_disparity(0),p1(10),p2_int(150),uniqueness(0.95f),subpixel(false),LR_max_diff(1) {}
};


class SGM
{
private:
    SGMOption option_;
    std::unique_ptr<sgm::StereoSGM> ssgm_;

    cv::Mat grayImageL_;
    cv::Mat grayImageR_;
    
    cv::Mat disparity_;
    cv::Mat xyz_;

public:
    SGM(SGMOption option);
    ~SGM();
    
    void getXYZ(cv::Mat& grayImageL,cv::Mat& grayImageR,cv::Mat& xyz,cv::Mat& disparityOut);
    void compute3d(cv::Mat& disparity,cv::Mat& xyz,float force,float baseline,float cxl,float cyl);

    cv::Mat Q_;
};



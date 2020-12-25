#pragma once 
#include<iostream>
#include<cstring>
#include <limits>
#include<vector>
// #include <cstdint>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

typedef int8_t			sint8;		
typedef uint8_t			uint8;		
typedef int16_t			sint16;		
typedef uint16_t		uint16;		
typedef int32_t			sint32;		
typedef uint32_t		uint32;		
typedef int64_t			sint64;		
typedef uint64_t		uint64;		
typedef float			float32;	
typedef double			float64;

constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

struct BoundingBox{
    std::string Class;
    float64 probability;
    sint64 xmin;
    sint64 ymin;
    sint64 xmax;
    sint64 ymax;

    sint32 worldX;
    sint32 worldY;
    sint32 worldZ;

    BoundingBox(){}
};



struct CamParam
{
    int imageWidth;
    int imageHeight;

    cv::Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
    cv::Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
    cv::Mat xyz;              //三维坐标

    /*相机内参*/
    cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) <<
        903.619734199452,0,952.971392111169,
        0,954.715461789098,542.112445049920,
        0,0,1);
    cv::Mat distCoeffL = (cv::Mat_<double>(4, 1) << 
        -0.309957088078872,0.0837818975629608,0,0);
    
    cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << 
        908.100241725721,0,928.649652459136,
        0,959.573864137008,550.978004165741,
        0,0,1);
    cv::Mat distCoeffR = (cv::Mat_<double>(5, 1) << 
        -0.306614842052119,0.0746103582700683,0,0);
    
    /*相机外参*/
    cv::Mat T = (cv::Mat_<double>(3, 1) << -181.951646922374,-3.78508082860841,2.56476205178126);//T平移向量
    //Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
    cv::Mat R = (cv::Mat_<double>(3,3)<<
        0.999877619025646,0.0125714116902012,0.00931163679068092,
        -0.0125563015687748,0.999919756227700,-0.00167940360014598,
        -0.00933200206386997,0.00156227835345441,0.999955235509983);//R 旋转矩阵
    
};
